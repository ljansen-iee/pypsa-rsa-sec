# SPDX-FileCopyrightText:  PyPSA-ZA2, PyPSA-ZA, PyPSA-Earth and PyPSA-Eur Authors
# # SPDX-License-Identifier: MIT
# coding: utf-8
"""
Prepare PyPSA network for solving according to :ref:`opts` and :ref:`ll`, such as

- adding an annual **limit** of carbon-dioxide emissions,
- adding an exogenous **price** per tonne emissions of carbon-dioxide (or other kinds),
- setting an **N-1 security margin** factor for transmission line capacities,
- specifying an expansion limit on the **cost** of transmission expansion,
- specifying an expansion limit on the **volume** of transmission expansion, and
- reducing the **temporal** resolution by averaging over multiple hours
  or segmenting time series into chunks of varying lengths using ``tsam``.

Relevant Settings
-----------------

.. code:: yaml

    costs:
        emission_prices:
        USD2013_to_EUR2013:
        discountrate:
        marginal_cost:
        capital_cost:

    electricity:
        co2limit:
        max_hours:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`costs_cf`, :ref:`electricity_cf`

Inputs
------

- ``data/costs.csv``: The database of cost assumptions for all included technologies for specific years from various sources; e.g. discount rate, lifetime, investment (CAPEX), fixed operation and maintenance (FOM), variable operation and maintenance (VOM), fuel costs, efficiency, carbon-dioxide intensity.
- ``networks/elec_s{simpl}_{clusters}.nc``: confer :ref:`cluster`

Outputs
-------

- ``networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc``: Complete PyPSA network that will be handed to the ``solve_network`` rule.

Description
-----------

.. tip::
    The rule :mod:`prepare_all_networks` runs
    for all ``scenario`` s in the configuration file
    the rule :mod:`prepare_network`.

"""
import logging
import re

import numpy as np
import pandas as pd
import pypsa
from pypsa.linopt import get_var, write_objective, define_constraints, linexpr
from pypsa.descriptors import get_switchable_as_dense as get_as_dense, expand_series
from pypsa.optimization.common import reindex

from _helpers import configure_logging, clean_pu_profiles, remove_leap_day, normalize_and_rename_df, assign_segmented_df_to_network
from add_electricity import load_extendable_parameters, update_transmission_costs
from concurrent.futures import ProcessPoolExecutor
import xarray as xr
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Comment out for debugging and development
from custom_constraints import set_operational_limits, ccgt_steam_constraints, define_reserve_margin
idx = pd.IndexSlice
logger = logging.getLogger(__name__)

"""
********************************************************************************
    Build limit constraints
********************************************************************************
"""
def set_extendable_limits_global(n):
    ext_years = n.investment_periods if n.multi_invest else [n.snapshots[0].year]
    sense = {"max": "<=", "min": ">="}
    ignore = {"max": "unc", "min": 0}


    global_limits = {
        lim: pd.read_excel(
            model_file,
            sheet_name=f'extendable_{lim}_build',
            index_col=[0, 1, 3, 2, 4],
        ).loc[(model_setup["extendable_build_limits"], "global", slice(None), slice(None)), ext_years]
        for lim in ["max", "min"]
    }

    for lim, global_limit in global_limits.items():
        global_limit.index = global_limit.index.droplevel([0, 1, 2, 3])
        global_limit = global_limit.loc[~(global_limit == ignore[lim]).all(axis=1)]
        constraints = [
            {
                "name": f"global_{lim}-{carrier}-{y}",
                "carrier_attribute": carrier,
                "sense": sense[lim],
                "type": "tech_capacity_expansion_limit",
                **({"investment_period": y} if n.multi_invest else {}),
                "constant": global_limit.loc[carrier, y],
            }
            for carrier in global_limit.index
            for y in ext_years
            if global_limit.loc[carrier, y] != ignore[lim]
        ]

        for constraint in constraints:
            n.add("GlobalConstraint", **constraint)


def set_extendable_limits_per_bus(n):
    ext_years = n.investment_periods if n.multi_invest else [n.snapshots[0].year]
    ignore = {"max": "unc", "min": 0}

    bus_limits = {
        lim: pd.read_excel(
            model_file,
            sheet_name=f'extendable_{lim}_build',
            index_col=[0, 1, 3, 2, 4],
        ).loc[(model_setup["extendable_build_limits"], snakemake.wildcards.regions, slice(None)), ext_years]
        for lim in ["max", "min"]
    }

    ext_carriers = (
        list(n.generators.carrier[n.generators.p_nom_extendable].unique())
        + list(n.storage_units.carrier[n.storage_units.p_nom_extendable].unique())
    )
    for lim, bus_limit in bus_limits.items():
        bus_limit.index = bus_limit.index.droplevel([0, 1, 2])
        bus_limit = bus_limit.loc[~(bus_limit == ignore[lim]).all(axis=1)]
        bus_limit = bus_limit.loc[bus_limit.index.get_level_values(1).isin(ext_carriers)]

        for idx in bus_limit.index:
            for y in ext_years:
                if bus_limit.loc[idx, y] != ignore[lim]:
                    n.buses.loc[idx[0],f"nom_{lim}_{idx[1]}_{y}"] = bus_limit.loc[idx, y]


"""
********************************************************************************
    Emissions limits and pricing
********************************************************************************
"""

def add_co2limit(n):
    n.add("GlobalConstraint", "CO2Limit",
          carrier_attribute="co2_emissions", sense="<=",
          constant=snakemake.config["electricity"]["co2limit"])

def add_emission_prices(n, emission_prices=None, exclude_co2=False):
    if emission_prices is None:
        emission_prices = snakemake.config["costs"]["emission_prices"]
    if exclude_co2: emission_prices.pop("co2")
    ep = (pd.Series(emission_prices).rename(lambda x: x+"_emissions") * n.carriers).sum(axis=1)
    n.generators["marginal_cost"] += n.generators.carrier.map(ep)
    n.storage_units["marginal_cost"] += n.storage_units.carrier.map(ep)

def add_co2limit(n):
    n.add("GlobalConstraint", "CO2Limit",
          carrier_attribute="co2_emissions", 
          sense="<=",
          constant=snakemake.config['electricity']['co2limit'])


def add_emission_prices(n, emission_prices={"co2": 0.0}, exclude_co2=False):
    if exclude_co2:
        emission_prices.pop("co2")
    ep = (
        pd.Series(emission_prices).rename(lambda x: x + "_emissions")
        * n.carriers.filter(like="_emissions")
    ).sum(axis=1)
    gen_ep = n.generators.carrier.map(ep) / n.generators.efficiency
    n.generators["marginal_cost"] += gen_ep
    su_ep = n.storage_units.carrier.map(ep) / n.storage_units.efficiency_dispatch
    n.storage_units["marginal_cost"] += su_ep




"""
********************************************************************************
    Transmission constraints
********************************************************************************
"""

def set_line_s_max_pu(n):
    s_max_pu = snakemake.config["lines"]["s_max_pu"]
    n.lines["s_max_pu"] = s_max_pu
    logger.info(f"N-1 security margin of lines set to {s_max_pu}")


def set_transmission_limit(n, ll_type, factor, costs, Nyears=1):
    links_dc_b = n.links.carrier == "DC" if not n.links.empty else pd.Series()

    _lines_s_nom = (
        np.sqrt(3)
        * n.lines.type.map(n.line_types.i_nom)
        * n.lines.num_parallel
        * n.lines.bus0.map(n.buses.v_nom)
    )
    lines_s_nom = n.lines.s_nom.where(n.lines.type == "", _lines_s_nom)

    col = "capital_cost" if ll_type == "c" else "length"
    ref = (
        lines_s_nom @ n.lines[col]
        + n.links.loc[links_dc_b, "p_nom"] @ n.links.loc[links_dc_b, col]
    )

    update_transmission_costs(n, costs)

    if factor == "opt" or float(factor) > 1.0:
        n.lines["s_nom_min"] = lines_s_nom
        n.lines["s_nom_extendable"] = True

        n.links.loc[links_dc_b, "p_nom_min"] = n.links.loc[links_dc_b, "p_nom"]
        n.links.loc[links_dc_b, "p_nom_extendable"] = True

    if factor != "opt":
        con_type = "expansion_cost" if ll_type == "c" else "volume_expansion"
        rhs = float(factor) * ref
        n.add(
            "GlobalConstraint",
            f"l{ll_type}_limit",
            type=f"transmission_{con_type}_limit",
            sense="<=",
            constant=rhs,
            carrier_attribute="AC, DC",
        )
    return n

def set_line_nom_max(n, s_nom_max_set=np.inf, p_nom_max_set=np.inf):
    n.lines.s_nom_max.clip(upper=s_nom_max_set, inplace=True)
    n.links.p_nom_max.clip(upper=p_nom_max_set, inplace=True)

"""
********************************************************************************
    Time step reduction
********************************************************************************
"""

def average_every_nhours(n, offset):
    logger.info(f"Resampling the network to {offset}")
    m = n.copy()#with_time=False)
    snapshots_unstacked = n.snapshots.get_level_values(1)

    snapshot_weightings = n.snapshot_weightings.copy().set_index(snapshots_unstacked).resample(offset).sum()
    snapshot_weightings = remove_leap_day(snapshot_weightings)
    snapshot_weightings=snapshot_weightings[snapshot_weightings.index.year.isin(n.investment_periods)]
    snapshot_weightings.index = pd.MultiIndex.from_arrays([snapshot_weightings.index.year, snapshot_weightings.index])
    m.set_snapshots(snapshot_weightings.index)
    m.snapshot_weightings = snapshot_weightings

    for c in n.iterate_components():
        pnl = getattr(m, c.list_name + "_t")
        for k, df in c.pnl.items():
            if not df.empty:
                resampled = df.set_index(snapshots_unstacked).resample(offset).mean()
                resampled = remove_leap_day(resampled)
                resampled=resampled[resampled.index.year.isin(n.investment_periods)]
                resampled.index = snapshot_weightings.index
                pnl[k] = resampled
    return m

def single_year_segmentation(n, snapshots, segments, config):

    p_max_pu, p_max_pu_max = normalize_and_rename_df(n.generators_t.p_max_pu, snapshots, 1, 'max')
    load, load_max = normalize_and_rename_df(n.loads_t.p_set, snapshots, 1, "load")
    inflow, inflow_max = normalize_and_rename_df(n.storage_units_t.inflow, snapshots, 0, "inflow")

    raw = pd.concat([p_max_pu, load, inflow], axis=1, sort=False)

    multi_index = False
    if isinstance(raw.index, pd.MultiIndex):
        multi_index = True
        raw.index = raw.index.droplevel(0)
        
    y = snapshots.get_level_values(0)[0] if multi_index else snapshots[0].year

    agg = tsam.TimeSeriesAggregation(
        raw,
        hoursPerPeriod=len(raw),
        noTypicalPeriods=1,
        noSegments=int(segments),
        segmentation=True,
        solver=config['solver'],
    )

    segmented_df = agg.createTypicalPeriods()
    weightings = segmented_df.index.get_level_values("Segment Duration")
    cumsum = np.cumsum(weightings[:-1])
    
    if np.floor(y/4)-y/4 == 0: # check if leap year and add Feb 29 
            cumsum = np.where(cumsum >= 1416, cumsum + 24, cumsum) # 1416h from start year to Feb 29
    
    offsets = np.insert(cumsum, 0, 0)
    start_snapshot = snapshots[0][1] if n.multi_invest else snapshots[0]
    snapshots = pd.DatetimeIndex([start_snapshot + pd.Timedelta(hours=offset) for offset in offsets])
    snapshots = pd.MultiIndex.from_arrays([snapshots.year, snapshots]) if multi_index else snapshots
    weightings = pd.Series(weightings, index=snapshots, name="weightings", dtype="float64")
    segmented_df.index = snapshots

    segmented_df[p_max_pu.columns] *= p_max_pu_max
    segmented_df[load.columns] *= load_max
    segmented_df[inflow.columns] *= inflow_max
     
    logging.info(f"Segmentation complete for period: {y}")

    return segmented_df, weightings

def apply_time_segmentation(n, segments, config):
    logging.info(f"Aggregating time series to {segments} segments.")    
    years = n.investment_periods if n.multi_invest else [n.snapshots[0].year]

    if len(years) == 1:
        segmented_df, weightings = single_year_segmentation(n, n.snapshots, segments, config)
    else:

        with ProcessPoolExecutor(max_workers = min(len(years),config['nprocesses'])) as executor:
            parallel_seg = {
                year: executor.submit(
                    single_year_segmentation,
                    n,
                    n.snapshots[n.snapshots.get_level_values(0) == year],
                    segments,
                    config
                )
                for year in years
            }

        segmented_df = pd.concat(
            [parallel_seg[year].result()[0] for year in parallel_seg], axis=0
        )
        weightings = pd.concat(
            [parallel_seg[year].result()[1] for year in parallel_seg], axis=0
        )

    n.set_snapshots(segmented_df.index)
    n.snapshot_weightings = weightings   
    
    assign_segmented_df_to_network(segmented_df, "_load", "", n.loads_t.p_set)
    assign_segmented_df_to_network(segmented_df, "_max", "", n.generators_t.p_max_pu)
    assign_segmented_df_to_network(segmented_df, "_min", "", n.generators_t.p_min_pu)
    assign_segmented_df_to_network(segmented_df, "_inflow", "", n.storage_units_t.inflow)

    return n



"""
********************************************************************************
    Setup unit commitment for steam generators
********************************************************************************
"""
# def set_unit_committment(n, snapshots, model_file, model_setup):
#     ccgt_hrsg = n.generators.query("carrier == 'ccgt_steam' & p_nom_extendable").index
#     n.generators[ccgt_hrsg].p_nom_extendable = False
#     n.generators[ccgt_hrsg].committable = True

#     # Get unit committment parameters
#     uc_params = pd.read_excel(
#         model_file,
#         sheet_name="extendable_parameters",
#         index_col=[0, 1],
#     ).loc[(model_setup["extendable_parameters"], "ccgt_steam")]
#     uc_params = uc_params.reset_index(drop=True).set_index("parameter")

    # # Set unit committment parameters
    # uc_list = ["p_min_pu","min_up_time", "min_down_time", "ramp_limit_up", "ramp_limit_down", "start_up_cost", "shut_down_cost"]
    



def solve_network(n, sns):
    
    n.optimize.create_model(snapshots = sns, multi_investment_periods = n.multi_invest)
    # Custom constraints
    set_operational_limits(n, sns, model_file, model_setup)
    ccgt_steam_constraints(n, sns, model_file, model_setup, snakemake)
    define_reserve_margin(n, sns, model_file, model_setup, snakemake)

    solver_name = snakemake.config["solving"]["solver"].pop("name")
    solver_options = snakemake.config["solving"]["solver"].copy()
    n.optimize.solve_model(solver_name=solver_name, solver_options=solver_options)

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake(
            'prepare_and_solve_network', 
            **{
                'model_file':'val-LC-UNC',
                'regions':'1-supply',
                'resarea':'redz',
                'll':'copt',
                'opts':'LC'
            }
        )
    #configure_logging(snakemake)
    logging.info("Preparing costs")
    n = pypsa.Network(snakemake.input[0])
    model_file = pd.ExcelFile(snakemake.input.model_file)
    model_setup = (
        pd.read_excel(
            model_file, 
            sheet_name="model_setup",
            index_col=[0])
            .loc[snakemake.wildcards.model_file]
    )
    #set_unit_committment(n, n.snapshots, model_file, model_setup)

    opts = snakemake.wildcards.opts.split("-")
    for o in opts:
        m = re.match(r"^\d+h$", o, re.IGNORECASE)
        if m is not None:
            n = average_every_nhours(n, m[0])
            break

    for o in opts:
        m = re.match(r"^\d+SEG$", o, re.IGNORECASE)
        if m is not None:
            try:
                import tsam.timeseriesaggregation as tsam
            except:
                raise ModuleNotFoundError(
                    "Optional dependency 'tsam' not found." "Install via 'pip install tsam'"
                )
            n = apply_time_segmentation(n, m[0][:-3], snakemake.config["tsam_clustering"])
            break


    for o in opts:
        if "Co2L" in o:
            m = re.findall("[0-9]*\.?[0-9]+$", o)
            if len(m) > 0:
                co2limit = float(m[0]) * snakemake.config["electricity"]["co2base"]
                add_co2limit(n)
                logging.info("Setting CO2 limit according to wildcard value.")
            else:
                add_co2limit(n)
                logging.info("Setting CO2 limit according to config value.")
            break

    logging.info("Setting transmission constraints")
    Nyears = n.snapshot_weightings.objective.sum() / 8760.0        
    set_line_s_max_pu(n)
    ll_type, factor = snakemake.wildcards.ll[0], snakemake.wildcards.ll[1:]
    param = load_extendable_parameters(n, model_file, model_setup, snakemake)

    #set_transmission_limit(n, ll_type, factor, costs, Nyears)

    set_line_nom_max(
        n,
        s_nom_max_set=snakemake.config["lines"].get("s_nom_max,", np.inf),
        p_nom_max_set=snakemake.config["links"].get("p_nom_max,", np.inf),
    )
    logging.info("Setting global and regional build limits")
    if snakemake.wildcards.regions != "1-supply": #covered under single bus limits
        set_extendable_limits_global(n) 
    set_extendable_limits_per_bus(n)

    logging.info("Solving network")
    solve_network(n, n.snapshots)
    
    n.export_to_netcdf(snakemake.output[0])

