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
from pypsa.descriptors import get_switchable_as_dense as get_as_dense
from pypsa.descriptors import expand_series
from pypsa.optimization.common import reindex

from _helpers import configure_logging, clean_pu_profiles, remove_leap_day, normalize_and_rename_df, assign_segmented_df_to_network
from add_electricity import load_costs, update_transmission_costs
from concurrent.futures import ProcessPoolExecutor
import xarray as xr
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Comment out for debugging and development

idx = pd.IndexSlice
logger = logging.getLogger(__name__)

"""
********************************************************************************
    Build limit constraints
********************************************************************************
"""
def set_extendable_limits_global(n, model_file, model_setup):
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
                "investment_period": y,
                "type": "tech_capacity_expansion_limit",

                "constant": global_limit.loc[carrier, y],
            }
            for carrier in global_limit.index
            for y in ext_years
            if global_limit.loc[carrier, y] != ignore[lim]
        ]

        for constraint in constraints:
            n.add("GlobalConstraint", **constraint)


def set_extendable_limits_per_bus(n, model_file, model_setup):
    ext_years = n.investment_periods if n.multi_invest else [n.snapshots[0].year]
    ignore = {"max": "unc", "min": 0}

    global_limits = {
        lim: pd.read_excel(
            model_file,
            sheet_name=f'extendable_{lim}_build',
            index_col=[0, 1, 3, 2, 4],
        ).loc[(model_setup["extendable_build_limits"], snakemake.wildcards.regions, slice(None)), ext_years]
        for lim in ["max", "min"]
    }

    for lim, global_limit in global_limits.items():
        global_limit.index = global_limit.index.droplevel([0, 1, 2])
        global_limit = global_limit.loc[~(global_limit == ignore[lim]).all(axis=1)]

        for idx in global_limit.index:
            for y in ext_years:
                if global_limit.loc[idx, y] != ignore[lim]:
                    n.buses.loc[idx[0],f"nom_{lim}_{idx[1]}_{y}"] = global_limit.loc[idx, y]



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
    Reserve margin
********************************************************************************
"""

# def add_peak_demand_hour_without_variable_feedin(n):
#     new_hour = n.snapshots[-1] + pd.Timedelta(hours=1)
#     n.set_snapshots(n.snapshots.append(pd.Index([new_hour])))

#     # Don"t value new hour for energy totals
#     n.snapshot_weightings[new_hour] = 0.

#     # Don"t allow variable feed-in in this hour
#     n.generators_t.p_max_pu.loc[new_hour] = 0.

#     n.loads_t.p_set.loc[new_hour] = (
#         n.loads_t.p_set.loc[n.loads_t.p_set.sum(axis=1).idxmax()]
#         * (1.+snakemake.config["electricity"]["SAFE_reservemargin"])
#     )


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
    start_snapshot = snapshots[0]
    snapshots = pd.DatetimeIndex([start_snapshot[1] + pd.Timedelta(hours=offset) for offset in offsets])

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



# def apply_time_segmentation_perfect(
#     n, segments, solver_name="cbc", overwrite_time_dependent=True
# ):
#     """
#     Aggregating time series to segments with different lengths.

#     Input:
#         n: pypsa Network
#         segments: (int) number of segments in which the typical period should be
#                   subdivided
#         solver_name: (str) name of solver
#         overwrite_time_dependent: (bool) overwrite time dependent data of pypsa network
#         with typical time series created by tsam
#     """
#     try:
#         import tsam.timeseriesaggregation as tsam
#     except:
#         raise ModuleNotFoundError(
#             "Optional dependency 'tsam' not found." "Install via 'pip install tsam'"
#         )

#     # get all time-dependent data
#     columns = pd.MultiIndex.from_tuples([], names=["component", "key", "asset"])
#     raw = pd.DataFrame(index=n.snapshots, columns=columns)
#     for c in n.iterate_components():
#         for attr, pnl in c.pnl.items():
#             # exclude e_min_pu which is used for SOC of EVs in the morning
#             if not pnl.empty and attr != "e_min_pu":
#                 df = pnl.copy()
#                 df.columns = pd.MultiIndex.from_product([[c.name], [attr], df.columns])
#                 raw = pd.concat([raw, df], axis=1)
#     raw = raw.dropna(axis=1)
#     sn_weightings = {}

#     for year in raw.index.levels[0]:
#         logger.info(f"Find representative snapshots for {year}.")
#         raw_t = raw.loc[year]
#         # normalise all time-dependent data
#         annual_max = raw_t.max().replace(0, 1)
#         raw_t = raw_t.div(annual_max, level=0)
#         # get representative segments
#         agg = tsam.TimeSeriesAggregation(
#             raw_t,
#             hoursPerPeriod=len(raw_t),
#             noTypicalPeriods=1,
#             noSegments=int(segments),
#             segmentation=True,
#             solver=solver_name,
#         )
#         segmented = agg.createTypicalPeriods()

#         weightings = segmented.index.get_level_values("Segment Duration")
#         offsets = np.insert(np.cumsum(weightings[:-1]), 0, 0)
#         timesteps = [raw_t.index[0] + pd.Timedelta(f"{offset}h") for offset in offsets]
#         snapshots = pd.DatetimeIndex(timesteps)
#         sn_weightings[year] = pd.Series(
#             weightings, index=snapshots, name="weightings", dtype="float64"
#         )

#     sn_weightings = pd.concat(sn_weightings)
#     n.set_snapshots(sn_weightings.index)
#     n.snapshot_weightings = n.snapshot_weightings.mul(sn_weightings, axis=0)

#     return n

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake(
            'prepare_network', 
            **{
                'model_file':'grid-expansion',
                'regions':'11-supply',
                'resarea':'redz',
                'll':'copt',
                'opts':'LC'
            }
        )
    #configure_logging(snakemake)
    n = pypsa.Network(snakemake.input[0])
    model_file = pd.ExcelFile(snakemake.input.model_file)
    model_setup = (
        pd.read_excel(
            model_file, 
            sheet_name="model_setup",
            index_col=[0])
            .loc[snakemake.wildcards.model_file]
    )


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

    Nyears = n.snapshot_weightings.objective.sum() / 8760.0        
    set_line_s_max_pu(n)
    ll_type, factor = snakemake.wildcards.ll[0], snakemake.wildcards.ll[1:]
    costs = load_costs(n, model_file, model_setup.costs, snakemake)

    #set_transmission_limit(n, ll_type, factor, costs, Nyears)

    set_line_nom_max(
        n,
        s_nom_max_set=snakemake.config["lines"].get("s_nom_max,", np.inf),
        p_nom_max_set=snakemake.config["links"].get("p_nom_max,", np.inf),
    )
    

    set_extendable_limits_global(n, model_file, model_setup)
    set_extendable_limits_per_bus(n, model_file, model_setup)

    #apply_time_segmentation(n, 100, {"solver":"cbc", "nprocesses":5})

    n.export_to_netcdf(snakemake.output[0])


    # opts = snakemake.wildcards.opts.split("-")

    # Nyears = n.snapshot_weightings.objective.sum() / 8760.0
    # costs = load_costs(
    #     snakemake.input.model_file,
    #     model_setup.costs,
    #     snakemake.config["costs"],
    #     snakemake.config["electricity"],
    #     n.investment_periods,
    # )


    # set_line_s_max_pu(n)

    # for o in opts:
    #     m = re.match(r"^\d+h$", o, re.IGNORECASE)
    #     if m is not None:
    #         n = average_every_nhours(n, m.group(0))
    #         break


    # for o in opts:
    #     m = re.match(r"^\d+SEG$", o, re.IGNORECASE)
    #     if m is not None:
    #         n = apply_time_segmentation(n, m.group(0)[:-3],snakemake.config["tsam_clustering"])
    #         break

    # for o in opts:
    #     if "Co2L" in o:
    #         m = re.findall("[0-9]*\.?[0-9]+$", o)
    #         if len(m) > 0:
    #             co2limit = float(m[0]) * snakemake.config["electricity"]["co2base"]
    #             add_co2limit(n)
    #             logger.info("Setting CO2 limit according to wildcard value.")
    #         else:
    #             add_co2limit(n)
    #             logger.info("Setting CO2 limit according to config value.")
    #         break

    # for o in opts:
    #     if "CH4L" in o:
    #         m = re.findall("[0-9]*\.?[0-9]+$", o)
    #         if len(m) > 0:
    #             limit = float(m[0]) * 1e6
    #             add_gaslimit(n, limit, Nyears)
    #             logging.info("Setting gas usage limit according to wildcard value.")
    #         else:
    #             add_gaslimit(n, snakemake.config["electricity"].get("gaslimit"), Nyears)
    #             logging.info("Setting gas usage limit according to config value.")

    #     for o in opts:
    #         oo = o.split("+")
    #         suptechs = map(lambda c: c.split("-", 2)[0], n.carriers.index)
    #         if oo[0].startswith(tuple(suptechs)):
    #             carrier = oo[0]
    #             # handles only p_nom_max as stores and lines have no potentials
    #             attr_lookup = {
    #                 "p": "p_nom_max",
    #                 "c": "capital_cost",
    #                 "m": "marginal_cost",
    #             }
    #             attr = attr_lookup[oo[1][0]]
    #             factor = float(oo[1][1:])
    #             if carrier == "AC":  # lines do not have carrier
    #                 n.lines[attr] *= factor
    #             else:
    #                 comps = {"Generator", "Link", "StorageUnit", "Store"}
    #                 for c in n.iterate_components(comps):
    #                     sel = c.df.carrier.str.contains(carrier)
    #                     c.df.loc[sel, attr] *= factor

    #     for o in opts:
    #         if "Ep" in o:
    #             m = re.findall("[0-9]*\.?[0-9]+$", o)
    #             if len(m) > 0:
    #                 logging.info("Setting emission prices according to wildcard value.")
    #                 add_emission_prices(n, dict(co2=float(m[0])))
    #             else:
    #                 logging.info("Setting emission prices according to config value.")
    #                 add_emission_prices(n, snakemake.config["costs"]["emission_prices"])
    #             break


    #n.export_to_netcdf(snakemake.output[0])
