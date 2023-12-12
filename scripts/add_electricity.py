# SPDX-FileCopyrightText:  PyPSA-ZA2, PyPSA-ZA, PyPSA-Earth and PyPSA-Eur Authors
# # SPDX-License-Identifier: MIT

# coding: utf-8

"""
Adds existing and extendable components to the base network. The primary functions run inside main are:

    attach_load
    attach_existing_generators
    attach_extendable_generators
    attach_existing_storage
    attach_extendable_storage

Relevant Settings
-----------------

.. code:: yaml

    costs:
        year:
        USD_to_ZAR:
        EUR_to_ZAR:
        marginal_cost:
        dicountrate:
        emission_prices:
        load_shedding:

    electricity:
        max_hours:
        marginal_cost:
        capital_cost:
        conventional_carriers:
        co2limit:
        extendable_carriers:
        include_renewable_capacities_from_OPSD:
        estimate_renewable_capacities_from_capacity_stats:

    load:
        scale:
        ssp:
        weather_year:
        prediction_year:
        region_load:

    renewable:
        hydro:
            carriers:
            hydro_max_hours:
            hydro_capital_cost:

    lines:
        length_factor:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at :ref:`costs_cf`,
    :ref:`electricity_cf`, :ref:`load_cf`, :ref:`renewable_cf`, :ref:`lines_cf`

Inputs
------
- ``model_file.xlsx``: The database to setup different scenarios based on cost assumptions for all included technologies for specific years from various sources; e.g. discount rate, lifetime, investment (CAPEX), fixed operation and maintenance (FOM), variable operation and maintenance (VOM), fuel costs, efficiency, carbon-dioxide intensity.
- ``data/Eskom EAF data.xlsx``: Hydropower plant store/discharge power capacities, energy storage capacity, and average hourly inflow by country.  Not currently used!
- ``data/eskom_pu_profiles.csv``: alternative to capacities above; not currently used!
- ``data/bundle/SystemEnergy2009_22.csv`` Hourly country load profiles produced by GEGIS
- ``resources/regions_onshore.geojson``: confer :ref:`busregions`
- ``resources/gadm_shapes.geojson``: confer :ref:`shapes`
- ``data/bundle/supply_regions/{regions}.shp``: confer :ref:`powerplants`
- ``resources/profile_{}_{regions}_{resarea}.nc``: all technologies in ``config["renewables"].keys()``, confer :ref:`renewableprofiles`.
- ``networks/base_{model_file}_{regions}.nc``: confer :ref:`base`

Outputs
-------

- ``networks/elec_{model_file}_{regions}_{resarea}.nc``:

    .. image:: ../img/elec.png
            :scale: 33 %

Description
-----------

The rule :mod:`add_electricity` ties all the different data inputs from the preceding rules together into a detailed PyPSA network that is stored in ``networks/elec.nc``. It includes:

- today"s transmission topology and transfer capacities (in future, optionally including lines which are under construction according to the config settings ``lines: under_construction`` and ``links: under_construction``),
- today"s thermal and hydro power generation capacities (for the technologies listed in the config setting ``electricity: conventional_carriers``), and
- today"s load time-series (upsampled in a top-down approach according to population and gross domestic product)

It further adds extendable ``generators`` with **zero** capacity for

- photovoltaic, onshore and AC- as well as DC-connected offshore wind installations with today"s locational, hourly wind and solar capacity factors (but **no** current capacities),
- additional open- and combined-cycle gas turbines (if ``OCGT`` and/or ``CCGT`` is listed in the config setting ``electricity: extendable_carriers``)
"""

import geopandas as gpd
import logging
logger = logging.getLogger(__name__)
import numpy as np
import pandas as pd
import pypsa
from pypsa.descriptors import get_switchable_as_dense as get_as_dense
from pypsa.io import import_components_from_dataframe
from shapely.geometry import Point
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning) # Comment out for debugging and development


from _helpers import (
    _add_missing_carriers_from_costs,
    convert_cost_units,
    load_disaggregate, 
    map_component_parameters, 
    read_and_filter_generators,
    remove_leap_day,
    drop_non_pypsa_attrs,
    normed,
    get_start_year,
    get_snapshots,
)

"""
********************************************************************************
    Cost related functions
********************************************************************************
"""
def annualise_costs(investment, lifetime, discount_rate, FOM):
    """
    Annualises the costs of an investment over its lifetime.The costs are annualised using the 
    Capital Recovery Factor (CRF) formula.

    Args:
    - investment: The overnight investment cost.
    - lifetime: The lifetime of the investment.
    - discount_rate: The discount rate used for annualisation.
    - FOM: The fixed operating and maintenance costs.

    Returns:
    A Series containing the annualised costs.
    """
    CRF = discount_rate / (1 - 1 / (1 + discount_rate) ** lifetime)
    return (investment * CRF + FOM).fillna(0)

def load_costs(n, model_file, cost_scenario, snakemake):
    """
    set all asset costs tab in the model file
    """

    costs = pd.read_excel(
        model_file, 
        sheet_name = "costs",
        index_col = [0,2,1],
    ).sort_index().loc[cost_scenario]

    costs.drop("source", axis=1, inplace=True)
    
    # Interpolate for years in config file but not in cost_data excel file
    ext_years = n.investment_periods if n.multi_invest else [n.snapshots[0].year]
    ext_years_array = np.array(ext_years)
    missing_year = ext_years_array[~np.isin(ext_years_array,costs.columns)]
    if len(missing_year) > 0:
        for i in missing_year: 
            costs.insert(0,i,np.nan) # add columns of missing year to dataframe
        costs_tmp = costs.drop("unit", axis=1).sort_index(axis=1)
        costs_tmp = costs_tmp.interpolate(axis=1)
        costs = pd.concat([costs_tmp, costs["unit"]], ignore_index=False, axis=1)

    # correct units to MW and ZAR
    costs_yr = costs.columns.drop("unit")

    costs = convert_cost_units(costs, snakemake.config["costs"]["USD_to_ZAR"], snakemake.config["costs"]["EUR_to_ZAR"])
    
    full_costs = pd.DataFrame(
        index = pd.MultiIndex.from_product(
            [
                costs.index.get_level_values(0).unique(),
                costs.index.get_level_values(1).unique()]),
        columns = costs.columns
    )
    # full_costs adds default values when missing from costs table
    for default in costs.index.get_level_values(0):
        full_costs.loc[costs.loc[(default, slice(None)),:].index, :] = costs.loc[(default, slice(None)),:]
        full_costs.loc[(default, slice(None)), costs_yr] = full_costs.loc[(default, slice(None)), costs_yr].fillna(snakemake.config["costs"]["defaults"][default])
    full_costs = full_costs.fillna("default")
    costs = full_costs.copy()

    # Get entries where FOM is specified as % of CAPEX
    fom_perc_capex=costs.loc[costs.unit.str.contains("%/year") == True, costs_yr]
    fom_perc_capex_idx=fom_perc_capex.index.get_level_values(1)

    add_costs = pd.DataFrame(
        index = pd.MultiIndex.from_product([["capital_cost","marginal_cost"],costs.loc["FOM"].index]),
        columns = costs.columns
    )
    
    costs = pd.concat([costs, add_costs],axis=0)
    costs.loc[("FOM",fom_perc_capex_idx), costs_yr] = (costs.loc[("investment", fom_perc_capex_idx),costs_yr]).values/100.0
    costs.loc[("FOM",fom_perc_capex_idx), "unit"] = costs.loc[("investment", fom_perc_capex_idx),"unit"].values

    capital_costs = annualise_costs(
        costs.loc["investment", costs_yr],
        costs.loc["lifetime", costs_yr], 
        costs.loc["discount_rate", costs_yr],
        costs.loc["FOM", costs_yr],
    )

    costs.loc["capital_cost", costs_yr] = capital_costs.fillna(0).values
    costs.loc["capital_cost","unit"] = "R/MWe"

    vom = costs.loc["VOM", costs_yr].fillna(0)
    fuel = (costs.loc["fuel", costs_yr] / costs.loc["efficiency", costs_yr]).fillna(0)

    costs.loc[("marginal_cost", vom.index), costs_yr] = vom.values
    costs.loc[("marginal_cost", fuel.index), costs_yr] += fuel.values
    costs.loc["marginal_cost","unit"] = "R/MWhe"

    max_hours = snakemake.config["electricity"]["max_hours"]
    costs.loc[("capital_cost","battery"), :] = costs.loc[("capital_cost","battery inverter"),:]
    costs.loc[("capital_cost","battery"), costs_yr] += max_hours["battery"]*costs.loc[("capital_cost", "battery storage"), costs_yr]
    
    return costs

def update_transmission_costs(n, costs, length_factor=1.0, simple_hvdc_costs=False):
    # Currently only average transmission costs are implemented
    n.lines["capital_cost"] = (
        n.lines["length"] * length_factor * costs.loc[("capital_cost","HVAC_overhead"), costs.columns.drop("unit")].mean()
    )

    if n.links.empty:
        return

    dc_b = n.links.carrier == "DC"
    # If there are no "DC" links, then the "underwater_fraction" column
    # may be missing. Therefore we have to return here.
    # TODO: Require fix
    if n.links.loc[n.links.carrier == "DC"].empty:
        return

    if simple_hvdc_costs:
        hvdc_costs = (
            n.links.loc[dc_b, "length"]
            * length_factor
            * costs.loc[("capital_cost","HVDC_overhead"),:].mean()
        )
    else:
        hvdc_costs = (
            n.links.loc[dc_b, "length"]
            * length_factor
            * (
                (1.0 - n.links.loc[dc_b, "underwater_fraction"])
                * costs.loc[("capital_cost","HVDC_overhead"),:].mean()
                + n.links.loc[dc_b, "underwater_fraction"]
                * costs.loc[("capital_cost","HVDC_submarine"),:].mean()
            )
            + costs.loc[("capital_cost","HVDC inverter_pair"),:].mean()
        )
    n.links.loc[dc_b, "capital_cost"] = hvdc_costs


"""
********************************************************************************
    Add load to the network 
********************************************************************************
"""

def attach_load(n, annual_demand):
    """
    Attaches load to the network based on the provided annual demand.
    Demand is disaggregated by the load_disaggreate function according to either population 
    or GVA (GDP) in each region. 

    Args:
    - n: The network object.
    - annual_demand: A DataFrame containing the annual demand values.

        """

    load = pd.read_csv(snakemake.input.load,index_col=[0],parse_dates=True)
    
    annual_demand = annual_demand.drop("unit")*1e6
    profile_demand = normed(remove_leap_day(load.loc[str(snakemake.config["years"]["reference_demand_year"]),"system_energy"]))
    
    if n.multi_invest:
        demand=pd.Series(0,index=n.snapshots)
        for y in n.investment_periods:
            demand.loc[y]=profile_demand.values*annual_demand[y]
    else:
        demand = pd.Series(profile_demand.values*annual_demand[n.snapshots[0].year], index = n.snapshots)

    if snakemake.wildcards.regions == "1-supply":
        n.add("Load", n.buses.index,
            bus="RSA",
            p_set=demand)
    else:
        n.madd("Load", n.buses.index,
            bus = n.buses.index,
            p_set = load_disaggregate(demand, normed(n.buses[snakemake.config["electricity"]["demand_disaggregation"]])))


"""
********************************************************************************
    Function to define p_max_pu and p_min_pu profiles 
********************************************************************************
"""
def init_pu_profiles(gens, snapshots):
    pu_profiles = pd.DataFrame(
        index = pd.MultiIndex.from_product(
            [["max", "min"], snapshots], 
            names=["profile", "snapshots"]
            ), 
        columns = gens.index
    )
    pu_profiles.loc["max"] = 1 
    pu_profiles.loc["min"] = 0

    return pu_profiles


def extend_reference_data(n, ref_data, snapshots):
    ext_years = snapshots.year.unique()
    if len(ref_data.shape) > 1:
        extended_data = pd.DataFrame(0, index=snapshots, columns=ref_data.columns)
    else:
        extended_data = pd.Series(0, index=snapshots)     
    ref_years = ref_data.index.year.unique()

    for _ in range(int(np.ceil(len(ext_years) / len(ref_years)))-1):
        ref_data = pd.concat([ref_data, ref_data],axis=0)

    extended_data.iloc[:] = ref_data.iloc[range(len(extended_data)),:].values

    return extended_data.clip(lower=0., upper=1.)

def get_eskom_eaf(ref_yrs, snapshots):
    # Add plant availability based on actual Eskom data provided
    eskom_data  = pd.read_excel(
        snakemake.input.existing_generators_eaf, 
        sheet_name="eskom_data", 
        na_values=["-"],
        index_col=[1,0],
        parse_dates=["date"]
    )

    eaf = (eskom_data["EAF %"]/100).unstack(level=0)

    eaf = eaf.loc[eaf.index.year.isin(ref_yrs)]
    eaf_mnthly = eaf.groupby(eaf.index.month).mean()

    eaf_hrly = pd.DataFrame(1, index = snapshots, columns = eaf_mnthly.columns)
    eaf_hrly = eaf_mnthly.loc[eaf_hrly.index.month].reset_index(drop=True).set_index(eaf_hrly.index) 

    return eaf_hrly

def proj_eaf_override(eaf_hrly, projections, snapshots, include = "_EAF", exclude = "extendable"):
    """
    Overrides the hourly EAF (Equivalent Availability Factor) values with projected EAF values, if these are defined
    under the project_parameters tab in the model_file.xlsx. Existing generators have suffix _EAF and extendable generators
    have the suffix _extendable_EAF by convention.  

    Args:
    - eaf_hrly: A DataFrame containing the hourly EAF values.
    - projections: A DataFrame containing the projected EAF values.
    - snapshots: A Series containing the snapshots.
    - include: A string used to filter the projections based on the index.
    - exclude: A string used to exclude certain projections based on the index.

    Relevant model_file.xlsx settings:
        project_parameters: parameters with _EAF or _extendable_EAF suffix  
    
    """

    eaf_yrly = eaf_hrly.groupby(eaf_hrly.index.year).mean()
    proj_eaf = projections.loc[(projections.index.str.contains(include) & ~projections.index.str.contains(exclude)), snapshots.year.unique()]
    proj_eaf.index = proj_eaf.index.str.replace(include,"")

    # remove decom_stations
    proj_eaf = proj_eaf[proj_eaf.index.isin(eaf_yrly.columns)]
    scaling = proj_eaf.T.div(eaf_yrly[proj_eaf.index], axis="columns", level="year").fillna(1)

    for y in snapshots.year.unique():
        eaf_hrly.loc[str(y), scaling.columns] *= scaling.loc[y, :]  

    return eaf_hrly

def generate_eskom_re_profiles(n):
    """
    Generates Eskom renewable energy profiles for the network, based on the Eskom Data Portal information, found under
    https://www.eskom.co.za/dataportal/. Data is available from 2018 to 2023 for aggregate carriers (e.g. all solar_pv, biomass, hydro etc).
    The user can specify whether to use this data under config.yaml. These Eskom profiles for biomass, hydro and hydro_import are used by default.

    Args:
    - n: The PyPSA network object.

    Relevant config.yaml settings:
    electricity:
        renewable_generators:
            carriers:
    years:
        reference_weather_years:
    enable:
        use_eskom_wind_solar
    """
    ext_years = n.investment_periods if n.multi_invest else [n.snapshots[0].year]
    carriers = snakemake.config["electricity"]["renewable_generators"]["carriers"]
    ref_years = snakemake.config["years"]["reference_weather_years"]

    if snakemake.config["enable"]["use_excel_wind_solar"][0]:
        carriers = [elem for elem in carriers if elem not in ["wind","solar_pv"]]

    eskom_data = (
        pd.read_csv(
            snakemake.input.eskom_profiles,skiprows=[1], 
            index_col=0,parse_dates=True
        )
        .resample("1h").mean()
    )

    eskom_data = remove_leap_day(eskom_data)
    eskom_profiles = pd.DataFrame(0, index=n.snapshots, columns=carriers)

    for carrier in carriers:
        weather_years = ref_years[carrier].copy()
        if n.multi_invest:
            weather_years *= int(np.ceil(len(ext_years) / len(weather_years)))

        for cnt, y in enumerate(ext_years):
            y = y if n.multi_invest else str(y)
            eskom_profiles.loc[y, carrier] = (eskom_data.loc[str(weather_years[cnt]), carrier]
                                            .clip(lower=0., upper=1.)).values
    return eskom_profiles
def generate_existing_wind_solar_profiles_from_excel(n, gens, ref_data, snapshots, pu_profiles):
    """
    Generates existing wind and solar PV profiles for the network based on timeseries data supplied in Excel format.


    Args:
    - n: The network object.
    - gens: A DataFrame containing the generator information.
    - ref_data: The reference data file.
    - snapshots: A Series containing the snapshots.
    - pu_profiles: The DataFrame to store the pu profiles.

    Relevant config.yaml settings:
        enable:
            use_excel_wind_solar
    
            
    Relevant Excel data:
    Spreadsheet with the following tabs:
        existing_wind_pu: Existing wind profiles
        existing_solar_pv_pu: Existing solar PV profiles
        {regions}_wind_pu: Extendable wind profiles
        {regions}_solar_pv_pu: Extendable solar PV profiles

    """

    for carrier in ["wind", "solar_pv"]:
        pu = pd.read_excel(
            ref_data,
            sheet_name=f"existing_{carrier}_pu",
            index_col=0,
            skiprows=[1],
            parse_dates=["date"],
        )

        pu = remove_leap_day(pu)
        mapping = gens.loc[(gens["Model Key"] != np.nan) & (gens["carrier"] == carrier),"Model Key"]
        mapping = pd.Series(mapping.index,index=mapping.values)
        pu = pu[mapping.index]
        pu.columns = mapping[pu.columns].values
        pu = extend_reference_data(n, pu, snapshots)
        pu_profiles.loc["max", pu.columns] = pu.values
        pu_profiles.loc["min", pu.columns] = 0.95*pu.values # Existing REIPPP take or pay constraint (100% can cause instabilitites)

    return pu_profiles

def generate_extendable_wind_solar_profiles_from_excel(n, gens, ref_data, snapshots, pu_profiles):
    years = snapshots.year.unique()
    re_carries = ["wind", "solar_pv"]
    gens = gens.query("carrier == @re_carries & p_nom_extendable")
        
    for carrier in re_carries:
        pu_ref = pd.read_excel(
            ref_data,
            sheet_name=f"{snakemake.wildcards.regions}_{carrier}_pu",
            index_col=0,
            skiprows=[1],
            parse_dates=["date"],
        )

        pu_ref = remove_leap_day(pu_ref)
        pu = pu_ref.copy()
        for y in years:
            pu.columns = pu_ref.columns + f"-{carrier}-{y}"
            pu_profiles.loc["max", pu.columns] = extend_reference_data(n, pu, snapshots).values

    return pu_profiles

def group_pu_profiles(pu_profiles, component_df):
    years = pu_profiles.index.get_level_values(1).year.unique()
    p_nom_pu = pd.DataFrame(1, index = pu_profiles.index.get_level_values(1), columns = [])
    pu_mul_p_nom = pu_profiles * component_df["p_nom"]

    filtered_df = component_df[component_df["apply_grouping"]].copy().fillna(0)

    for bus in filtered_df.bus.unique():
        for carrier in filtered_df.carrier.unique():
            carrier_list = filtered_df[(filtered_df["carrier"] == carrier) & (filtered_df["bus"] == bus)].index

            for y in years:
                active = carrier_list[(component_df.loc[carrier_list, "lifetime"] - (y-years[0]))>=0]
                if len(active)>0:
                    key_list = filtered_df.loc[active, "Grouping"]
                    for key in key_list.unique():
                        active_key = active[filtered_df.loc[active, "Grouping"] == key]
                        init_active_key = carrier_list[filtered_df.loc[carrier_list, "Grouping"] == key]
                        pu_profiles.loc[(slice(None), str(y)), bus + "-" + carrier + "_" + key] = pu_mul_p_nom.loc[(slice(None), str(y)), active_key].sum(axis=1) / component_df.loc[init_active_key, "p_nom"].sum()
                        p_nom_pu.loc[str(y), bus + "-" + carrier + "_" + key] = component_df.loc[active_key, "p_nom"].sum() / component_df.loc[init_active_key, "p_nom"].sum()
            pu_profiles.drop(columns = carrier_list, inplace=True)

    return pu_profiles.fillna(0), p_nom_pu.fillna(0) # TODO check .fillna(0) doesn't make ramp_rate infeasible on p_max_pu

"""
********************************************************************************
    Functions to define and attach generators to the network  
********************************************************************************
"""
def load_components_from_model_file(model_file, model_setup, carriers, start_year, config):
    """
    Load components from a model file based on specified filters and configurations.

    Args:
        model_file: The file path to the model file.
        model_setup: The model setup object.
        carriers: A list of carriers to filter the generators.
        start_year: The start year for the components.
        config: A dictionary containing configuration settings.

    Returns:
        A DataFrame containing the loaded components.
    """
    conv_gens = read_and_filter_generators(model_file, "existing_conventional", model_setup.existing_eskom, carriers)
    re_gens = read_and_filter_generators(model_file, "existing_renewables", model_setup.existing_non_eskom, carriers)

    conv_gens["apply_grouping"] = config["conventional_generators"]["apply_grouping"]
    re_gens["apply_grouping"] = config["renewable_generators"]["apply_grouping"]
    re_gens.set_index((re_gens["Model Key"] + "_" + re_gens["Carrier"]).values,inplace=True)

    gens = pd.concat([conv_gens, re_gens])
    gens = map_component_parameters(gens, start_year)
    gens = gens.query("(p_nom > 0) & x.notnull() & y.notnull() & (lifetime >= 0)")
    
    return gens

def map_components_to_buses(component_df, regions, crs_config):
    """
    Associate every generator/storage_unit with the bus of the region based on GPS coords.

    Args:
        component_df: A DataFrame containing generator/storage_unit data.
        regions: The file path to the regions shapefile.
        crs_config: A dictionary containing coordinate reference system configurations.

    Returns:
        A DataFrame with the generators associated with their respective bus.
    """

    regions_gdf = gpd.read_file(regions).to_crs(snakemake.config["crs"]["distance_crs"]).set_index("name")
    gps_gdf = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries([Point(o.x, o.y) for o in component_df[["x", "y"]].itertuples()],
        index=component_df.index, 
        crs=crs_config["geo_crs"]
    ).to_crs(crs_config["distance_crs"]))
    joined = gpd.sjoin(gps_gdf, regions_gdf, how="left", op="within")
    component_df["bus"] = joined["index_right"].copy()

    if empty_bus := list(component_df[~component_df["bus"].notnull()].index):
        logger.warning(f"Dropping generators/storage units with no bus assignment {empty_bus}")
        component_df = component_df[component_df["bus"].notnull()]

    return component_df

def attach_existing_generators(n, costs, model_setup, model_file):
    # setup carrier info
    config = snakemake.config["electricity"]
    fix_ref_years = config["conventional_generators"]["fix_ref_years"]
    conv_carriers = config["conventional_generators"]["carriers"]
    re_carriers = config["renewable_generators"]["carriers"]
    carriers = conv_carriers + re_carriers
        
    start_year = get_start_year(n)
    snapshots = get_snapshots(n)
    
    # load generators from model file
    gens = load_components_from_model_file(model_file, model_setup, carriers, start_year, snakemake.config["electricity"])
    gens = map_components_to_buses(gens, snakemake.input.supply_regions, snakemake.config["crs"])
    pu_profiles = init_pu_profiles(gens, snapshots)

    # Monthly average EAF for conventional plants from Eskom  
    conv_pu = get_eskom_eaf(fix_ref_years, snapshots)
    conv_pu = proj_eaf_override(conv_pu, projections, snapshots, include = "_EAF", exclude = "extendable")
    eskom_carriers = [carrier for carrier in conv_carriers if carrier not in ["nuclear", "hydro", "hydro_import"]]
    for col in gens.query("Grouping == 'eskom' & carrier in @eskom_carriers").index:
        pu_profiles.loc["max", col] = conv_pu[col.split("*")[0]].values

    # Hourly data from Eskom data portal
    eskom_re_pu = generate_eskom_re_profiles(n)
    eskom_re_carriers = eskom_re_pu.columns
    for col in gens.query("carrier in @eskom_re_carriers").index:
        pu_profiles.loc["max", col] = eskom_re_pu[gens.loc[col, "carrier"]].values

    # Wind and solar profiles if not using Eskom data portal
    if snakemake.config["enable"]["use_excel_wind_solar"][0]:
        ref_data = pd.ExcelFile(snakemake.config["enable"]["use_excel_wind_solar"][1])
        pu_profiles = generate_existing_wind_solar_profiles_from_excel(n, gens, ref_data, snapshots, pu_profiles)

    pu_profiles, p_nom_pu = group_pu_profiles(pu_profiles, gens) #includes both grouped an non-grouped generators
    grouped_gens, non_grouped_gens = group_components(gens)
    grouped_gens["build_year"], grouped_gens["p_nom_extendable"] = start_year, False
    non_grouped_gens["build_year"], non_grouped_gens["p_nom_extendable"] = start_year, False

    n.import_components_from_dataframe(non_grouped_gens, "Generator")
    n.import_components_from_dataframe(grouped_gens, "Generator")

    n.generators_t.p_nom_pu = p_nom_pu
    n.generators_t.p_max_pu = pu_profiles.loc["max"]
    n.generators_t.p_min_pu = pu_profiles.loc["min"]
    
    _add_missing_carriers_from_costs(n, costs, gens.carrier.unique())

def group_components(component_df):
    """
    Apply grouping of similar carrier if specified in snakemake config.

    Args:
        component_df: A DataFrame containing generator/storage_unit data.

    Returns:
        A tuple containing two DataFrames: grouped_df, non_grouped_df
    """
    
    params = ["bus", "carrier", "lifetime", "p_nom", "efficiency", "ramp_limit_up", "ramp_limit_down", "marginal_cost", "capital_cost"]
    param_cols = [p for p in params if p not in ["bus","carrier","p_nom"]]

    filtered_df = component_df.query("apply_grouping").copy().fillna(0)#[component_df["apply_grouping"]].copy().fillna(0)

    grouped_df = pd.DataFrame(index=filtered_df.groupby(["Grouping", "carrier", "bus"]).sum().index, columns = param_cols)
    grouped_df["p_nom"] = filtered_df.groupby(["Grouping", "carrier", "bus"]).sum()["p_nom"]
    grouped_df["lifetime"] = filtered_df.groupby(["Grouping", "carrier", "bus"]).max()["lifetime"]

    # calculate weighted average of remaining parameters in gens dataframe
    for param in [p for p in params if p not in ["bus","carrier","p_nom", "lifetime"]]:
        weighted_sum = filtered_df.groupby(["Grouping", "carrier", "bus"]).apply(lambda x: (x[param] * x["p_nom"]).sum())
        total_p_nom = filtered_df.groupby(["Grouping", "carrier", "bus"])["p_nom"].sum()
        weighted_average = weighted_sum / total_p_nom 
        grouped_df.loc[weighted_average.index, param] = weighted_average.values
    
    rename_idx = grouped_df.index.get_level_values(2) +  "-" + grouped_df.index.get_level_values(1) +  "_" + grouped_df.index.get_level_values(0)
    grouped_df = grouped_df.reset_index(level=[1,2]).replace(0, np.nan).set_index(rename_idx) # replace 0 with nan to ignore in pypsa

    non_grouped_df = component_df[~component_df["apply_grouping"]][params].copy()

    return grouped_df, non_grouped_df

def extendable_max_build_per_bus_per_carrier(model_file, model_setup):

    ext_max_build = (
        pd.read_excel(
            model_file, 
            sheet_name='extendable_max_build',
            index_col=[0,1,2,3])
    ).loc[model_setup]
    ext_max_build.replace("unc", np.inf, inplace=True)

    return ext_max_build.loc[snakemake.wildcards.regions]

def define_extendable_tech(model_file, model_setup, years, type_):
    ext_max_build = pd.read_excel(
        model_file, 
        sheet_name='extendable_max_build',
        index_col= [0,1,3,2,4],
    ).loc[(model_setup["extendable_build_limits"], snakemake.wildcards.regions, type_, slice(None)), years]
    ext_max_build.replace("unc", np.inf, inplace=True)
    ext_max_build.index = ext_max_build.index.droplevel([0, 1, 2])
    ext_max_build = ext_max_build.loc[~(ext_max_build==0).all(axis=1)]

    return (
        ext_max_build[ext_max_build != 0].stack().index.to_series().apply(lambda x: "-".join([x[0], x[1], str(x[2])]))
    ).values


def attach_extendable_generators(n, costs):
    config = snakemake.config["electricity"]
    ext_carriers = config["extendable_carriers"]["Generator"]
    ext_ref_years = config["conventional_generators"]["ext_ref_years"]
    ext_ref_gens = config["conventional_generators"]["extendable_reference"]
    ext_years = n.investment_periods if n.multi_invest else [n.snapshots[0].year]
    snapshots = n.snapshots.get_level_values(1) if n.multi_invest else n.snapshots

    #bus_carrier_years = [f"{bus}-{carrier}-{year}" for bus in n.buses.index for carrier in ext_carriers for year in ext_years]
    ext_gens_list = define_extendable_tech(model_file, model_setup, ext_years, "Generator")
    gens = set_default_extendable_params("Generator", ext_gens_list)
    
    pu_profiles = init_pu_profiles(gens, snapshots)
    
    # Monthly average EAF for conventional plants from Eskom  
    conv_pu = get_eskom_eaf(ext_ref_years, snapshots)[ext_ref_gens.values()]
    conv_pu.columns = ext_ref_gens.keys()
    conv_pu = proj_eaf_override(conv_pu, projections, snapshots, include = "_extendable_EAF", exclude = "NA")

    conv_carriers = [carrier for carrier in conv_pu.columns if carrier in ext_carriers]
    for col in gens.query("carrier in @conv_carriers & p_nom_extendable == True").index:
        pu_profiles.loc["max", col] = conv_pu[col.split("-")[1]].values

    # Hourly data from Eskom data portal
    eskom_ref_re_pu = generate_eskom_re_profiles(n)
    eskom_ref_re_carriers = [carrier for carrier in eskom_ref_re_pu.columns if carrier in ext_carriers] 
    for col in gens.query("carrier in @eskom_ref_re_carriers & p_nom_extendable == True").index:
        pu_profiles.loc["max", col] = eskom_ref_re_pu[gens.loc[col, "carrier"]].values

    # Wind and solar profiles if not using Eskom data portal
    if snakemake.config["enable"]["use_excel_wind_solar"][0]:
        ref_data = pd.ExcelFile(snakemake.config["enable"]["use_excel_wind_solar"][1])
        pu_profiles = generate_extendable_wind_solar_profiles_from_excel(n, gens, ref_data, snapshots, pu_profiles)
    
    gens = drop_non_pypsa_attrs(n, "Generator", gens)
    n.import_components_from_dataframe(gens, "Generator")
    n.generators["plant_name"] = n.generators.index.str.split("*").str[0]

    in_network = [g for g in pu_profiles.columns if g in n.generators.index]
    n.generators_t.p_max_pu = pd.concat([n.generators_t.p_max_pu, pu_profiles.loc["max", in_network]], axis = 1)
    n.generators_t.p_min_pu = pd.concat([n.generators_t.p_min_pu, pu_profiles.loc["min", in_network]], axis = 1)

    _add_missing_carriers_from_costs(n, costs, ext_carriers)

def set_default_extendable_params(c, bus_carrier_years, **config):
    default_param = [
                "bus",
                "p_nom_extendable",
                "carrier",
                "build_year",
                "lifetime",
                "capital_cost",
                "marginal_cost",
    ]
    if c == "Generator":
        default_param += ["efficiency"]
    elif c == "StorageUnit":
        default_param += ["max_hours", "efficiency_store", "efficiency_dispatch"]

    component_df = pd.DataFrame(index = bus_carrier_years, columns = default_param)

    component_df["p_nom_extendable"] = True
    component_df["bus"] = component_df.index.str.split("-").str[0]
    component_df["carrier"] = component_df.index.str.split("-").str[1]
    component_df["build_year"] = component_df.index.str.split("-").str[2].astype(int)

    for param in ["lifetime", "capital_cost", "marginal_cost", "efficiency"]:
        component_df[param] =  component_df.apply(lambda row: costs.loc[(param, row["carrier"]), row["build_year"]], axis=1)

    if c == "StorageUnit":
        component_df["cyclic_state_of_charge"] = True
        component_df["cyclic_state_of_charge_per_period"] = True
        component_df["efficiency_store"] = component_df["efficiency"]**0.5
        component_df["efficiency_dispatch"] = component_df["efficiency"]**0.5
        component_df["max_hours"] = component_df["carrier"].map(config["max_hours"])
        component_df = component_df.drop("efficiency", axis=1)
    return component_df

"""
********************************************************************************
    Functions to define and attach storage units to the network  
********************************************************************************
"""
def attach_existing_storage(n, model_setup, model_file): 
    carriers = ["phs", "battery"]
    start_year = get_start_year(n)
    
    storage = load_components_from_model_file(model_file, model_setup, carriers, start_year, snakemake.config["electricity"])
    storage = map_components_to_buses(storage, snakemake.input.supply_regions, snakemake.config["crs"])

    max_hours_col = [col for col in storage.columns if "_max_hours" in col]
    efficiency_col = [col for col in storage.columns if "_efficiency" in col]

    storage["max_hours"] = storage[max_hours_col].sum(axis=1)
    storage["efficiency_store"] = storage[efficiency_col].sum(axis=1)**0.5
    storage["efficiency_dispatch"] = storage[efficiency_col].sum(axis=1)**0.5
    storage["cyclic_state_of_charge"], storage["p_nom_extendable"] = True, False
    
    storage = drop_non_pypsa_attrs(n, "StorageUnit", storage)
    n.import_components_from_dataframe(storage, "StorageUnit")

def attach_extendable_storage(n, costs):
    config = snakemake.config["electricity"]
    carriers = config["extendable_carriers"]["StorageUnit"]
    ext_years = n.investment_periods if n.multi_invest else [n.snapshots[0].year]
    _add_missing_carriers_from_costs(n, costs, carriers)

    #bus_carrier_years = [f"{bus}-{carrier}-{year}" for bus in n.buses.index for carrier in carriers for year in ext_years]
    ext_storage_list = define_extendable_tech(model_file, model_setup, ext_years, "StorageUnit")
    storage = set_default_extendable_params("StorageUnit", ext_storage_list, **config)
    
    n.import_components_from_dataframe(storage, "StorageUnit")

"""
********************************************************************************
    Other functions
********************************************************************************
"""

def add_nice_carrier_names(n):

    carrier_i = n.carriers.index
    nice_names = (
        pd.Series(snakemake.config["plotting"]["nice_names"])
        .reindex(carrier_i)
        .fillna(carrier_i.to_series().str.title())
    )
    n.carriers["nice_name"] = nice_names
    colors = pd.Series(snakemake.config["plotting"]["tech_colors"]).reindex(carrier_i)
    if colors.isna().any():
        missing_i = list(colors.index[colors.isna()])
        logger.warning(
            f"tech_colors for carriers {missing_i} not defined " "in config."
        )
    n.carriers["color"] = colors


def add_load_shedding(n, cost):
    n.add("Carrier", "load_shedding")
    buses_i = n.buses.index
    n.madd(
        "Generator",
        buses_i,
        "_load_shedding",
        bus = buses_i,
        p_nom = 1e6,  # MW
        carrier = "load_shedding",
        build_year = get_start_year(n),
        lifetime = 100,
        marginal_cost = cost,
    )


"""
********************************************************************************
        MAIN  
********************************************************************************
"""


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake(
            "add_electricity", 
            **{
                "model_file":"grid-2040",
                "regions":"11-supply",
                "resarea":"redz",
            }
        )
    #configure_logging(snakemake, skip_handlers=False)
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(asctime)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    model_file = pd.ExcelFile(snakemake.input.model_file)
    model_setup = (
        pd.read_excel(
            model_file, 
            sheet_name="model_setup",
            index_col=[0])
            .loc[snakemake.wildcards.model_file]
    )

    projections = (
        pd.read_excel(
            model_file, 
            sheet_name="projected_parameters",
            index_col=[0,1])
            .loc[model_setup["projected_parameters"]]
    )

    #opts = snakemake.wildcards.opts.split("-")
    logging.info("Loading base network {snakemake.input.base_network}")
    n = pypsa.Network(snakemake.input.base_network)

    logging.info("Preparing costs")
    costs = load_costs(n, model_file, model_setup.costs, snakemake)

    #wind_solar_profiles = xr.open_dataset(snakemake.input.wind_solar_profiles).to_dataframe()
    #eskom_profiles = generate_eskom_re_profiles(n)
    logging.info("Attaching load")
    attach_load(n, projections.loc["annual_demand",:])

    if snakemake.wildcards.regions!="1-supply":
        update_transmission_costs(n, costs)

    logging.info("Attaching existing generators")
    attach_existing_generators(n, costs, model_setup, model_file)

    logging.info("Attaching extendable generators")
    attach_extendable_generators(n, costs)

    logging.info("Attaching existing storage")
    attach_existing_storage(n, model_setup, model_file)

    logging.info("Attaching extendable storage")
    attach_extendable_storage(n, costs) 

    if snakemake.config["solving"]["options"]["load_shedding"]:
        ls_cost = snakemake.config["costs"]["load_shedding"]
        logging.info("Adding load shedding")
        add_load_shedding(n, ls_cost) 

    add_nice_carrier_names(n)
    n.export_to_netcdf(snakemake.output[0])
