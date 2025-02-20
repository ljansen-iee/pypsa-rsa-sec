# SPDX-FileCopyrightText:  PyPSA-ZA2, PyPSA-ZA, PyPSA-Earth and PyPSA-Eur Authors
# # SPDX-License-Identifier: MIT

# coding: utf-8

"""
Adds fixed and extendable components to the base network. The primary functions run inside main are:

    attach_load
    attach_fixed_generators
    attach_extendable_generators
    attach_fixed_storage
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
    Documentation of the configuration file ``config/config.yaml`` at :ref:`costs_cf`,
    :ref:`electricity_cf`, :ref:`load_cf`, :ref:`renewable_cf`, :ref:`lines_cf`

Inputs
------
- ``config/model_file.xlsx``: The database to setup different scenarios based on cost assumptions for all included technologies for specific years from various sources; e.g. discount rate, lifetime, investment (CAPEX), fixed operation and maintenance (FOM), variable operation and maintenance (VOM), fuel costs, efficiency, carbon-dioxide intensity.
- ``data/Eskom EAF data.xlsx``: Hydropower plant store/discharge power capacities, energy storage capacity, and average hourly inflow by country.  Not currently used!
- ``data/bundle/eskom_pu_profiles.csv``: alternative to capacities above; not currently used!
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

import os
import geopandas as gpd
import logging
logger = logging.getLogger(__name__)
import numpy as np
import pandas as pd
import pypsa
from pypsa.descriptors import get_switchable_as_dense as get_as_dense, get_activity_mask
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
    get_investment_periods,
    adjust_by_p_max_pu,
    apply_default_attr,
    initial_ramp_rate_fix,
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

def load_extendable_parameters(n, model_file, model_setup, snakemake):
    """
    set all asset costs tab in the model file
    """

    param = pd.read_excel(
        model_file, 
        sheet_name = "extendable_parameters",
        index_col = [0,2,1],
    ).sort_index().loc[model_setup["extendable_parameters"]]

    param.drop("source", axis=1, inplace=True)
    
    # Interpolate for years in config file but not in cost_data excel file
    ext_years = get_investment_periods(n.snapshots, n.multi_invest)
    ext_years_array = np.array(ext_years)
    missing_year = ext_years_array[~np.isin(ext_years_array,param.columns)]
    if len(missing_year) > 0:
        for i in missing_year: 
            param.insert(0,i,np.nan) # add columns of missing year to dataframe
        param_tmp = param.drop("unit", axis=1).sort_index(axis=1)
        param_tmp = param_tmp.interpolate(axis=1)
        param= pd.concat([param_tmp, param["unit"]], ignore_index=False, axis=1)

    # correct units to MW and ZAR
    param_yr = param.columns.drop("unit")

    param = convert_cost_units(param, snakemake.config["costs"]["USD_to_ZAR"], 
                               snakemake.config["costs"]["EUR_to_ZAR"])
    
    full_param = pd.DataFrame(
        index = pd.MultiIndex.from_product(
            [
                param.index.get_level_values(0).unique(),
                param.index.get_level_values(1).unique()]),
        columns = param.columns
    )
    full_param.loc[param.index] = param.values

    # full_costs adds default values when missing from costs table
    config_defaults = snakemake.config["electricity"]["extendable_parameters"]["defaults"]
    for default in param.index.get_level_values(0).intersection(config_defaults.keys()):
        full_param.loc[param.loc[(default, slice(None)),:].index, :] = param.loc[(default, slice(None)),:]
        full_param.loc[(default, slice(None)), param_yr] = full_param.loc[(default, slice(None)), param_yr].fillna(config_defaults[default])
    #full_param = full_param.fillna("default")
    param = full_param.copy()

    # Get entries where FOM is specified as % of CAPEX
    fom_perc_capex=param.loc[param.unit.str.contains("%/year") == True, param_yr]
    fom_perc_capex_idx=fom_perc_capex.index.get_level_values(1)

    add_param = pd.DataFrame(
        index = pd.MultiIndex.from_product([["capital_cost","marginal_cost"],param.loc["FOM"].index]),
        columns = param.columns
    )
    
    param = pd.concat([param, add_param],axis=0)
    param.loc[("FOM",fom_perc_capex_idx), param_yr] = (param.loc[("investment", fom_perc_capex_idx),param_yr]).values/100.0
    param.loc[("FOM",fom_perc_capex_idx), "unit"] = param.loc[("investment", fom_perc_capex_idx),"unit"].values

    capital_costs = annualise_costs(
        param.loc["investment", param_yr],
        param.loc["lifetime", param_yr], 
        param.loc["discount_rate", param_yr],
        param.loc["FOM", param_yr],
    )

    param.loc["capital_cost", param_yr] = capital_costs.fillna(0).values
    param.loc["capital_cost","unit"] = "R/MWe"

    vom = param.loc["VOM", param_yr].fillna(0)
    fuel = (param.loc["fuel", param_yr] / param.loc["efficiency", param_yr]).fillna(0)

    param.loc[("marginal_cost", vom.index), param_yr] = vom.values
    param.loc[("marginal_cost", fuel.index), param_yr] += fuel.values
    param.loc["marginal_cost","unit"] = "R/MWhe"

    #max_hours = snakemake.config["electricity"]["max_hours"]
    #param.loc[("capital_cost","battery"), :] = param.loc[("capital_cost","battery inverter"),:]
    #param.loc[("capital_cost","battery"), param_yr] += max_hours["battery"]*param.loc[("capital_cost", "battery storage"), param_yr]
    
    return param

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
        n.add("Load", "RSA",
            bus="RSA",
            p_set=demand)
    else:
        n.madd("Load", list(n.buses.index),
            bus = list(n.buses.index),
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


def get_eaf_profiles(snapshots, type):
    outages = pd.read_excel(
            model_file, 
            sheet_name='outage_profiles',
            index_col=[0,1,2],
            header=[0,1],
    ).loc[model_setup["outage_profiles"]]
    outages = outages[type+"_generators"]
    eaf_mnthly = 1 - (outages.loc["planned"] + outages.loc["unplanned"])
    eaf_hrly = pd.DataFrame(1, index = snapshots, columns = eaf_mnthly.columns)
    eaf_hrly = eaf_mnthly.loc[eaf_hrly.index.month].reset_index(drop=True).set_index(eaf_hrly.index) 
    
    return eaf_hrly

def get_eskom_eaf(ref_yrs, snapshots):
    # Add plant availability based on actual Eskom data provided
    eskom_data  = pd.read_excel(
        snakemake.input.fixed_generators_eaf, 
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

def clip_eskom_eaf(eaf_hrly, gen_list, lower=0, upper=1):
    return eaf_hrly[gen_list].clip(lower=lower, upper=upper)

def clip_pu_profiles(n, pu, gen_list, lower=0, upper=1):
    n.generators_t[pu] = n.generators_t[pu].copy()
    n.generators_t[pu].loc[:, gen_list] = get_as_dense(n, "Generator", pu)[gen_list].clip(lower=lower, upper=upper)


def proj_eaf_override(eaf_hrly, projections, snapshots, include = "_EAF", exclude = "extendable"):
    """
    Overrides the hourly EAF (Energy Availability Factor) values with projected EAF values, if these are defined
    under the project_parameters tab in the config/model_file.xlsx. Existing generators have suffix _EAF and extendable generators
    have the suffix _extendable_EAF by convention.  

    Args:
    - eaf_hrly: A DataFrame containing the hourly EAF values.
    - projections: A DataFrame containing the projected EAF values.
    - snapshots: A Series containing the snapshots.
    - include: A string used to filter the projections based on the index.
    - exclude: A string used to exclude certain projections based on the index.

    Relevant config/model_file.xlsx settings:
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
    The user can specify whether to use this data under config/config.yaml. These Eskom profiles for biomass, hydro and hydro_import are used by default.

    Args:
    - n: The PyPSA network object.

    Relevant config/config.yaml settings:
    electricity:
        renewable_generators:
            carriers:
    years:
        reference_weather_years:
    enable:
        use_eskom_wind_solar
    """
    ext_years = get_investment_periods(n.snapshots, n.multi_invest)
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
def generate_fixed_wind_solar_profiles_from_excel(n, gens, ref_data, snapshots, pu_profiles):
    """
    Generates fixed wind and solar PV profiles for the network based on timeseries data supplied in Excel format.


    Args:
    - n: The network object.
    - gens: A DataFrame containing the generator information.
    - ref_data: The reference data file.
    - snapshots: A Series containing the snapshots.
    - pu_profiles: The DataFrame to store the pu profiles.

    Relevant config/config.yaml settings:
        enable:
            use_excel_wind_solar
    
            
    Relevant Excel data:
    Spreadsheet with the following tabs:
        fixed_wind_pu: Existing wind profiles
        fixed_solar_pv_pu: Existing solar PV profiles
        {regions}_wind_pu: Extendable wind profiles
        {regions}_solar_pv_pu: Extendable solar PV profiles

    """

    for carrier in ["wind", "solar_pv"]:
        pu = pd.read_excel(
            ref_data,
            sheet_name=f"fixed_{carrier}_pu",
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
        pu_ref = pu_ref[pu_ref.index.year.isin(snakemake.config["years"]["reference_weather_years"][carrier])]
        pu_ref = remove_leap_day(pu_ref)
        pu = pu_ref.copy()
        
        for y in years:
            pu.columns = pu_ref.columns + f"-{carrier}-{y}"
            pu_profiles.loc["max", pu.columns] = extend_reference_data(n, pu, snapshots).values

    return pu_profiles

def group_pu_profiles(pu_profiles, component_df):
    years = pu_profiles.index.get_level_values(1).year.unique()
    p_nom_pu = pd.DataFrame(1, index = pu_profiles.loc["max"].index, columns = [])
    pu_mul_p_nom = pu_profiles * component_df["p_nom"]

    filtered_df = component_df[component_df["apply_grouping"]].copy().fillna(0)

    for bus in filtered_df.bus.unique():
        for carrier in filtered_df.carrier.unique():
            carrier_list = filtered_df[(filtered_df["carrier"] == carrier) & (filtered_df["bus"] == bus)].index

            for y in years:
                active = carrier_list[(component_df.loc[carrier_list, ["build_year", "lifetime"]].sum(axis=1) >= y) & (component_df.loc[carrier_list, "build_year"] <= y)]
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
def load_components_from_model_file(carriers, start_year, config):
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
    conv_tech = read_and_filter_generators(model_file, "fixed_conventional", model_setup.fixed_conventional, carriers)
    re_tech = read_and_filter_generators(model_file, "fixed_renewables", model_setup.fixed_renewables, carriers)

    conv_tech["apply_grouping"] = config["conventional_generators"]["apply_grouping"]
    re_tech["apply_grouping"] = config["renewable_generators"]["apply_grouping"]
    re_tech.set_index((re_tech["Model Key"] + "_" + re_tech["Carrier"]).values,inplace=True)

    tech= pd.concat([conv_tech, re_tech])
    tech = map_component_parameters(tech, start_year)
    tech = tech.query("(p_nom > 0) & x.notnull() & y.notnull() & (lifetime >= 0)")
    
    return tech

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


def group_components(component_df, attrs):
    """
    Apply grouping of similar carrier if specified in snakemake config.

    Args:
        component_df: A DataFrame containing generator/storage_unit data.

    Returns:
        A tuple containing two DataFrames: grouped_df, non_grouped_df
    """
    
    params = ["bus", "carrier", "lifetime", "build_year", "p_nom", "efficiency", "ramp_limit_up", "ramp_limit_down", "marginal_cost", "capital_cost"]
    uc_params = ["ramp_limit_start_up","ramp_limit_shut_down", "start_up_cost", "shut_down_cost", "min_up_time", "min_down_time"] #,"p_min_pu"]
    params += uc_params    
    param_cols = [p for p in params if p not in ["bus","carrier","p_nom"]]

    filtered_df = component_df.query("apply_grouping").copy().fillna(0)#[component_df["apply_grouping"]].copy().fillna(0)

    grouped_df = pd.DataFrame(index=filtered_df.groupby(["Grouping", "carrier", "bus"]).sum().index, columns = param_cols)
    grouped_df["p_nom"] = filtered_df.groupby(["Grouping", "carrier", "bus"]).sum()["p_nom"]
    grouped_df["build_year"] = filtered_df.groupby(["Grouping", "carrier", "bus"]).min()["build_year"]
    grouped_df["lifetime"] = filtered_df.groupby(["Grouping", "carrier", "bus"]).max()["lifetime"]
    
    # calculate weighted average of remaining parameters in gens dataframe
    for param in [p for p in params if p not in ["bus","carrier","p_nom", "lifetime", "build_year"]]:
        weighted_sum = filtered_df.groupby(["Grouping", "carrier", "bus"]).apply(lambda x: (x[param] * x["p_nom"]).sum())
        total_p_nom = filtered_df.groupby(["Grouping", "carrier", "bus"])["p_nom"].sum()
        weighted_average = weighted_sum / total_p_nom 
        grouped_df.loc[weighted_average.index, param] = weighted_average.values
    
    rename_idx = grouped_df.index.get_level_values(2) +  "-" + grouped_df.index.get_level_values(1) +  "_" + grouped_df.index.get_level_values(0)
    grouped_df = grouped_df.reset_index(level=[1,2]).replace(0, np.nan).set_index(rename_idx) 
    non_grouped_df = component_df[~component_df["apply_grouping"]][params].copy()

    # Fill missing values with default values (excluding defaults that are NaN)    
    grouped_df = apply_default_attr(grouped_df, attrs)
    non_grouped_df = apply_default_attr(non_grouped_df, attrs)

    return grouped_df, non_grouped_df

def attach_fixed_generators(n, costs):
    # setup carrier info
    gen_attrs = n.component_attrs["Generator"]
    config = snakemake.config["electricity"]
    fix_ref_years = config["conventional_generators"]["fix_ref_years"]
    ext_years = get_investment_periods(n.snapshots, n.multi_invest)
    conv_carriers = config["conventional_generators"]["carriers"]
    re_carriers = config["renewable_generators"]["carriers"]
    carriers = conv_carriers + re_carriers

    start_year = get_start_year(n.snapshots, n.multi_invest)
    snapshots = get_snapshots(n.snapshots, n.multi_invest)
    
    # load generators from model file
    gens = load_components_from_model_file(carriers, start_year, snakemake.config["electricity"])
    gens = map_components_to_buses(gens, snakemake.input.supply_regions, snakemake.config["crs"])
    
    if len(ext_years) == 1: 
        gens = gens.loc[
            (gens.loc[:, ["build_year", "lifetime"]].sum(axis=1) >= ext_years[0]) 
            & (gens.loc[:, "build_year"] <= ext_years[0])]
    
    pu_profiles = init_pu_profiles(gens, snapshots)

    unique_entries = set()
    coal_gens =  [unique_entries.add(g.split("*")[0]) or g.split("*")[0] for g in gens[gens.carrier == 'coal'].index if g.split("*")[0] not in unique_entries]
    # Monthly average EAF for conventional plants from Eskom  
  
    #conv_pu = get_eskom_eaf(fix_ref_years, snapshots)
    conv_pu = get_eaf_profiles(snapshots, "fixed")
    
    not_in_pu = [g for g in coal_gens if g not in conv_pu.columns]
    conv_pu[not_in_pu] = 1    

    conv_pu[coal_gens] = clip_eskom_eaf(conv_pu, coal_gens, lower=0.3, upper=1)
    conv_pu = proj_eaf_override(conv_pu, projections, snapshots, include = "_EAF", exclude = "extendable")
    eskom_carriers = [carrier for carrier in conv_carriers if carrier not in ["nuclear", "hydro", "hydro_import"]]
    for col in gens.query("Grouping == 'eskom' & carrier in @eskom_carriers").index:
        pu_profiles.loc["max", col] = conv_pu[col.split("*")[0]].values

    # Hourly data from Eskom data portal
    eskom_re_pu = generate_eskom_re_profiles(n)
    eskom_re_carriers = eskom_re_pu.columns
    for col in gens.query("carrier in @eskom_re_carriers").index:
        pu_profiles.loc["max", col] = eskom_re_pu[gens.loc[col, "carrier"]].values
        if snakemake.config["electricity"]["min_hourly_station_gen"]["enable_for_eskom_re"]:
            pu_profiles.loc["min", col] = eskom_re_pu[gens.loc[col, "carrier"]].values

    # Wind and solar profiles if not using Eskom data portal
    if snakemake.config["enable"]["use_excel_wind_solar"][0]:
        ref_data = pd.ExcelFile(snakemake.config["enable"]["use_excel_wind_solar"][1])
        pu_profiles = generate_fixed_wind_solar_profiles_from_excel(n, gens, ref_data, snapshots, pu_profiles)

    pu_profiles, p_nom_pu = group_pu_profiles(pu_profiles, gens) #includes both grouped an non-grouped generators
    grouped_gens, non_grouped_gens = group_components(gens, gen_attrs)
    grouped_gens["p_nom_extendable"] = False
    non_grouped_gens["p_nom_extendable"] = False
    
    n.import_components_from_dataframe(drop_non_pypsa_attrs(n, "Generator", non_grouped_gens), "Generator")
    n.import_components_from_dataframe(drop_non_pypsa_attrs(n, "Generator", grouped_gens), "Generator")

    pu_max, pu_min = pu_profiles.loc["max"], pu_profiles.loc["min"]
    pu_max.index, pu_min.index, p_nom_pu.index = n.snapshots, n.snapshots, n.snapshots

    n.generators_t.p_nom_pu = p_nom_pu
    n.generators_t.p_max_pu = pu_max.clip(lower=0.0, upper=1.0)
    n.generators_t.p_min_pu = pu_min.clip(lower=0.0, upper=1.0)
    
    for carrier, value in snakemake.config["electricity"]["min_hourly_station_gen"]["fixed"].items():
        clip_pu_profiles(n, "p_min_pu", n.generators.query("carrier == @carrier & p_nom_extendable == False").index, lower=value, upper=1.0)


def extendable_max_build_per_bus_per_carrier():

    ext_max_build = (
        pd.read_excel(
            model_file, 
            sheet_name='extendable_max_build',
            index_col=[0,1,2,3])
    ).loc[model_setup]
    ext_max_build.replace("unc", np.inf, inplace=True)

    return ext_max_build.loc[snakemake.wildcards.regions]

def define_extendable_tech(years, type_, ext_param):

    ext_max_build = pd.read_excel(
        model_file, 
        sheet_name='extendable_max_build',
        index_col= [0,1,3,2,4],
    ).loc[(model_setup["extendable_build_limits"], snakemake.wildcards.regions, type_, slice(None)), years]
    ext_max_build.replace("unc", np.inf, inplace=True)
    ext_max_build.index = ext_max_build.index.droplevel([0, 1, 2])
    ext_max_build = ext_max_build.loc[~(ext_max_build==0).all(axis=1)]

    carrier_names = ext_max_build.index.get_level_values(1)
    if bad_name := list(carrier_names[carrier_names.str.contains("-")]):
        logger.warning(f"Carrier names in extendable_max_build sheet must not contain the character '-'. The following carriers will be ignored: {bad_name}")
        ext_max_build = ext_max_build[~carrier_names.str.contains("-")]
    
    carrier_names = ext_max_build.index.get_level_values(1)
    if bad_name := list(carrier_names[~carrier_names.isin(ext_param.index.get_level_values(1))]):
        logger.warning(f"Carrier names in extendable_max_build sheet must be in the extendable_paramaters sheet. The following carriers will be ignored: {bad_name}")
        ext_max_build = ext_max_build[carrier_names.isin(ext_param.index.get_level_values(1))]

    return (
        ext_max_build[ext_max_build != 0].stack().index.to_series().apply(lambda x: "-".join([x[0], x[1], str(x[2])]))
    ).values


def attach_extendable_generators(n, ext_param):
    gen_attrs = n.component_attrs["Generator"]
    config = snakemake.config["electricity"]
    ext_ref_years = config["conventional_generators"]["ext_ref_years"]
    ext_ref_gens = config["conventional_generators"]["extendable_reference"] 
    
    ext_years = get_investment_periods(n.snapshots, n.multi_invest)
    snapshots = get_snapshots(n.snapshots, n.multi_invest)

    ext_gens_list = define_extendable_tech(ext_years, "Generator", ext_param) 
    gens = set_extendable_params("Generator", ext_gens_list, ext_param)
    pu_profiles = init_pu_profiles(gens, snapshots)
    
    # Monthly average EAF for conventional plants from Eskom
    #conv_pu = get_eskom_eaf(ext_ref_years, snapshots)[ext_ref_gens.values()]
    conv_pu = get_eaf_profiles(snapshots, "extendable")
    conv_pu["coal"] = clip_eskom_eaf(conv_pu, gen_list = ["coal"], lower=0.3, upper=1)
    conv_pu.columns = ext_ref_gens.keys()
    conv_pu = proj_eaf_override(conv_pu, projections, snapshots, include = "_extendable_EAF", exclude = "NA")

    conv_carriers = [carrier for carrier in conv_pu.columns if carrier in n.generators.carrier.unique()]

    for col in gens.query("carrier in @conv_carriers & p_nom_extendable == True").index:
        pu_profiles.loc["max", col] = conv_pu[col.split("-")[1]].values

    # Hourly data from Eskom data portal
    eskom_ref_re_pu = generate_eskom_re_profiles(n)  
    eskom_ref_re_carriers = [carrier for carrier in eskom_ref_re_pu.columns if carrier in n.generators.carrier.unique()]# and carrier not in committable_carriers

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

    pu_max, pu_min = pu_profiles.loc["max", in_network], pu_profiles.loc["min", in_network]
    pu_max.index, pu_min.index = n.snapshots, n.snapshots
    n.generators_t.p_max_pu[in_network] = pu_max[in_network].clip(lower=0.0, upper=1.0)
    n.generators_t.p_min_pu[in_network] = pu_min[in_network].clip(lower=0.0, upper=1.0)

    for carrier, value in snakemake.config["electricity"]["min_hourly_station_gen"]["fixed"].items():
        clip_pu_profiles(n, "p_min_pu", n.generators.query("carrier == @carrier & p_nom_extendable").index, lower=value, upper=1.0)


def set_extendable_params(c, bus_carrier_years, ext_param, **config):
    default_param = [
        "bus",
        "p_nom_extendable",
        "carrier",
        "build_year",
        "lifetime",
        "capital_cost",
        "marginal_cost",
        "ramp_limit_up",
        "ramp_limit_down",
        "efficiency",
    ]
    uc_param = [
        "ramp_limit_start_up",
        "ramp_limit_shut_down",
        "min_up_time",
        "min_down_time",
        "start_up_cost",
        "shut_down_cost",
    ]


    if c == "StorageUnit":
        default_param += ["max_hours", "efficiency_store", "efficiency_dispatch"]

    default_col = [p for p in default_param if p not in ["bus", "carrier", "build_year", "p_nom_extendable", "efficiency_store", "efficiency_dispatch"]]

    component_df = pd.DataFrame(index = bus_carrier_years, columns = default_param)
    component_df["p_nom_extendable"] = True
    component_df["p_nom"] = 0
    component_df["bus"] = component_df.index.str.split("-").str[0]
    component_df["carrier"] = component_df.index.str.split("-").str[1]
    component_df["build_year"] = component_df.index.str.split("-").str[2].astype(int)
    
    if c == "Generator":
        component_df = pd.concat([component_df, pd.DataFrame(index = bus_carrier_years, columns = uc_param)],axis=1)
        for param in default_col + uc_param:
            component_df[param] =  component_df.apply(lambda row: ext_param.loc[(param, row["carrier"]), row["build_year"]], axis=1)
            component_df = apply_default_attr(component_df, n.component_attrs[c])
    elif c == "StorageUnit":
        for param in default_col:
            component_df[param] =  component_df.apply(lambda row: ext_param.loc[(param, row["carrier"]), row["build_year"]], axis=1)
        
        component_df["cyclic_state_of_charge"] = True
        component_df["cyclic_state_of_charge_per_period"] = True
        component_df["efficiency_store"] = component_df["efficiency"]**0.5
        component_df["efficiency_dispatch"] = component_df["efficiency"]**0.5
        component_df = component_df.drop("efficiency", axis=1)
    
    return component_df

"""
********************************************************************************
    Functions to define and attach storage units to the network  
********************************************************************************
"""
def attach_fixed_storage(n): 
    carriers = ["phs", "battery"]
    start_year = get_start_year(n.snapshots, n.multi_invest)
    
    storage = load_components_from_model_file(carriers, start_year, snakemake.config["electricity"])
    storage = map_components_to_buses(storage, snakemake.input.supply_regions, snakemake.config["crs"])

    max_hours_col = [col for col in storage.columns if "_max_hours" in col]
    efficiency_col = [col for col in storage.columns if "_efficiency" in col]

    storage["max_hours"] = storage[max_hours_col].sum(axis=1)
    storage["efficiency_store"] = storage[efficiency_col]**0.5
    storage["efficiency_dispatch"] = storage[efficiency_col]**0.5
    storage["cyclic_state_of_charge"], storage["p_nom_extendable"] = True, False
    storage["p_min_pu"] = -1

    storage = drop_non_pypsa_attrs(n, "StorageUnit", storage)
    n.import_components_from_dataframe(storage, "StorageUnit")

def attach_extendable_storage(n, ext_param):
    config = snakemake.config["electricity"]
    ext_years = get_investment_periods(n.snapshots, n.multi_invest)
    ext_storage_list = define_extendable_tech(ext_years, "StorageUnit", ext_param)
    storage = set_extendable_params("StorageUnit", ext_storage_list, ext_param, **config)
    storage = drop_non_pypsa_attrs(n, "StorageUnit", storage)
    n.import_components_from_dataframe(storage, "StorageUnit")


"""
********************************************************************************
    Transmission network functions  
********************************************************************************
"""

def convert_lines_to_links(n):
    """
    Convert AC lines to DC links for multi-decade optimisation with line
    expansion.

    Losses of DC links are assumed to be 3% per 1000km
    """
    years = get_investment_periods(n.snapshots, n.multi_invest)

    logger.info("Convert AC lines to DC links to perform multi-decade optimisation.")

    for y in years:
        lines = n.lines.copy()
        lines.index = lines.index + "-" + str(y)
        
        n.madd(
            "Link",
            lines.index,
            bus0 = lines.bus0,
            bus1 = lines.bus1,
            carrier="DC link",
            p_nom = lines.s_nom if y == years[0] else 0,
            #p_nom_min = lines.s_nom,
            p_max_pu = lines.s_max_pu,
            p_min_pu = -1 * lines.s_max_pu,
            build_year = y,
            #lifetime = 100,
            efficiency = 1 - 0.03 * lines.length / 1000,
            marginal_cost = 0,
            length = lines.length,
            capital_cost = lines.capital_cost,
            p_nom_extendable = False if y == years[0] else False,
        )

    # Remove AC lines
    logger.info("Removing AC lines")
    lines_rm = n.lines.index
    n.mremove("Line", lines_rm)



"""
********************************************************************************
    Other functions
********************************************************************************
"""
# def overwrite_extendable_with_committable(n, model_file, model_setup, param):

#     com_i = n.generators.query("committable & p_nom_extendable").index

#     if len(com_i) >= 0:
#             logging.warning("""A generator cannot be both extendable and committable, as this would make the problem non linear. 
#             Setting p_nom for committable generators to the extendable_max_build value in the model_file. If no build limit is
#             specified (i.e 'unc'), the generator is returned to being extendable and the committable flag is ignored. """
#             )
#     else:
#         return

#     ext_years = get_investment_periods(n)
#     fix_capacity = pd.read_excel(
#         model_file,
#         sheet_name='extendable_max_build',
#         index_col=[0, 1, 3, 2, 4],
#     ).sort_index().loc[
#         (
#             model_setup["extendable_build_limits"],
#             snakemake.wildcards.regions,
#             "Generator",
#         ),
#         ext_years,
#     ]


#     for _ in range(len(com_i)):
#         bus = com_i.str.split("-").str[0][_]
#         carrier = com_i.str.split("-").str[1][_]
#         y = int(com_i.str.split("-").str[2][_])
#         capacity = fix_capacity.loc[(bus, carrier), y]
#         if capacity != "unc":
#             n.generators.loc[com_i[_], "p_nom"] = capacity
#             n.generators.loc[com_i[_], "p_nom_extendable"] = False
#             n.generators_t.p_max_pu[com_i[_]] = 1
#             n.generators_t.p_min_pu[com_i[_]] = 0
#             n.generators.loc[com_i[_], "p_min_pu"] = param.loc[("min_stable_level", carrier), y]
#         else:
#             logging.warning(f"Removing committable flag from generator {com_i[_]} and defaulting to extendable.")
#             n.generators.loc[com_i[_], "committable"] = False
            
        
def add_nice_carrier_names(n, config):

    carrier_i = n.carriers.index
    nice_names = (
        pd.Series(config["plotting"]["nice_names"])
        .reindex(carrier_i)
        .fillna(carrier_i.to_series().str.title())
    )
    n.carriers["nice_name"] = nice_names
    colors = pd.Series(config["plotting"]["tech_colors"]).reindex(carrier_i)
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
        build_year = get_start_year(n.snapshots, n.multi_invest),
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
        from _helpers import mock_snakemake, sets_path_to_root
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        snakemake = mock_snakemake(
            "add_electricity", 
            **{
                "model_file":"NZ-2040",
                "regions":"11-supply",
                "resarea":"corridors",
            }
        )
        sets_path_to_root("pypsa-rsa-sec")
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
    logging.info(f"Loading base network {snakemake.input.base_network}")
    n = pypsa.Network(snakemake.input.base_network)

    logging.info("Preparing extendable parameters")
    param = load_extendable_parameters(n, model_file, model_setup, snakemake)

    #wind_solar_profiles = xr.open_dataset(snakemake.input.wind_solar_profiles).to_dataframe()
    #eskom_profiles = generate_eskom_re_profiles(n)
    logging.info("Attaching load")
    attach_load(n, projections.loc["annual_demand",:])

    if snakemake.wildcards.regions!="1-supply":
        update_transmission_costs(n, param)

    logging.info("Attaching fixed generators")
    attach_fixed_generators(n, param)

    logging.info("Attaching extendable generators")
    attach_extendable_generators(n, param)

    logging.info("Attaching fixed storage")
    attach_fixed_storage(n)

    logging.info("Attaching extendable storage")
    attach_extendable_storage(n, param)

    adj_by_pu = snakemake.config["electricity"]["adjust_by_p_max_pu"]
    logging.info(f"Adjusting by p_max_pu for {list(adj_by_pu.keys())}")
    adjust_by_p_max_pu(n, adj_by_pu)

    if snakemake.config["solving"]["options"]["load_shedding"]:
        ls_cost = snakemake.config["costs"]["load_shedding"]
        logging.info("Adding load shedding")
        add_load_shedding(n, ls_cost) 

    _add_missing_carriers_from_costs(n, param, n.generators.carrier.unique())
    _add_missing_carriers_from_costs(n, param, n.storage_units.carrier.unique())

    add_nice_carrier_names(n, snakemake.config)

    logging.info("Exporting network.")
    if n.multi_invest:
        initial_ramp_rate_fix(n)

    n.export_to_netcdf(snakemake.output[0])
