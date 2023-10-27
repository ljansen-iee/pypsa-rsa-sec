# SPDX-FileCopyrightText:  PyPSA-ZA2, PyPSA-ZA, PyPSA-Earth and PyPSA-Eur Authors
# # SPDX-License-Identifier: MIT

# coding: utf-8

"""
Adds electrical generators, load and existing hydro storage units to a base network.

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


from email import generator
import logging
import os
import geopandas as gpd
import numpy as np
import pandas as pd
import powerplantmatching as pm
import pypsa
import re
import xarray as xr
from _helpers import (
    convert_cost_units,
    configure_logging, 
    update_p_nom_max, 
    pdbcast, 
    map_generator_parameters, 
    clean_pu_profiles,
    remove_leap_day,
    add_row_multi_index_df
)

from shapely.validation import make_valid
from shapely.geometry import Point
from vresutils import transfer as vtransfer
idx = pd.IndexSlice
logger = logging.getLogger(__name__)
from pypsa.descriptors import get_switchable_as_dense as get_as_dense

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning) # Comment out for debugging and development

def normed(s):
    return s / s.sum()

def annual_costs(investment, lifetime, discount_rate, FOM):
    CRF = discount_rate / (1 - 1 / (1 + discount_rate) ** lifetime)
    return (investment * CRF + FOM).fillna(0)

def _add_missing_carriers_from_costs(n, costs, carriers):
    start_year = n.snapshots.get_level_values(0)[0] if n._multi_invest else n.snapshots[0].year
    missing_carriers = pd.Index(carriers).difference(n.carriers.index)
    if missing_carriers.empty: return

    emissions = costs.loc[("co2_emissions",missing_carriers),start_year]
    emissions.index = emissions.index.droplevel(0)
    n.madd("Carrier", missing_carriers, co2_emissions=emissions)
    
def load_costs(model_file, cost_scenario):
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
    ext_years = n.investment_periods if n._multi_invest else [n.snapshots[0].year]
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

    capital_costs = annual_costs(
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

def add_generator_availability(n, projections):
    config = snakemake.config["electricity"]["conventional_generators"]
    fix_ref_years = config["fix_ref_years"]
    ext_ref_years = config["ext_ref_years"]
    conv_carriers = config["carriers"]
    ext_unit_ref = config["extendable_reference"]
    conv_extendable = snakemake.config["electricity"]["extendable_carriers"]['Generator']

    fix_i = n.generators[(n.generators.carrier.isin(conv_carriers)) & (~n.generators.p_nom_extendable)].index
    fix_i_station = pd.Series(index=fix_i, data= [name.split('*')[0].strip() for name in fix_i])
    ext_i = n.generators[(n.generators.carrier.isin(conv_extendable)) & (n.generators.p_nom_extendable)].index
   
    # Add plant availability based on actual Eskom data provided
    eskom_data  = pd.read_excel(
        snakemake.input.existing_generators_eaf, 
        sheet_name="eskom_data", 
        na_values=["-"],
        index_col=[0,1],
        parse_dates=True
    )
    snapshots = n.snapshots.get_level_values(1) if n._multi_invest else n.snapshots
    
    # check for stations without eskom data and add default for extendables
    fix_i_missing = fix_i_station[~fix_i_station.isin(eskom_data.index.get_level_values(1).unique())].index
    fix_i_missing_car = n.generators.carrier.loc[fix_i_missing]
    
    # Non -extendable generators
    eaf = eskom_data.loc[(slice(None),fix_i_station[fix_i.drop(fix_i_missing)]), "EAF %"]/100
    
    def process_eaf(eskom_data, ref_yrs, snapshots, carriers):
        eaf = eskom_data["EAF %"]/100#
        eaf = eaf.loc[eaf.index.get_level_values(0).year.isin(ref_yrs)]  
        eaf_m = eaf.groupby(["station", eaf.index.get_level_values(0).month]).mean().unstack(level=0)[carriers]

        eaf_h = pd.DataFrame(1, index = snapshots, columns = eaf_m.columns)
        eaf_h = eaf_m.loc[eaf_h.index.month].reset_index(drop=True).set_index(eaf_h.index)
        eaf_y = eaf_h.groupby(eaf_h.index.year).mean()
        return eaf_h, eaf_y
    
    def proj_eaf_override(projections, eaf_h, eaf_y, include = "_EAF", exclude = "extendable"):
        proj_eaf = projections.loc[(projections.index.str.contains(include) & ~projections.index.str.contains(exclude)),snapshots.year.unique()]
        proj_eaf.index = proj_eaf.index.str.replace(include,"")

        # remove decom_stations
        proj_eaf = proj_eaf[proj_eaf.index.isin(eaf_y.columns)]
        scaling = proj_eaf.T.div(eaf_y[proj_eaf.index], axis="columns", level="year").fillna(1)

        for y in snapshots.year.unique():
            eaf_h.loc[str(y), scaling.columns] *= scaling.loc[y, :]  

        return eaf_h
    
    fix_st_eaf_h, fix_st_eaf_y = process_eaf(eskom_data, fix_ref_years, snapshots, eaf.index.get_level_values(1).unique())
    fix_st_eaf_h = proj_eaf_override(projections, fix_st_eaf_h, fix_st_eaf_y, include = "_EAF", exclude = "extendable")

    # map back to station units
    fix_eaf_h = pd.DataFrame(index = snapshots, columns = fix_i.drop(fix_i_missing))
    # Using .loc indexer to align data based on the mapping from fix_i_station
    fix_eaf_h = fix_st_eaf_h.loc[:, fix_i_station[fix_i.drop(fix_i_missing).values]].copy()
    fix_eaf_h.columns = fix_i.drop(fix_i_missing).values

    # Extendable generators and missing fixed generators
    # check if carriers from fix_i_missing_car are all in car_extendable
    for carrier in fix_i_missing_car[~fix_i_missing_car.isin(conv_extendable)].unique():
        conv_extendable.append(carrier)
    conv_extendable = [c for c in conv_extendable if c in ext_unit_ref]


    car_eaf_h, car_eaf_y = process_eaf(eskom_data, ext_ref_years, snapshots, ext_unit_ref.values())
    car_eaf_h.columns, car_eaf_y.columns = ([k for k, v in ext_unit_ref.items() if v in ext_unit_ref.values()] for _ in range(2))
    car_eaf_h = proj_eaf_override(projections, car_eaf_h, car_eaf_y, include = "_extendable_EAF", exclude="NA")
    
    ext_eaf_h = pd.DataFrame(index = snapshots, columns = ext_i.append(fix_i_missing_car.index))
    for g in ext_i.append(fix_i_missing_car.index):
        carrier = n.generators.loc[g, "carrier"]
        if carrier in car_eaf_h.columns:
            ext_eaf_h[g] = car_eaf_h[carrier].values
    ext_eaf_h = ext_eaf_h.fillna(1) # if reference not specified

    eaf_h = pd.concat([fix_eaf_h, ext_eaf_h],axis=1).fillna(1)
    eaf_h[eaf_h >1] = 1
    eaf_h.index = n.snapshots
    n.generators_t.p_max_pu[eaf_h.columns] = eaf_h 


def add_min_stable_levels(n):
    min_st_lvl = snakemake.config["electricity"]["min_stable_levels"]

    static = [k for k in min_st_lvl.keys() if min_st_lvl[k][1] == 'static']
    dynamic = [k for k in min_st_lvl.keys() if min_st_lvl[k][1] == 'dynamic']

    for carrier in static:
        gen_lst = n.generators[n.generators.carrier == carrier].index
        n.generators_t.p_min_pu[gen_lst] = min_st_lvl[carrier][0]

    for carrier in dynamic:
        gen_lst = n.generators[n.generators.carrier == carrier].index
        p_max_pu = get_as_dense(n, "Generator", "p_max_pu")[gen_lst]
        n.generators_t.p_min_pu[gen_lst] = (min_st_lvl[carrier][0]*p_max_pu).fillna(0)

 ## Attach components
# ### Load

def attach_load(n, annual_demand):
    load = pd.read_csv(snakemake.input.load,index_col=[0],parse_dates=True)
    
    annual_demand = annual_demand.drop("unit")*1e6
    profile_demand = normed(remove_leap_day(load.loc[str(snakemake.config["years"]["reference_demand_year"]),"system_energy"]))
    
    if n._multi_invest:
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
            bus=n.buses.index,
            p_set=pdbcast(demand, normed(n.buses[snakemake.config["electricity"]["demand_disaggregation"]])))


### Generate pu profiles for other_re based on Eskom data
def generate_eskom_profiles(n):
    carriers = snakemake.config["electricity"]["renewable_carriers"]
    ref_years = snakemake.config["years"]["reference_weather_years"]

    if snakemake.config["enable"]["use_excel_wind_solar"][0]:
        carriers = [ elem for elem in carriers if elem not in ["onwind","solar_pv"]]

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
        weather_years = ref_years[carrier]
        for _ in range(int(np.ceil(len(n.investment_periods) / len(weather_years)) - 1)):
            weather_years += weather_years

        # Use the default RSA hourly data (from Eskom) and extend to multiple weather years
        for cnt, y in enumerate(n.investment_periods):
            eskom_profiles.loc[y, carrier] = (eskom_data.loc[str(weather_years[cnt]), carrier]
                                              .clip(lower=0., upper=1.)).values
    return eskom_profiles

def generate_excel_wind_solar_profiles(n):
    ref_years = snakemake.config["years"]["reference_weather_years"]
    snapshots = n.snapshots.get_level_values(1) if n._multi_invest else n.snapshots
    profiles=pd.DataFrame(index=pd.MultiIndex.from_product([["onwind", "solar_pv"], snapshots], names=["Generator", "snapshots"]), columns=n.buses.index)
    ext_years = n.investment_periods if n._multi_invest else [n.snapshots[0].year]


    for carrier in ["onwind","solar_pv"]:
        raw_profiles= (
            pd.read_excel(snakemake.config["enable"]["use_excel_wind_solar"][1],
            sheet_name=snakemake.wildcards.regions+"_"+carrier+"_pu",
            skiprows=[1], 
            index_col=0,parse_dates=True)
            .resample("1h").mean()
        )
        raw_profiles = remove_leap_day(raw_profiles)

        weather_years=ref_years[carrier]
        for _ in range(int(np.ceil(len(ext_years)/len(weather_years))-1)):
            weather_years+=weather_years

        # Use the default RSA hourly data (from Eskom) and extend to multiple weather years
        for cnt, y in enumerate(ext_years):    
            profiles.loc[(carrier, str(y)), n.buses.index] = (
                raw_profiles.loc[str(weather_years[cnt]),n.buses.index]
                .clip(lower=0., upper=1.)
            ).values
    if n._multi_invest:
        profiles["periods"] = profiles.index.get_level_values(1).year
        profiles = profiles.reset_index().set_index(["Generator", "periods", "snapshots"])
    return profiles


### Set line costs
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


# ### Generators - TODO Update from pypa-eur
def attach_wind_and_solar(n, costs,input_profiles, model_setup, eskom_profiles):
    start_year = n.snapshots.get_level_values(0)[0] if n._multi_invest else n.snapshots[0].year
    ext_years = n.investment_periods if n._multi_invest else [n.snapshots[0].year]

    # Aggregate existing REIPPPP plants per region
    eskom_gens = pd.read_excel(
        snakemake.input.model_file, 
        sheet_name="existing_eskom", 
        na_values=["-"],
        index_col=[0,1]
    ).loc[model_setup.existing_eskom]
    eskom_gens = eskom_gens[eskom_gens["Carrier"].isin(["solar_pv","onwind"])] # Currently only Sere wind farm

    ipp_gens = pd.read_excel(
        snakemake.input.model_file, 
        sheet_name="existing_non_eskom", 
        na_values=["-"],
        index_col=[0,1]
    ).loc[model_setup.existing_non_eskom]
    ipp_gens=ipp_gens[ipp_gens["Carrier"].isin(["solar_pv","onwind"])] # add existing wind and PV IPP generators 

    gens = pd.concat([eskom_gens,ipp_gens])
    gens["bus"]=np.nan
    # Calculate fields where pypsa uses different conventions
    gens = map_generator_parameters(gens, start_year) 

    # Associate every generator with the bus of the region it is in or closest to
    pos = gpd.GeoSeries([Point(o.x, o.y) for o in gens[["x", "y"]].itertuples()], index=gens.index)
    regions = gpd.read_file(
        snakemake.input.supply_regions,
    ).to_crs(snakemake.config["crs"]["geo_crs"]).set_index("name")

    for bus, region in regions.geometry.items():
        pos_at_bus_b = pos.within(region)
        if pos_at_bus_b.any():
            gens.loc[pos_at_bus_b, "bus"] = bus
    gens.loc[gens.bus.isnull(), "bus"] = pos[gens.bus.isnull()].map(lambda p: regions.distance(p).idxmin())
    gens.loc["Sere","Grouping"] = "REIPPPP_BW1" #add Sere wind farm to BW1 for simplification #TODO fix this to be general

    # Aggregate REIPPPP bid window generators at each bus #TODO use capacity weighted average for lifetime, costs
    for carrier in ["solar_pv","onwind"]:
        plant_data = gens.loc[gens["carrier"]==carrier,["Grouping","bus","p_nom"]].groupby(["Grouping","bus"]).sum()
        for param in ["lifetime","capital_cost","marginal_cost"]:
            plant_data[param]=gens.loc[gens["carrier"]==carrier,["Grouping","bus",param]].groupby(["Grouping","bus"]).mean()

        resource_carrier=pd.DataFrame(0,index=n.snapshots,columns=n.buses.index)
        if ((snakemake.config["enable"]["use_eskom_wind_solar"]==False) &
            (snakemake.config["enable"]["use_excel_wind_solar"][0]==False)):
            ds = xr.open_dataset(getattr(input_profiles, "profile_" + carrier))
            for y in n.investment_periods: 
                atlite_data = ds["profile"].transpose("time", "bus").to_pandas()
                    #resource_carrier.loc[y] = (atlite_data.loc[str(weather_years[cnt])].clip(lower=0., upper=1.)).values
                resource_carrier.loc[y] = atlite_data.clip(lower=0., upper=1.).values[:8760]
        elif (snakemake.config["enable"]["use_excel_wind_solar"][0]):
            excel_wind_solar_profiles = generate_excel_wind_solar_profiles(n)    
            resource_carrier[n.buses.index] = excel_wind_solar_profiles.loc[carrier, n.buses.index]
        else:
            for bus in n.buses.index:
                resource_carrier[bus] = eskom_profiles[carrier].values # duplicate aggregate Eskom profile if specified

        for group in plant_data.index.levels[0]:
            n.madd(
                "Generator", plant_data.loc[group].index, suffix=" "+group+"_"+carrier,
                bus = plant_data.loc[group].index,
                carrier = carrier,
                build_year = start_year,
                lifetime = plant_data.loc[group,"lifetime"],
                p_nom = plant_data.loc[group,"p_nom"],
                p_nom_extendable = False,
                marginal_cost = plant_data.loc[group,"marginal_cost"],
                p_max_pu = resource_carrier[plant_data.loc[group].index].values,
                p_min_pu = resource_carrier[plant_data.loc[group].index].values*0.95, # for existing PPAs force to buy all energy produced
            )


    # Add new generators
        for y in ext_years:
            #TODO add check here to exclude buses where p_nom_max = 0 
            n.madd(
                "Generator", n.buses.index, suffix=" "+carrier+"_"+str(y),
                bus = n.buses.index,
                carrier = carrier,
                build_year = y,
                lifetime = costs.loc[("lifetime",carrier), y],
                p_nom_extendable = True,
                #p_nom_max=ds["p_nom_max"].to_pandas(), # For multiple years a further constraint is applied in prepare_network.py
                #weight=ds["weight"].to_pandas(),
                marginal_cost = costs.loc[("marginal_cost",carrier), y],#costs[y].at[carrier, "marginal_cost"],
                capital_cost = costs.loc[("capital_cost",carrier), y],#costs[y].at[carrier, "capital_cost"],
                efficiency = costs.loc[("efficiency",carrier), y],#costs[y].at[carrier, "efficiency"],
                p_max_pu = resource_carrier[n.buses.index].values
            )

# # Generators
def attach_existing_generators(n, costs, eskom_profiles, model_setup):

    # conventional generators
    conv_carriers = snakemake.config["electricity"]["conventional_generators"]["carriers"]
    start_year = n.snapshots.get_level_values(0)[0] if n._multi_invest else n.snapshots[0].year

    # Add existing conventional generators that are active
    eskom_gens = pd.read_excel(
        snakemake.input.model_file, 
        sheet_name="existing_eskom",
        na_values=["-"],
        index_col=[0,1]
    ).loc[model_setup.existing_eskom]
    
    eskom_gens = eskom_gens[eskom_gens["Carrier"].isin(conv_carriers)]
    
    ipp_gens = pd.read_excel(
        snakemake.input.model_file,
        sheet_name="existing_non_eskom",
        na_values=["-"],
        index_col=[0,1]
    ).loc[model_setup.existing_non_eskom]
    
    ipp_gens=ipp_gens[ipp_gens["Carrier"].isin(conv_carriers)] # add existing non eskom generators (excluding solar, onwind)  
    
    gens = pd.concat([eskom_gens,ipp_gens])
    gens = map_generator_parameters(gens, start_year)

    # Drop power plants where we don"t have coordinates or capacity
    gens = pd.DataFrame(gens.loc[lambda df: (df.p_nom>0.) & df.x.notnull() & df.y.notnull()])
    gens = gens[gens["lifetime"] >= 0]  #drop any generators that have already been decomissioned by start year

    # Associate every generator with the bus of the region it is in or closest to
    gens_gps = gpd.GeoSeries([Point(o.x, o.y) for o in gens[["x", "y"]].itertuples()], index=gens.index, crs=snakemake.config["crs"]["geo_crs"])
    gens_gps = gens_gps.to_crs(snakemake.config["crs"]["distance_crs"])
    regions = gpd.read_file(
        snakemake.input.supply_regions,
    ).to_crs(snakemake.config["crs"]["distance_crs"]).set_index("name")

    for bus, region in regions.geometry.items():
        pos_at_bus_b = gens_gps.within(region)
        if pos_at_bus_b.any():
            gens.loc[pos_at_bus_b, "bus"] = bus
    gens.loc[gens.bus.isnull(), "bus"] = gens_gps[gens.bus.isnull()].map(lambda p: regions.distance(p).idxmin())



    gen_index=gens[gens.carrier.isin(conv_carriers)].index
    n.madd(
        "Generator", gen_index,
        bus  =gens.loc[gen_index,"bus"],
        carrier = gens.loc[gen_index,"carrier"],
        build_year = start_year,
        lifetime = gens.loc[gen_index,"lifetime"],
        p_nom = gens.loc[gen_index,"p_nom"],
        p_nom_extendable=False,
        efficiency = gens.loc[gen_index,"efficiency"],
        ramp_limit_up = gens.loc[gen_index,"ramp_limit_up"],
        ramp_limit_down = gens.loc[gen_index,"ramp_limit_down"],
        marginal_cost = gens.loc[gen_index,"marginal_cost"],
        capital_cost = gens.loc[gen_index,"capital_cost"],
        #p_max_pu - added later under generator availability function
    )  
    n.generators["plant_name"] = n.generators.index.str.split("*").str[0]

    for carrier in ["solar_csp","biomass"]:
        n.add("Carrier", name=carrier)
        plant_data = gens.loc[gens["carrier"]==carrier,["Grouping","bus","p_nom"]].groupby(["Grouping","bus"]).sum()
        for param in ["lifetime","efficiency","capital_cost","marginal_cost"]:
            plant_data[param]=gens.loc[gens["carrier"]==carrier,["Grouping","bus",param]].groupby(["Grouping","bus"]).mean()

        for group in plant_data.index.levels[0]:
            # Duplicate Aggregate Eskom Data across the regions
            eskom_data = pd.concat([eskom_profiles[carrier]] * (len(plant_data.loc[group].index)), axis=1, ignore_index=True)
            eskom_data.columns = plant_data.loc[group].index
            capacity_factor = (eskom_data[plant_data.loc[group].index]).mean()[0]
            annual_cost = capacity_factor * 8760 * plant_data.loc[group,"marginal_cost"]

            n.madd(
                "Generator", plant_data.loc[group].index, suffix=" "+group+"_"+carrier,
                bus=plant_data.loc[group].index,
                carrier=carrier,
                build_year=start_year,
                lifetime=plant_data.loc[group,"lifetime"],
                p_nom = plant_data.loc[group,"p_nom"],
                p_nom_extendable=False,
                efficiency = plant_data.loc[group,"efficiency"],
                capital_cost=annual_cost,
                p_max_pu=eskom_data.values,
                p_min_pu=eskom_data.values*0.95, #purchase at least 95% of power under existing PPAs despite higher cost
            )    

    # ## HYDRO and PHS    
    # # Cohora Bassa imports to South Africa - based on Actual Eskom data from 2017-2022
    n.generators_t.p_max_pu["CahoraBassa"] = eskom_profiles["hydro_import"].values
    # Hydro power generation - based on actual Eskom data from 2017-2022
    for tech in n.generators[n.generators.carrier=="hydro"].index:
        n.generators_t.p_max_pu[tech] = eskom_profiles["hydro"].values

    for tech in n.generators[n.generators.carrier=="hydro_import"].index:
        n.generators_t.p_max_pu[tech] = eskom_profiles["hydro_import"].values

    # PHS
    phs = gens[gens.carrier=="phs"]
    n.madd(
        "StorageUnit", 
        phs.index, 
        carrier="phs",
        bus=phs["bus"],
        p_nom=phs["p_nom"],
        max_hours=phs["PHS_max_hours"],
        capital_cost=phs["capital_cost"],
        marginal_cost=phs["marginal_cost"],
        efficiency_dispatch=phs["PHS_efficiency"]**(0.5),
        efficiency_store=phs["PHS_efficiency"]**(0.5),
        cyclic_state_of_charge=True
        #inflow=inflow_t.loc[:, hydro.index]) #TODO add in
    )

    _add_missing_carriers_from_costs(n, costs, gens.carrier.unique())


def attach_extendable_generators(n, costs):
    elec_opts = snakemake.config["electricity"]
    carriers = elec_opts["extendable_carriers"]["Generator"]
    if snakemake.wildcards.regions=="1-supply":
        buses = dict(zip(carriers,["RSA"]*len(carriers)))
    else:
        buses = elec_opts["buses"][snakemake.wildcards.regions]
    start_year = n.investment_periods[0] if n._multi_invest else n.snapshots[0].year
    ext_years = n.investment_periods if n._multi_invest else [n.snapshots[0].year]
    _add_missing_carriers_from_costs(n, costs, carriers)


    for y in ext_years: 
        for carrier in carriers:
            buses_i = buses.get(carrier, n.buses.index)
            gen_params = {
                "bus": buses_i,
                "p_nom_extendable": True,
                "carrier": carrier,
                "build_year": y,
                "lifetime": costs.loc[("lifetime",carrier),y],
                "capital_cost": costs.loc[("capital_cost",carrier),y],
                "marginal_cost": costs.loc[("marginal_cost",carrier),y],
                "efficiency": costs.loc[("efficiency",carrier),y],
            }
            if snakemake.wildcards.regions == "1-supply":
                n.add(
                    "Generator", buses_i + " " + carrier + "_"+str(y),
                    **gen_params
                )
            else:
                n.madd(
                    "Generator", buses_i, suffix=" " + carrier + "_"+str(y),
                    **gen_params
                )


    # for y in ext_years: 
    #     for carrier in carriers:
    #         buses_i = buses.get(carrier, n.buses.index)
    #         if snakemake.wildcards.regions == "1-supply":
    #             n.add(
    #                 "Generator", buses_i + " " + carrier + "_"+str(y),
    #                 bus=buses_i,
    #                 p_nom_extendable=True,
    #                 carrier=carrier,
    #                 build_year=y,
    #                 lifetime=costs.loc[("lifetime",carrier),y],
    #                 capital_cost=costs.loc[("capital_cost",carrier),y],
    #                 marginal_cost=costs.loc[("marginal_cost",carrier),y],
    #                 efficiency=costs.loc[("efficiency",carrier),y],
    #             )
    #         else:
    #             n.madd(
    #                 "Generator", buses_i, suffix=" " + carrier + "_"+str(y),
    #                 bus=buses_i,
    #                 p_nom_extendable=True,
    #                 carrier=carrier,
    #                 build_year=y,
    #                 lifetime=costs.loc[("lifetime",carrier),y],
    #                 capital_cost=costs.loc[("capital_cost",carrier),y],
    #                 marginal_cost=costs.loc[("marginal_cost",carrier),y],
    #                 efficiency=costs.loc[("efficiency",carrier),y]
    #             )


def attach_storage(n, costs):
    elec_opts = snakemake.config["electricity"]
    carriers = elec_opts["extendable_carriers"]["StorageUnit"]
    max_hours = elec_opts["max_hours"]
    buses = elec_opts["buses"]
    
    _add_missing_carriers_from_costs(n, costs, carriers)
    ext_years = n.investment_periods if n._multi_invest else [n.snapshots[0].year]

    for y in ext_years:
        for carrier in carriers:
            buses_i = buses.get(carrier, n.buses.index)
            n.madd(
                "StorageUnit", buses_i, " " + carrier + "_" + str(y),
                bus=buses_i,
                p_nom_extendable=True,
                carrier=carrier,
                build_year=y,
                lifetime=costs.loc[("lifetime",carrier),y],
                capital_cost=costs.loc[("capital_cost",carrier),y],
                marginal_cost=costs.loc[("marginal_cost",carrier),y],
                efficiency_store=costs.loc[("efficiency",carrier),y] ** 2,
                efficiency_dispatch=costs.loc[("efficiency",carrier),y] ** 2,
                max_hours=max_hours[carrier],
                cyclic_state_of_charge=True,
                )

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

def add_peak_demand_hour_without_variable_feedin(n):
    new_hour = n.snapshots[-1] + pd.Timedelta(hours=1)
    n.set_snapshots(n.snapshots.append(pd.Index([new_hour])))

    # Don"t value new hour for energy totals
    n.snapshot_weightings[new_hour] = 0.

    # Don"t allow variable feed-in in this hour
    n.generators_t.p_max_pu.loc[new_hour] = 0.

    n.loads_t.p_set.loc[new_hour] = (
        n.loads_t.p_set.loc[n.loads_t.p_set.sum(axis=1).idxmax()]
        * (1.+snakemake.config["electricity"]["SAFE_reservemargin"])
    )

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


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake(
            "add_electricity", 
            **{
                "model_file":"grid-2040",
                "regions":"11-supply",
                "resarea":"redz",
                "ll":"copt",
                "attr":"p_nom"
            }
        )

    model_setup = (
        pd.read_excel(
            snakemake.input.model_file, 
            sheet_name="model_setup",
            index_col=[0])
            .loc[snakemake.wildcards.model_file]
    )

    projections = (
        pd.read_excel(
            snakemake.input.model_file, 
            sheet_name="projected_parameters",
            index_col=[0,1])
            .loc[model_setup["projected_parameters"]]
    )

    #opts = snakemake.wildcards.opts.split("-")
    n = pypsa.Network(snakemake.input.base_network)
    costs = load_costs(snakemake.input.model_file, model_setup.costs)

    #wind_solar_profiles = xr.open_dataset(snakemake.input.wind_solar_profiles).to_dataframe()
    eskom_profiles = generate_eskom_profiles(n)

    attach_load(n, projections.loc["annual_demand",:])
    if snakemake.wildcards.regions!="1-supply":
        update_transmission_costs(n, costs)
    attach_existing_generators(n, costs, eskom_profiles, model_setup)
    attach_wind_and_solar(n, costs, snakemake.input, model_setup, eskom_profiles)
    attach_extendable_generators(n, costs)
    attach_storage(n, costs)
    if snakemake.config["electricity"]["conventional_generators"]["implement_availability"]==True:
        add_generator_availability(n, projections)
        add_min_stable_levels(n)      
    add_nice_carrier_names(n)
    n.export_to_netcdf(snakemake.output[0])
