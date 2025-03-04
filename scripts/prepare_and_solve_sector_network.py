import logging
import os
import re
from types import SimpleNamespace
from pathlib import Path

import pypsa
import pandas as pd
idx = pd.IndexSlice
import numpy as np
import geopandas as gpd

from concurrent.futures import ProcessPoolExecutor

from _helpers import (
    configure_logging, 
    prepare_costs, 
    remove_leap_day, 
    NA_VALUES, 
    create_network_topology,
    locate_bus,
    cycling_shift,
    normalize_and_rename_df,
    assign_segmented_df_to_network)
from prepare_and_solve_network import (
    average_every_nhours, 
    add_co2limit, 
    set_extendable_limits_global, 
    set_extendable_limits_per_bus)
from add_electricity import load_extendable_parameters, update_transmission_costs
from custom_constraints import (
    set_operational_limits, 
    ccgt_steam_constraints, 
    define_reserve_margin,
    add_co2_sequestration_limit,
    add_battery_constraints,
    custom_define_tech_capacity_expansion_limit)

logger = logging.getLogger(__name__)

spatial = SimpleNamespace()


def get(item, investment_year=None):
    """Check whether item depends on investment year"""
    if isinstance(item, dict):
        return item[investment_year]
    else:
        return item

def add_carrier_buses(n, carrier, nodes=None):
    """
    Add buses to connect e.g. coal, nuclear and oil plants
    """

    if nodes is None:
        nodes = vars(spatial)[carrier].nodes
    location = vars(spatial)[carrier].locations

    # skip if carrier already exists
    if carrier in n.carriers.index:
        return

    if not isinstance(nodes, pd.Index):
        nodes = pd.Index(nodes)

    n.add("Carrier", carrier, co2_emissions=costs.at[carrier, "CO2 intensity"])

    n.add("Bus", nodes, location=location, carrier=carrier)

    # capital cost could be corrected to e.g. 0.2 Currency/kWh * annuity and O&M
    n.add(
        "Store",
        nodes + " Store",
        bus=nodes,
        e_nom_extendable=True,
        e_cyclic=True,
        carrier=carrier,
    )

    n.add(
        "Generator",
        nodes,
        bus=nodes,
        p_nom_extendable=True,
        carrier=carrier,
        marginal_cost=costs.at[carrier, "fuel"],
    )

def add_missing_co2_emissions_from_model_file(n, carriers, model_file):
    carrier_co2_intensity = (
        pd.read_excel(
            model_file, 
            sheet_name="carriers",
            index_col=[0],
            na_values=NA_VALUES)
    )["value"]
    n.carriers.loc[carriers,"co2_emissions"] = n.carriers.loc[carriers,:].index.map(carrier_co2_intensity)
    
def add_generation(n, costs):
    """Adds conventional generation as specified in config


    Args:
        n (network): PyPSA prenetwork
        costs (dataframe): _description_

    Returns:
        _type_: _description_
    """ """"""

    logger.info("adding electricity generation")


    fallback = {"OCGT": "gas"}
    conventionals = options.get("conventional_generation", fallback)

    for generator, carrier in conventionals.items():
        add_carrier_buses(n, carrier)
        carrier_nodes = vars(spatial)[carrier].nodes
        n.add(
            "Link",
            nodes + " " + generator,
            bus0=carrier_nodes,
            bus1=nodes,
            bus2="co2 atmosphere",
            marginal_cost=costs.at[generator, "efficiency"]
            * costs.at[generator, "VOM"],  # NB: VOM is per MWel
            # NB: fixed cost is per MWel
            capital_cost=costs.at[generator, "efficiency"]
            * costs.at[generator, "fixed"],
            p_nom_extendable=True,
            carrier=generator,
            efficiency=costs.at[generator, "efficiency"],
            efficiency2=costs.at[carrier, "CO2 intensity"],
            lifetime=costs.at[generator, "lifetime"],
        )


def add_oil(n, costs):
    """
    Function to add oil carrier and bus to network. If-Statements are required in
    case oil was already added from config ['sector']['conventional_generation']
    Oil is copper plated
    """
    # TODO function will not be necessary if conventionals are added using "add_carrier_buses()"
    # TODO before using add_carrier_buses: remove_elec_base_techs(n), otherwise carriers are added double
    # spatial.gas = SimpleNamespace()

    spatial.oil = SimpleNamespace()

    if options["oil"]["spatial_oil"]:
        spatial.oil.nodes = nodes + " oil"
        spatial.oil.locations = nodes
    else:
        spatial.oil.nodes = ["ZA oil"]
        spatial.oil.locations = ["ZA"]

    if "oil" not in n.carriers.index:
        n.add("Carrier", "oil")

    # Set the "co2_emissions" of the carrier "oil" to 0, because the emissions of oil usage taken from the spatial.oil.nodes are accounted seperately (directly linked to the co2 atmosphere bus). Setting the carrier to 0 here avoids double counting. Be aware to link oil emissions to the co2 atmosphere bus.
    # n.carriers.loc["oil", "co2_emissions"] = 0
    # print("co2_emissions of oil set to 0 for testing")  # TODO add logger.info

    n.add(
        "Bus",
        spatial.oil.nodes,
        location=spatial.oil.locations,
        carrier="oil",
    )

    # if "ZA oil" not in n.buses.index:

    #     n.add("Bus", "ZA oil", location="ZA", carrier="oil")

    # if "ZA oil Store" not in n.stores.index:

    # e_initial = (snakemake.config["fossil_reserves"]).get("ctl_oil", 0) * 1e6
    # e_initial = e_initial / len(spatial.oil.nodes)
    
    n.add(
        "Store",
        [oil_bus + " Store" for oil_bus in spatial.oil.nodes],
        bus=spatial.oil.nodes,
        e_nom_extendable=True,
        e_cyclic=True,
        carrier="oil",
        capital_cost = 0.02
    )

    n.add(
        "Generator",
        spatial.oil.nodes,
        bus=spatial.oil.nodes,
        p_nom_extendable=True,
        carrier="oil",
        marginal_cost=costs.at["oil", "fuel"],
    )


def add_gas(n, costs):
    spatial.gas = SimpleNamespace()

    if options["gas"]["spatial_gas"]:
        spatial.gas.nodes = nodes + " gas"
        spatial.gas.locations = nodes
        spatial.gas.biogas = nodes + " biogas"
        spatial.gas.industry = nodes + " gas for industry"
        if snakemake.config["sector"]["cc"]:
            spatial.gas.industry_cc = nodes + " gas for industry CC"
        spatial.gas.biogas_to_gas = nodes + " biogas to gas"
    else:
        spatial.gas.nodes = ["ZA gas"]
        spatial.gas.locations = ["ZA"]
        spatial.gas.biogas = ["ZA biogas"]
        spatial.gas.industry = ["gas for industry"]
        if snakemake.config["sector"]["cc"]:
            spatial.gas.industry_cc = ["gas for industry CC"]
        spatial.gas.biogas_to_gas = ["ZA biogas to gas"]

    spatial.gas.df = pd.DataFrame(vars(spatial.gas), index=nodes)

    gas_nodes = vars(spatial)["gas"].nodes

    add_carrier_buses(n, "gas", gas_nodes)
    

def H2_liquid_fossil_conversions(n, costs):
    """
    Function to add conversions between H2 and liquid fossil
    Carrier and bus is added in add_oil, which later on might be switched to add_generation
    """
    n.add(
        "Bus",
        nodes + " Fischer-Tropsch",
        location=nodes,
        carrier="Fischer-Tropsch",
        x=n.buses.loc[list(nodes)].x.values,
        y=n.buses.loc[list(nodes)].y.values,
    )
    
    n.add(
        "Link",
        nodes + " Fischer-Tropsch",
        bus0=spatial.h2.nodes,
        bus1=nodes + " Fischer-Tropsch",
        bus2=spatial.co2.nodes,
        carrier="Fischer-Tropsch",
        efficiency=costs.at["Fischer-Tropsch", "efficiency"],
        capital_cost=costs.at["Fischer-Tropsch", "fixed"]
        * costs.at[
            "Fischer-Tropsch", "efficiency"
        ],  # Use efficiency to convert from Currency/MW_FT/a to Currency/MW_H2/a
        efficiency2=-costs.at["oil", "CO2 intensity"]
        * costs.at["Fischer-Tropsch", "efficiency"],
        p_nom_extendable=True,
        #p_min_pu=options.get("min_part_load_fischer_tropsch", 0),
        lifetime=costs.at["Fischer-Tropsch", "lifetime"],
    )
    
    n.add(
        "Store",
        nodes + " Fischer-Tropsch Store",
        bus=nodes + " Fischer-Tropsch",
        e_nom_extendable=True,
        e_cyclic=True,
        carrier="Fischer-Tropsch Store",
        capital_cost = 0.02
    )
    
    n.add(
        "Link",
        nodes + " Fischer-Tropsch -> oil",
        bus0=nodes + " Fischer-Tropsch",
        bus1=spatial.oil.nodes,
        carrier="Fischer-Tropsch -> oil",
        efficiency=1,
        p_nom_extendable=True,
    )
    


def add_hydrogen_and_desalination(n, costs):
    n.add("Carrier", "H2")
    n.add("Carrier", "H2 Electrolysis")
    n.add("Carrier", "freshwater", unit="H20_m3")
    n.add("Carrier", "seawater", unit="H20_m3")

    n.add(
        "Bus",
        spatial.freshwater.nodes,
        carrier="freshwater",
    )
    
    n.add(
        "Bus",
        spatial.seawater.nodes,
        carrier="seawater",
    )

    n.add(
        "Generator",
        spatial.coastal_nodes + " seawater",
        bus=spatial.seawater.nodes,
        p_nom_extendable=True,
        carrier="seawater",
    )
    
    n.add(
        "Link",
        spatial.coastal_nodes + " desalination",
        bus0=spatial.seawater.nodes,
        bus1=spatial.freshwater.nodes,
        bus2=spatial.coastal_nodes,
        p_nom_extendable=True,
        carrier="seawater desalination",
        efficiency=1.0,
        efficiency2=-costs.at["seawater desalination", "electricity-input"],
        capital_cost=costs.at["seawater desalination", "fixed"],
        )
    
    n.add(
        "Bus",
        spatial.h2.nodes,
        location=nodes,
        carrier="H2",
        x=n.buses.loc[list(nodes)].x.values,
        y=n.buses.loc[list(nodes)].y.values,
    )

    n.add(
        "Link",
        nodes + " H2 Electrolysis",
        bus0=nodes,
        bus1=spatial.h2.nodes,
        bus2=spatial.freshwater.nodes,
        p_nom_extendable=True,
        carrier="H2 Electrolysis",
        efficiency=costs.at["electrolysis PEMEC", "efficiency"],
        efficiency2=-costs.at["electrolysis", "water-input"] 
                    * costs.at["electrolysis PEMEC", "efficiency"], # convert to water-input in m3/MWh_el
        capital_cost=costs.at["electrolysis PEMEC", "fixed"],
        lifetime=costs.at["electrolysis PEMEC", "lifetime"],
    )

    n.add(
        "Link",
        nodes + " H2 Fuel Cell",
        bus0=spatial.h2.nodes,
        bus1=nodes,
        p_nom_extendable=True,
        carrier="H2 Fuel Cell",
        efficiency=costs.at["fuel cell", "efficiency"],
        # NB: fixed cost is per MWel
        capital_cost=costs.at["fuel cell", "fixed"]
        * costs.at["fuel cell", "efficiency"],
        lifetime=costs.at["fuel cell", "lifetime"],
    )

    cavern_nodes = pd.DataFrame()

    # hydrogen stored overground (where not already underground)
    h2_capital_cost = costs.at[
        "hydrogen storage tank type 1 including compressor", "fixed"
    ]
    nodes_overground = cavern_nodes.index.symmetric_difference(nodes)

    n.add(
        "Store",
        nodes_overground + " H2 Store",
        bus=nodes_overground + " H2",
        e_nom_extendable=True,
        e_cyclic=True,
        carrier="H2 Store",
        capital_cost=h2_capital_cost,
    )

    if not snakemake.config["sector"]["hydrogen"]["network_routes"] == "greenfield":
        
        raise NotImplementedError("this feature is not yet implemented for South Africa")
        
        h2_links = pd.read_csv(snakemake.input.pipelines)

        # Order buses to detect equal pairs for bidirectional pipelines
        buses_ordered = h2_links.apply(lambda p: sorted([p.bus0, p.bus1]), axis=1)

        if snakemake.config["clustering_options"]["alternative_clustering"]:
            # Appending string for carrier specification '_AC'
            h2_links["bus0"] = buses_ordered.str[0] + "_AC"
            h2_links["bus1"] = buses_ordered.str[1] + "_AC"

            # Conversion of GADM id to from 3 to 2-digit
            h2_links["bus0"] = (
                h2_links["bus0"]
                .str.split(".")
                .apply(lambda id: three_2_two_digits_country(id[0]) + "." + id[1])
            )
            h2_links["bus1"] = (
                h2_links["bus1"]
                .str.split(".")
                .apply(lambda id: three_2_two_digits_country(id[0]) + "." + id[1])
            )

        # Create index column
        h2_links["buses_idx"] = (
            "H2 pipeline " + h2_links["bus0"] + " -> " + h2_links["bus1"]
        )

        # Aggregate pipelines applying mean on length and sum on capacities
        h2_links = h2_links.groupby("buses_idx").agg(
            {"bus0": "first", "bus1": "first", "length": "mean", "capacity": "sum"}
        )
    else:
        attrs = ["bus0", "bus1", "length"]
        h2_links = pd.DataFrame(columns=attrs)

        candidates = pd.concat(
            {
                "lines": n.lines[attrs],
                "links": n.links.loc[n.links.carrier == "DC", attrs],
            }
        )

        for candidate in candidates.index:
            buses = [candidates.at[candidate, "bus0"], candidates.at[candidate, "bus1"]]
            buses.sort()
            name = f"H2 pipeline {buses[0]} -> {buses[1]}"
            if name not in h2_links.index:
                h2_links.at[name, "bus0"] = buses[0]
                h2_links.at[name, "bus1"] = buses[1]
                h2_links.at[name, "length"] = candidates.at[candidate, "length"]

    # TODO Add efficiency losses
    if snakemake.config["sector"]["hydrogen"]["network"]:
        n.add(
            "Link",
            h2_links.index,
            bus0=h2_links.bus0.values + " H2",
            bus1=h2_links.bus1.values + " H2",
            p_min_pu=-1,
            p_nom_extendable=True,
            length=h2_links.length.values,
            capital_cost=costs.at["H2 (g) pipeline", "fixed"]
            * h2_links.length.values,
            carrier="H2 pipeline",
            lifetime=costs.at["H2 (g) pipeline", "lifetime"],
        )


def add_ammonia(n, costs):
    
    n.add("Carrier", "N2", unit="t_N2")
    
    n.add(
        "Bus",
        spatial.nitrogen.nodes,
        carrier="N2",
    )
    # NB: electricity input for ASU is included in the Haber-Bosch electricity-input.
    n.add(
        "Generator",
        spatial.nitrogen.nodes,
        bus=spatial.nitrogen.nodes,
        p_nom_extendable=True,
        carrier="air separation unit",
        capital_cost=costs.at["air separation unit", "fixed"], #Currency/t_N2
    )
    
    n.add("Carrier", "NH3")
    
    n.add(
        "Bus",
        spatial.ammonia.nodes,
        carrier="NH3",
    )
    
    n.add(
        "Link",
        nodes + " Haber-Bosch",
        bus0=nodes,
        bus1=spatial.ammonia.nodes,
        bus2=spatial.h2.nodes,
        bus3=spatial.nitrogen.nodes,
        p_nom_extendable=True,
        carrier="Haber-Bosch",
        efficiency=1 / costs.at["Haber-Bosch", "electricity-input"],
        efficiency2=-costs.at["Haber-Bosch", "hydrogen-input"]
        / costs.at["Haber-Bosch", "electricity-input"],
        efficiency3=-costs.at["Haber-Bosch", "nitrogen-input"]
        / costs.at["Haber-Bosch", "electricity-input"],
        capital_cost=costs.at["Haber-Bosch", "fixed"]
        / costs.at["Haber-Bosch", "electricity-input"],
        marginal_cost=costs.at["Haber-Bosch", "VOM"]
        / costs.at["Haber-Bosch", "electricity-input"],
    )
    
    # Ammonia Storage
    n.add(
        "Store",
        spatial.ammonia.nodes,
        bus=spatial.ammonia.nodes,
        e_nom_extendable=True,
        e_cyclic=True,
        carrier="ammonia store",
        capital_cost=costs.at["NH3 (l) storage tank incl. liquefaction", "fixed"],
        lifetime=costs.at["NH3 (l) storage tank incl. liquefaction", "lifetime"],
    )

def define_spatial(nodes):
    """
    Namespace for spatial

    Parameters
    ----------
    nodes : list-like
    """

    global spatial
    global options

    spatial.nodes = nodes

    # biomass

    spatial.biomass = SimpleNamespace()

    if options["biomass_transport"]:
        spatial.biomass.nodes = nodes + " solid biomass"
        spatial.biomass.locations = nodes
        spatial.biomass.industry = nodes + " solid biomass for industry"
        spatial.biomass.industry_cc = nodes + " solid biomass for industry CC"
    else:
        spatial.biomass.nodes = ["ZA solid biomass"]
        spatial.biomass.locations = ["ZA"]
        spatial.biomass.industry = ["solid biomass for industry"]
        spatial.biomass.industry_cc = ["solid biomass for industry CC"]

    spatial.biomass.df = pd.DataFrame(vars(spatial.biomass), index=nodes)

    # co2

    spatial.co2 = SimpleNamespace()

    if options["co2_network"]:
        spatial.co2.nodes = nodes + " co2 stored"
        spatial.co2.locations = nodes
        spatial.co2.vents = nodes + " co2 vent"
        spatial.co2.x = (n.buses.loc[list(nodes)].x.values,)
        spatial.co2.y = (n.buses.loc[list(nodes)].y.values,)
    else:
        spatial.co2.nodes = ["co2 stored"]
        spatial.co2.locations = ["ZA"]
        spatial.co2.vents = ["co2 vent"]
        spatial.co2.x = (0,)
        spatial.co2.y = 0

    spatial.co2.df = pd.DataFrame(vars(spatial.co2), index=nodes)

    spatial.nitrogen = SimpleNamespace()
    spatial.nitrogen.nodes = ["N2"]
    
    spatial.ammonia = SimpleNamespace()
    spatial.ammonia.nodes = nodes + " NH3"
    
    spatial.h2 = SimpleNamespace()
    spatial.h2.nodes = nodes + " H2"

    spatial.coastal_nodes = pd.Index(['Western Cape', 'KwaZulu Natal','Northern Cape','Eastern Cape'])
    
    spatial.seawater = SimpleNamespace()
    spatial.seawater.nodes = spatial.coastal_nodes + " seawater"
    
    spatial.freshwater = SimpleNamespace()
    spatial.freshwater.nodes = ["freshwater"]
    
    

def add_biomass(n, costs):
    logger.info("adding biomass")

    # TODO get biomass potentials dataset and enable spatially resolved potentials

    # Get biomass and biogas potentials from config and convert from TWh to MWh
    biomass_pot = snakemake.config["sector"]["solid_biomass_potential"] * 1e6  # MWh
    biogas_pot = snakemake.config["sector"]["biogas_potential"] * 1e6  # MWh
    logger.info("Biomass and Biogas potential fetched from config")

    # Convert from total to nodal potentials,
    biomass_pot_spatial = biomass_pot / len(spatial.biomass.nodes)
    biogas_pot_spatial = biogas_pot / len(spatial.gas.biogas)
    logger.info("Biomass potentials spatially resolved equally across all nodes")

    n.add("Carrier", "biogas")
    n.add("Carrier", "solid biomass")

    n.add(
        "Bus", spatial.gas.biogas, location=spatial.biomass.locations, carrier="biogas"
    )

    n.add(
        "Bus",
        spatial.biomass.nodes,
        location=spatial.biomass.locations,
        carrier="solid biomass",
    )

    n.add(
        "Store",
        spatial.gas.biogas,
        bus=spatial.gas.biogas,
        carrier="biogas",
        e_nom=biogas_pot_spatial,
        marginal_cost=costs.at["biogas", "fuel"],
        e_initial=biogas_pot_spatial,
    )

    n.add(
        "Store",
        spatial.biomass.nodes,
        bus=spatial.biomass.nodes,
        carrier="solid biomass",
        e_nom=biomass_pot_spatial,
        marginal_cost=costs.at["solid biomass", "fuel"],
        e_initial=biomass_pot_spatial,
    )

    biomass_gen = "biomass EOP"
    n.add(
        "Link",
        nodes + " biomass EOP",
        bus0=spatial.biomass.nodes,
        bus1=nodes,
        # bus2="co2 atmosphere",
        marginal_cost=costs.at[biomass_gen, "efficiency"]
        * costs.at[biomass_gen, "VOM"],  # NB: VOM is per MWel
        # NB: fixed cost is per MWel
        capital_cost=costs.at[biomass_gen, "efficiency"]
        * costs.at[biomass_gen, "fixed"],
        p_nom_extendable=True,
        carrier=biomass_gen,
        efficiency=costs.at[biomass_gen, "efficiency"],
        # efficiency2=costs.at["solid biomass", "CO2 intensity"],
        lifetime=costs.at[biomass_gen, "lifetime"],
    )
    n.add(
        "Link",
        spatial.gas.biogas_to_gas,
        bus0=spatial.gas.biogas,
        bus1=spatial.gas.nodes,
        bus2="co2 atmosphere",
        carrier="biogas to gas",
        capital_cost=costs.loc["biogas upgrading", "fixed"],
        marginal_cost=costs.loc["biogas upgrading", "VOM"],
        efficiency2=-costs.at["gas", "CO2 intensity"],
        p_nom_extendable=True,
    )

    if options["biomass_transport"]:
        # TODO add biomass transport costs
        transport_costs = pd.read_csv(
            snakemake.input.biomass_transport_costs,
            index_col=0,
            keep_default_na=False,
        ).squeeze()

        # add biomass transport
        biomass_transport = create_network_topology(
            n, "biomass transport ", bidirectional=False
        )

        # costs
        countries_not_in_index = set(countries) - set(biomass_transport.index)
        if countries_not_in_index:
            logger.info(
                "No transport values found for {0}, using default value of {1}".format(
                    ", ".join(countries_not_in_index),
                    snakemake.config["sector"]["biomass_transport_default_cost"],
                )
            )

        bus0_costs = biomass_transport.bus0.apply(
            lambda x: transport_costs.get(
                x[:2], snakemake.config["sector"]["biomass_transport_default_cost"]
            )
        )
        bus1_costs = biomass_transport.bus1.apply(
            lambda x: transport_costs.get(
                x[:2], snakemake.config["sector"]["biomass_transport_default_cost"]
            )
        )
        biomass_transport["costs"] = pd.concat([bus0_costs, bus1_costs], axis=1).mean(
            axis=1
        )

        n.add(
            "Link",
            biomass_transport.index,
            bus0=biomass_transport.bus0 + " solid biomass",
            bus1=biomass_transport.bus1 + " solid biomass",
            p_nom_extendable=True,
            length=biomass_transport.length.values,
            marginal_cost=biomass_transport.costs * biomass_transport.length.values,
            capital_cost=1,
            carrier="solid biomass transport",
        )

    # n.add(
    #         "Link",
    #         urban_central + " urban central solid biomass CHP",
    #         bus0=spatial.biomass.df.loc[urban_central, "nodes"].values,
    #         bus1=urban_central,
    #         bus2=urban_central + " urban central heat",
    #         carrier="urban central solid biomass CHP",
    #         p_nom_extendable=True,
    #         capital_cost=costs.at[key, "fixed"] * costs.at[key, "efficiency"],
    #         marginal_cost=costs.at[key, "VOM"],
    #         efficiency=costs.at[key, "efficiency"],
    #         efficiency2=costs.at[key, "efficiency-heat"],
    #         lifetime=costs.at[key, "lifetime"],
    #     )

    # AC buses with district heating
    urban_central = n.buses.index[n.buses.carrier == "urban central heat"]
    if not urban_central.empty and options["chp"]:
        urban_central = urban_central.str[: -len(" urban central heat")]

        key = "central solid biomass CHP"

        n.add(
            "Link",
            urban_central + " urban central solid biomass CHP",
            bus0=spatial.biomass.df.loc[urban_central, "nodes"].values,
            bus1=urban_central,
            bus2=urban_central + " urban central heat",
            carrier="urban central solid biomass CHP",
            p_nom_extendable=True,
            capital_cost=costs.at[key, "fixed"] * costs.at[key, "efficiency"],
            marginal_cost=costs.at[key, "VOM"],
            efficiency=costs.at[key, "efficiency"],
            efficiency2=costs.at[key, "efficiency-heat"],
            lifetime=costs.at[key, "lifetime"],
        )

        if snakemake.config["sector"]["cc"]:
            n.add(
                "Link",
                urban_central + " urban central solid biomass CHP CC",
                bus0=spatial.biomass.df.loc[urban_central, "nodes"].values,
                bus1=urban_central,
                bus2=urban_central + " urban central heat",
                bus3="co2 atmosphere",
                bus4=spatial.co2.df.loc[urban_central, "nodes"].values,
                carrier="urban central solid biomass CHP CC",
                p_nom_extendable=True,
                capital_cost=costs.at[key, "fixed"] * costs.at[key, "efficiency"]
                + costs.at["biomass CHP capture", "fixed"]
                * costs.at["solid biomass", "CO2 intensity"],
                marginal_cost=costs.at[key, "VOM"],
                efficiency=costs.at[key, "efficiency"]
                - costs.at["solid biomass", "CO2 intensity"]
                * (
                    costs.at["biomass CHP capture", "electricity-input"]
                    + costs.at["biomass CHP capture", "compression-electricity-input"]
                ),
                efficiency2=costs.at[key, "efficiency-heat"]
                + costs.at["solid biomass", "CO2 intensity"]
                * (
                    costs.at["biomass CHP capture", "heat-output"]
                    + costs.at["biomass CHP capture", "compression-heat-output"]
                    - costs.at["biomass CHP capture", "heat-input"]
                ),
                efficiency3=-costs.at["solid biomass", "CO2 intensity"]
                * costs.at["biomass CHP capture", "capture_rate"],
                efficiency4=costs.at["solid biomass", "CO2 intensity"]
                * costs.at["biomass CHP capture", "capture_rate"],
                lifetime=costs.at[key, "lifetime"],
            )

 

def add_co2(n, costs):
    "add carbon carrier, it's networks and storage units"

    # minus sign because opposite to how fossil fuels used:
    # CH4 burning puts CH4 down, atmosphere up
    n.add("Carrier", "co2", co2_emissions=-1.0)

    # this tracks CO2 in the atmosphere
    n.add(
        "Bus",
        "co2 atmosphere",
        location="ZA",  # TODO Ignoed by pypsa check
        carrier="co2",
    )

    # can also be negative
    n.add(
        "Store",
        "co2 atmosphere",
        e_nom_extendable=True,
        e_min_pu=-1,
        carrier="co2",
        bus="co2 atmosphere",
    )

    # this tracks CO2 stored, e.g. underground
    n.add(
        "Bus",
        spatial.co2.nodes,
        location=spatial.co2.locations,
        carrier="co2 stored",
        # x=spatial.co2.x[0],
        # y=spatial.co2.y[0],
    )
    """
    co2_stored_x = n.buses.filter(like="co2 stored", axis=0).loc[:, "x"]
    co2_stored_y = n.buses.loc[n.buses[n.buses.carrier == "co2 stored"].location].y

    n.buses[n.buses.carrier == "co2 stored"].x = co2_stored_x.values
    n.buses[n.buses.carrier == "co2 stored"].y = co2_stored_y.values
    """

    n.add(
        "Link",
        spatial.co2.vents,
        bus0=spatial.co2.nodes,
        bus1="co2 atmosphere",
        carrier="co2 vent",
        efficiency=1.0,
        p_nom_extendable=True,
    )

    # logger.info("Adding CO2 network.")
    # co2_links = create_network_topology(n, "CO2 pipeline ")

    # cost_onshore = (
    #     (1 - co2_links.underwater_fraction)
    #     * costs.at["CO2 pipeline", "fixed"]
    #     * co2_links.length
    # )
    # cost_submarine = (
    #     co2_links.underwater_fraction
    #     * costs.at["CO2 submarine pipeline", "fixed"]
    #     * co2_links.length
    # )
    # capital_cost = cost_onshore + cost_submarine

    # n.add(
    #     "Link",
    #     co2_links.index,
    #     bus0=co2_links.bus0.values + " co2 stored",
    #     bus1=co2_links.bus1.values + " co2 stored",
    #     p_min_pu=-1,
    #     p_nom_extendable=True,
    #     length=co2_links.length.values,
    #     capital_cost=capital_cost.values,
    #     carrier="CO2 pipeline",
    #     lifetime=costs.at["CO2 pipeline", "lifetime"],
    # )

    n.add(
        "Store",
        spatial.co2.nodes,
        e_nom_extendable=True,
        e_cyclic=True,
        #e_nom_max=options["co2_sequestration_potential"]*1e6,#np.inf, #TODO global constraint would be needed for co2_network
        capital_cost=options["co2_sequestration_cost"],
        carrier="co2 stored",
        bus=spatial.co2.nodes,
    )



def add_storage(n, costs):
    "function to add the different types of storage systems"
    
    remove_elec_base_battery(n)
    
    n.add("Carrier", "battery")

    n.add(
        "Bus",
        nodes + " battery",
        location=nodes,
        carrier="battery",
        x=n.buses.loc[list(nodes)].x.values,
        y=n.buses.loc[list(nodes)].y.values,
    )

    n.add(
        "Store",
        nodes + " battery",
        bus=nodes + " battery",
        e_cyclic=True,
        e_nom_extendable=True,
        carrier="battery",
        capital_cost=costs.at["battery storage", "fixed"],
        lifetime=costs.at["battery storage", "lifetime"],
    )

    n.add(
        "Link",
        nodes + " battery charger",
        bus0=nodes,
        bus1=nodes + " battery",
        carrier="battery charger",
        efficiency=costs.at["battery inverter", "efficiency"] ** 0.5,
        capital_cost=costs.at["battery inverter", "fixed"],
        p_nom_extendable=True,
        lifetime=costs.at["battery inverter", "lifetime"],
    )

    n.add(
        "Link",
        nodes + " battery discharger",
        bus0=nodes + " battery",
        bus1=nodes,
        carrier="battery discharger",
        efficiency=costs.at["battery inverter", "efficiency"] ** 0.5,
        p_nom_extendable=True,
        lifetime=costs.at["battery inverter", "lifetime"],
    )


def h2_hc_conversions(n, costs):
    "function to add the conversion technologies between H2 and hydrocarbons"
    if options["methanation"]:
        n.add(
            "Link",
            spatial.nodes,
            suffix=" Sabatier",
            bus0=spatial.h2.nodes,
            bus1=spatial.gas.nodes,
            bus2=spatial.co2.nodes,
            p_nom_extendable=True,
            carrier="Sabatier",
            efficiency=costs.at["methanation", "efficiency"],
            efficiency2=-costs.at["methanation", "efficiency"]
            * costs.at["gas", "CO2 intensity"],
            # costs given per kW_gas
            capital_cost=costs.at["methanation", "fixed"]
            * costs.at["methanation", "efficiency"],
            lifetime=costs.at["methanation", "lifetime"],
        )

    if options["helmeth"]:
        n.add(
            "Link",
            spatial.nodes,
            suffix=" helmeth",
            bus0=nodes,
            bus1=spatial.gas.nodes,
            bus2=spatial.co2.nodes,
            carrier="helmeth",
            p_nom_extendable=True,
            efficiency=costs.at["helmeth", "efficiency"],
            efficiency2=-costs.at["helmeth", "efficiency"]
            * costs.at["gas", "CO2 intensity"],
            capital_cost=costs.at["helmeth", "fixed"],
            lifetime=costs.at["helmeth", "lifetime"],
        )

    if options["SMR"]:
        n.add(
            "Link",
            spatial.nodes,
            suffix=" SMR CC",
            bus0=spatial.gas.nodes,
            bus1=spatial.h2.nodes,
            bus2="co2 atmosphere",
            bus3=spatial.co2.nodes,
            p_nom_extendable=True,
            carrier="SMR CC",
            efficiency=costs.at["SMR CC", "efficiency"],
            efficiency2=costs.at["gas", "CO2 intensity"] * (1 - options["cc_fraction"]),
            efficiency3=costs.at["gas", "CO2 intensity"] * options["cc_fraction"],
            capital_cost=costs.at["SMR CC", "fixed"],
            lifetime=costs.at["SMR CC", "lifetime"],
        )

        n.add(
            "Link",
            nodes + " SMR",
            bus0=spatial.gas.nodes,
            bus1=spatial.h2.nodes,
            bus2="co2 atmosphere",
            p_nom_extendable=True,
            carrier="SMR",
            efficiency=costs.at["SMR", "efficiency"],
            efficiency2=costs.at["gas", "CO2 intensity"],
            capital_cost=costs.at["SMR", "fixed"],
            lifetime=costs.at["SMR", "lifetime"],
        )



    
def add_shipping(n, costs):
    ports = pd.read_csv(
        snakemake.input.ports, index_col=None, keep_default_na=False
    ).squeeze()
    #ports = ports[ports.country.isin(["ZA"])]

    all_navigation = ["total international navigation", "total domestic navigation"]

    navigation_demand = (
        energy_totals.loc[["ZA"], all_navigation].sum(axis=1).sum()  # * 1e6 / 8760
    )

    # efficiency = (
    #     options["shipping_average_efficiency"] / costs.at["fuel cell", "efficiency"]
    # )

    # check whether item depends on investment year
    shipping_hydrogen_share = get(
        options["shipping_hydrogen_share"], investment_year
    )

    ports["location"] = ports[["x", "y", "country"]].apply(
        lambda port: locate_bus(
            port[["x", "y"]],
            regions,
        ),
        axis=1,
    )

    ports = ports.set_index("location")

    ind = pd.DataFrame(n.buses.index[n.buses.carrier == "AC"])
    ind = ind.set_index(n.buses.index[n.buses.carrier == "AC"])

    ports["p_set"] = ports["fraction"].apply(
        lambda frac: shipping_hydrogen_share
        * frac
        * navigation_demand
        #* efficiency
        * 1e6
        / 8760
        # TODO double check the use of efficiency
    )  # TODO use real data here

    ports = pd.concat([ports, ind]).drop("Bus", axis=1)

    # ports = ports.fillna(0.0)
    ports = ports.groupby(ports.index).sum()

    if options["shipping_hydrogen_liquefaction"]:
        n.add("Bus", nodes, suffix=" H2 liquid", carrier="H2 liquid", location=nodes)

        # link the H2 supply to liquified H2
        n.add(
            "Link",
            nodes + " H2 liquefaction",
            bus0=spatial.h2.nodes,
            bus1=nodes + " H2 liquid",
            carrier="H2 liquefaction",
            efficiency=costs.at["H2 liquefaction", "efficiency"],
            capital_cost=costs.at["H2 liquefaction", "fixed"],
            p_nom_extendable=True,
            lifetime=costs.at["H2 liquefaction", "lifetime"],
        )

        shipping_bus = nodes + " H2 liquid"
    else:
        shipping_bus = spatial.h2.nodes

    n.add(
        "Load",
        nodes,
        suffix=" H2 for shipping",
        bus=shipping_bus,
        carrier="H2 for shipping",
        p_set=ports["p_set"][nodes],
    )

    if shipping_hydrogen_share < 1:
        shipping_oil_share = 1 - shipping_hydrogen_share

        ports["p_set"] = ports["fraction"].apply(
            lambda frac: shipping_oil_share * frac * navigation_demand * 1e6 / 8760
        )

        n.add(
            "Load",
            nodes,
            suffix=" shipping oil",
            bus=spatial.oil.nodes,
            carrier="shipping oil",
            p_set=ports["p_set"][nodes],
        )

        co2 = ports["p_set"].sum() * costs.at["oil", "CO2 intensity"]

        n.add(
            "Load",
            "shipping oil emissions",
            bus="co2 atmosphere",
            carrier="shipping oil emissions",
            p_set=-co2,
        )

    if "oil" not in n.buses.carrier.unique():
        n.add("Bus", spatial.oil.nodes, location=spatial.oil.locations, carrier="oil")
    if "oil" not in n.stores.carrier.unique():
        # could correct to e.g. 0.001 Currency/kWh * annuity and O&M
        n.add(
            "Store",
            [oil_bus + " Store" for oil_bus in spatial.oil.nodes],
            bus=spatial.oil.nodes,
            e_nom_extendable=True,
            e_cyclic=True,
            carrier="oil",
        )

    if "oil" not in n.generators.carrier.unique():
        n.add(
            "Generator",
            spatial.oil.nodes,
            bus=spatial.oil.nodes,
            p_nom_extendable=True,
            carrier="oil",
            marginal_cost=costs.at["oil", "fuel"],
        )

# def remove_elec_base_techs(n):
#     """
#     Remove conventional generators (e.g. OCGT) and storage units (e.g.
#     batteries and H2) from base electricity-only network, since they're added
#     here differently using links.
#     """
#     for c in n.iterate_components(snakemake.params.pypsa_Currency):
#         to_keep = snakemake.params.pypsa_Currency[c.name]
#         to_remove = pd.Index(c.df.carrier.unique()).symmetric_difference(to_keep)
#         if to_remove.empty: 
#             continue
#         logger.info(f"Removing {c.list_name} with carrier {list(to_remove)}")
#         names = c.df.index[c.df.carrier.isin(to_remove)]
#         n.mremove(c.name, names)
#         n.carriers.drop(to_remove, inplace=True, errors="ignore")

def remove_elec_base_battery(n):
    """
    Remove conventional batteries electricity-only network, since they're added
    here differently using links.
    """
    to_remove = "battery"
    names = n.storage_units[n.storage_units.carrier == to_remove].index
    n.mremove("StorageUnit",names)
    n.carriers.drop(to_remove, inplace=True, errors="ignore")


def add_aviation(n, cost):
    all_aviation = ["total international aviation", "total domestic aviation"]

    aviation_demand = (
        energy_totals.loc[["ZA"], all_aviation].sum(axis=1).sum()  # * 1e6 / 8760
    )

    airports = pd.read_csv(snakemake.input.airports, keep_default_na=False)
    airports = airports[airports.country.isin(["ZA"])]


    airports["location"] = airports[["x", "y", "country"]].apply(
        lambda port: locate_bus(
            port[["x", "y"]],
            regions,
        ),
        axis=1,
    )

    airports = airports.set_index("location")

    ind = pd.DataFrame(n.buses.index[n.buses.carrier == "AC"])

    ind = ind.set_index(n.buses.index[n.buses.carrier == "AC"])
    airports["p_set"] = airports["fraction"].apply(
        lambda frac: frac * aviation_demand * 1e6 / 8760
    )

    airports = pd.concat([airports, ind])

    # airports = airports.fillna(0)

    airports = airports.groupby(airports.index).sum()
    n.add(
        "Load",
        nodes,
        suffix=" kerosene for aviation",
        bus=spatial.oil.nodes,
        carrier="kerosene for aviation",
        p_set=airports["p_set"][nodes],
    )

    co2 = airports["p_set"].sum() * costs.at["oil", "CO2 intensity"]

    n.add(
        "Load",
        "aviation oil emissions",
        bus="co2 atmosphere",
        carrier="oil emissions",
        p_set=-co2,
    )



def add_rail_transport(n, costs):
    p_set_elec = nodal_energy_totals.loc[nodes, "electricity rail"]
    p_set_oil = (nodal_energy_totals.loc[nodes, "total rail"]) - p_set_elec

    n.add(
        "Load",
        nodes,
        suffix=" rail transport oil",
        bus=spatial.oil.nodes,
        carrier="rail transport oil",
        p_set=p_set_oil * 1e6 / 8760,
    )

    n.add(
        "Load",
        nodes,
        suffix=" rail transport electricity",
        bus=nodes,
        carrier="rail transport electricity",
        p_set=p_set_elec * 1e6 / 8760,
    )

    loads_i = n.loads.index[n.loads.carrier == "AC"]
    factor = (1 - (p_set_elec * 1e6).sum() / n.loads_t.p_set[loads_i].sum().sum())
    n.loads_t.p_set[loads_i] *= factor

def add_industry(n, costs):
    logger.info("adding industrial demand")
    # 1e6 to convert TWh to MWh



    # industrial_demand.reset_index(inplace=True)

    # Add carrier Biomass

    n.add(
        "Bus",
        spatial.biomass.industry,
        location=spatial.biomass.locations,
        carrier="solid biomass for industry",
    )

    if options["biomass_transport"]:
        p_set = (
            industrial_demand.loc[spatial.biomass.locations, "solid biomass"].rename(
                index=lambda x: x + " solid biomass for industry"
            )
            / 8760
        )
    else:
        p_set = industrial_demand["solid biomass"].sum() / 8760

    n.add(
        "Load",
        spatial.biomass.industry,
        bus=spatial.biomass.industry,
        carrier="solid biomass for industry",
        p_set=p_set,
    )

    n.add(
        "Link",
        spatial.biomass.industry,
        bus0=spatial.biomass.nodes,
        bus1=spatial.biomass.industry,
        carrier="solid biomass for industry",
        p_nom_extendable=True,
        efficiency=1.0,
    )
    if snakemake.config["sector"]["cc"]:
        n.add(
            "Link",
            spatial.biomass.industry_cc,
            bus0=spatial.biomass.nodes,
            bus1=spatial.biomass.industry,
            bus2="co2 atmosphere",
            bus3=spatial.co2.nodes,
            carrier="solid biomass for industry CC",
            p_nom_extendable=True,
            capital_cost=costs.at["cement capture", "fixed"]
            * costs.at["solid biomass", "CO2 intensity"],
            efficiency=0.9,  # TODO: make config option
            efficiency2=-costs.at["solid biomass", "CO2 intensity"]
            * costs.at["cement capture", "capture_rate"],
            efficiency3=costs.at["solid biomass", "CO2 intensity"]
            * costs.at["cement capture", "capture_rate"],
            lifetime=costs.at["cement capture", "lifetime"],
        )

    # CARRIER = FOSSIL GAS

    nodes = (
        pop_layout.index
    )  # TODO where to change country code? 2 letter country codes.

    # industrial_demand['TWh/a (MtCO2/a)'] = industrial_demand['TWh/a (MtCO2/a)'].apply(
    #     lambda cocode: two_2_three_digits_country(cocode[:2]) + "." + cocode[3:])

    # industrial_demand.set_index("TWh/a (MtCO2/a)", inplace=True)

    # n.add("Bus", "gas for industry", location="ZA", carrier="gas for industry")
    n.add(
        "Bus",
        spatial.gas.industry,
        location=spatial.gas.locations,
        carrier="gas for industry",
    )

    gas_demand = industrial_demand.loc[nodes, "gas"] / 8760.0

    if options["gas"]["spatial_gas"]:
        spatial_gas_demand = gas_demand.rename(index=lambda x: x + " gas for industry")
    else:
        spatial_gas_demand = gas_demand.sum()

    n.add(
        "Load",
        spatial.gas.industry,
        bus=spatial.gas.industry,
        carrier="gas for industry",
        p_set=spatial_gas_demand,
    )

    n.add(
        "Link",
        spatial.gas.industry,
        # bus0="ZA gas",
        bus0=spatial.gas.nodes,
        # bus1="gas for industry",
        bus1=spatial.gas.industry,
        bus2="co2 atmosphere",
        carrier="gas for industry",
        p_nom_extendable=True,
        efficiency=1.0,
        efficiency2=costs.at["gas", "CO2 intensity"],
    )
    if snakemake.config["sector"]["cc"]:
        n.add(
            "Link",
            spatial.gas.industry_cc,
            # suffix=" gas for industry CC",
            # bus0="ZA gas",
            bus0=spatial.gas.nodes,
            bus1=spatial.gas.industry,
            bus2="co2 atmosphere",
            bus3=spatial.co2.nodes,
            carrier="gas for industry CC",
            p_nom_extendable=True,
            capital_cost=costs.at["cement capture", "fixed"]
            * costs.at["gas", "CO2 intensity"],
            efficiency=0.9,
            efficiency2=costs.at["gas", "CO2 intensity"]
            * (1 - costs.at["cement capture", "capture_rate"]),
            efficiency3=costs.at["gas", "CO2 intensity"]
            * costs.at["cement capture", "capture_rate"],
            lifetime=costs.at["cement capture", "lifetime"],
        )

    #################################################### CARRIER = HYDROGEN

    n.add(
        "Load",
        nodes,
        suffix=" H2 for industry",
        bus=spatial.h2.nodes,
        carrier="H2 for industry",
        p_set=industrial_demand["hydrogen"].apply(lambda frac: frac / 8760),
    )

    # CARRIER = LIQUID HYDROCARBONS
    n.add(
        "Load",
        nodes,
        suffix=" naphtha for industry",
        bus=spatial.oil.nodes,
        carrier="naphtha for industry",
        p_set=industrial_demand["oil"].apply(lambda frac: frac / 8760),
    )

    #     #NB: CO2 gets released again to atmosphere when plastics decay or kerosene is burned
    #     #except for the process emissions when naphtha is used for petrochemicals, which can be captured with other industry process emissions
    #     #tco2 per hour
    # TODO kerosene for aviation should be added too but in the right func.
    co2_release = [" naphtha for industry"]
    # check land tranport

    co2 = (
        n.loads.loc[nodes + co2_release, "p_set"].sum()
        * costs.at["oil", "CO2 intensity"]
        # - industrial_demand["process emission from feedstock"].sum()
        # / 8760
    )

    n.add(
        "Load",
        "industry oil emissions",
        bus="co2 atmosphere",
        carrier="industry oil emissions",
        p_set=-co2,
    )

    co2 = (
        industrial_demand["coal"].sum()
        * costs.at["coal", "CO2 intensity"]
        # - industrial_demand["process emission from feedstock"].sum()
        / 8760
    )

    n.add(
        "Load",
        "industry coal emissions",
        bus="co2 atmosphere",
        carrier="industry coal emissions",
        p_set=-co2,
    )

    ########################################################### CARIER = HEAT
    # TODO simplify bus expression
    # n.add(
    #     "Load",
    #     nodes,
    #     suffix=" low-temperature heat for industry",
    #     bus=[
    #         node + " urban central heat"
    #         if node + " urban central heat" in n.buses.index
    #         else node + " services urban decentral heat"
    #         for node in nodes
    #     ],
    #     carrier="low-temperature heat for industry",
    #     p_set=industrial_demand.loc[nodes, "low-temperature heat"] / 8760,
    # )

    ################################################## CARRIER = ELECTRICITY

    # remove today's industrial electricity demand by scaling down total electricity demand
    loads_i = n.loads.index[n.loads.carrier == "AC"]

    factor = (
        1
        - industrial_demand.loc[loads_i, "electricity"].sum()
        / n.loads_t.p_set[loads_i].sum().sum()
    )
    
    n.loads_t.p_set[loads_i] *= factor
    
    
    industrial_elec = industrial_demand["electricity"].apply(
        lambda frac: frac / 8760
    )

    n.add(
        "Load",
        nodes,
        suffix=" industry electricity",
        bus=nodes,
        carrier="industry electricity",
        p_set=industrial_elec,
    )

    n.add("Bus", "process emissions", location="ZA", carrier="process emissions")

    # this should be process emissions fossil+feedstock
    # then need load on atmosphere for feedstock emissions that are currently going to atmosphere via Link Fischer-Tropsch demand
    n.add(
        "Load",
        nodes,
        suffix=" process emissions",
        bus="process emissions",
        carrier="process emissions",
        p_set=-(
            #    industrial_demand["process emission from feedstock"]+
            industrial_demand["process emissions"]
        )
        / 8760,
    )

    n.add(
        "Link",
        "process emissions",
        bus0="process emissions",
        bus1="co2 atmosphere",
        carrier="process emissions",
        p_nom_extendable=True,
        efficiency=1.0,
    )

    # assume enough local waste heat for CC
    if snakemake.config["sector"]["cc"]:
        n.add(
            "Link",
            spatial.co2.locations,
            suffix=" process emissions CC",
            bus0="process emissions",
            bus1="co2 atmosphere",
            bus2=spatial.co2.nodes,
            carrier="process emissions CC",
            p_nom_extendable=True,
            capital_cost=costs.at["cement capture", "fixed"],
            efficiency=1 - costs.at["cement capture", "capture_rate"],
            efficiency2=costs.at["cement capture", "capture_rate"],
            lifetime=costs.at["cement capture", "lifetime"],
        )
        

def add_land_transport(n, costs):
    """
    Function to add land transport to network
    """
    # TODO options?

    logger.info("adding land transport")


    fuel_cell_share = round(energy_totals.at["ZA","total road fcev"] /
        energy_totals.at["ZA","total road"], 4
    )
    
    electric_share  = round(energy_totals.at["ZA","total road ev"] /
        energy_totals.at["ZA","total road"], 4
    )

    ice_share = 1 - fuel_cell_share - electric_share

    logger.info("FCEV share: {}".format(fuel_cell_share))
    logger.info("EV share: {}".format(electric_share))
    logger.info("ICEV share: {}".format(ice_share))

    assert ice_share >= 0, "Error, more FCEV and EV share than 1."

    if electric_share > 0:
        n.add("Carrier", "Li ion")

        n.add(
            "Bus",
            nodes,
            location=nodes,
            suffix=" EV battery",
            carrier="Li ion",
            x=n.buses.loc[list(nodes)].x.values,
            y=n.buses.loc[list(nodes)].y.values,
        )

        p_set = (
            electric_share
            * (
                transport[nodes]
                + cycling_shift(transport[nodes], 1)
                + cycling_shift(transport[nodes], 2)
            )
            / 3
        )

        n.add(
            "Load",
            nodes,
            suffix=" land transport EV",
            bus=nodes + " EV battery",
            carrier="land transport EV",
            p_set=p_set,
        )

    #    # remove land transport EV electricity demand by scaling down total electricity demand
        loads_i = n.loads.index[n.loads.carrier == "AC"]

        factor = (
            1
            - p_set.sum().sum()
            / n.loads_t.p_set[loads_i].sum().sum()
        )
        
        n.loads_t.p_set[loads_i] *= factor

        p_nom = (
            nodal_transport_data["number cars"]
            * options.get("bev_charge_rate", 0.011)
            * electric_share
        )

        n.add(
            "Link",
            nodes,
            suffix=" BEV charger",
            bus0=nodes,
            bus1=nodes + " EV battery",
            p_nom=p_nom,
            carrier="BEV charger",
            p_max_pu=avail_profile[nodes],
            efficiency=options.get("bev_charge_efficiency", 0.9),
            # These were set non-zero to find LU infeasibility when availability = 0.25
            # p_nom_extendable=True,
            # p_nom_min=p_nom,
            # capital_cost=1e6,  #i.e. so high it only gets built where necessary
        )

    if electric_share > 0 and options["v2g"]:
        n.add(
            "Link",
            nodes,
            suffix=" V2G",
            bus1=nodes,
            bus0=nodes + " EV battery",
            p_nom=p_nom,
            carrier="V2G",
            p_max_pu=avail_profile[nodes],
            efficiency=options.get("bev_charge_efficiency", 0.9),
        )

    if electric_share > 0 and options["bev_dsm"]:
        e_nom = (
            nodal_transport_data["number cars"]
            * options.get("bev_energy", 0.05)
            * options["bev_availability"]
            * electric_share
        )

        n.add(
            "Store",
            nodes,
            suffix=" battery storage",
            bus=nodes + " EV battery",
            carrier="battery storage",
            e_cyclic=True,
            e_nom=e_nom,
            e_max_pu=1,
            e_min_pu=dsm_profile[nodes],
        )

    if fuel_cell_share > 0:
        n.add(
            "Load",
            nodes,
            suffix=" land transport fuel cell",
            bus=spatial.h2.nodes,
            carrier="land transport fuel cell",
            p_set=fuel_cell_share * transport[nodes],
        )

    if ice_share > 0:
        if "oil" not in n.buses.carrier.unique():
            n.add(
                "Bus", spatial.oil.nodes, location=spatial.oil.locations, carrier="oil"
            )

        n.add(
            "Load",
            nodes,
            suffix=" land transport oil",
            bus=spatial.oil.nodes,
            carrier="land transport oil",
            p_set=ice_share * transport[nodes],
        )

        co2 = (
            ice_share
            * transport[nodes].sum().sum()
            / 8760
            * costs.at["oil", "CO2 intensity"]
        )

        n.add(
            "Load",
            "land transport oil emissions",
            bus="co2 atmosphere",
            carrier="land transport oil emissions",
            p_set=-co2,
        )


def create_nodes_for_heat_sector():
    # TODO pop_layout

    # rural are areas with low heating density and individual heating
    # urban are areas with high heating density
    # urban can be split into district heating (central) and individual heating (decentral)

    ct_urban = pop_layout.urban.groupby(pop_layout.ct).sum()
    # distribution of urban population within a country
    pop_layout["urban_ct_fraction"] = pop_layout.urban / pop_layout.ct.map(ct_urban.get)

    sectors = ["residential", "services"]

    nodes = {}
    urban_fraction = pop_layout.urban / pop_layout[["rural", "urban"]].sum(axis=1)

    for sector in sectors:
        nodes[sector + " rural"] = pop_layout.index
        nodes[sector + " urban decentral"] = pop_layout.index

    # maximum potential of urban demand covered by district heating
    central_fraction = options["district_heating"]["potential"]
    # district heating share at each node
    dist_fraction_node = (
        district_heat_share["district heat share"]
        * pop_layout["urban_ct_fraction"]
        / pop_layout["fraction"]
    )
    nodes["urban central"] = dist_fraction_node.index
    # if district heating share larger than urban fraction -> set urban
    # fraction to district heating share
    urban_fraction = pd.concat([urban_fraction, dist_fraction_node], axis=1).max(axis=1)
    # difference of max potential and today's share of district heating
    diff = (urban_fraction * central_fraction) - dist_fraction_node
    progress = options["district_heating"]["progress"]
    dist_fraction_node += diff * progress
    # logger.info(
    #     "The current district heating share compared to the maximum",
    #     f"possible is increased by a progress factor of\n{progress}",
    #     "resulting in a district heating share of",  # "\n{dist_fraction_node}", #TODO fix district heat share
    # )

    return nodes, dist_fraction_node, urban_fraction

def add_heat(n, costs):

    logger.info("adding heat")

    sectors = ["residential", "services"]

    nodes, dist_fraction, urban_fraction = create_nodes_for_heat_sector()

    # NB: must add costs of central heating afterwards (Currency 400 / kWpeak, 50a, 1% FOM from Fraunhofer ISE)

    heat_systems = [
        "residential rural",
        "services rural",
        "residential urban decentral",
        "services urban decentral",
        "urban central",
    ]

    for name in heat_systems:
        name_type = "central" if name == "urban central" else "decentral"

        n.add("Carrier", name + " heat")

        n.add(
            "Bus",
            nodes[name] + " {} heat".format(name),
            location=nodes[name],
            carrier=name + " heat",
        )

        ## Add heat load

        for sector in sectors:
            # heat demand weighting
            if "rural" in name:
                factor = 1 - urban_fraction[nodes[name]]
            elif "urban central" in name:
                factor = dist_fraction[nodes[name]]
            elif "urban decentral" in name:
                factor = urban_fraction[nodes[name]] - dist_fraction[nodes[name]]
            else:
                raise NotImplementedError(
                    f" {name} not in heat systems: {heat_systems}"
                )

            if sector in name:
                heat_load = (
                    heat_demand[[sector + " water", sector + " space"]]
                    .groupby(level=1, axis=1)
                    .sum()[nodes[name]]
                    .multiply(factor)
                )

        if name == "urban central":
            heat_load = (
                heat_demand.groupby(level=1, axis=1)
                .sum()[nodes[name]]
                .multiply(
                    factor * (1 + options["district_heating"]["district_heating_loss"])
                )
            )

        if options["add_useful_heat_demands"]:
            n.add(
                "Load",
                nodes[name],
                suffix=f" {name} heat",
                bus=nodes[name] + f" {name} heat",
                carrier=name + " heat",
                p_set=heat_load,
            )

        ## Add heat pumps

        if "rural" in name:
            heat_pump_type = options["heat_pump_type"]["rural"]
        elif "urban central" in name:
            heat_pump_type = options["heat_pump_type"]["urban central"]
        elif "urban decentral" in name:
            heat_pump_type = options["heat_pump_type"]["urban decentral"]
        else:
            raise NotImplementedError(f"No heat pump type assigned for {name}")
        
        costs_name = f"{name_type} {heat_pump_type}-sourced heat pump"
        cop = {"air": ashp_cop, "ground": gshp_cop}
        efficiency = cop[heat_pump_type][nodes[name]]

        n.add(
            "Link",
            nodes[name],
            suffix=f" {name} {heat_pump_type} heat pump",
            bus0=nodes[name],
            bus1=nodes[name] + f" {name} heat",
            carrier=f"{name} {heat_pump_type} heat pump",
            efficiency=efficiency,
            capital_cost=costs.at[costs_name, "efficiency"]
            * costs.at[costs_name, "fixed"],
            p_nom_extendable=True,
            lifetime=costs.at[costs_name, "lifetime"],
        )

        if options["tes"]:
            n.add("Carrier", name + " water tanks")

            n.add(
                "Bus",
                nodes[name] + f" {name} water tanks",
                location=nodes[name],
                carrier=name + " water tanks",
            )

            n.add(
                "Link",
                nodes[name] + f" {name} water tanks charger",
                bus0=nodes[name] + f" {name} heat",
                bus1=nodes[name] + f" {name} water tanks",
                efficiency=costs.at["water tank charger", "efficiency"],
                carrier=name + " water tanks charger",
                p_nom_extendable=True,
            )

            n.add(
                "Link",
                nodes[name] + f" {name} water tanks discharger",
                bus0=nodes[name] + f" {name} water tanks",
                bus1=nodes[name] + f" {name} heat",
                carrier=name + " water tanks discharger",
                efficiency=costs.at["water tank discharger", "efficiency"],
                p_nom_extendable=True,
            )

            if isinstance(options["tes_tau"], dict):
                tes_time_constant_days = options["tes_tau"][name_type]
            else:  # TODO add logger
                # logger.warning("Deprecated: a future version will require you to specify 'tes_tau' ",
                # "for 'decentral' and 'central' separately.")
                tes_time_constant_days = (
                    options["tes_tau"] if name_type == "decentral" else 180.0
                )

            # conversion from Currency/m^3 to Currency/MWh for 40 K diff and 1.17 kWh/m^3/K
            capital_cost = (
                costs.at[name_type + " water tank storage", "fixed"] / 0.00117 / 40
            )

            n.add(
                "Store",
                nodes[name] + f" {name} water tanks",
                bus=nodes[name] + f" {name} water tanks",
                e_cyclic=True,
                e_nom_extendable=True,
                carrier=name + " water tanks",
                standing_loss=1 - np.exp(-1 / 24 / tes_time_constant_days),
                capital_cost=capital_cost,
                lifetime=costs.at[name_type + " water tank storage", "lifetime"],
            )

        if options["boilers"]:
            key = f"{name_type} resistive heater"

            n.add(
                "Link",
                nodes[name] + f" {name} resistive heater",
                bus0=nodes[name],
                bus1=nodes[name] + f" {name} heat",
                carrier=name + " resistive heater",
                efficiency=costs.at[key, "efficiency"],
                capital_cost=costs.at[key, "efficiency"] * costs.at[key, "fixed"],
                p_nom_extendable=True,
                lifetime=costs.at[key, "lifetime"],
            )

            key = f"{name_type} gas boiler"

            n.add(
                "Link",
                nodes[name] + f" {name} gas boiler",
                p_nom_extendable=True,
                bus0=spatial.gas.nodes,
                bus1=nodes[name] + f" {name} heat",
                bus2="co2 atmosphere",
                carrier=name + " gas boiler",
                efficiency=costs.at[key, "efficiency"],
                efficiency2=costs.at["gas", "CO2 intensity"],
                capital_cost=costs.at[key, "efficiency"] * costs.at[key, "fixed"],
                lifetime=costs.at[key, "lifetime"],
            )

        if options["solar_thermal"]:
            n.add("Carrier", name + " solar thermal")

            n.add(
                "Generator",
                nodes[name],
                suffix=f" {name} solar thermal collector",
                bus=nodes[name] + f" {name} heat",
                carrier=name + " solar thermal",
                p_nom_extendable=True,
                capital_cost=costs.at[name_type + " solar thermal", "fixed"],
                p_max_pu=solar_thermal[nodes[name]],
                lifetime=costs.at[name_type + " solar thermal", "lifetime"],
            )

        if options["chp"] and name == "urban central":
            # add gas CHP; biomass CHP is added in biomass section
            n.add(
                "Link",
                nodes[name] + " urban central gas CHP",
                bus0=spatial.gas.nodes,
                bus1=nodes[name],
                bus2=nodes[name] + " urban central heat",
                bus3="co2 atmosphere",
                carrier="urban central gas CHP",
                p_nom_extendable=True,
                capital_cost=costs.at["central gas CHP", "fixed"]
                * costs.at["central gas CHP", "efficiency"],
                marginal_cost=costs.at["central gas CHP", "VOM"],
                efficiency=costs.at["central gas CHP", "efficiency"],
                efficiency2=costs.at["central gas CHP", "efficiency"]
                / costs.at["central gas CHP", "c_b"],
                efficiency3=costs.at["gas", "CO2 intensity"],
                lifetime=costs.at["central gas CHP", "lifetime"],
            )
            if snakemake.config["sector"]["cc"]:
                n.add(
                    "Link",
                    nodes[name] + " urban central gas CHP CC",
                    # bus0="Africa gas",
                    bus0=spatial.gas.nodes,
                    bus1=nodes[name],
                    bus2=nodes[name] + " urban central heat",
                    bus3="co2 atmosphere",
                    bus4=spatial.co2.df.loc[nodes[name], "nodes"].values,
                    carrier="urban central gas CHP CC",
                    p_nom_extendable=True,
                    capital_cost=costs.at["central gas CHP", "fixed"]
                    * costs.at["central gas CHP", "efficiency"]
                    + costs.at["biomass CHP capture", "fixed"]
                    * costs.at["gas", "CO2 intensity"],
                    marginal_cost=costs.at["central gas CHP", "VOM"],
                    efficiency=costs.at["central gas CHP", "efficiency"]
                    - costs.at["gas", "CO2 intensity"]
                    * (
                        costs.at["biomass CHP capture", "electricity-input"]
                        + costs.at[
                            "biomass CHP capture", "compression-electricity-input"
                        ]
                    ),
                    efficiency2=costs.at["central gas CHP", "efficiency"]
                    / costs.at["central gas CHP", "c_b"]
                    + costs.at["gas", "CO2 intensity"]
                    * (
                        costs.at["biomass CHP capture", "heat-output"]
                        + costs.at["biomass CHP capture", "compression-heat-output"]
                        - costs.at["biomass CHP capture", "heat-input"]
                    ),
                    efficiency3=costs.at["gas", "CO2 intensity"]
                    * (1 - costs.at["biomass CHP capture", "capture_rate"]),
                    efficiency4=costs.at["gas", "CO2 intensity"]
                    * costs.at["biomass CHP capture", "capture_rate"],
                    lifetime=costs.at["central gas CHP", "lifetime"],
                )

        if options["chp"] and options["micro_chp"] and name != "urban central":
            n.add(
                "Link",
                nodes[name] + f" {name} micro gas CHP",
                p_nom_extendable=True,
                # bus0="Africa gas",
                bus0=spatial.gas.nodes,
                bus1=nodes[name],
                bus2=nodes[name] + f" {name} heat",
                bus3="co2 atmosphere",
                carrier=name + " micro gas CHP",
                efficiency=costs.at["micro CHP", "efficiency"],
                efficiency2=costs.at["micro CHP", "efficiency-heat"],
                efficiency3=costs.at["gas", "CO2 intensity"],
                capital_cost=costs.at["micro CHP", "fixed"],
                lifetime=costs.at["micro CHP", "lifetime"],
            )

def add_dac(n, costs):
    heat_carriers = ["urban central heat", "services urban decentral heat"]
    heat_buses = n.buses.index[n.buses.carrier.isin(heat_carriers)]
    locations = n.buses.location[heat_buses]

    efficiency2 = -(
        costs.at["direct air capture", "electricity-input"]
        + costs.at["direct air capture", "compression-electricity-input"]
    )
    efficiency3 = -(
        costs.at["direct air capture", "heat-input"]
        - costs.at["direct air capture", "compression-heat-output"]
    )


    n.add(
        "Link",
        heat_buses.str.replace(" heat", " DAC"),
        bus0="co2 atmosphere",
        bus1=spatial.co2.df.loc[locations, "nodes"].values,
        bus2=locations.values,
        bus3=heat_buses,
        carrier="DAC",
        capital_cost=costs.at["direct air capture", "fixed"],
        efficiency=1.0,
        efficiency2=efficiency2,
        efficiency3=efficiency3,
        p_nom_extendable=True,
        lifetime=costs.at["direct air capture", "lifetime"],
    )



def add_services(n, costs):
    
    profile_elec_today = n.loads_t.p_set[nodes] / n.loads_t.p_set[nodes].sum().sum()

    p_set_elec = (
        profile_elec_today
        * energy_totals.loc[["ZA"], [
            "services electricity",
            "electricity services space", # TODO: The heat demand is added as final energy demand and not as end use demand
            "electricity services water"]].sum().sum()
        * 1e6
    )

    n.add(
        "Load",
        nodes,
        suffix=" services electricity",
        bus=nodes,
        carrier="services electricity",
        p_set=p_set_elec,
    )
    
    loads_i = n.loads.index[n.loads.carrier == "AC"]
    factor = (1 - p_set_elec.sum() / n.loads_t.p_set[loads_i].sum().sum())
    n.loads_t.p_set[loads_i] *= factor
    
    p_set_biomass = (
        profile_elec_today
        * energy_totals.loc[["ZA"], "services biomass"].sum()
        * 1e6
    )

    n.add(
        "Load",
        nodes,
        suffix=" services biomass",
        bus=spatial.biomass.nodes,
        carrier="services biomass",
        p_set=p_set_biomass,
    )

    # co2 = (
    #     p_set_biomass.sum().sum() * costs.at["solid biomass", "CO2 intensity"]
    # ) / 8760

    # n.add(
    #     "Load",
    #     "services biomass emissions",
    #     bus="co2 atmosphere",
    #     carrier="biomass emissions",
    #     p_set=-co2,
    # )
    p_set_oil = (
        profile_elec_today * energy_totals.loc[["ZA"], "services oil"].sum() * 1e6
    )

    n.add(
        "Load",
        nodes,
        suffix=" services oil",
        bus=spatial.oil.nodes,
        carrier="services oil",
        p_set=p_set_oil,
    )

    co2 = (p_set_oil.sum().sum() * costs.at["oil", "CO2 intensity"]) / 8760

    n.add(
        "Load",
        "services oil emissions",
        bus="co2 atmosphere",
        carrier="oil emissions",
        p_set=-co2,
    )

    p_set_gas = (
        profile_elec_today * energy_totals.loc[["ZA"], "services gas"].sum() * 1e6
    )

    n.add(
        "Load",
        nodes,
        suffix=" services gas",
        bus=spatial.gas.nodes,
        carrier="services gas",
        p_set=p_set_gas,
    )

    co2 = (p_set_gas.sum().sum() * costs.at["gas", "CO2 intensity"]) / 8760

    n.add(
        "Load",
        "services gas emissions",
        bus="co2 atmosphere",
        carrier="gas emissions",
        p_set=-co2,
    )


def add_agriculture(n, costs):
    
    p_set_elec = nodal_energy_totals.loc[nodes, "agriculture electricity"] * 1e6
    
    n.add(
        "Load",
        nodes,
        suffix=" agriculture electricity",
        bus=nodes,
        carrier="agriculture electricity",
        p_set=p_set_elec / 8760,
    )

    loads_i = n.loads.index[n.loads.carrier == "AC"]
    factor = (1 - p_set_elec.sum() / n.loads_t.p_set[loads_i].sum().sum())
    n.loads_t.p_set[loads_i] *= factor

    n.add(
        "Load",
        nodes,
        suffix=" agriculture oil",
        bus=spatial.oil.nodes,
        carrier="agriculture oil",
        p_set=nodal_energy_totals.loc[nodes, "agriculture oil"] * 1e6 / 8760,
    )
    co2 = (
        nodal_energy_totals.loc[nodes, "agriculture oil"]
        * 1e6
        / 8760
        * costs.at["oil", "CO2 intensity"]
    ).sum()

    n.add(
        "Load",
        "agriculture oil emissions",
        bus="co2 atmosphere",
        carrier="oil emissions",
        p_set=-co2,
    )


def add_residential(n, costs):
    
    profile_elec_today = n.loads_t.p_set[nodes] / n.loads_t.p_set[nodes].sum().sum()

    p_set_elec = (
        profile_elec_today
        * energy_totals.loc[["ZA"], [
            "electricity residential",
            "electricity residential space", # TODO: The heat demand is added as final energy demand and not as end use demand
            "electricity residential water"]].sum().sum()
        * 1e6
    )

    n.add(
        "Load",
        nodes,
        suffix=" residential electricity",
        bus=nodes,
        carrier="residential electricity",
        p_set=p_set_elec,
    )
    
    loads_i = n.loads.index[n.loads.carrier == "AC"]
    factor = (1 - p_set_elec.sum() / n.loads_t.p_set[loads_i].sum().sum())
    n.loads_t.p_set[loads_i] *= factor

    profile_elec_today = n.loads_t.p_set[nodes] / n.loads_t.p_set[nodes].sum().sum()

    p_set_oil = (
        profile_elec_today
        * energy_totals.loc[["ZA"], "residential oil"].sum()
        * 1e6
    ) 

    p_set_biomass = (
        profile_elec_today
        * energy_totals.loc[["ZA"], "residential biomass"].sum()
        * 1e6
    ) 

    p_set_gas = (
        profile_elec_today
        * energy_totals.loc[["ZA"], "residential gas"].sum()
        * 1e6
    )

    n.add(
        "Load",
        nodes,
        suffix=" residential oil",
        bus=spatial.oil.nodes,
        carrier="residential oil",
        p_set=p_set_oil,
    )
    co2 = (p_set_oil.sum().sum() * costs.at["oil", "CO2 intensity"]) / 8760

    n.add(
        "Load",
        "residential oil emissions",
        bus="co2 atmosphere",
        carrier="oil emissions",
        p_set=-co2,
    )
    n.add(
        "Load",
        nodes,
        suffix=" residential biomass",
        bus=spatial.biomass.nodes,
        carrier="residential biomass",
        p_set=p_set_biomass,
    )

    n.add(
        "Load",
        nodes,
        suffix=" residential gas",
        bus=spatial.gas.nodes,
        carrier="residential gas",
        p_set=p_set_gas,
    )
    co2 = (p_set_gas.sum().sum() * costs.at["gas", "CO2 intensity"]) / 8760

    n.add(
        "Load",
        "residential gas emissions",
        bus="co2 atmosphere",
        carrier="gas emissions",
        p_set=-co2,
    )

def adjust_elec_base_loads(n):
    logger.info("""
        All final energy electriciy loads were added based on UCT statistics.
        The total electricity load is lower than the historical loads presented by CSIR or IRPs. 
        According to our understanding, the UCT dataset excludes self-consumptions of coal plants, 
        expects mid-term GDP drop, is pessmistic regarding load shedding and excludes exports. 
        We keep and slightly adjust basic load to represent self-consumption, load-shedding and exports.
        """)
    
    self_consumption = 10
    load_shedding = 10
    exports = 10
    
    loads_i = n.loads.index[n.loads.carrier == "AC"]
            
    if investment_year <= 2040:
        elec_base = self_consumption + load_shedding + exports
        elec_base = elec_base / n.snapshot_weightings.generators.mean()
        
        factor = elec_base * 1e6 / n.loads_t.p_set[loads_i].sum().sum()
        n.loads_t.p_set[loads_i] *= factor
    else:
        elec_base = exports
        elec_base = elec_base / n.snapshot_weightings.generators.mean()
        factor = elec_base * 1e6 / n.loads_t.p_set[loads_i].sum().sum()
        n.loads_t.p_set[loads_i] *= factor       

def add_export(n, export_regions_index, carrier, export_volume):
    regions["country"] = "ZA"
    country_shape = regions.dissolve(by="country")
    country_shape = country_shape.to_crs("EPSG:3395")  
    # Get coordinates of the most western and northern point of the country and add a buffer of 2 degrees (equiv. to approx 220 km)
    x_export = country_shape.geometry.centroid.x.min() - 2
    y_export = country_shape.geometry.centroid.y.max() + 2
    
    buses_ports = n.buses.loc[export_regions_index + " " + carrier]
    buses_ports.index.name = "Bus"
    
    # add export bus
    n.add(
        "Bus",
        carrier + " export bus",
        carrier=carrier + " export bus",
        x=x_export,
        y=y_export,
    )

    logger.info(f"Adding export links and loads for {carrier}")
    n.add(
        "Link",
        buses_ports.index + " export",
        bus0=buses_ports.index,
        bus1=carrier + " export bus",
        carrier=carrier + " export",
        p_nom_extendable=True,
    )

    export_links = n.links[n.links.index.str.contains("export")]
    logger.info(export_links.index)
    n.add(
        "Load",
        carrier+" export load",
        bus=carrier+" export bus",
        carrier=carrier+" export load",
        p_set=export_volume/8760,
    )
    
    if carrier == "Fischer-Tropsch":
        co2 = export_volume * costs.at["oil", "CO2 intensity"] / 8760

        n.add(
            "Load",
            "Fischer-Tropsch export emissions",
            bus="co2 atmosphere",
            carrier="oil export emissions",
            p_set=-co2,
        )

def add_waste_heat(n, costs):

    logger.info("Add possibility to use industrial waste heat in district heating")

    # AC buses with district heating
    urban_central = n.buses.index[n.buses.carrier == "urban central heat"]
    if not urban_central.empty:
        urban_central = urban_central.str[: -len(" urban central heat")]

        link_carriers = n.links.carrier.unique()

        if (
            options["use_fischer_tropsch_waste_heat"]
            and "Fischer-Tropsch" in link_carriers
        ):
            n.links.loc[urban_central + " Fischer-Tropsch", "bus3"] = (
                urban_central + " urban central heat"
            )
            n.links.loc[urban_central + " Fischer-Tropsch", "efficiency3"] = (
                costs.at["Fischer-Tropsch","thermal-output"]
            ) * options["use_fischer_tropsch_waste_heat"]

        if options["use_methanation_waste_heat"] and "Sabatier" in link_carriers:
            n.links.loc[urban_central + " Sabatier", "bus3"] = (
                urban_central + " urban central heat"
            )
            n.links.loc[urban_central + " Sabatier", "efficiency3"] = (
                costs.at["methanation","thermal-output"]
            ) * options["use_methanation_waste_heat"]

        if options["use_haber_bosch_waste_heat"] and "Haber-Bosch" in link_carriers:
            n.links.loc[urban_central + " Haber-Bosch", "bus4"] = (
                urban_central + " urban central heat"
            )
            n.links.loc[urban_central + " Haber-Bosch", "efficiency4"] = (
               costs.at["Haber-Bosch","thermal-output"] # thermal-output per H2-input
               * -n.links.loc[urban_central + " Haber-Bosch", "efficiency2"] # H2-input/elec-input
            ) * options["use_haber_bosch_waste_heat"]
 
        if (
            options["use_electrolysis_waste_heat"]
            and "H2 Electrolysis" in link_carriers
        ):
            n.links.loc[urban_central + " H2 Electrolysis", "bus3"] = (
                urban_central + " urban central heat"
            )
            n.links.loc[urban_central + " H2 Electrolysis", "efficiency3"] = (
                0.84 - n.links.loc[urban_central + " H2 Electrolysis", "efficiency"]
            ) * options["use_electrolysis_waste_heat"]

        if options["use_fuel_cell_waste_heat"] and "H2 Fuel Cell" in link_carriers:
            n.links.loc[urban_central + " H2 Fuel Cell", "bus2"] = (
                urban_central + " urban central heat"
            )
            n.links.loc[urban_central + " H2 Fuel Cell", "efficiency2"] = (
                0.95 - n.links.loc[urban_central + " H2 Fuel Cell", "efficiency"]
            ) * options["use_fuel_cell_waste_heat"]
        
        # add a heat sink to compensate for potentially unusable waste heat
        n.add(
            "Generator",
            urban_central + " unused waste heat",
            bus=urban_central + " urban central heat",
            carrier="unused waste heat",
            p_nom_extendable=True,
            p_max_pu=0, 
            p_min_pu=-1,
        )

def set_transmission_limit(n, ll_type, factor, costs, Nyears=1):
    links_dc_b = n.links.carrier == "DC" if not n.links.empty else pd.Series()

    _lines_s_nom = (
        np.sqrt(3)
        * n.lines.type.map(n.line_types.i_nom)
        * n.lines.num_parallel
        * n.lines.bus0.map(n.buses.v_nom)
    )
    lines_s_nom = n.lines.s_nom.where(n.lines.type == "Al/St 240/40 4-bundle 380.0", _lines_s_nom)

    col = "capital_cost" if ll_type == "c" else "length"
    ref = (
        lines_s_nom @ n.lines[col]
        + n.links.loc[links_dc_b, "p_nom"] @ n.links.loc[links_dc_b, col]
    )

    #update_transmission_costs(n, costs)

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

def set_line_s_max_pu(n):
    s_max_pu = snakemake.config["lines"]["s_max_pu"]
    n.lines["s_max_pu"] = s_max_pu
    logger.info(f"N-1 security margin of lines set to {s_max_pu}")


def set_extendable_limits_per_link(n, model_file, model_setup, wc_regions):
    ext_years = n.investment_periods if n.multi_invest else [n.snapshots[0].year]
    ignore = {"max": "unc", "min": 0}
    
    link_limits = {
        lim: pd.read_excel(
            model_file,
            sheet_name=f'extendable_{lim}_build',
            index_col=[0, 1, 3, 2, 4],
        ).loc[(model_setup["extendable_build_limits"], wc_regions, slice(None)), ext_years]
        for lim in ["min", "max"]
    }

    ext_links = list(n.links.carrier[n.links.p_nom_extendable].unique())
    
    for lim, link_limit in link_limits.items():
        link_limit = link_limit.loc[link_limit.index.get_level_values(2) == "Link"]
        link_limit.index = link_limit.index.droplevel([0, 1, 2])
        link_limit = link_limit.loc[~(link_limit == ignore[lim]).all(axis=1)]
        link_limit = link_limit.loc[link_limit.index.get_level_values(1).isin(ext_links)]
        
        if link_limit.empty:
            continue
        else:
            for (location, carrier) in link_limit.index:
                for y in ext_years:
                    assert f"{location} {carrier}" in n.links.index, f"'{location} {carrier}' is not in n.links"
                    logger.info(f"Limiting '{location} {carrier}' link.")
                    n.links.at[f"{location} {carrier}",f"p_nom_{lim}"] = \
                        link_limit.at[(location,carrier),y] 





def add_co2limit(n, constant):
    n.add("GlobalConstraint", "CO2Limit",
          carrier_attribute="co2_emissions", sense="<=",
          constant=constant)

def add_emission_prices(n, emission_prices={"co2": 100.0}):
    ep = (
        pd.Series(emission_prices).rename(lambda x: x + "_emissions")
        * n.carriers.filter(like="_emissions")
    ).sum(axis=1)
    gen_ep = n.generators.carrier.map(ep) / n.generators.efficiency
    n.generators["marginal_cost"] += gen_ep
    su_ep = n.storage_units.carrier.map(ep) / n.storage_units.efficiency_dispatch
    n.storage_units["marginal_cost"] += su_ep

def average_every_nhours(n, offset):
    logger.info(f"Resampling the network to {offset}")
    m = n.copy(with_time=False)
    m.multi_invest = n.multi_invest # TODO create issue in PyPSA repo

    snapshot_weightings = n.snapshot_weightings.resample(offset).sum()
    snapshot_weightings = remove_leap_day(snapshot_weightings)
    m.set_snapshots(snapshot_weightings.index)
    m.snapshot_weightings = snapshot_weightings

    for c in n.iterate_components():
        pnl = getattr(m, c.list_name + "_t")
        for k, df in c.pnl.items():
            if not df.empty:
                resampled = df.resample(offset).mean()
                resampled = remove_leap_day(resampled)
                #resampled.index = snapshot_weightings.index
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


def solve_sector_network(n, sns):
    n.optimize.create_model(snapshots = sns, multi_investment_periods = n.multi_invest)
    # Custom constraints
    set_operational_limits(n, sns, model_file, model_setup)
    ccgt_steam_constraints(n, sns, model_file, model_setup, snakemake)
    #define_reserve_margin(n, sns, model_file, model_setup, snakemake)
    add_battery_constraints(n)
    
    # if n.stores.carrier.eq("co2 stored").any():
    #     limit = options["co2_sequestration_potential"]
    #     add_co2_sequestration_limit(n, sns, limit=limit) # TODO global limit only needed for co2_network
    

    custom_define_tech_capacity_expansion_limit(n, sns)

    solver_name = snakemake.config["solving"]["solver"]["name"]
    solver_options = snakemake.config["solving"]["solver_options"][solver_name]
    n.optimize.solve_model(solver_name=solver_name, solver_options=solver_options)

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake, sets_path_to_root
        
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        snakemake = mock_snakemake(
            'prepare_and_solve_sector_network', 
            **{
                'model_file':'NZEE-2040',
                'regions':'11-supply',
                'resarea':'corridors',
                'll':'v1.10',
                'opts':'73h-Co2L',
            }
        )
        
        sets_path_to_root("pypsa-rsa-sec")
        
    regions = gpd.read_file(
        snakemake.input.supply_regions, 
        layer=snakemake.wildcards.regions
        ).set_crs(crs=snakemake.config["crs"]["geo_crs"])

    n = pypsa.Network(snakemake.input[0])
    
    nodes = n.buses[n.buses.carrier == "AC"].index
    
    pop_layout = pd.read_csv(snakemake.input.clustered_pop_layout, index_col=0)
    
    n.loads.loc[nodes, "carrier"] = "AC"

    Nyears = n.snapshot_weightings.generators.sum() / 8760    
    
    model_file = pd.ExcelFile(snakemake.input.model_file)
    model_setup = (
        pd.read_excel(
            model_file, 
            sheet_name="model_setup",
            index_col=[0],
            na_values=NA_VALUES)
            .loc[snakemake.wildcards.model_file]
    )
    
    investment_year = int(model_setup["simulation_years"])
        
    add_missing_co2_emissions_from_model_file(
        n, 
        n.generators.carrier.unique(), 
        model_file)
    
    
    co2price = float(model_setup["co2price"])
    
    co2limit = float(model_setup["co2limit"]) 
    co2limit = co2limit * 1e6 if not np.isnan(co2limit) else None 
    
    projected_exports = (
        pd.read_excel(
            model_file, 
            sheet_name="projected_exports",
            index_col=[0],)
            .loc[model_setup["projected_exports"]]
            .set_index(["carrier","attribute"])
    )

    costs_file = snakemake.params.costs_filepath+f"costs_{investment_year}.csv"
    cost_options = snakemake.params.cost_options
    costs = prepare_costs(costs_file, 
                cost_options["USD_to_ZAR"],
                cost_options["discount_rate"], 
                Nyears, 
                cost_options["lifetime"])
    
    options = snakemake.params.sector_options

    define_spatial(nodes)


    nodal_energy_totals = pd.read_csv(
        snakemake.input.nodal_energy_totals,
        index_col=0,
        keep_default_na=False,
        na_values=[""],
    )
    
    energy_totals = pd.read_csv(
        snakemake.input.energy_totals,
        index_col=[0,1],
        keep_default_na=False,
        na_values=[""],
    ).xs(investment_year, level="year")

    transport = pd.read_csv(snakemake.input.transport, index_col=0, parse_dates=True)

    avail_profile = pd.read_csv(
        snakemake.input.avail_profile, index_col=0, parse_dates=True
    )
    dsm_profile = pd.read_csv(
        snakemake.input.dsm_profile, index_col=0, parse_dates=True
    )
    nodal_transport_data = pd.read_csv(  # TODO This only includes no. of cars, change name to something descriptive?
        snakemake.input.nodal_transport_data, index_col=0
    )

    # Load data required for the heat sector
    heat_demand = pd.read_csv(
        snakemake.input.heat_demand, index_col=0, header=[0, 1], parse_dates=True
    ).fillna(0)
    # Ground-sourced heatpump coefficient of performance
    gshp_cop = pd.read_csv(
        snakemake.input.gshp_cop, index_col=0, parse_dates=True
    )  # only needed with heat dep. hp cop allowed from config
    # TODO add option heat_dep_hp_cop to the config

    # Air-sourced heatpump coefficient of performance
    ashp_cop = pd.read_csv(
        snakemake.input.ashp_cop, index_col=0, parse_dates=True
    )  # only needed with heat dep. hp cop allowed from config

    # Solar thermal availability profiles
    solar_thermal = pd.read_csv(
        snakemake.input.solar_thermal, index_col=0, parse_dates=True
    )
    
    gshp_cop = pd.read_csv(snakemake.input.gshp_cop, index_col=0, parse_dates=True)

    # Share of district heating at each node
    district_heat_share = pd.read_csv(snakemake.input.district_heat_share, index_col=0)
    
    # Load industry demand data
    industrial_demand = pd.read_csv(
        snakemake.input.industrial_demand, index_col=[0,1], header=0
    ).xs(investment_year, level="year")  # * 1e6
    cols_to_convert = industrial_demand.columns.difference(["process emissions"])
    industrial_demand.loc[:,cols_to_convert] *= 1e6
    
    ##########################################################################
    ############## Functions adding different carrires and sectors ###########
    ##########################################################################

    add_co2(n, costs)  # TODO add costs

    # TODO This might be transferred to add_generation, but before apply remove_elec_base_techs(n) from PyPSA-Currency-Sec
    add_oil(n, costs)

    add_gas(n, costs)
    
    add_generation(n, costs)

    add_hydrogen_and_desalination(n, costs)  
            
    add_ammonia(n, costs)
            
    add_storage(n, costs)

    H2_liquid_fossil_conversions(n, costs)

    h2_hc_conversions(n, costs)
    
    add_heat(n, costs)
    
    add_waste_heat(n, costs)
    
    add_biomass(n, costs)

    add_industry(n, costs)

    add_shipping(n, costs)

    add_aviation(n, costs)

    add_land_transport(n, costs)

    add_rail_transport(n, costs)

    add_agriculture(n, costs)
    
    add_residential(n, costs)
    
    add_services(n, costs)
    
    # def remove_elec_base_loads(n):
    #     logger.info("Removing base AC loads from the network, because all electricity loads were added sector-wise.")
    #     loads_i = n.loads.index[n.loads.carrier == "AC"]
    #     n.mremove("Load", loads_i)
    # remove_elec_base_loads(n)    
    
    adjust_elec_base_loads(n)
    
    
    if options["dac"]:
        add_dac(n, costs)
        
        
    export_regions_index = pd.Index(spatial.coastal_nodes)
    
    for product in projected_exports.index.get_level_values(0).unique():
        export_volume = projected_exports.at[(product,"export_volume"),investment_year].round(2)*1e6
        logging.info(f"Setting {product} export volume of {export_volume} " 
                      f"exportable via {export_regions_index}.")
        add_export(n, export_regions_index, product, export_volume)
            

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
        
    if co2limit:
        logging.info(f"Setting CO2 limit according to model_setup value of {co2limit}.")
        add_co2limit(n,co2limit)

    logging.info(f"Setting CO2 price according to model_setup value of {co2price}.")
    add_emission_prices(n, emission_prices={"co2": co2price})
    
    
    logging.info("Setting transmission constraints")
    Nyears = n.snapshot_weightings.objective.sum() / 8760.0        
    set_line_s_max_pu(n)
    ll_type, factor = snakemake.wildcards.ll[0], snakemake.wildcards.ll[1:]
    param = load_extendable_parameters(n, model_file, model_setup, snakemake)
    set_transmission_limit(n, ll_type, factor, costs, Nyears)

    set_line_nom_max(
        n,
        s_nom_max_set=snakemake.config["lines"].get("s_nom_max,", np.inf),
        p_nom_max_set=snakemake.config["links"].get("p_nom_max,", np.inf),
    )
    logging.info("Setting global and regional build limits")
    if snakemake.wildcards.regions != "1-supply": #covered under single bus limits
        set_extendable_limits_global(n, model_file, model_setup) 
    set_extendable_limits_per_bus(n, model_file, model_setup, snakemake.wildcards.regions)
    set_extendable_limits_per_link(n, model_file, model_setup, snakemake.wildcards.regions)

    print(n.links.query("carrier == 'H2 Electrolysis'").p_nom_max)

    logging.info("Solving sector coupled network")
    solve_sector_network(n, n.snapshots)
    
    n.export_to_netcdf(snakemake.output[0])

