# SPDX-FileCopyrightText:  PyPSA-ZA2, PyPSA-ZA, PyPSA-Earth and PyPSA-Eur Authors
# # SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-

"""
Creates the network topology for South Africa from either South Africa"s shape file, GCCA map extract for 10 supply regions or 27-supply regions shape file as a PyPSA
network.

Relevant Settings
-----------------

.. code:: yaml

    snapshots:

    supply_regions:

    electricity:
        voltages:

    lines:
        types:
        s_max_pu:
        under_construction:

    links:
        p_max_pu:
        under_construction:
        include_tyndp:

    transformers:
        x:
        s_nom:
        type:

.. seealso::
    Documentation of the configuration file ``config/config.yaml`` at
    :ref:`snapshots_cf`, :ref:`toplevel_cf`, :ref:`electricity_cf`, :ref:`load_cf`,
    :ref:`lines_cf`, :ref:`links_cf`, :ref:`transformers_cf`

Inputs
------

- ``data/bundle/supply_regions/{regions}.shp``:  Shape file for different supply regions.
- ``data/bundle/South_Africa_100m_Population/ZAF15adjv4.tif``: Raster file of South African population from https://hub.worldpop.org/doi/10.5258/SOTON/WP00246
- ``data/num_lines.xlsx``: confer :ref:`lines`


Outputs
-------

- ``networks/base_{model_file}_{regions}.nc``

    .. image:: ../img/base.png
        :scale: 33 %
"""

import geopandas as gpd
import logging
import numpy as np
import pandas as pd
import pypsa
import re

def create_network():
    n = pypsa.Network()
    n.name = "PyPSA-ZA"
    return n

def load_buses_and_lines(n, line_config):
    buses = gpd.read_file(snakemake.input.buses)
    buses.set_index("name", drop=True,inplace=True)
    buses = buses[["POP2016", "GVA2016"]]
    buses["v_nom"] = line_config["v_nom"]
    if snakemake.wildcards.regions != "1-supply":
        lines = gpd.read_file(snakemake.input.lines, index_col=[1])
        lines = lines[["bus0","bus1","length", line_config["s_rating"] + "_limit"]]
    else:
        lines = []
    return buses, lines

def set_snapshots(n, years):
    def create_snapshots(year):
        snapshots = pd.date_range(start = f"{year}-01-01 00:00", end = f"{year}-12-31 23:00", freq="H")
        return snapshots[~((snapshots.month == 2) & (snapshots.day == 29))]  # exclude Feb 29 for leap years

    if n.multi_invest:
        snapshots = pd.DatetimeIndex([])
        for y in years:
            snapshots = snapshots.append(create_snapshots(y))
        n.set_snapshots(pd.MultiIndex.from_arrays([snapshots.year, snapshots]))
    else:
        n.set_snapshots(create_snapshots(years))


def set_investment_periods(n, years):
    n.investment_periods = years
    n.investment_period_weightings["years"] = list(np.diff(years)) + [5]
    T = 0
    for period, nyears in n.investment_period_weightings.years.items():
        discounts = [(1 / (1 + snakemake.config["costs"]["discount_rate"]) ** t) for t in range(T, T + nyears)]
        n.investment_period_weightings.at[period, "objective"] = sum(discounts)
        T += nyears

def add_components_to_network(n, buses, lines, line_config):
    n.import_components_from_dataframe(buses, "Bus")
    if snakemake.wildcards.regions != "1-supply":
        lines["type"] = line_config["type"][line_config["v_nom"]]
        lines = lines.rename(columns={line_config["s_rating"] + "_limit": "s_nom_min"})
        lines = lines.assign(
            s_nom = lines["s_nom_min"],
            s_nom_extendable=True, 
            type=line_config["type"][line_config["v_nom"]])
        n.import_components_from_dataframe(lines, "Line")

def get_years():
    years = (
        pd.read_excel(
            snakemake.input.model_file,
            sheet_name="model_setup",
            index_col=0
        )
        .loc[snakemake.wildcards.model_file,"simulation_years"]
    )

    if len(str(years)) > 4: #if not isinstance(years, int):
        years = list(map(int, re.split(",\s*", years))) 
        n.multi_invest = 1
    else:
        
        n.multi_invest = 0 

    return years

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake(
            "base_network", 
            **{
                "model_file":"Existing-2023",
                "regions":"11-supply",
            }
        )
    line_config = snakemake.config["lines"]
    
    # Create network and load buses and lines data
    n = create_network()
    buses, lines = load_buses_and_lines(n, line_config)
  
    # Set snapshots and investment periods
    years = get_years()    
    set_snapshots(n,years)
    if n.multi_invest:
        set_investment_periods(n,years)
    add_components_to_network(n, buses, lines, line_config)
    
    n.export_to_netcdf(snakemake.output[0])
