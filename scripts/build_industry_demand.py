#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

adds process emissions to industry totals

industry totals --> industrial_energy_demand_per_node

"""

import logging
import os
from itertools import product

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from _helpers import read_csv_nafix


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake, sets_path_to_root

        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        snakemake = mock_snakemake(
            "build_industry_demand",
            regions="11-supply",
            model_file="NZ-2050"
        )

        sets_path_to_root("pypsa-rsa-sec")


    industry_totals = pd.read_csv(snakemake.input.industry_totals, index_col=[0, 1])
    
    dist_keys = pd.read_csv(
        snakemake.input.industrial_distribution_key, 
        index_col=0, keep_default_na=False, na_values=[""]
    )

    # # material demand per node and industry (kton/a)
    # nodal_production_tom = country_to_nodal(production_tom, dist_keys)

    clean_industry_list = [
        "iron and steel",
        "chemical and petrochemical",
        "non-ferrous metals",
        "non-metallic minerals",
        "transport equipment",
        "machinery",
        "mining and quarrying",
        "food and tobacco",
        "paper pulp and print",
        "wood and wood products",
        "textile and leather",
        "construction",
        "other",
    ]

    emission_factors = {  # Based on JR data following PyPSA-EUR
        "iron and steel": 0.025,
        "chemical and petrochemical": 0.51,  # taken from HVC including process and feedstock
        "non-ferrous metals": 1.5,  # taken from Aluminum primary
        "non-metallic minerals": 0.542,  # taken for cement
        "transport equipment": 0,
        "machinery": 0,
        "mining and quarrying": 0,  # assumed
        "food and tobacco": 0,
        "paper pulp and print": 0,
        "wood and wood products": 0,
        "textile and leather": 0,
        "construction": 0,  # assumed
        "other": 0,
    }

    geo_locs = pd.read_csv(
        snakemake.input.industrial_database,
        sep=",",
        header=0,
        keep_default_na=False,
        index_col=0,
    )
    geo_locs["capacity"] = pd.to_numeric(geo_locs.capacity)

    def match_technology(df):
        industry_mapping = {
            "Integrated steelworks": "iron and steel",
            "DRI + Electric arc": "iron and steel",
            "Electric arc": "iron and steel",
            "Cement": "non-metallic minerals",
            "HVC": "chemical and petrochemical",
            "Paper": "paper pulp and print",
        }

        return df["technology"].map(industry_mapping)

    
    industry_util_factor = snakemake.config["sector"]["industry_util_factor"]

    geo_locs["industry"] = match_technology(geo_locs)

    alu_production = read_csv_nafix(snakemake.input["alu_production"], index_col=0)
    alu_production = (
        alu_production["production[ktons/a]"]
        .loc[(alu_production!=0).any(axis=1) & alu_production.index.duplicated(keep="last")]
        .loc[["ZA"]])
    alu_emissions = alu_production * emission_factors["non-ferrous metals"]

    Steel_emissions = (
        geo_locs[geo_locs.industry == "iron and steel"]
        .groupby("country")
        .sum()
        .capacity
        * 1000
        * emission_factors["iron and steel"]
        * industry_util_factor
    )
    NMM_emissions = (
        geo_locs[geo_locs.industry == "non-metallic minerals"]
        .groupby("country")
        .sum()
        .capacity
        * 1000
        * emission_factors["non-metallic minerals"]
        * industry_util_factor
    )
    refinery_emissons = (
        geo_locs[geo_locs.industry == "chemical and petrochemical"]
        .groupby("country")
        .sum()
        .capacity
        * emission_factors["chemical and petrochemical"]
        * 0.136
        * 365
        * industry_util_factor
    )

    for year in industry_totals.index.levels[0]:
        industry_totals.loc[(year, "process emissions"), :] = 0
        try:
            industry_totals.loc[
                (year, "process emissions"), "non-metallic minerals"
            ] = NMM_emissions.loc["ZA"]
        except KeyError:
            pass

        try:
            industry_totals.loc[
                (year, "process emissions"), "iron and steel"
            ] = Steel_emissions.loc["ZA"]
        except KeyError:
            pass  # # Code to handle the KeyError
        try:
            industry_totals.loc[
                (year, "process emissions"), "non-ferrous metals"
            ] = alu_emissions.loc["ZA"]
        except KeyError:
            pass  # Code to handle the KeyError
        try:
            industry_totals.loc[
                (year, "process emissions"), "chemical and petrochemical"
            ] = refinery_emissons.loc["ZA"]
        except KeyError:
            pass  # Code to handle the KeyError
    industry_totals = industry_totals.sort_index()

    all_carriers = [
        "electricity",
        "gas",
        "coal",
        "oil",
        "hydrogen",
        "biomass", 
        "low-temperature heat",
    ]

    for year in industry_totals.index.levels[0]:
        carriers_present = industry_totals.xs(year, level="year").index
        missing_carriers = set(all_carriers) - set(carriers_present)
        for carrier in missing_carriers:
            # Add the missing carrier with a value of 0
            industry_totals.loc[(year, carrier), :] = 0

    # fill_missing_dist_keys(dist_keys, industry_totals)
    
    nodal_dist_keys = pd.DataFrame(
        index=dist_keys.index, columns=industry_totals.columns, dtype=float
    )

    sectors = industry_totals.columns

    for sector in sectors:
        if sector not in dist_keys.columns or dist_keys[sector].sum() == 0:
            mapping = "gva"
        else:
            mapping = sector

        nodal_dist_keys[sector] = dist_keys[mapping]
        print(sector)
    
    nodal_df = pd.DataFrame()

    for year in industry_totals.index.levels[0]:
        industry_totals_yr = industry_totals.loc[year]
        # final energy consumption per node and industry (TWh/a)
        nodal_df_yr = pd.concat(
            [nodal_dist_keys.dot(industry_totals_yr.T)], keys=[year],names=["year"])
        nodal_df = pd.concat([nodal_df, nodal_df_yr])

    rename_sectors = {
        "elec": "electricity",
        "biomass": "solid biomass",
        "heat": "low-temperature heat",
    }
    nodal_df.rename(columns=rename_sectors, inplace=True)

    nodal_df.index.names = ["year","region"]
    nodal_df.columns.name = "carrier"
    
    nodal_df.to_csv(
        snakemake.output.industrial_energy_demand_per_node, float_format="%.4f"
    )
