# -*- coding: utf-8 -*-
"""Build industrial distribution keys from hotmaps database."""

import os
import uuid
from distutils.version import StrictVersion
from itertools import product

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

gpd_version = StrictVersion(gpd.__version__)

def build_nodal_distribution_key(industrial_database, regions):
        
    keys = pd.DataFrame(index=regions.name, columns=industries, dtype=float)
    keys["population"] = regions["POP2016"].values / regions["POP2016"].sum()
    keys["gva"] = regions["GVA2016"].values / regions["GVA2016"].sum()
    
    for industry in industrial_database.industry.unique():
        facilities = (
            industrial_database
            .query("country == 'ZA' and industry == @industry")
            .set_index("name"))
        
        indicator = facilities["capacity"]
        
        if not facilities.empty:
            if indicator.sum() == 0:
                key = pd.Series(1 / len(facilities), facilities.index) 
            else:
                # TODO BEWARE: this is a strong assumption, because facilities might be incomplete
                # indicator = indicator.fillna(0)
                key = indicator / indicator.sum()
            key = (
                key.groupby(facilities.index).sum().reindex(regions.name, fill_value=0.0)
            )
        else:
            key = keys["gva"]
            
        keys.loc[regions.name, industry] = key
        
    return keys


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


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake, sets_path_to_root

        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        snakemake = mock_snakemake(
            "build_industrial_distribution_key",
            regions="11-supply",
        )
        sets_path_to_root("pypsa-rsa-sec")

    options = snakemake.params["sector_options"]
    
    regions = gpd.read_file(
        snakemake.input.supply_regions, 
        layer=snakemake.wildcards.regions
        ).set_crs(crs=snakemake.config["crs"]["geo_crs"])

    geo_locs = pd.read_csv(
        snakemake.input.industrial_database,
        sep=",",
        header=0,
        keep_default_na=False,  # , index_col=0
    )
    geo_locs["capacity"] = pd.to_numeric(geo_locs.capacity)
    
    geo_locs["industry"] = match_technology(geo_locs)

    geo_locs.capacity = pd.to_numeric(geo_locs.capacity)

    geo_locs = geo_locs[geo_locs.quality != "nonexistent"]

    industries = geo_locs.industry.unique()

    geo_points = gpd.GeoDataFrame(geo_locs, 
                              geometry=gpd.points_from_xy(geo_locs.x, geo_locs.y)
                              ).set_crs(snakemake.config["crs"]["geo_crs"])
    
    industrial_database = gpd.sjoin(geo_points, regions, op = 'within')
    
       
    keys = build_nodal_distribution_key(industrial_database, regions)

    if "ammonia" not in keys.columns:
        keys["ammonia"] = keys["chemical and petrochemical"]
    if "refinery" not in keys.columns:
        keys["refinery"] = keys["chemical and petrochemical"]

    keys.to_csv(snakemake.output.industrial_distribution_key)


    industrial_database["marker_size"] = \
        1.5 + industrial_database["capacity"].div(1500)
    
    hvc_bool = industrial_database["technology"] == "HVC"
    industrial_database.loc[hvc_bool,"marker_size"] = \
        industrial_database.loc[hvc_bool,"capacity"].div(1e5)
        
    industrial_database.to_csv('data/bundle/geospatial/industrial_database_with_markersize.csv')