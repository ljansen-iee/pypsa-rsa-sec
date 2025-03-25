# -*- coding: utf-8 -*-
"""Build clustered population layouts."""
import os

import atlite
import geopandas as gpd
import pandas as pd
import xarray as xr

if __name__ == "__main__":
    if "snakemake" not in globals():
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        from _helpers import mock_snakemake, sets_path_to_root

        snakemake = mock_snakemake(
            'build_clustered_population_layouts', 
            **{
                'regions':'11-supply',
            }
        )
        sets_path_to_root("pypsa-rsa-sec")

    cutout_path = (
        snakemake.input.cutout
    )  # os.path.abspath(snakemake.config["atlite"]["cutout"])
    cutout = atlite.Cutout(cutout_path)
    # cutout = atlite.Cutout(snakemake.config['atlite']['cutout'])

    regions = (
        gpd.read_file(
            snakemake.input.supply_regions, 
            layer=snakemake.wildcards.regions)
        .set_crs(crs=snakemake.config["crs"]["geo_crs"])
        .set_index("name"))

    regions["country"] = "ZA"

    # Set value of population to same dimension as in PyPSA-Eur-Sec, where the value is given in 1e3
    regions["pop"] = regions["POP2016"] / 1000
    regions["gdp"] = regions["GVA2016"] #TODO

    I = cutout.indicatormatrix(regions.buffer(0).squeeze())

    pop = {}
    for item in ["total", "urban", "rural"]:
        pop_layout = xr.open_dataarray(snakemake.input[f"pop_layout_{item}"])
        pop[item] = I.dot(pop_layout.stack(spatial=("y", "x")))

    pop = pd.DataFrame(pop, index=regions.index)

    pop["ct"] = regions.country
    country_population = pop.total.groupby(pop.ct).sum()
    pop["fraction"] = pop.total / pop.ct.map(country_population)
    pop.to_csv(snakemake.output.clustered_pop_layout)

    gdp_layout = xr.open_dataarray(snakemake.input["gdp_layout"])
    gdp = I.dot(gdp_layout.stack(spatial=("y", "x")))
    gdp = pd.DataFrame(gdp, index=regions.index, columns=["total"])

    gdp["ct"] = regions.country
    country_gdp = gdp.total.groupby(gdp.ct).sum()
    gdp["fraction"] = gdp.total / gdp.ct.map(country_gdp)
    gdp.to_csv(snakemake.output.clustered_gdp_layout)
