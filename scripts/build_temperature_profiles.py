# -*- coding: utf-8 -*-
"""Build temperature profiles."""
import os

import atlite
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake, sets_path_to_root

        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        snakemake = mock_snakemake(
            "build_temperature_profiles",
            regions="11-supply"
        )
        sets_path_to_root("pypsa-rsa-sec")

    time = pd.date_range(freq="h", **snakemake.config["atlite"]["cutout_snapshots"])
    cutout_path = (
        snakemake.input.cutout
    )  # os.path.abspath(snakemake.config["atlite"]["cutout"])

    cutout = atlite.Cutout(cutout_path).sel(time=time)

    regions = (
        gpd.read_file(snakemake.input.supply_regions, layer=snakemake.wildcards.regions)
        .to_crs(snakemake.config["crs"]["geo_crs"])
        .set_index("name")
        .buffer(0)
        .squeeze()
        )

    I = cutout.indicatormatrix(regions)

    for area in ["total", "rural", "urban"]:
        pop_layout = xr.open_dataarray(snakemake.input[f"pop_layout_{area}"])

        stacked_pop = pop_layout.stack(spatial=("y", "x"))
        M = I.T.dot(np.diag(I.dot(stacked_pop)))

        nonzero_sum = M.sum(axis=0, keepdims=True)
        nonzero_sum[nonzero_sum == 0.0] = 1.0
        M_tilde = M / nonzero_sum

        temp_air = cutout.temperature(matrix=M_tilde.T, index=regions.index)

        temp_air.to_netcdf(snakemake.output[f"temp_air_{area}"])

        temp_soil = cutout.soil_temperature(
            matrix=M_tilde.T, index=regions.index
        )

        temp_soil.to_netcdf(snakemake.output[f"temp_soil_{area}"])
