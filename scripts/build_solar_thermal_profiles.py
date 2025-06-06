# -*- coding: utf-8 -*-
"""Build solar thermal collector time series."""

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
            "build_solar_thermal_profiles",
            regions="11-supply",
        )
        sets_path_to_root("pypsa-rsa-sec")

    config = snakemake.config["solar_thermal"]

    time = pd.date_range(freq="h", **snakemake.config["atlite"]["cutout_snapshots"])
    cutout_config = snakemake.input.cutout
    cutout = atlite.Cutout(cutout_config).sel(time=time)

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

        solar_thermal = cutout.solar_thermal(
            **config, matrix=M_tilde.T, index=regions.index
        )

        solar_thermal.to_netcdf(snakemake.output[f"solar_thermal_{area}"])
