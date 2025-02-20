# -*- coding: utf-8 -*-
"""Build mapping between grid cells and population (total, urban, rural)"""
import multiprocessing as mp
import os

import atlite
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from _helpers import read_csv_nafix
from vresutils import shapes as vshapes

if __name__ == "__main__":
    if "snakemake" not in globals():
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        from _helpers import mock_snakemake, sets_path_to_root

        snakemake = mock_snakemake(
            'build_population_layouts', 
            **{
                'regions':'11-supply',
            }
        )
        sets_path_to_root("pypsa-rsa-sec")

    cutout_path = (
        snakemake.input.cutout
    )  # os.path.abspath(snakemake.config["atlite"]["cutout"])
    cutout = atlite.Cutout(cutout_path)

    grid_cells = cutout.grid.geometry.to_list()

    # nuts3 has columns country, gdp, pop, geometry
    #nuts3 = gpd.read_file(snakemake.input.nuts3_shapes).set_index("GADM_ID")
        
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

    # Indicator matrix regions -> grid cells
    I = atlite.cutout.compute_indicatormatrix(regions.geometry, grid_cells)

    # Indicator matrix grid_cells -> regions; inprinciple Iinv*I is identity
    # but imprecisions mean not perfect
    Iinv = cutout.indicatormatrix(regions.geometry)

    countries = np.sort(regions.country.unique())

    urban_percent_df = read_csv_nafix(
        snakemake.input.urban_percent,
        usecols=[0, 1, 4],
        index_col=0,
    )

    # Filter for the year used in the workflow
    urban_percent_df = urban_percent_df.loc[
        (
            urban_percent_df["Year"]
            == 2030 #TODO
        )
    ]

    # Filter for urban percent column
    urban_percent_df = urban_percent_df[
        ["Urban population as percentage of total population"]
    ]

    # Remove index header
    urban_percent_df.index.name = None

    # Squeeze into a Series
    urban_fraction = urban_percent_df.squeeze() / 100.0

    # population in each grid cell
    pop_cells = pd.Series(I.dot(regions["pop"]))
    gdp_cells = pd.Series(I.dot(regions["gdp"]))

    # in km^2
    with mp.Pool(processes=snakemake.threads) as pool:
        cell_areas = pd.Series(pool.map(vshapes.area, grid_cells)) / 1e6

    # pop per km^2
    density_cells_pop = pop_cells / cell_areas
    density_cells_gdp = gdp_cells / cell_areas

    # rural or urban population in grid cell
    pop_rural = pd.Series(0.0, density_cells_pop.index)
    pop_urban = pd.Series(0.0, density_cells_pop.index)

    for ct in countries:
        indicator_regions_ct = regions.country.apply(lambda x: 1.0 if x == ct else 0.0)

        indicator_cells_ct = pd.Series(Iinv.T.dot(indicator_regions_ct))

        density_cells_pop_ct = indicator_cells_ct * density_cells_pop
        density_cells_gdp_ct = indicator_cells_ct * density_cells_gdp

        pop_cells_ct = indicator_cells_ct * pop_cells
        gdp_cells_ct = indicator_cells_ct * gdp_cells
        # correct for imprecision of Iinv*I
        pop_ct = regions.loc[regions.country == ct, "pop"].sum()
        pop_cells_ct *= pop_ct / pop_cells_ct.sum()

        gdp_ct = regions.loc[regions.country == ct, "gdp"].sum()
        gdp_cells_ct *= gdp_ct / gdp_cells_ct.sum()

        # The first low density grid cells to reach rural fraction are rural
        asc_density_i = density_cells_pop_ct.sort_values().index
        asc_density_cumsum = pop_cells_ct[asc_density_i].cumsum() / pop_cells_ct.sum()
        rural_fraction_ct = 1 - urban_fraction[ct].mean() # to handle duplicates (ZA); 
        pop_ct_rural_b = asc_density_cumsum < rural_fraction_ct
        pop_ct_urban_b = ~pop_ct_rural_b

        pop_ct_rural_b[indicator_cells_ct == 0.0] = False
        pop_ct_urban_b[indicator_cells_ct == 0.0] = False

        pop_rural += pop_cells_ct.where(pop_ct_rural_b, 0.0)
        pop_urban += pop_cells_ct.where(pop_ct_urban_b, 0.0)

    pop_cells = {"total": pop_cells}
    pop_cells["rural"] = pop_rural
    pop_cells["urban"] = pop_urban

    for key, pop in pop_cells.items():
        ycoords = ("y", cutout.coords["y"].data)
        xcoords = ("x", cutout.coords["x"].data)
        values = pop.values.reshape(cutout.shape)
        pop_layout = xr.DataArray(values, [ycoords, xcoords])

        pop_layout.to_netcdf(snakemake.output[f"pop_layout_{key}"])

    # for key, gdp in gdp_cells.items():
    ycoords = ("y", cutout.coords["y"].data)
    xcoords = ("x", cutout.coords["x"].data)
    values = gdp_cells.values.reshape(cutout.shape)
    gdp_layout = xr.DataArray(values, [ycoords, xcoords])
    gdp_layout.to_netcdf(snakemake.output[f"gdp_layout"])
