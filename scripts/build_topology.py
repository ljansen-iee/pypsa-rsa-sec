# SPDX-FileCopyrightText:  PyPSA-ZA2, PyPSA-ZA Authors
# SPDX-License-Identifier: MIT
# coding: utf-8

"""
Creates the network topology (buses and lines).


Relevant Settings
-----------------

.. code:: yaml

    snapshots:

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

.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`snapshots_cf`, :ref:`toplevel_cf`, :ref:`electricity_cf`, :ref:`load_cf`,
    :ref:`lines_cf`, :ref:`links_cf`, :ref:`transformers_cf`

Inputs
------

- ``data/bundle/supply_regions/{regions}.shp``:  Shape file for different supply regions.
- ``data/num_lines.xlsx``: confer :ref:`links`


Outputs
-------
- ``resources/buses_{regions}.geojson``
- ``resources/lines_{regions}.geojson``

"""

import networkx as nx
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
import numpy as np
from operator import attrgetter
import os
import pypsa
from _helpers import save_to_geojson

def check_centroid_in_region(regions,centroids):
    idx = regions.index[~centroids.intersects(regions['geometry'])]
    buffered_regions = regions.buffer(-200)
    boundary = buffered_regions.boundary
    for i in idx:
        # Initialize a variable to store the minimum distance
        min_distance = np.inf
        # Iterate over a range of distances along the boundary
        for d in np.arange(0, boundary[i].length, 200):
            # Interpolate a point at the current distance
            p = boundary[i].interpolate(d)
            # Calculate the distance between the centroid and the interpolated point
            distance = centroids[i].distance(p)
            # If the distance is less than the minimum distance, update the minimum distance and the closest point
            if distance < min_distance:
                min_distance = distance
                closest_point = p
        centroids[i] = closest_point
    return centroids

def build_line_topology(lines, regions):
    # Extract starting and ending points of each line
    lines = lines.explode()
    start_points = lines["geometry"].apply(lambda line: line.coords[0])
    end_points = lines["geometry"].apply(lambda line: line.coords[-1])

    # Convert start and end points to Point geometries
    start_points = start_points.apply(Point)
    end_points = end_points.apply(Point)

    # Map starting and ending points to regions
    lines["bus0"] = start_points.apply(
        lambda point: regions[regions.geometry.contains(point)].index.values[0] 
        if len(regions[regions.geometry.contains(point)].index.values) > 0 else None
    )
    lines["bus1"] = end_points.apply(
        lambda point: regions[regions.geometry.contains(point)].index.values[0] 
        if len(regions[regions.geometry.contains(point)].index.values) > 0 else None
    )
    lines['id']=range(len(lines))
    lines = lines[lines['bus0']!=lines['bus1']]
    lines = lines.dropna(subset=['bus0','bus1'])
    lines.reset_index(drop=True,inplace=True)       
    lines['bus0'], lines['bus1'] = np.sort(lines[['bus0', 'bus1']].values, axis=1).T # sort bus0 and bus1 alphabetically

    return lines

def calc_inter_region_lines(lines):
    inter_region_lines = lines.groupby(['bus0','bus1','DESIGN_VOL']).count()['id'].reset_index().rename(columns={'id':'count'})
    inter_region_lines = inter_region_lines[inter_region_lines['bus0']!=inter_region_lines['bus1']]
    inter_region_lines['bus0'], inter_region_lines['bus1'] = np.sort(inter_region_lines[['bus0', 'bus1']].values, axis=1).T
    inter_region_lines = inter_region_lines.pivot_table(index=["bus0", "bus1"], columns="DESIGN_VOL", values="count", aggfunc='sum',fill_value=0).reset_index()
    inter_region_lines.columns = [col if isinstance(col, str) else str(int(col)) for col in inter_region_lines.columns]

    limits = lines[['bus0','bus1','thermal_limit','SIL_limit','St_Clair_limit']].groupby(['bus0','bus1']).sum()
    inter_region_lines = inter_region_lines.merge(limits[['thermal_limit','SIL_limit','St_Clair_limit']],on=['bus0','bus1'],how='left')
       
    return inter_region_lines

def extend_topology(lines, regions, centroids):
    # get a list of lines between all adjacent regions
    adj_lines = gpd.sjoin(regions, regions, op='touches')['index_right'].reset_index()
    adj_lines.columns = ['bus0', 'bus1']
    adj_lines['bus0'], adj_lines['bus1'] = np.sort(adj_lines[['bus0', 'bus1']].values, axis=1).T # sort bus0 and bus1 alphabetically
    adj_lines = adj_lines.drop_duplicates(subset=['bus0', 'bus1'])

    missing_lines = adj_lines.merge(lines, on=['bus0', 'bus1'], how='left', indicator=True)
    missing_lines = missing_lines[missing_lines['_merge'] == 'left_only'][['bus0', 'bus1']]
    missing_lines['DESIGN_VOL'] = 0
    missing_lines['status'] = 'missing'
    missing_lines['geometry'] = missing_lines.apply(lambda row: LineString([centroids[row['bus0']],centroids[row['bus1']]]),axis=1)
    missing_lines = missing_lines.drop_duplicates(subset=['bus0', 'bus1'])
    lines = pd.concat([lines,missing_lines])

    return lines

def calc_line_limits(length, voltage, line_config):
    # digitised from https://www.researchgate.net/figure/The-St-Clair-curve-as-based-on-the-results-of-14-retrieved-from-15-is-used-to_fig3_318692193
    if voltage in [220, 275, 400, 765]:
        thermal = line_config['thermal'][voltage]
        SIL = line_config['SIL'][voltage]
        St_Clair = thermal if length <= 80 else SIL * 53.736 * (length/1000) ** -0.65
    else:
        thermal = np.nan
        SIL = np.nan
        St_Clair = np.nan

    return pd.Series([thermal, SIL, St_Clair])

def build_regions(line_config):
    # Load supply regions and calculate population per region
    regions = gpd.read_file(
        snakemake.input.supply_regions,
        layer=snakemake.wildcards.regions,
    ).to_crs(snakemake.config["crs"]["distance_crs"])  
    index_column = 'name' if 'name' in regions.columns else 'Name' # some layers use Name or name
    regions = regions.set_index(index_column)
    regions.index.name = 'name'

    centroids = regions['geometry'].centroid
    centroids = check_centroid_in_region(regions,centroids)
    centroids = centroids.to_crs(snakemake.config["crs"]["geo_crs"])

    v_nom = line_config['v_nom']
    buses = (
        regions.assign(
            x=centroids.map(attrgetter('x')),
            y=centroids.map(attrgetter('y')),
            v_nom=v_nom
        )
    )
    buses.index.name='name' # ensure consistency for other scripts

    return buses, regions, centroids

def build_topology(regions, centroids, line_config):

    # Read in Eskom GIS data for existing and planned transmission lines
    lines = gpd.read_file(snakemake.input.existing_lines)
    lines['status'] = 'existing'
    lines = lines.to_crs(snakemake.config["crs"]["distance_crs"])

    if 'planned' in snakemake.config["lines"]["status"]:
        planned_lines = gpd.read_file(snakemake.input.planned_lines)
        planned_lines = planned_lines.to_crs(snakemake.config["crs"]["distance_crs"])
        planned_lines['DESIGN_VOL'] = planned_lines['Voltage']
        planned_lines['status'] = 'planned'
        lines = pd.concat([lines[['status','DESIGN_VOL','geometry']],planned_lines[['status','DESIGN_VOL','geometry']]])

    lines = build_line_topology(lines, regions)
  
    # Line length between regions if lines is empty, return empty dataframe
    def haversine_length(row):
        lon1, lat1, lon2, lat2 = map(np.radians, [centroids[row['bus0']].x, centroids[row['bus0']].y, centroids[row['bus1']].x, centroids[row['bus1']].y])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return c * 6371
    
    if line_config['extend_topology']:
        lines = extend_topology(lines, regions, centroids)
    
    lines['actual_length'] = lines['geometry'].length
    line_limits = lines.apply(lambda row: calc_line_limits(row['actual_length'], row['DESIGN_VOL'], line_config), axis=1)
    lines['thermal_limit'], lines['SIL_limit'], lines['St_Clair_limit'] = line_limits[0], line_limits[1], line_limits[2]

    inter_region_lines = calc_inter_region_lines(lines)
    inter_region_lines['geometry'] = inter_region_lines.apply(
        lambda row: LineString([centroids[row['bus0']], centroids[row['bus1']]]),
        axis=1
    )

    inter_region_lines['length'] = inter_region_lines.apply(haversine_length, axis=1) * snakemake.config['lines']['length_factor']
    inter_region_lines = gpd.GeoDataFrame(inter_region_lines, geometry='geometry', crs = snakemake.config["crs"]["geo_crs"])

    return lines, inter_region_lines

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake(
            'build_topology', 
            **{
                'regions':'11-supply',
            }
        )
    line_config = snakemake.config['lines']
    buses, regions, centroids = build_regions(line_config)
    save_to_geojson(buses.to_crs(snakemake.config["crs"]["geo_crs"]),snakemake.output.buses)

    if snakemake.wildcards.regions != '1-supply':
        lines, inter_region_lines= build_topology(regions, centroids, line_config)
        save_to_geojson(inter_region_lines.to_crs(snakemake.config["crs"]["geo_crs"]),snakemake.output.lines)
    else:
        save_to_geojson(buses.to_crs(snakemake.config["crs"]["geo_crs"]),snakemake.output.lines) # Dummy file will not get used if single node model  
        
