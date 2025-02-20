configfile: "config/config.yaml"

from os.path import normpath, exists, isdir

ATLITE_NPROCESSES = config["atlite"].get("nprocesses", 4)

wildcard_constraints:
    resarea="[a-zA-Z0-9]+",
    model_file="[-a-zA-Z0-9]+",
    regions="[-+a-zA-Z0-9]+",
    opts="[-+a-zA-Z0-9]+"

rule solve_scenario_matrix:
    input:
        expand(
            "networks/solved_{model_file}_{regions}_{resarea}_l{ll}_{opts}.nc",
            **config["scenario_matrix"]
        ),

rule solve_sector_scenario_matrix:
    input:
        expand(
            "networks/solved_sector_{model_file}_{regions}_{resarea}_l{ll}_{opts}.nc",
            **config["scenario_matrix"]
        ),

if config["enable"]["build_natura_raster"]: 
    rule build_natura_raster:
        input:
            protected_areas = "data/bundle_extended/SAPAD_OR_2017_Q2", #https://egis.environment.gov.za/
            conservation_areas = "data/bundle_extended/SACAD_OR_2017_Q2", #https://egis.environment.gov.za/
            cutouts=expand("cutouts/{cutouts}.nc", **config["atlite"]),
        output:
            "resources/natura.tiff",
        resources:
            mem_mb=5000,
        log:
            "logs/build_natura_raster.log",
        script:
            "scripts/build_natura_raster.py"

if config['enable']['build_cutout']:
    rule build_cutout:
        input:
            regions_onshore='data/bundle/rsa_supply_regions.gpkg',
        output:
            "cutouts/{cutout}.nc",
        log:
            "logs/build_cutout/{cutout}.log",
        benchmark:
            "benchmarks/build_cutout_{cutout}"
        threads: ATLITE_NPROCESSES
        resources:
            mem_mb=ATLITE_NPROCESSES * 1000,
        script:
            "scripts/build_cutout.py"

if not config['hydro_inflow']['disable']:
    rule build_inflow_per_country:
        input: EIA_hydro_gen="data/bundle/EIA_hydro_generation_2011_2014.csv"
        output: "resources/hydro_inflow.csv"
        benchmark: "benchmarks/inflow_per_country"
        threads: 1
        resources: mem_mb=1000
        script: "scripts/build_inflow_per_country.py"


if config['enable']['build_topology']: 
    rule build_topology:
        input:
            supply_regions='data/bundle/rsa_supply_regions.gpkg',
            existing_lines='data/bundle/Eskom/Existing_Lines.shp',
            planned_lines='data/bundle/Eskom/Planned_Lines.shp',        
        output:
            buses='resources/buses_{regions}.geojson',
            lines='resources/lines_{regions}.geojson',
            #parallel_lines='resources/parallel_lines_{regions}.csv',
        threads: 1
        script: "scripts/build_topology.py"

rule base_network:
    input:
        model_file="config/model_file.xlsx",
        buses='resources/buses_{regions}.geojson',
        lines='resources/lines_{regions}.geojson',
    output: "networks/base_{model_file}_{regions}.nc",
    benchmark: "benchmarks/base_{model_file}_{regions}"
    threads: 1
    resources: mem_mb=1000
    script: "scripts/base_network.py"

if config['enable']['build_renewable_profiles'] & ~config['enable']['use_eskom_wind_solar']: 
    rule build_renewable_profiles:
        input:
            regions = 'resources/buses_{regions}.geojson',#'resources/onshore_shapes_{regions}.geojson',
            resarea = lambda w: "data/bundle/" + config['data']['resarea'][w.resarea],
            natura=lambda w: (
                "data/bundle_extended/landuse_without_protected_conservation.tiff"
                if config["renewable"][w.technology]["natura"]
                else []
            ),
            cutout=lambda w: "cutouts/"+ config["renewable"][w.technology]["cutout"] + ".nc",
            gwa_map="data/bundle/ZAF_wind-speed_100m.tif",
            salandcover = 'data/bundle_extended/SALandCover_OriginalUTM35North_2013_GTI_72Classes/sa_lcov_2013-14_gti_utm35n_vs22b.tif'
        output:
            profile="resources/profile_{technology}_{regions}_{resarea}.nc",
            
        log:
            "logs/build_renewable_profile_{technology}_{regions}_{resarea}.log",
        benchmark:
            "benchmarks/build_renewable_profiles_{technology}_{regions}_{resarea}"
        threads: ATLITE_NPROCESSES
        resources:
            mem_mb=ATLITE_NPROCESSES * 5000,
        wildcard_constraints:
            technology="(?!hydro).*",  # Any technology other than hydro
        script:
            "scripts/build_renewable_profiles.py"

if ~config['enable']['use_eskom_wind_solar']:
    renewable_carriers = config["renewable"] 
else:
    renewable_carriers=[]

rule add_electricity:
    input:
        # **{
        #     f"profile_{tech}": f"resources/profile_{tech}_"+ "{regions}_{resarea}.nc"
        #     for tech in renewable_carriers
        # },
        base_network='networks/base_{model_file}_{regions}.nc',
        supply_regions='resources/buses_{regions}.geojson',
        load='data/bundle/SystemEnergy2009_22.csv',
        #onwind_area='resources/area_wind_{regions}_{resarea}.csv',
        #solar_area='resources/area_solar_{regions}_{resarea}.csv',
        eskom_profiles="data/bundle/eskom_pu_profiles.csv",
        model_file="config/model_file.xlsx",
        #fixed_generators_eaf="data/Eskom EAF data.xlsx",
    output: "networks/elec_{model_file}_{regions}_{resarea}.nc",
    benchmark: "benchmarks/add_electricity/elec_{model_file}_{regions}_{resarea}"
    script: "scripts/add_electricity.py"

rule prepare_and_solve_network:
    input:
        network="networks/elec_{model_file}_{regions}_{resarea}.nc",
        model_file="config/model_file.xlsx",
    output:"networks/solved_{model_file}_{regions}_{resarea}_l{ll}_{opts}.nc",
    log:"logs/prepare_and_solve_network/solved_{model_file}_{regions}_{resarea}_l{ll}_{opts}.log",
    benchmark:"benchmarks/prepare_and_solve_network/solved_{model_file}_{regions}_{resarea}_l{ll}_{opts}",
    script:
        "scripts/prepare_and_solve_network.py"

rule solve_network_dispatch:
    input:
        network="networks/solved_{model_file}_{regions}_{resarea}_l{ll}_{opts}.nc",
        model_file="config/model_file.xlsx",
    output:"networks/dispatch_{model_file}_{regions}_{resarea}_l{ll}_{opts}_{years}.nc",
    log:"logs/solve_network_dispatch:/dispatch_{model_file}_{regions}_{resarea}_l{ll}_{opts}_{years}.log",
    benchmark:"benchmarks/solve_network_dispatch:/dispatch_{model_file}_{regions}_{resarea}_l{ll}_{opts}_{years}",
    script:
        "scripts/solve_network_dispatch.py"


# rule solve_network:
#     input: 
#         network="networks/pre_{model_file}_{regions}_{resarea}_l{ll}_{opts}.nc",
#         model_file="config/model_file.xlsx",
#     output: "results/networks/solved_{model_file}_{regions}_{resarea}_l{ll}_{opts}.nc"
#     shadow: "shallow"
#     log:
#         solver=normpath(
#             "logs/solve_network/solved_{model_file}_{regions}_{resarea}_l{ll}_{opts}_solver.log"
#         ),
#         python="logs/solve_network/solved_{model_file}_{regions}_{resarea}_l{ll}_{opts}_python.log",
#         memory="logs/solve_network/solved_{model_file}_{regions}_{resarea}_l{ll}_{opts}_memory.log",
#     benchmark: "benchmarks/solve_network/solved_{model_file}_{regions}_{resarea}_l{ll}_{opts}"
#     script: "scripts/solve_network.py"


rule plot_network:
    input:
        network='results/networks/solved_{model_file}_{regions}_{resarea}_l{ll}_{opts}.nc',
        model_file="config/model_file.xlsx",
        supply_regions='data/bundle/rsa_supply_regions.gpkg',
        resarea = lambda w: "data/bundle/" + config['data']['resarea'][w.resarea]
    output:
        only_map='results/plots/{model_file}_{regions}_{resarea}_l{ll}_{opts}_{attr}.{ext}',
        ext='results/plots/{model_file}_{regions}_{resarea}_l{ll}_{opts}_{attr}_ext.{ext}',
    log: 'logs/plot_network/{model_file}_{regions}_{resarea}_l{ll}_{opts}_{attr}.{ext}.log'
    script: "scripts/plot_network.py"

rule build_population_layouts:
    input:
        supply_regions='data/bundle/rsa_supply_regions.gpkg',
        urban_percent="data/bundle/urban_percent.csv",
        cutout="cutouts/RSA-2020_22-era5.nc",
    output:
        pop_layout_total="resources/population_shares/pop_layout_total_{regions}.nc",
        pop_layout_urban="resources/population_shares/pop_layout_urban_{regions}.nc",
        pop_layout_rural="resources/population_shares/pop_layout_rural_{regions}.nc",
        gdp_layout="resources/gdp_shares/gdp_layout_{regions}.nc",
    resources:
        mem_mb=16000,
    threads: 8
    script:
        "scripts/build_population_layouts.py"

rule build_clustered_population_layouts:
    input:
        pop_layout_total="resources/population_shares/pop_layout_total_{regions}.nc",
        pop_layout_urban="resources/population_shares/pop_layout_urban_{regions}.nc",
        pop_layout_rural="resources/population_shares/pop_layout_rural_{regions}.nc",
        gdp_layout="resources/gdp_shares/gdp_layout_{regions}.nc",
        supply_regions='data/bundle/rsa_supply_regions.gpkg',
        cutout="cutouts/RSA-2020_22-era5.nc",
    output:
        clustered_pop_layout="resources/population_shares/pop_layout_base_{regions}.csv",
        clustered_gdp_layout="resources/gdp_shares/gdp_layout_base_{regions}.csv",
    resources:
        mem_mb=10000,
    script:
        "scripts/build_clustered_population_layouts.py"

rule build_daily_heat_demand:
    input:
        pop_layout_total="resources/population_shares/pop_layout_total_{regions}.nc",
        pop_layout_urban="resources/population_shares/pop_layout_urban_{regions}.nc",
        pop_layout_rural="resources/population_shares/pop_layout_rural_{regions}.nc",
        supply_regions='data/bundle/rsa_supply_regions.gpkg',
        cutout="cutouts/RSA-2020_22-era5.nc",
    output:
        heat_demand_urban="resources/demand/heat/heat_demand_urban_{regions}.nc",
        heat_demand_rural="resources/demand/heat/heat_demand_rural_{regions}.nc",
        heat_demand_total="resources/demand/heat/heat_demand_total_{regions}.nc",
    benchmark:
        "benchmarks/build_daily_heat_demand/{regions}"
    script:
        "scripts/build_daily_heat_demand.py"

rule build_solar_thermal_profiles:
    input:
        pop_layout_total="resources/population_shares/pop_layout_total_{regions}.nc",
        pop_layout_urban="resources/population_shares/pop_layout_urban_{regions}.nc",
        pop_layout_rural="resources/population_shares/pop_layout_rural_{regions}.nc",
        supply_regions='data/bundle/rsa_supply_regions.gpkg',
        cutout="cutouts/RSA-2020_22-era5.nc",
    output:
        solar_thermal_total="resources/demand/heat/solar_thermal_total_{regions}.nc",
        solar_thermal_urban="resources/demand/heat/solar_thermal_urban_{regions}.nc",
        solar_thermal_rural="resources/demand/heat/solar_thermal_rural_{regions}.nc",
    benchmark:
        "benchmarks/build_solar_thermal_profiles/{regions}"
    script:
        "scripts/build_solar_thermal_profiles.py"

rule build_temperature_profiles:
    input:
        pop_layout_total="resources/population_shares/pop_layout_total_{regions}.nc",
        pop_layout_urban="resources/population_shares/pop_layout_urban_{regions}.nc",
        pop_layout_rural="resources/population_shares/pop_layout_rural_{regions}.nc",
        supply_regions='data/bundle/rsa_supply_regions.gpkg',
        cutout="cutouts/RSA-2020_22-era5.nc",
    output:
        temp_soil_total="resources/temperatures/temp_soil_total_{regions}.nc",
        temp_soil_rural="resources/temperatures/temp_soil_rural_{regions}.nc",
        temp_soil_urban="resources/temperatures/temp_soil_urban_{regions}.nc",
        temp_air_total="resources/temperatures/temp_air_total_{regions}.nc",
        temp_air_rural="resources/temperatures/temp_air_rural_{regions}.nc",
        temp_air_urban="resources/temperatures/temp_air_urban_{regions}.nc",
    benchmark:
        "benchmarks/build_temperature_profiles/{regions}"
    script:
        "scripts/build_temperature_profiles.py"

rule build_cop_profiles:
    input:
        temp_soil_total="resources/temperatures/temp_soil_total_{regions}.nc",
        temp_soil_rural="resources/temperatures/temp_soil_rural_{regions}.nc",
        temp_soil_urban="resources/temperatures/temp_soil_urban_{regions}.nc",
        temp_air_total="resources/temperatures/temp_air_total_{regions}.nc",
        temp_air_rural="resources/temperatures/temp_air_rural_{regions}.nc",
        temp_air_urban="resources/temperatures/temp_air_urban_{regions}.nc",
    output:
        cop_soil_total="resources/cops/cop_soil_total_{regions}.nc",
        cop_soil_rural="resources/cops/cop_soil_rural_{regions}.nc",
        cop_soil_urban="resources/cops/cop_soil_urban_{regions}.nc",
        cop_air_total="resources/cops/cop_air_total_{regions}.nc",
        cop_air_rural="resources/cops/cop_air_rural_{regions}.nc",
        cop_air_urban="resources/cops/cop_air_urban_{regions}.nc",
    benchmark:
        "benchmarks/build_cop_profiles/{regions}"
    script:
        "scripts/build_cop_profiles.py"

rule build_energy_totals_from_UCT:
    input: 
        model_file="config/model_file.xlsx",
        energy_demands_jetip = "data/bundle/demands/energy_demands_JETIP_from_UCT.csv", #https://www.climatecommission.org.za/south-africas-jet-ip
        energy_totals_template = "data/bundle/demands/energy_totals_template.csv",
    output: 
        energy_totals="resources/demand/energy_totals_{model_file}.csv", 
        industry_totals="resources/demand/industry_totals_{model_file}.csv",
        refinery_totals="resources/demand/refinery_totals_{model_file}.csv",
    log:"logs/build_energy_totals_from_UCT/energy_totals_{model_file}.log",
    benchmark:"benchmarks/build_energy_totals_from_UCT/energy_totals_{model_file}",
    script:
        "scripts/build_energy_totals_from_UCT.py"

rule build_heat_data:
    input:
        model_file="config/model_file.xlsx",
        network='networks/base_{model_file}_{regions}.nc',
        energy_totals="resources/demand/energy_totals_{model_file}.csv", 
        clustered_pop_layout="resources/population_shares/pop_layout_base_{regions}.csv",
        temp_air_total="resources/temperatures/temp_air_total_{regions}.nc",
        cop_soil_total="resources/cops/cop_soil_total_{regions}.nc",
        cop_air_total="resources/cops/cop_air_total_{regions}.nc",
        solar_thermal_total="resources/demand/heat/solar_thermal_total_{regions}.nc",
        heat_demand_total="resources/demand/heat/heat_demand_total_{regions}.nc",
        heat_profile="data/bundle/heat_load_profile_BDEW.csv",
    output:
        nodal_energy_totals="resources/demand/heat/nodal_energy_heat_totals_{model_file}_{regions}.csv",
        heat_demand="resources/demand/heat/heat_demand_{model_file}_{regions}.csv",
        ashp_cop="resources/demand/heat/ashp_cop_{model_file}_{regions}.csv",
        gshp_cop="resources/demand/heat/gshp_cop_{model_file}_{regions}.csv",
        solar_thermal="resources/demand/heat/solar_thermal_{model_file}_{regions}.csv",
        district_heat_share="resources/demand/heat/district_heat_share_{model_file}_{regions}.csv",
    log:"logs/build_heat_data/{model_file}_{regions}.log",
    benchmark:"benchmarks/build_heat_data/{model_file}_{regions}",
    script:
        "scripts/build_heat_data.py"


rule build_transport_data:
    input:
        model_file="config/model_file.xlsx",
        network='networks/base_{model_file}_{regions}.nc',
        energy_totals="resources/demand/energy_totals_{model_file}.csv", 
        traffic_data_KFZ="data/bundle/traffic_data/KFZ__count",
        traffic_data_Pkw="data/bundle/traffic_data/Pkw__count",
        transport_name="data/bundle/transport_data.csv",
        clustered_pop_layout="resources/population_shares/pop_layout_base_{regions}.csv",
        temp_air_total="resources/temperatures/temp_air_total_{regions}.nc",
    output:
        transport="resources/demand/transport_{model_file}_{regions}.csv",
        avail_profile="resources/pattern_profiles/avail_profile_{model_file}_{regions}.csv",
        dsm_profile="resources/pattern_profiles/dsm_profile_{model_file}_{regions}.csv",
        nodal_transport_data="resources/demand/nodal_transport_data_{model_file}_{regions}.csv",
    log:"logs/build_transport_data/{model_file}_{regions}.log",
    benchmark:"benchmarks/build_transport_data/{model_file}_{regions}",
    script:
        "scripts/build_transport_data.py"

rule build_industrial_distribution_key:
    params:
        sector_options=config["sector"]
    input:
        supply_regions='data/bundle/rsa_supply_regions.gpkg',
        industrial_database='data/bundle/geospatial/industrial_database.csv',
    output:
        industrial_distribution_key='resources/demand/industrial_distribution_key_{regions}.csv',
    log:"logs/build_industrial_distribution_key/industrial_distribution_key_{regions}.log",
    benchmark:"benchmarks/build_industrial_distribution_key/industrial_distribution_key_{regions}",
    script:
        "scripts/build_industrial_distribution_keys.py"

rule build_industry_demand:
    params:
        sector_options=config["sector"]
    input:
        industrial_database='data/bundle/geospatial/industrial_database.csv',
        industrial_distribution_key='resources/demand/industrial_distribution_key_{regions}.csv',
        industry_totals='resources/demand/industry_totals_{model_file}.csv',
        alu_production="data/bundle/AL_production.csv",
    output:
        industrial_energy_demand_per_node='resources/demand/industrial_energy_demand_per_node{model_file}_{regions}.csv',
    log:"logs/build_industry_demand/industry_demand_{model_file}_{regions}.log",
    benchmark:"benchmarks/build_industry_demand/industry_demand_{model_file}_{regions}",
    script:
        "scripts/build_industry_demand.py"
        
rule prepare_and_solve_sector_network:
    params:
        sector_options=config["sector"],
        cost_options=config["costs"],
        costs_filepath="data/bundle/costs/"
    input:
        network="networks/elec_{model_file}_{regions}_{resarea}.nc",
        model_file="config/model_file.xlsx",
        supply_regions='data/bundle/rsa_supply_regions.gpkg',
        energy_totals="resources/demand/energy_totals_{model_file}.csv",
        nodal_energy_totals="resources/demand/heat/nodal_energy_heat_totals_{model_file}_{regions}.csv",
        transport="resources/demand/transport_{model_file}_{regions}.csv",
        avail_profile="resources/pattern_profiles/avail_profile_{model_file}_{regions}.csv",
        dsm_profile="resources/pattern_profiles/dsm_profile_{model_file}_{regions}.csv",
        nodal_transport_data="resources/demand/nodal_transport_data_{model_file}_{regions}.csv",
        clustered_pop_layout="resources/population_shares/pop_layout_base_{regions}.csv",
        industrial_demand='resources/demand/industrial_energy_demand_per_node{model_file}_{regions}.csv',
        refinery_totals="resources/demand/refinery_totals_{model_file}.csv",
        airports="data/bundle/geospatial/airports.csv",
        ports="data/bundle/geospatial/ports.csv",
        heat_demand="resources/demand/heat/heat_demand_{model_file}_{regions}.csv",
        ashp_cop="resources/demand/heat/ashp_cop_{model_file}_{regions}.csv",
        gshp_cop="resources/demand/heat/gshp_cop_{model_file}_{regions}.csv",
        solar_thermal="resources/demand/heat/solar_thermal_{model_file}_{regions}.csv",
        district_heat_share="resources/demand/heat/district_heat_share_{model_file}_{regions}.csv",
        biomass_transport_costs="data/bundle/biomass_transport_costs.csv",
        export_ports="data/bundle/geospatial/export_ports.csv"
    output:"networks/solved_sector_{model_file}_{regions}_{resarea}_l{ll}_{opts}.nc",
    log:"logs/prepare_and_solve_network/solved_sector_{model_file}_{regions}_{resarea}_l{ll}_{opts}.log",
    benchmark:"benchmarks/prepare_and_solve_network/solved_sector_{model_file}_{regions}_{resarea}_l{ll}_{opts}.nc",
    script:
        "scripts/prepare_and_solve_sector_network.py"
