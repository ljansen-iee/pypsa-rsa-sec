import logging
logger = logging.getLogger(__name__)
import os
from pathlib import Path

import pandas as pd
idx = pd.IndexSlice
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white+xgridoff+ygridoff"
pio.renderers.default = "plotly_mimetype+notebook_connected"
#pio.kaleido.scope.mathjax= None


def rename_commodity_names(commodity):
    if commodity == "Coal":
        return "coal"
    elif commodity == "Electricity":
        return "electricity"
    elif commodity == "RefinedProducts":
        return "oil"
    elif commodity == "Gas":
        return "gas"
    elif commodity in ["Biomass","Canola grain crop", "Sugar cane crop"]:
        return "biomass"
    elif commodity == "Hydrogen":
        return "hydrogen"
    else:
        return commodity
    
def rename_description_to_subsector(description):
    """ rename where useful for later assignments"""
    
    if "rail" in description or "train" in description:
        return "rail"
    elif "aviation int" in description:
        return "international aviation"
    elif "aviation" in description:
        return "domestic aviation"
    elif any(x in description for x in ["hcv", "lcv", "brt", "bus", "car", "moto priv.", "suv"]):
        return "road"
    elif "space" in description:
        return "space"
    elif "water" in description:
        return "water"
    elif "cooking" in description:
        return "cooking"
    elif "ammonia plant" in description:
        return "ammonia"
    elif "chemical" in description:
        return "chemical and petrochemical"
    elif "..." in description:
        return "construction"
    elif "food" in description:
        return "food and tobacco"
    elif "iron" in description or "ferr" in description:
        return "iron and steel"
    # elif "..." in description:
    #     return "machinery"
    elif "mining" in description or "mine and refine" in description:
        return "mining and quarrying"
    elif "non-ferrous" in description or "aluminium" in description:
        return "non-ferrous metals"
    elif "minerals" in description:
        return "non-metallic minerals"
    elif "paper" in description:
        return "paper pulp and print"
    elif "other" in description:
        return "other"
    elif "electr" in description:
        return "electricity"
    elif "refinery ctl" in description:
        return "refinery ctl"
    elif "refinery crude oil" in description:
        return "refinery crude oil"
    elif "biofuel refinery" in description:
        return "refinery biofuel"
    else:
        return description
    


jetip_scenario_mapping = {
    "moderate": "85T2E2S0P0G0", # delayed transport transition with price parity after 2030, no efficiency
    "ambitious": "78T1E2S0P1G0", # price parity 2027, with efficiency
}

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake, sets_path_to_root
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        snakemake = mock_snakemake(
            'build_energy_totals_from_UCT', 
            **{
                'model_file':'CP-2050',
            }
        )
        sets_path_to_root("pypsa-rsa-sec") 

    logger.info("Preparing energy totals from UCT data")
    
    model_file = pd.ExcelFile(snakemake.input.model_file)
    model_setup = (
        pd.read_excel(
            model_file, 
            sheet_name="model_setup",
            index_col=[0])
            .loc[snakemake.wildcards.model_file]
    )
    
    demand_scenario = jetip_scenario_mapping[model_setup["projected_demands"]]
    
    years = ["2021", "2030", "2040", "2050"]
    
    df = (
        pd.read_csv(snakemake.input["energy_demands_jetip"])
        .fillna(0.0)
        .query("Scenario == @demand_scenario").drop("Scenario", axis=1)
    )
    df.loc[:,years] /= 3.6 # from PJ to TWh
    #df.loc[:,years] *= 1e6 # from TWh to Mwh
    
    df["Commodity"] = df["Commodity"].map(rename_commodity_names)
    df["Sector"] = df["Sector"].str.lower()
    df["TechDescription"] = df["TechDescription"].str.lower()
    df["SubSector"] = df["TechDescription"].map(rename_description_to_subsector)

    energy_totals = pd.read_csv(snakemake.input["energy_totals_template"], index_col=[0,1])
    # energy = pd.read_csv(snakemake.input["energy_demands_jetip"]).fillna(0.0).query("Scenario == @scenario").drop("Scenario", axis=1)
    # energy[["2030", "2040", "2050"]] /= 3.6 
    # energy = energy.query("Scenario == @scenario").drop("Scenario", axis=1)

    df_commodity = df.groupby(["Sector","Commodity"]).sum(numeric_only=True)
    df_subsector = df.groupby(["Sector","Commodity","SubSector"]).sum(numeric_only=True)

    energy_totals.loc["ZA","agriculture electricity"] = \
        df_commodity.loc[("agriculture", "electricity"),:].values
        
    energy_totals.loc["ZA","agriculture oil"] = \
        df_commodity.loc[("agriculture", "oil"),:].values
        
    energy_totals.loc["ZA","agriculture coal"] = \
        df_commodity.loc[("agriculture", "coal"),:].values
        
    energy_totals.loc["ZA","electricity residential"] = \
        df_subsector.loc[idx["residential","electricity",
                             ["electricity","other","cooking"]]].sum().values
        
    energy_totals.loc["ZA","electricity residential space"] = \
        df_subsector.loc[idx["residential","electricity",
                             "space"]].values
        
    energy_totals.loc["ZA","electricity residential water"] = \
        df_subsector.loc[idx["residential","electricity",
                             "water"]].values

    energy_totals.loc["ZA","residential biomass"] = \
        df_commodity.loc[idx["residential","biomass"]].values
        
    energy_totals.loc["ZA","residential gas"] = \
        df_commodity.loc[idx["residential","gas"]].values
        
    energy_totals.loc["ZA","residential coal"] = \
        df_commodity.loc[idx["residential","coal"]].values

    energy_totals.loc["ZA","residential oil"] = \
        df_commodity.loc[idx["residential","oil"]].values
             
    energy_totals.loc["ZA","residential heat biomass"] = \
        df_subsector.loc[idx["residential","biomass",["space", "water"]]].sum().values

    energy_totals.loc["ZA","residential heat gas"] = \
        df_subsector.loc[idx["residential","gas",["space", "water"]]].sum().values
        
    energy_totals.loc["ZA","residential heat oil"] = \
        df_subsector.loc[idx["residential","oil",["space", "water"]]].sum().values

    energy_totals.loc["ZA","residential heat coal"] = \
        df_subsector.loc[idx["residential","coal",["space", "water"]]].sum().values
        
    energy_totals.loc["ZA","total residential space"] = \
        df_subsector.loc[idx["residential",:,"space"]].sum().values

    energy_totals.loc["ZA","total residential water"] = \
        df_subsector.loc[idx["residential",:,"water"]].sum().values



    energy_totals.loc["ZA","services electricity"] = \
        df_subsector.loc[idx["commerce","electricity",
                             ["electricity","other","cooking"]]].sum().values
        
    energy_totals.loc["ZA","electricity services space"] = \
        df_subsector.loc[idx["commerce","electricity",
                             "space"]].values
        
    energy_totals.loc["ZA","electricity services water"] = \
        df_subsector.loc[idx["commerce","electricity",
                             "water"]].values

    energy_totals.loc["ZA","services coal"] = \
        df_commodity.loc[idx["commerce","coal"]].values
        
    energy_totals.loc["ZA","services gas"] = \
        df_commodity.loc[idx["commerce","gas"]].values

    energy_totals.loc["ZA","services oil"] = \
        df_commodity.loc[idx["commerce","oil"]].values
             
    energy_totals.loc["ZA","services heat coal"] = \
        df_subsector.loc[idx["commerce","coal",["space", "water"]]].sum().values

    energy_totals.loc["ZA","services heat gas"] = \
        df_subsector.loc[idx["commerce","gas",["space", "water"]]].sum().values
        
    energy_totals.loc["ZA","services heat oil"] = \
        df_subsector.loc[idx["commerce","oil",["space", "water"]]].sum().values
        
    energy_totals.loc["ZA","total services space"] = \
        df_subsector.loc[idx["commerce",:,"space"]].sum().values

    energy_totals.loc["ZA","total services water"] = \
        df_subsector.loc[idx["commerce",:,"water"]].sum().values

    
    energy_totals.loc["ZA","total domestic aviation"] = \
        df_subsector.loc[idx["transport",:,"domestic aviation"]].sum().values

    energy_totals.loc["ZA","total international aviation"] = \
        df_subsector.loc[idx["transport",:,"international aviation"]].sum().values
    
    energy_totals.loc["ZA","total navigation oil"] = \
        energy_totals.loc[
            "ZA",
            ["total international navigation","total domestic navigation"]].sum(1).values # From UNSD
    
    energy_totals.loc["ZA","electricity rail"] = \
        df_subsector.loc[idx["transport","electricity","rail"]].values
        
    energy_totals.loc["ZA","total rail"] = \
        df_subsector.loc[idx["transport",:,"rail"]].sum().values

    energy_totals.loc["ZA","total road"] = \
        df_subsector.loc[idx["transport",:,"road"]].sum().values

    energy_totals.loc["ZA","total road ev"] = \
        df_subsector.loc[idx["transport","electricity","road"]].values
        
    energy_totals.loc["ZA","total road fcev"] = \
        df_subsector.loc[idx["transport","hydrogen","road"]].values

    energy_totals.loc["ZA","total road ice"] = \
        df_subsector.loc[idx["transport",["gas","oil"],"road"]].sum().values

    # build industry totals
    
    df_industry = df_subsector.loc["industry"].reset_index()
    industry_totals = (
        df_industry
        .melt(id_vars=["Commodity", "SubSector"], 
                var_name="year", 
                value_vars=years)
        .pivot(index=["year","Commodity"],columns="SubSector", values="value")
        .fillna(0.))
    
    
    # Replace 2Mt nitro-fertiliser imports with local green fertilisers in net-zero scenarios
    ## 1.8 Mt nitrogen fertiliser per 1 Mt ammonia (NH3), 5.166 MWh_NH3/tNH3, 
    ## 1.205,MWh_H2/MWh_NH3,"DECHEMA 2017
    ## replacing and producing 3Mt fertilizer ~ 8.6 TWh_NH3 ~ 10.4 TWh_H2 local demand
    ## JETP assumes 3.6 TWh_H2 in 2050 for ammonia production in all scenario
    ## Here, we adapt this assumption for our scenarios moderate and ambitious
    
    if model_setup["projected_demands"] == "moderate":
        ammonia_2021 = industry_totals.loc[idx[['2021'],:],"ammonia"].values
        industry_totals.loc[idx[['2030'],:],"ammonia"] = ammonia_2021
        industry_totals.loc[idx[['2040'],:],"ammonia"] = ammonia_2021
        industry_totals.loc[idx[['2050'],:],"ammonia"] = ammonia_2021
        
    elif model_setup["projected_demands"] == "ambitious":
        # replace imports
        industry_totals.loc[idx[['2030'],"hydrogen"],"ammonia"] += .5
        industry_totals.loc[idx[['2040'],"hydrogen"],"ammonia"] += 3.
        industry_totals.loc[idx[['2050'],"hydrogen"],"ammonia"] += 7.
    
    df_refinery = df_subsector.loc["refineries"].reset_index()
    refinery_totals = (
        df_refinery
        .melt(id_vars=["Commodity", "SubSector"], 
                var_name="year", 
                value_vars=years)
        .pivot(index=["year","Commodity"],columns="SubSector", values="value")
        .fillna(0.))
    
    logger.info("Drop coal from refinery totals. CtL emissions must be considered in CO2 limit.")
    refinery_totals = refinery_totals.drop("coal", axis=0, level=1)
    industry_totals["refinery"] = refinery_totals["refinery crude oil"]
    industry_totals["refinery"] += refinery_totals["refinery ctl"]
    industry_totals.fillna(0., inplace=True)


    energy_totals.to_csv(snakemake.output.energy_totals, index=True)
    industry_totals.to_csv(snakemake.output.industry_totals, index=True)
    refinery_totals.to_csv(snakemake.output.refinery_totals, index=True)
    
    
    print(df_subsector.xs("electricity", level="Commodity").sum())
    print(df_subsector.xs("electricity", level="Commodity").groupby("Sector").sum().round(0))
    
    print(df_subsector.xs("hydrogen", level="Commodity").sum())
    print(df_subsector.xs("hydrogen", level="Commodity").groupby("Sector").sum().round(1))
    
    print(df.query("Commodity == 'electricity' and Sector == 'industry'").groupby("TechDescription").sum().round(0))