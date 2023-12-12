import logging
import pandas as pd
import pypsa

from pypsa.descriptors import get_switchable_as_dense as get_as_dense, expand_series, get_activity_mask
from pypsa.optimization.common import reindex

from _helpers import get_investment_periods
# from add_electricity import load_costs, update_transmission_costs

import xarray as xr
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Comment out for debugging and development

idx = pd.IndexSlice
logger = logging.getLogger(__name__)


"""
********************************************************************************
    Operational limits
********************************************************************************
"""

def calc_max_gen_potential(n, sns, gens, incl_pu, weightings, active, cf_limit, extendable=False):
    suffix = "" if extendable == False else "-ext"
    p_max_pu = get_as_dense(n, 'Generator', "p_max_pu", sns)[gens] if incl_pu else pd.DataFrame(1, index=sns, columns=gens)
    p_max_pu.columns.name = f'Generator{suffix}'
    
    if n.multi_invest:
        cf_limit_h = pd.DataFrame(0, index=sns, columns=gens)
        for y in cf_limit_h.index.get_level_values(0).unique():
            cf_limit_h.loc[y] = cf_limit[y]
    else:
        cf_limit_h = pd.DataFrame(cf_limit[y], index=sns, columns=gens) * weightings[gens]
    
    if not extendable:
        return cf_limit_h[gens] * active[gens] * p_max_pu * weightings[gens] * n.generators.loc[gens, "p_nom"]
    p_nom = n.model.variables["Generator-p_nom"].sel({f"Generator{suffix}": gens})
    potential = xr.DataArray(cf_limit_h[gens] * active[gens] * p_max_pu * weightings[gens])
    potential = potential.rename({"Generator":"Generator-ext"}) if "Generator" in potential.dims else potential
    
    return (potential * p_nom).sum(f'Generator{suffix}')

def group_and_sum(data, groupby_func):
    grouped_data = data.groupby(groupby_func).sum()
    return grouped_data.sum(axis=1) if len(grouped_data) > 1 else grouped_data
    
def apply_operational_constraints(n, sns, **kwargs):
    energy_unit_conversion = {"GJ": 1/3.6, "TJ": 1000/3.6, "PJ": 1e6/3.6, "GWh": 1e3, "TWh": 1e6}
    apply_to = kwargs["apply_to"]
    carrier = kwargs["carrier"]
    if carrier in ["gas", "diesel"]:
        carrier = n.carriers[n.carriers.index.str.contains(carrier)].index
    bus = kwargs["bus"]
    period = kwargs["period"]

    type_ = "energy" if kwargs["type"] in ["primary_energy", "output_energy"] else "capacity_factor"

    if (period  == "week") & (max(n.snapshot_weightings["generators"])>1):
        logger.warning(
            "Applying weekly operational limits and time segmentation should be used with caution as the snapshot weightings might not align with the weekly grouping."
        )
    incl_pu = kwargs["incl_pu"]
    limit = kwargs["limit"]

    sense = "<=" if limit == "max" else ">="

    cf_limit = 0 * kwargs["values"] if type_ == "energy" else kwargs["values"]
    en_limit = 0 * kwargs["values"] if type_ == "capacity_factor" else kwargs["values"]

    if (type_ == "energy") & (kwargs["units"] != "MWh"):
        en_limit *= energy_unit_conversion[kwargs["units"]]

    years = get_investment_periods(sns, n.multi_invest)

    filtered_gens = n.generators.query("carrier in @carrier") if len(carrier)>1 else n.generators.query("carrier == @carrier")
    if bus != "global":
        filtered_gens = filtered_gens.query("bus == @bus")
    fix_i = filtered_gens.query("not p_nom_extendable").index if apply_to in ["fixed", "all"] else []
    ext_i = filtered_gens.query("p_nom_extendable").index if apply_to in ["extendable", "all"] else []
    filtered_gens = filtered_gens.loc[list(fix_i) + list(ext_i)]

    efficiency = get_as_dense(n, "Generator", "efficiency", inds=filtered_gens.index) if kwargs["type"] == "primary_energy" else pd.DataFrame(1, index=n.snapshots, columns = filtered_gens.index)
    weightings = (1/efficiency).multiply(n.snapshot_weightings.generators, axis=0)

    # if only extendable generators only select snapshots where generators are active
    min_year = n.generators.loc[filtered_gens.index, "build_year"].min()
    sns_active = sns[sns.get_level_values(0) >= min_year] if n.multi_invest else sns[sns.year >= min_year]
    act_gen = (n.model.variables['Generator-p'].loc[sns_active, filtered_gens.index] * weightings.loc[sns_active]).sel(Generator=filtered_gens.index).sum('Generator')

    timestep = "timestep" if n.multi_invest else "snapshot"
    groupby_dict = {
        "year": f"{timestep}.year",
        "month": f"{timestep}.month",
        "week": f"{timestep}.week",
        "hour": None
    }

    active = get_activity_mask(n, "Generator", sns).astype(int)
    if type_ != "energy":
        max_gen_fix = calc_max_gen_potential(n, sns, fix_i, incl_pu, weightings, active, cf_limit, extendable=False) if len(fix_i)>0 else 0
        max_gen_ext = calc_max_gen_potential(n, sns, ext_i, incl_pu, weightings, active, cf_limit, extendable=True) if len(ext_i)>0 else 0

    if groupby := groupby_dict[period]:
        for y in years:
            year_sns = sns_active[sns_active.get_level_values(0)==y] if n.multi_invest else sns_active
            if len(year_sns) > 0:
                lhs = (act_gen - max_gen_ext) if type_ == "capacity_factor" else act_gen
                lhs = lhs.sel(snapshot=year_sns)
                if (isinstance(max_gen_fix, int)) | (isinstance(max_gen_fix, float)):
                    rhs = max_gen_fix
                else:
                    rhs = max_gen_fix.loc[y] if n.multi_invest else max_gen_fix.loc[year_sns] if type_ == "capacity_factor" else en_limit[y]

                lhs_p = lhs.sum() if period == "year" else lhs.groupby(groupby).sum()

                if period == "month":
                    if isinstance(rhs, int):
                        rhs_p = rhs
                    else:
                        rhs_p = group_and_sum(rhs, lambda x: x.index.month)
                        rhs_p.index.name = period
                elif period == "week":
                    if isinstance(rhs, int):
                        rhs_p = rhs
                    else:
                        rhs_p = group_and_sum(rhs, lambda x: x.index.isocalendar().week)
                        rhs_p.index.name = period
                else:  # period == "year"
                    rhs_p = rhs if isinstance(rhs, int) else rhs.sum().sum()

                n.model.add_constraints(lhs_p, sense, rhs_p, name=f'{limit}-{kwargs["carrier"]}-{period}-{kwargs["apply_to"][:3]}-{y}')

    else:

        lhs = (act_gen - max_gen_ext).sel(snapshot = sns_active) if type_ == "capacity_factor" else act_gen.sel(snapshot = sns_active)

        if type_ == "capacity_factor":
            if isinstance(max_gen_fix, int):
                rhs = max_gen_fix
            else:
                rhs = xr.DataArray(max_gen_fix.loc[sns_active].sum(axis=1)).rename({"dim_0":"snapshot"})

        else:
            logging.warning("Energy limits are not yet implemented for hourly operational limits.")
        n.model.add_constraints(lhs, sense, rhs, name = f'{limit}-{kwargs["carrier"]}-hour-{kwargs["apply_to"][:3]}')

def set_operational_limits(n, sns, model_file, model_setup):

    op_limits = pd.read_excel(
        model_file,
        sheet_name='operational_limits',
        index_col=list(range(9)),
    )
    
    if model_setup["operational_limits"] not in op_limits.index.get_level_values(0).unique():
        return
    op_limits = op_limits.loc[model_setup["operational_limits"]]

    #drop rows where all NaN
    op_limits = op_limits.loc[~(op_limits.isna().all(axis=1))]
    for idx, row in op_limits.iterrows():
        apply_operational_constraints(
            n, sns, 
            bus = idx[0], carrier = idx[1], 
            type = idx[2], values = row, 
            period = idx[3], incl_pu = idx[4],
            limit = idx[5], apply_to = idx[6],
            units = idx[7],
        )


def ccgt_steam_constraints(n, sns, model_file, model_setup, snakemake):
    # At each bus HRSG power is limited by what OCGT_gas or diesel is producing
    config = snakemake.config["electricity"]
    p_nom_ratio = config["ccgt_gt_to_st_ratio"]
    ocgt_carriers = ["ocgt_diesel", "ocgt_gas"]
    for bus in n.buses.index:
        ocgt_gens = n.generators.query("bus == bus & carrier in @ocgt_carriers").index
        ccgt_hrsg = n.generators.query("bus == bus & carrier == 'ccgt_steam'").index
        
        lhs = (n.model.variables['Generator-p'].loc[sns, ccgt_hrsg] - p_nom_ratio*n.model.variables['Generator-p'].loc[sns, ocgt_gens]).sum("Generator")
        rhs = 0
        n.model.add_constraints(lhs, "<=", rhs, name = f'ccgt_steam_limit-{bus}')


"""
********************************************************************************
    Reserve margin
********************************************************************************
"""
def check_active(n, c, y, list):
    active = n.df(c).index[n.get_active_assets(c, y)] if n.multi_invest else list
    return list.intersection(active)

def define_reserve_margin(n, sns, model_file, model_setup, snakemake):
    ###################################################################################
    # Reserve margin above maximum peak demand in each year
    # The sum of res_margin_carriers multiplied by their assumed constribution factors 
    # must be higher than the maximum peak demand in each year by the reserve_margin value

    projections = (
        pd.read_excel(
            model_file, 
            sheet_name="projected_parameters",
            index_col=[0,1])
            .loc[model_setup["projected_parameters"]]
    ).drop("unit", axis=1)

    res_mrgn_active = projections.loc["reserve_margin_active"]
    res_mrgn = projections.loc["reserve_margin"]

    peak = n.loads_t.p_set.loc[sns].sum(axis=1).groupby(sns.get_level_values(0)).max() if n.multi_invest else n.loads_t.p_set.loc[sns].sum(axis=1).max()
    peak = peak if n.multi_invest else pd.Series(peak, index = sns.year.unique())
    capacity_credit = snakemake.config["electricity"]["reserves"]["capacity_credit"]

    for y in peak.index:
        if res_mrgn_active[y]:    

            fix_i = n.generators.query("not p_nom_extendable & carrier in @capacity_credit").index
            ext_i = n.generators.query("p_nom_extendable & carrier in @capacity_credit").index
    
            fix_cap = 0
            lhs = 0
            for c in ["Generator", "StorageUnit"]:
                fix_i = n.df(c).query("not p_nom_extendable & carrier in @capacity_credit").index
                fix_i = check_active(n, c, y, fix_i)

                fix_cap += (
                    n.df(c).loc[fix_i, "carrier"].map(capacity_credit)
                    * n.df(c).loc[fix_i, "p_nom"]
                ).sum()
            
                ext_i = n.df(c).query("p_nom_extendable & carrier in @capacity_credit").index
                ext_i = check_active(n, c, y, ext_i)
    
                lhs += (
                    n.model.variables[f"{c}-p_nom"].sel({f"{c}-ext":ext_i}) 
                    *xr.DataArray(n.df(c).loc[ext_i, "carrier"].map(capacity_credit)).rename({f"{c}":f"{c}-ext"})
                ).sum(f"{c}-ext")

            rhs = peak.loc[y]*(1+res_mrgn[y]) - fix_cap 

            n.model.add_constraints(lhs, ">=", rhs, name = f"reserve_margin_{y}")    


if __name__ == "__main__":

    test_file = '../networks/elec_val-LC-UNC_1-supply_redz.nc'
    model_file = pd.ExcelFile("../data/model_file.xlsx")
    model_setup = (
        pd.read_excel(
            model_file, 
            sheet_name="model_setup",
            index_col=[0])
            .loc["grid-expansion"]
    )

    n = pypsa.Network(test_file)

    n.optimize.create_model(multi_investment_periods = True)
    set_operational_limits(n, n.snapshots, model_file, model_setup)
    ccgt_steam_constraints(n, n.snapshots, model_file, model_setup, snakemake)

