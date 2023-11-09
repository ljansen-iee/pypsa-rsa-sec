import logging
import pandas as pd
import pypsa

from pypsa.descriptors import get_switchable_as_dense as get_as_dense
from pypsa.descriptors import expand_series
from pypsa.optimization.common import reindex

# from _helpers import configure_logging, clean_pu_profiles, remove_leap_day, normalize_and_rename_df, assign_segmented_df_to_network
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

def calc_max_gen_potential(n, sns, gens, extendable=False):
    suffix = "" if extendable == False else "-ext"
    p_max_pu = get_as_dense(n, 'Generator', "p_max_pu", sns)[gens]
    p_max_pu.columns.name = f'Generator{suffix}'
    if not extendable:
        return p_max_pu[gens] * n.generators.loc[gens, "p_nom"]
    p_nom = n.model.variables["Generator-p_nom"].sel({f"Generator{suffix}": gens})
    return (xr.DataArray(p_max_pu)*p_nom).sum(f'Generator{suffix}')

def apply_operational_constraints(n, sns, **kwargs):
    carrier = kwargs["carrier"]
    if carrier in ["gas", "diesel"]:
        carrier = n.carriers[n.carriers.index.str.contains(carrier)].index
    bus = kwargs["bus"]
    period = kwargs["period"]
    limit = kwargs["limit"]

    sense = "<=" if limit == "max" else ">="

    cf_limit = 0 * kwargs["values"] if kwargs["type"] == "energy" else kwargs["values"]
    en_limit = 0 * kwargs["values"] if kwargs["type"] == "capacity_factor" else kwargs["values"]

    years = sns.get_level_values(0).unique() if n.multi_invest else [sns.year.unique()]

    filtered_gens = n.generators.query("carrier in @carrier") if len(carrier)>1 else n.generators.query("carrier == @carrier")
    if bus != "global":
        filtered_gens = filtered_gens.query("bus == @bus")
    fix_i = filtered_gens.query("not p_nom_extendable").index
    ext_i = filtered_gens.query("p_nom_extendable").index

    # apply snapshot weightings to generator constraints
    weightings = n.snapshot_weightings.loc[sns, "generators"]

    max_gen_fix = expand_series(weightings, fix_i) * calc_max_gen_potential(n, sns, fix_i, extendable=False) if len(fix_i)>0 else 0
    max_gen_ext = calc_max_gen_potential(n, sns, ext_i, extendable=True) * weightings if len(ext_i)>0 else 0
    act_gen = (n.model.variables['Generator-p'] * weightings).sel(Generator=filtered_gens.index).sum('Generator')

    groupby_dict = {
        "year": "timestep.year",
        "month": "timestep.month",
        "week": "timestep.week",
        "hour": None
    }

    for y in years:
        year_sns = sns[sns.get_level_values(0)==y] if n.multi_invest else sns[sns.year==y]
        lhs = (act_gen - cf_limit[y] * max_gen_ext)
        lhs = lhs.sel(period=y) if n.multi_invest else lhs.sel(snapshot=year_sns)
        
        if groupby := groupby_dict[period]:
            lhs_p = lhs.groupby(groupby).sum()
            if len(fix_i) == 0: #i.e set to 0 above as empty
                rhs_p = en_limit[y]
            else:
                rhs = en_limit[y] + cf_limit[y] * (max_gen_fix.loc[y] if n.multi_invest else max_gen_fix.loc[year_sns]) 
                rhs_p = rhs.groupby(rhs.index.month if period == "month" else rhs.index.isocalendar().week).sum()
                rhs_p = rhs_p.sum(axis=1) if rhs_p.shape[1]>1 else rhs_p
                rhs_p.index.name = period
        n.model.add_constraints(lhs_p, sense, rhs_p, name = f'{limit}-{kwargs["carrier"]}-{period}-{y}')

def set_operational_limits(n, model_file, model_setup):

    op_limits = pd.read_excel(
        model_file,
        sheet_name='operational_limits',
        index_col=list(range(6)),
    ).loc[model_setup["operational_limits"]]

    #drop rows where all NaN
    op_limits = op_limits.loc[~(op_limits.isna().all(axis=1))]
    for idx, row in op_limits.iterrows():
        apply_operational_constraints(
            n, n.snapshots, 
            bus = idx[0], carrier = idx[1], 
            type = idx[2], values = row, 
            period = idx[3], limit = idx[4]
        )
    

if __name__ == "__main__":

    test_file = '../networks/pre_grid-expansion_11-supply_redz_lcopt_LC-3000SEG.nc'
    model_file = pd.ExcelFile("../data/model_file.xlsx")
    model_setup = (
        pd.read_excel(
            model_file, 
            sheet_name="model_setup",
            index_col=[0])
            .loc["grid-expansion"]
    )

    n = pypsa.Network(test_file)

    n.optimize.create_model()
    set_operational_limits(n, model_file, model_setup)
    tes=1
