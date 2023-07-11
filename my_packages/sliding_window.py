import pandas as pd
import numpy as np

def get_window_df(df, window_size, increment):
    res_df = pd.DataFrame(columns = df.columns)
    ps_col = [col for col in df.columns if 'mt_pseudotime' in col]

    for small_value in np.arange(0.0, 1.01 - window_size, increment):
        curr_df = df.loc[(df[ps_col[0]] >= small_value) & (df[ps_col[0]] < small_value + window_size)]
        if len(curr_df) >= 1:
            mean_df = pd.DataFrame(curr_df.mean(), index = curr_df.columns).T
            res_df = pd.concat([res_df, mean_df], ignore_index = True)

    return res_df, ps_col[0]

def get_sorted(df, ps_colname, ps = True):
    ''' Input: df
    Output: sorted df by columns sums'''
    if ps:
        temp = df.drop(ps_colname, axis = 1)
    else:
        temp = df.copy()
    s = temp.sum(axis = 0)
    sort_df = temp[s.sort_values(ascending = False).index[:temp.shape[1]]]

    if ps:
        sort_df = sort_df.join(df[[ps_colname]])

    return sort_df

def get_others_col(df, col_name, n, ps_colname, ps = True):
    ''' Input: df, others columns name, n number of columns to leave not in others
    Output: df with n columns and others column'''
    if ps:
        temp = df.drop(ps_colname, axis = 1)
    else:
        temp = df.copy()
    temp[col_name] = temp.iloc[:, n:temp.shape[1]].sum(axis = 1)
    final_df = pd.concat([temp.iloc[:, :n], temp[col_name]], axis = 1)

    if ps:
        final_df = final_df.join(df[ps_colname])

    return final_df

def all_window_proc(df, meta, branch, root_col, root,  window_size, increment, n = 12, remove = True):
    if remove:
        branch_df = meta.loc[(meta['CST'] == branch) & (meta[root_col] != root)]
    else:
        branch_df = meta.loc[(meta['CST'] == branch)]

    ## Branch df
    abun_df = df[df.index.isin(branch_df.index)]
    abun_df = abun_df.loc[:, (abun_df != 0).any(axis=0)]

    # Add branch name
    abun_df = abun_df.join(meta[['mt_pseudotime']])
    abun_df = abun_df.add_suffix('_' + branch)

    # Sliding window
    window_df, ps_colname = get_window_df(abun_df, window_size, increment)

    # Create others column
    sort_window_df = get_sorted(window_df, ps_colname)
    others_name = 'Others_' + branch
    sumsort_window_df = get_others_col(sort_window_df, others_name, n, ps_colname)

    return abun_df, window_df, sumsort_window_df

def get_order(df_lst):
    ''' Input: list of dfs from all branches
    Output: concencated df from list'''
    all_df = pd.concat(df_lst)
    all_df.fillna(0, inplace = True)

    col_list = [col for col in all_df.columns if 'mt_pseudotime' in col]
    all_df['mt_pseudotime'] = all_df[col_list].sum(axis=1)
    all_df.drop(col_list, axis = 1, inplace = True)
    all_df02 = all_df.set_index('mt_pseudotime', inplace = False)

    return all_df02


def get_melt(orig_df):
    ''' Input: df with species from all branches
    Output: melted df with columns of species, value and branch'''
    df = orig_df.reset_index()
    melt_branch_df = df.melt(id_vars=['mt_pseudotime'], var_name='species', value_name='value')
    plot_df = melt_branch_df.copy()
    plot_df = plot_df[plot_df['value'] > 0.0]
    spec_df = plot_df.species.str.rsplit('_', 1, expand=True).rename(lambda x: f'col{x + 1}', axis=1)
    spec_df.columns = ['species_only', 'branch']
    plot_df = plot_df.merge(spec_df, left_index=True, right_index=True)

    plot_df.drop('species', axis=1, inplace=True)
    plot_df.rename(columns={'species_only': 'species'}, inplace=True)

    return plot_df