import pandas as pd
import numpy as np

def get_window_df(df, window_size, increment):
    res_df = pd.DataFrame(columns = df.columns)

    for small_value in np.arange(0.0, 1.01 - window_size, increment):
        curr_df = df.loc[(df.mt_pseudotime >= small_value) & (df.mt_pseudotime < small_value + window_size)]
        if len(curr_df) >= 1:
            mean_df = pd.DataFrame(curr_df.mean(), index = curr_df.columns).T
            res_df = pd.concat([res_df, mean_df], ignore_index = True)

    return res_df

def get_sorted(df):
    ''' Input: df
    Output: sorted df by columns sums'''
    s = df.sum(axis = 0)
    sort_df = df[s.sort_values(ascending = False).index[:df.shape[1]]]

    return sort_df

def get_others_col(df, col_name, n):
    ''' Input: df, others columns name, n number of columns to leave not in others
    Output: df with n columns and others column'''
    df[col_name] = df.iloc[:, n:df.shape[1]].sum(axis = 1)
    final_df = pd.concat([df.iloc[:, :n + 1], df[col_name]], axis = 1)

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
    abun_df = abun_df.add_suffix('_' + branch)

    # Sliding window
    abun_df = abun_df.join(meta[['mt_pseudotime']])
    window_df = get_window_df(abun_df, window_size, increment)

    # Save pseudotime column
    ps_col = window_df['mt_pseudotime']
    window_df_forreturn = window_df.copy()
    # window_df.drop('mt_pseudotime', axis=1, inplace=True)

    # Create others column
    sort_window_df = get_sorted(window_df)
    others_name = 'Others_' + branch
    sumsort_window_df = get_others_col(sort_window_df, others_name, n)
    sumsort_window_df = sumsort_window_df.join(ps_col)

    return abun_df, window_df_forreturn, sumsort_window_df

def get_order(df_lst):
    ''' Input: list of dfs from all branches
    Output: concencated df from list'''
    all_df = pd.concat(df_lst)
    # all_df.set_index('mt_pseudotime', inplace = True)
    all_df.fillna(0, inplace = True)

    return all_df


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