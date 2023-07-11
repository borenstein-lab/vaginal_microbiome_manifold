import pandas as pd

def get_closest_nn(dist_df, ps_df, out_df):
    filt_dist = dist_df.loc[ps_df.index, out_df.index]
    res_df = pd.DataFrame(columns=['closest_sample', 'distance'], index=out_df.index)

    for col in filt_dist.columns:
        min_distance = filt_dist[col].min()
        min_indx = filt_dist[col].idxmin()
        closest_sample = min_indx

        ## insert results to res_df
        res_df.loc[col, 'distance'] = min_distance
        res_df.loc[col, 'closest_sample'] = closest_sample

    return res_df


def get_ps_from_n(knn_df, ps_meta, out_meta):
    knn_df = knn_df.reset_index(inplace=False)

    ps_meta = ps_meta.reset_index(inplace=False)
    ps_meta.rename(columns={'index': 'closest_sample'}, inplace=True)

    res_df = pd.merge(knn_df, ps_meta[['closest_sample', 'mt_pseudotime', 'subCST']], on='closest_sample')
    res_df.rename(columns={'subCST': 'closest_subCST'}, inplace=True)
    res_df = res_df.set_index('index', inplace=False)

    out_meta = out_meta.join(res_df)

    return out_meta