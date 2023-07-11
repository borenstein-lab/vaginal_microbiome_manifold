import pandas as pd
import xlsxwriter
import scanpy as sc
from anndata import AnnData
import numpy as np


def run_paga(adata, n_neighbors, n_pcs, ps_col, ps_value, metric = 'euclidean'):
    '''Input: adata, n_neighbors, n_pcs, ps_col, ps_value
     Ouput: adata with results
     Run: UMAP, Leiden, pseudotime'''
    ## Create UMAP
    if metric == 'euclidean':
        sc.pp.neighbors(adata, n_neighbors = n_neighbors, n_pcs = n_pcs, metric = metric)
    else:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs = n_pcs, use_rep = 'X', metric=metric)

    sc.tl.draw_graph(adata)
    sc.pl.draw_graph(adata, color=['subCST', 'nugent', 'ph', 'db'], legend_loc='right margin')

    ## Leiden clusters
    sc.tl.leiden(adata, resolution=1.0)
    sc.tl.paga(adata, groups='leiden')
    sc.pl.draw_graph(adata, color=['CST', 'leiden'], legend_loc='on data')
    sc.pl.paga(adata, color=['leiden'])

    adata.obs['leiden_anno'] = adata.obs['leiden']
    sc.tl.paga(adata, groups = 'leiden_anno')

    ## Pseudotime
    adata.uns['iroot'] = np.flatnonzero(adata.obs[ps_col] == ps_value)[0]
    sc.tl.dpt(adata)

    ## Plot pseudotime
    sc.pl.draw_graph(adata, color=['leiden', 'subCST', 'dpt_pseudotime'], legend_loc='on data')

    return adata


def save_excel(path, dict_res):
    writer = pd.ExcelWriter(path, engine='xlsxwriter')
    for key, value in dict_res.items():
        value.to_excel(writer, sheet_name=str(key))
    writer.save()
    writer.close()

    return


def paga_results(adata, df, meta, metric = 'euclidean'):
    ''' Input: adata, df, meta
    Output: dictionary of all dfs
    Run get 5 objects of adata results to dictionary'''
    names_lst = ['abundance', 'meta', 'umap', 'leiden']

    graph_df = pd.DataFrame(adata.obsm['X_draw_graph_fa'], index=meta.index, columns=['fa1', 'fa2'])

    ## Meta
    leiden_df = adata.obs['leiden'].to_frame()
    final_meta = meta.copy()
    final_meta.loc[:, 'dpt_pseudotime'] = adata.obs['dpt_pseudotime']
    final_meta.loc[:, 'mt_pseudotime'] = 1 - final_meta.loc[:, 'dpt_pseudotime']
    if metric == 'euclidean':
        pca_df = pd.DataFrame(adata.obsm['X_pca'])
        names_lst.append('pca')

    ## Save all in dict
    df_lst = [df, final_meta, graph_df, leiden_df]
    if metric == 'euclidean':
        df_lst.append(pca_df)
    dict_res = dict(zip(names_lst, df_lst))

    return dict_res
