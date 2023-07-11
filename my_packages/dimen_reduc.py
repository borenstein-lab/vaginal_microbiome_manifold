from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skbio.diversity import beta_diversity
from skbio.stats.ordination import pcoa, pcoa_biplot
import pandas as pd
import numpy as np

def get_center(df):
    ''' Input: df
    Output: centered df, 1 object '''
    mean_vec = df.mean(axis=0)
    centered_df = df - mean_vec

    return centered_df


def get_standarized(df):
    ''' Input: df
    Output: standarized df, 1 object'''
    scale = StandardScaler()
    stand_df = pd.DataFrame(StandardScaler().fit_transform(df), index=df.index, columns=df.columns)

    return stand_df


def get_pca_reg(df):
    ''' Input: df
    Output: PCA array and extra results, 4 objects
    Run PCA analysis'''
    pca = PCA(n_components = 50)
    pca.fit(df)
    pca_array = pca.transform(df)

    loadings = pca.components_.T
    variance_ratio = pca.explained_variance_ratio_
    variance = pca.explained_variance_

    return pca_array, loadings, variance_ratio, variance


def get_bc_dist(df):
    ''' Input: df
    Output: dataframe of distance matrix by Bray Curtis between all samples
    Run beta diversity with Bray Curtis metric'''
    arr = df.to_numpy()
    bc_dist = beta_diversity(counts = arr, ids = list(df.index), metric = "braycurtis", validate = False)
    bc_df = pd.DataFrame(bc_dist, columns=df.index, index=df.index)

    return bc_dist.to_data_frame()


def get_pcoa(dist, df, row_names):
    ''' Input: distance df, orig df, row_names column name
    Output: PCoA array and PCoA extra results
    Run PCoA for distance df with bplot'''
    pc = pcoa(dist)

    b_df = df.copy()
    b_df.reset_index(inplace=True)
    b_df.drop(row_names, axis=1, inplace=True)
    b_df.index = b_df.index.map(str)

    pc_bplot = pcoa_biplot(pc, b_df)

    samples = pc.samples.iloc[:, :50].to_numpy()
    loadings = pc.eigvals.to_numpy()
    variance = pc.proportion_explained[:50].to_numpy()
    variance_ratio = variance / np.sum(variance)

    return samples, loadings, variance_ratio, variance
