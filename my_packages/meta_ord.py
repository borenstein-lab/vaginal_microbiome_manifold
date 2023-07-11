import pandas as pd
from skbio.diversity import alpha_diversity, beta_diversity
import numpy as np

def get_shannon(df, meta):
    ''' Input: df, meta
    Output: meta with shannon_index columns
    Run shannon_index and merge with the original meta'''
    shannon_vec = alpha_diversity(counts = df.values, metric = 'shannon')
    shannon_df = pd.DataFrame(shannon_vec, columns = ['shannon_index'])
    shannon_df = shannon_df.set_index(df.index)

    meta = pd.merge(meta, shannon_df, left_index=True, right_index=True)

    return meta


def label_numeric_bin(row, col, value):
    ''' Input: row (with apply), columne name and value of bin separation
    Output: after apply, columns with binary values'''
    if pd.isnull(row[col]):
        return np.nan
    if row[col] >= value:
        return 1
    else:
        return 0


def label_menst(row):
    if pd.isnull(row['menst']) or row['menst'] == 0:
        return 0
    else:
        return 1


def all_menst_labels(df, n):
    new_df = df.copy()
    new_df['tmp_menst0'] = new_df.apply(lambda row: label_menst(row), axis=1)
    new_df['tmp_menst_plus_n'] = new_df['tmp_menst0'].shift()
    new_df['tmp_menst_minus_n'] = new_df['tmp_menst0'].shift(-1)

    new_df['temp_menst_final'] = new_df['tmp_menst0'] + new_df['tmp_menst_plus_n'] + new_df['tmp_menst_minus_n']
    new_df['bin_menst_temp'] = (new_df['temp_menst_final'] >= 1).astype(int)
    new_df.drop(['tmp_menst0', 'tmp_menst_plus_n', 'tmp_menst_minus_n', 'temp_menst_final'], axis=1, inplace=True)

    return new_df