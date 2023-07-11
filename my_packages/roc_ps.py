import pandas as pd
from sklearn import metrics

def get_shuffled_df(df, n, label_col):
    ## Order df
    filt_df = df[[label_col]]
    filt_df.reset_index(drop=True, inplace=True)

    ## Permutations
    for i in range(n):
        shuff_arr = filt_df[label_col].sample(frac=1).reset_index(drop=True)
        colname = 'shuff_' + str(i)
        filt_df[colname] = shuff_arr

    return filt_df


def get_roc_auc(label_arr, pred_arr, shuff_num):
    fpr, tpr, _ = metrics.roc_curve(label_arr, pred_arr)
    auc = metrics.roc_auc_score(label_arr, pred_arr)
    res_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'auc': auc})
    res_df['shuff_num'] = shuff_num

    return res_df


def get_roc(shuffled_df, pred_arr):
    ## Create df
    final_df = pd.DataFrame(columns=['fpr', 'tpr', 'auc', 'shuff_num'])

    for col in shuffled_df.columns:
        res_df = get_roc_auc(shuffled_df[col], pred_arr, col)
        final_df = pd.concat([final_df, res_df])

    return final_df


def all_func(df, n, label_col):
    filt_df = df[df[label_col].notna()]
    pred_arr = filt_df.mt_pseudotime

    shuffled_df = get_shuffled_df(filt_df, n, label_col)
    final_df = get_roc(shuffled_df, pred_arr)

    return final_df, shuffled_df
