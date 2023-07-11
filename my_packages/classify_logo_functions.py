from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import GroupKFold
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay
import xgboost as xgb
import shap
from boruta import BorutaPy
import numpy as np
import pandas as pd

def get_plot_df(lst, col_name):
    df = pd.DataFrame(lst).T
    melt_df = df.melt(var_name='run_index', value_name=col_name)
    melt_df = melt_df.loc[~ melt_df[col_name].isnull()]

    return melt_df

def logo(X, y, meta, col):
    groups = meta['db']
    logo = LeaveOneGroupOut()
    classifier = xgb.XGBClassifier(n_estimators=200, nthread=-1, max_depth=5)

    fprs = []
    tprs = []
    aucs = []

    for i, (train, test) in enumerate(logo.split(X, y, groups)):
        print('Study index: ' + str(i) + ', samples number in study: ' + str(len(test)))
        classifier.fit(X.iloc[train], y[train])
        y_proba = classifier.predict_proba(X.iloc[test])[:, 1]
        auc = roc_auc_score(y[test], classifier.predict_proba(X.iloc[test])[:, 1], average='macro')
        fpr, tpr, thresholds = roc_curve(y[test], y_proba)
        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(auc)

    return fprs, tprs, aucs


def get_logo_ord(lst1, lst2, lst3, names_lst):
    df_lst = []
    for i, lst in enumerate([lst1, lst2, lst3]):
        df = get_plot_df(lst, names_lst[i])
        df_lst.append(df)

    plot_df = df_lst[0].join(df_lst[1][[names_lst[1]]])
    plot_df = plot_df.merge(df_lst[2], on='run_index')

    return plot_df


def logo_all_func(df, meta, col, prcent, names_lst):
    # Feature selection
    #     X_filt, y = feature_select(df, meta, col, prcent)
    #     print('Feature selected: ' + str(X_filt.shape[1]))

    # Logo cross-validation
    fprs, tprs, aucs = logo(df, meta[col], meta, col)
    print('fprs: ' + str(len(fprs)))
    print('tprs: ' + str(len(tprs)))

    # Convert to df's
    all_df = get_logo_ord(fprs, tprs, aucs, names_lst)
    replacers = {0: 'car22', 1: 'cec19', 2: 'sri12'} #, 3: 'rav13'}
    all_df = all_df.replace({"run_index": replacers})

    return all_df