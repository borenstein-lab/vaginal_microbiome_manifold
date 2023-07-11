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


# Feature selection
def feature_select(df, meta, col, prcent):
    X = df.to_numpy()
    y = meta[col].to_numpy()

    model = xgb.XGBClassifier(n_estimators=200, nthread=-1, max_depth=5)
    model.fit(X, y)

    feat_selector = BorutaPy(model, n_estimators='auto', verbose=0, random_state=1, perc=prcent)
    feat_selector.fit(X, y)

    X_filt = df.loc[:, df.columns[feat_selector.support_]]
    y = meta[col]

    return X_filt, y


def get_plot_df(lst, col_name):
    df = pd.DataFrame(lst).T
    melt_df = df.melt(var_name='run_index', value_name=col_name)
    melt_df = melt_df.loc[~ melt_df[col_name].isnull()]

    return melt_df


def get_all_ord(lst1, lst2, lst3, names_lst):
    df_lst = []
    for i, lst in enumerate([lst1, lst2, lst3]):
        df = get_plot_df(lst, names_lst[i])
        df_lst.append(df)

    plot_df = df_lst[0].join(df_lst[1][[names_lst[1]]])
    plot_df = plot_df.merge(df_lst[2], on='run_index')

    return plot_df


# Repeated Kfold cross validation
def cros_val(X, y, k, n_reps, mean_fpr):
    cv = RepeatedKFold(n_splits=k, n_repeats=n_reps, random_state=1)
    classifier = xgb.XGBClassifier(n_estimators=200, nthread=-1, max_depth=5)

    importance_df = pd.DataFrame(index=X.columns)
    train_idx_df = pd.DataFrame()
    fprs = []
    tprs = []
    interp_tprs = []
    aucs = []
    fprs_rand = []
    tprs_rand = []
    interp_tprs_rand = []
    aucs_rand = []

    for i in range(2):
        if i == 1:
            y = np.random.permutation(y)

        for fold, (train, test) in enumerate(cv.split(X, y)):
            classifier.fit(X.iloc[train], y[train])
            y_proba = classifier.predict_proba(X.iloc[test])[:, 1]
            auc = roc_auc_score(y[test], classifier.predict_proba(X.iloc[test])[:, 1], average='macro')
            fpr, tpr, thresholds = roc_curve(y[test], y_proba)
            importance_df[str(fold)] = list(classifier.feature_importances_)
            train_idx_df.loc[:, str(fold)] = pd.Series(train)

            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0

            if i == 1:
                fprs_rand.append(fpr)
                tprs_rand.append(tpr)
                interp_tprs_rand.append(interp_tpr)
                aucs_rand.append(auc)
            else:
                fprs.append(fpr)
                tprs.append(tpr)
                interp_tprs.append(interp_tpr)
                aucs.append(auc)

    return fprs, tprs, interp_tprs, aucs, fprs_rand, tprs_rand, interp_tprs_rand, aucs_rand, importance_df, train_idx_df


def get_mean(fprs, mean_fpr, tprs, interp_tprs, auc):
    mean_tpr = np.mean(interp_tprs, axis=0)
    mean_tpr[-1] = 1.0

    mean_auc = metrics.auc(mean_fpr, mean_tpr)

    fprs.append(mean_fpr)
    tprs.append(mean_tpr)
    auc.append(mean_auc)

    return fprs, tprs, auc


def all_func(df, meta, col, prcent, k, n_reps, names_lst):
    # Feature selection
    X_filt, y = feature_select(df, meta, col, prcent)
    #     X_filt = df
    #     y = meta['BV_status']
    print('df columns number after feature selection: ' + str(X_filt.shape[1]))

    # K-fold cross validation
    mean_fpr = np.linspace(0, 1, 100)
    fprs, tprs, interp_tprs, aucs, fprs_rand, tprs_rand, interp_tprs_rand, aucs_rand, importance_df, train_idx_df = cros_val(
        X_filt, y, k, n_reps, mean_fpr)
    print('fprs: ' + str(sum([len(lst) for lst in fprs])))
    print('tprs: ' + str(sum([len(lst) for lst in tprs])))
    print('fprs_rand: ' + str(sum([len(lst) for lst in fprs_rand])))
    print('tprs_rand: ' + str(sum([len(lst) for lst in tprs_rand])))

    # Create mean values
    fprs, tprs, aucs = get_mean(fprs, mean_fpr, tprs, interp_tprs, aucs)
    fprs_rand, tprs_rand, aucs_rand = get_mean(fprs_rand, mean_fpr, tprs_rand, interp_tprs_rand, aucs_rand)
    print('tprs: ' + str(sum([len(lst) for lst in tprs])))
    print('tprs_rand: ' + str(sum([len(lst) for lst in tprs_rand])))

    # Create final df's
    all_df = get_all_ord(fprs, tprs, aucs, names_lst)
    rand_df = get_all_ord(fprs_rand, tprs_rand, aucs_rand, names_lst)
    print('final df row number: ' + str(all_df.shape[0]))
    print('random df row number: ' + str(rand_df.shape[0]))

    return all_df, rand_df, importance_df, train_idx_df, X_filt
