# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 15:28:42 2018

@author: i.serov
"""

# Logistin Regression Framework Functions
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set(style="white", color_codes=True)
from pandas import ExcelWriter
from sklearn.utils import resample
from tqdm import tqdm_notebook
import pandas.core.algorithms as algos
from pandas import Series
import scipy.stats.stats as stats
import statsmodels.api as sm
import re
import traceback
import string
import math
from datetime import datetime
import importlib
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LassoCV, RidgeClassifierCV
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score, train_test_split, KFold, StratifiedKFold, GridSearchCV 
from sklearn.metrics import roc_auc_score, roc_curve, log_loss, f1_score, confusion_matrix, precision_score, recall_score, classification_report;
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
# try:
#     import IDF
# except:
#     import autoscoring.IDF


def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    print('_done')
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = X.columns[new_pval.argmin()]
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        print('pre_done_sm.ols')
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = X.columns[pvalues.argmax()]
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

def feature_selection(X_train, y_train, X_test, y_test, n_features = 10, selection_type = 'sw', try_all = False):
    print('done included_00')
    '''
    Selection Types:
    'f' - forward
    'b' - backward
    'sw' - stepwise + n best features
    'ff' - forward floating
    'bf' - backward floating
    'kb' - K-Best based on performance
    'RFE' - recursive feature elimination
    'ALL' - Try All types (computatuionally expensive)
    '''
    
    clf_sw = LogisticRegressionCV(random_state=1, cv=3)
    
    X_train_cp = X_train.copy()
    X_test_cp = X_test.copy()
    print('done included_00')
    # Stepwise
    if selection_type == 'sw':
        print('done included_0')
        included = stepwise_selection(X_train, y_train, threshold_in=0.01, threshold_out = 0.05, verbose=False)
        print('done included')
        X_train = X_train[included]
        X_test = X_test[included]
        # If stepwise produces many features we cut only X best by regression performance
        if X_train.shape[1]>n_features:            
            clf_sw.fit(X_train, y_train)
            topX = list(pd.DataFrame(list(zip(X_test.columns, clf_sw.coef_[0])), columns=[
             'feat', 'imp']).sort_values(by='imp', ascending=False).feat[:n_features])
            X_train = X_train[topX]
            X_test = X_test[topX]

    # Forward    
    elif selection_type == 'f':
        sfs1 = SFS(estimator=clf_sw,k_features=(n_features-4, n_features),forward=True,
                   floating=False,scoring='roc_auc',n_jobs=3, verbose=0,cv=3)
        sfs1.fit(np.array(X_train), y_train)
        feat_cols = list(sfs1.k_feature_idx_)
        X_train = X_train.iloc[:, feat_cols]
        X_test = X_test.iloc[:, feat_cols]
    
    # Forward Floating
    elif selection_type == 'ff':
        sfs1 = SFS(estimator=clf_sw,k_features=(n_features-4, n_features),forward=True,
                   floating=True,scoring='roc_auc',n_jobs=3, verbose=0,cv=3)
        sfs1.fit(np.array(X_train), y_train)
        feat_cols = list(sfs1.k_feature_idx_)
        X_train = X_train.iloc[:, feat_cols]
        X_test = X_test.iloc[:, feat_cols]
    
    # Backward Floating    
    elif selection_type == 'bf':
        sfs1 = SFS(estimator=clf_sw,k_features=(n_features-4, n_features),forward=False,
                   floating=True,scoring='roc_auc',n_jobs=3, verbose=0,cv=3)
        sfs1.fit(np.array(X_train), y_train)
        feat_cols = list(sfs1.k_feature_idx_)
        X_train = X_train.iloc[:, feat_cols]
        X_test = X_test.iloc[:, feat_cols]
    # Backward 
    elif selection_type == 'b':
        sfs1 = SFS(estimator=clf_sw,k_features=(n_features-4, n_features),forward=False,
                   floating=False,scoring='roc_auc',n_jobs=3, verbose=0,cv=3)
        sfs1.fit(np.array(X_train), y_train)
        feat_cols = list(sfs1.k_feature_idx_)
        X_train = X_train.iloc[:, feat_cols]
        X_test = X_test.iloc[:, feat_cols]
    
    # K Best
    elif selection_type == 'kb':
        kb = SelectKBest(k=n_features).fit(X_train, y_train)
        mask = kb.get_support()
        X_train = X_train.loc[:, mask]
        X_test = X_test.loc[:, mask]
    
    # RFE
    elif selection_type == 'RFE':
        rfe = RFE(estimator=clf_sw, n_features_to_select=n_features)
        rfe.fit(X_train, y_train)
        mask = rfe.get_support()
        X_train = X_train.loc[:, mask]
        X_test = X_test.loc[:, mask]
    elif selection_type == 'ALL':
        print ('Trying ALL_ types...')
        clf_last = LogisticRegressionCV(random_state=42, cv=3)
        feat_cnt = n_features
        
        print ('Trying Stepwise_...')
        X_train_a, X_test_a, _, _ = feature_selection(X_train_cp, y_train, X_test_cp, y_test, n_features = feat_cnt, selection_type = 'sw')
        clf_last.fit(X_train_a, y_train)
        print ('Gini Train:', round(2*roc_auc_score(y_train, clf_last.predict_proba(X_train_a)[:,1]) - 1, 3))
        print ('Gini Test:', round(2*roc_auc_score(y_test, clf_last.predict_proba(X_test_a)[:,1]) - 1, 3))
        best = round(2*roc_auc_score(y_test, clf_last.predict_proba(X_test_a)[:,1]) - 1, 3)
        X_end_train = X_train_a.copy()
        X_end_test = X_test_a.copy()
        
        print ('Trying Backward Floating...')
        X_train_a, X_test_a, _, _ = feature_selection(X_train_cp, y_train, X_test_cp, y_test, n_features = feat_cnt, selection_type = 'b')
        clf_last.fit(X_train_a, y_train)
        print ('Gini Train:', round(2*roc_auc_score(y_train, clf_last.predict_proba(X_train_a)[:,1]) - 1, 3))
        print ('Gini Test:', round(2*roc_auc_score(y_test, clf_last.predict_proba(X_test_a)[:,1]) - 1, 3))
        if round(2*roc_auc_score(y_test, clf_last.predict_proba(X_test_a)[:,1]) - 1, 3)>best:
            X_end_train = X_train_a.copy()
            X_end_test = X_test_a.copy()
            
        print ('Trying Forward Floating...')
        X_train_a, X_test_a, _, _ = feature_selection(X_train_cp, y_train, X_test_cp, y_test, n_features = feat_cnt, selection_type = 'ff')
        clf_last.fit(X_train_a, y_train)
        print ('Gini Train:', round(2*roc_auc_score(y_train, clf_last.predict_proba(X_train_a)[:,1]) - 1, 3))
        print ('Gini Test:', round(2*roc_auc_score(y_test, clf_last.predict_proba(X_test_a)[:,1]) - 1, 3))
        if round(2*roc_auc_score(y_test, clf_last.predict_proba(X_test_a)[:,1]) - 1, 3)>best:
            X_end_train = X_train_a.copy()
            X_end_test = X_test_a.copy()
            
        print ('Trying RFE...')
        X_train_a, X_test_a, _, _ = feature_selection(X_train_cp, y_train, X_test_cp, y_test, n_features = feat_cnt, selection_type = 'RFE')
        clf_last.fit(X_train_a, y_train)
        print ('Gini Train:', round(2*roc_auc_score(y_train, clf_last.predict_proba(X_train_a)[:,1]) - 1, 3))
        print ('Gini Test:', round(2*roc_auc_score(y_test, clf_last.predict_proba(X_test_a)[:,1]) - 1, 3))
        if round(2*roc_auc_score(y_test, clf_last.predict_proba(X_test_a)[:,1]) - 1, 3)>best:
            X_end_train = X_train_a.copy()
            X_end_test = X_test_a.copy()
            
        print ('Trying K-Best...')
        X_train_a, X_test_a, _, _ = feature_selection(X_train_cp, y_train, X_test_cp, y_test, n_features = feat_cnt, selection_type = 'kb')
        clf_last.fit(X_train_a, y_train)
        print ('Gini Train:', round(2*roc_auc_score(y_train, clf_last.predict_proba(X_train_a)[:,1]) - 1, 3))
        print ('Gini Test:', round(2*roc_auc_score(y_test, clf_last.predict_proba(X_test_a)[:,1]) - 1, 3))
        if round(2*roc_auc_score(y_test, clf_last.predict_proba(X_test_a)[:,1]) - 1, 3)>best:
            X_end_train = X_train_a.copy()
            X_end_test = X_test_a.copy()
        
        # End
        X_train = X_end_train.copy()
        X_test = X_end_test.copy()
            
    return X_train, X_test, X_train_cp, X_test_cp

################ Preprocessing ####################
def filling(df):
    cat_vars = df.select_dtypes(include=[object]).columns
    num_vars = df.select_dtypes(include=[np.number]).columns
    df[cat_vars] = df[cat_vars].fillna('_MISSING_')
    df[num_vars] = df[num_vars].fillna(np.nan)
    return df

def replace_not_frequent(df, cols, perc_min=5, value_to_replace = "_ELSE_"):
    else_df = pd.DataFrame(columns=['var', 'list'])
    for i in cols:
        if i != 'date_requested' and i != 'credit_id':
            t = df[i].value_counts(normalize=True)
            q = list(t[t.values < perc_min/100].index)
            if q:
                else_df = else_df.append(pd.DataFrame([[i, q]], columns=['var', 'list']))
            df.loc[df[i].value_counts(normalize=True)[df[i]].values < perc_min/100, i] = value_to_replace
    else_df = else_df.set_index('var')
    return df, else_df
    
def replace_not_frequent_2(df, cols, num_min=100, value_to_replace = "_ELSE_"):
    else_df = pd.DataFrame(columns=['var', 'list'])
    for i in cols:
        if i != 'date_requested' and i != 'credit_id':
            t = df[i].value_counts()
            q = list(t[t.values < num_min].index)
            if q:
                else_df = else_df.append(pd.DataFrame([[i, q]], columns=['var', 'list']))
            df.loc[df[i].value_counts(dropna=False)[df[i]].values < num_min, i] = value_to_replace
    else_df = else_df.set_index('var')
    return df, else_df

def drop_single_value_column(df, except_cols=[]):
    except_cols = set(except_cols)
    df2 = df.copy()
    for i in df2.columns:
        if i in except_cols:
            continue
        if df2[i].value_counts(dropna=False).shape[0]==1:
            df2.drop(i, axis=1, inplace=True)
    return df2  

################## Binning ######################

def adjust_binning(df, bins_dict):
    for i in range(len(bins_dict)):
        key = list(bins_dict.keys())[i]
        if type(list(bins_dict.values())[i])==dict:
            df[key] = df[key].map(list(bins_dict.values())[i])
        else:
            #Categories labels
            categories = list()
            for j in range(len(list(bins_dict.values())[i])):
                if j == 0:
                    categories.append('<'+ str(list(bins_dict.values())[i][j]))
                    try:                        
                        categories.append('(' + str(list(bins_dict.values())[i][j]) +'; '+ str(list(bins_dict.values())[i][j+1]) + ']')
                    except:                       
                        categories.append('(' + str(list(bins_dict.values())[i][j]))
                elif j==len(list(bins_dict.values())[i])-1:
                    categories.append(str(list(bins_dict.values())[i][j]) +'>')
                else:
                    categories.append('(' + str(list(bins_dict.values())[i][j]) +'; '+ str(list(bins_dict.values())[i][j+1]) + ']')
            
            values = [df[key].min()] + list(bins_dict.values())[i]  + [df[key].max()]        
            df[key + '_bins'] = pd.cut(df[key], values, include_lowest=True, labels=categories).astype(object).fillna('_MISSING_').astype(str)
            df[key] = df[key + '_bins']#.map(df.groupby(key + '_bins')[key].agg('median'))
            df.drop([key + '_bins'], axis=1, inplace=True)
    return df

################## Correlation and Plots ######################
def delete_correlated_features(df, cut_off=0.75, is_plot_prev=True, exclude=[], IV_sort=False, iv_df=None):
    # Create correlation matrix
    corr_matrix = df.corr().abs()
    # Сортируем матрицу корреляций по значению IV, чтобы удалять признаки с наименьшим IV
    if IV_sort and iv_df is not None:
        # Sorting correlation matrix by IV value
        IV = iv_df[['VAR_NAME', 'IV']].drop_duplicates()
        # IV['VAR_NAME'] = 'WOE_' + IV['VAR_NAME']
        IV_sort = IV[IV['VAR_NAME'].isin(df.columns)].sort_values(by='IV')['VAR_NAME'].values[::-1]

        corr_matrix = corr_matrix[IV_sort]

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    
    if is_plot_prev:
        # Plotting All correlations
        f, ax = plt.subplots(figsize=(15, 10))
        plt.title('All correlations', fontsize=20)
        sns.heatmap(df.corr(), annot=True)
        
        # Plotting highly correlated
        try:
            f, ax = plt.subplots(figsize=(15, 10))
            plt.title('High correlated', fontsize=20)
            sns.heatmap(corr_matrix[(corr_matrix>cut_off) & (corr_matrix!=1)].dropna(axis=0, how='all').dropna(axis=1, how='all'), annot=True, linewidths=.5)
        except:
            print ('No highly correlated features found')

    # Find index of feature columns with correlation greater than cut_off
    to_drop = [column for column in upper.columns if any(upper[column] > cut_off)]
    to_drop = [column for column in to_drop if column not in exclude]
    print ('Dropped columns:', to_drop, '\n')
    df2 = df.drop(to_drop, axis=1)
    print ('Features left after correlation check: {}'.format(len(df.columns)-len(to_drop)), '\n')    
   
    print ('Not dropped columns:', list(df2.columns), '\n')
    
    # Plotting final correlations
    f, ax = plt.subplots(figsize=(15, 10))
    plt.title('Final correlations', fontsize=20)
    sns.heatmap(df2.corr(), annot=True)
    plt.show()
    
    return df2 

def plot_bin(ev, for_excel=False, pic_folder=''):
    ind = np.arange(len(ev.index)) 
    width = 0.35
    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax2 = ax1.twinx()
    p1 = ax1.bar(ind, ev['NONEVENT'], width, color=(24/254, 192/254, 196/254))
    p2 = ax1.bar(ind, ev['EVENT'], width, bottom=ev['NONEVENT'], color=(246/254, 115/254, 109/254))

    ax1.set_ylabel('Event Distribution', fontsize=15)
    ax2.set_ylabel('WOE', fontsize=15)

    plt.title(list(ev.VAR_NAME)[0], fontsize=20) 
    ax2.plot(ind, ev['WOE'], marker='o', color='blue')
    # Legend
    plt.legend((p2[0], p1[0]), ('bad', 'good'), loc='best', fontsize=10)

    #Set xticklabels
    q = list()
    for i in range(len(ev)):
        try:
            mn = str(round(ev.MIN_VALUE[i], 2))
            mx = '-' + str(round(ev.MAX_VALUE[i], 2))
        except:
            mn = str((ev.MIN_VALUE[i]))
            # mx = str((ev.MAX_VALUE[i]))
            mx = ''
        q.append(mn + mx)

    plt.xticks(ind, q, rotation='vertical')
    for tick in ax1.get_xticklabels():
        tick.set_rotation(60)
    plt.savefig(pic_folder + '{}.png'.format(ev.VAR_NAME[0]), dpi=100, bbox_inches = 'tight')
    plt.show()    
    
def plot_all_bins(iv_df, X_train, pic_folder=''):
    for i in [x.replace('WOE_','') for x in X_train.columns]:
        ev = iv_df[iv_df.VAR_NAME==i]
        ev.reset_index(inplace=True)
        plot_bin(ev, pic_folder=pic_folder)

################## To Excel Functions ######################
def iv_groups(iv_df2):
    iv_df2['WOE_group'] = np.nan
    iv_df2['WOE_group'][0] = int(1)
    for i in range(len(iv_df2)):
        if i==0: continue
        if iv_df2.WOE[i]==iv_df2.WOE[i-1] and iv_df2.VAR_NAME[i]==iv_df2.VAR_NAME[i-1]:
            iv_df2['WOE_group'][i] = iv_df2['WOE_group'][i-1]
        else:
            if iv_df2.VAR_NAME[i]==iv_df2.VAR_NAME[i-1]:
                iv_df2['WOE_group'][i] = iv_df2['WOE_group'][i-1]+1
            else: iv_df2['WOE_group'][i] = 1
    return iv_df2

def add_score(ivs, feat, y_train):
    ivs['Score'] = 0
    factor = 20/math.log(2)
    intercept = float(feat[feat.Feature=='_INTERCEPT_'].Coefficient)
    offset = 1000 - factor*math.log(y_train.value_counts()[0]/y_train.value_counts()[1])
    for i in range(len(ivs)): # int(-(coef_woe+intercept)*factor)
        ivs['Score'][i] = (-(intercept/len(feat) + (feat[feat.Feature=='WOE_' + ivs.VAR_NAME[i]].Coefficient)*ivs.WOE[i])*factor + offset/len(feat))
    return ivs

def coef_woe_columns(df3, feat):
    for i in list(feat.Feature):
        name = 'coef_'+ i
        try: df3[name] = float(feat[feat.Feature==i].Coefficient) * df3[i]
        except: pass
    return df3

def create_gini_stability(df, clf_lr, X_train, date_name='requested_month_year'):
    df3 = df.copy()
    months = sorted(list(df3[date_name].drop_duplicates()))
    Ginis = pd.DataFrame()
    Ginis['Months'] = df3[date_name].drop_duplicates().sort_values().reset_index(drop=True)
    Ginis['Gini'] = 1

    q = list()
    goods = list()
    bads = list()
    for i in range(len(months)):
        try:
            q.append(round(2 * roc_auc_score(df3[df3[date_name] == months[i]].target.astype(float), clf_lr.predict_proba(df3[df3[date_name] == months[i]][list(X_train.columns)])[:,1]) - 1, 3))
        except Exception as e:
            print(e)
            q.append("Can't calculate, only 1 class appears!")
        try:
            goods.append(df3[df3.requested_month_year==months[i]].target.value_counts()[0])            
        except:
            print ('No `Good`  events in one of the month {}!'.format(months[i]))
            goods.append('--')            
        try:
            bads.append(df3[df3[date_name] == months[i]].target.value_counts()[1])
        except:
            print ('No `Bad` events in one of the month {}!'.format(months[i]))
            bads.append('--')

    Ginis['Goods'] = goods
    Ginis['Bads'] = bads

    Ginis['Gini'] = q
    Ginis.set_index('Months', inplace=True)
    
    return Ginis

################## Graphs Gini and Score To Excel ######################
def gini_stability_chart(Ginis, pic_name='gini_stability', pic_folder=''):
    ind = np.arange(len(Ginis.index)) 
    width = 0.6
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()
    p1 = ax1.bar(ind, Ginis['Goods'], width)
    p2 = ax1.bar(ind, Ginis['Bads'], width, bottom=Ginis['Goods'])
    
    ax1.set_ylabel('Goods-Bads ratio', fontsize=15)
    ax2.set_ylabel('Gini', fontsize=15)

    plt.title('Gini Stability', fontsize=15) 
    plt.xticks(ind, Ginis.index)
    ax2.plot(ind, Ginis['Gini'], marker='o', color='red')
    plt.ylim([0, 1])
    plt.legend((p2[0], p1[0]), ('bad', 'good'), loc='best', fontsize=10)
    plt.savefig(pic_folder + pic_name + '.png', dpi=100)
    plt.show()
    
def score_stability_graph(table, pic_name='score_distribution', pic_folder=''):
    ax = table.div(table.sum(1)/100,0).plot(kind='bar', stacked=True, figsize=(10, 7), title='Score Distribution Stability')
    plt.savefig(pic_folder + pic_name + '.png', dpi=100)
    plt.show()
    
def ivs_to_excel(iv_d, X_test):
    iv_df = iv_d.copy()
    ivs = iv_df.loc[iv_df.VAR_NAME.isin([w.replace('WOE_', '') for w in X_test.columns])][['VAR_NAME', 'MIN_VALUE', 'MAX_VALUE', 'WOE', 'COUNT']].reset_index().drop('index',axis=1)
    ivs = ivs.fillna('_MISSING_')
    ivs['Variable range'] = np.nan
    for i in range(len(ivs)):
    # For categorical
        if ivs.MIN_VALUE[i]==ivs.MAX_VALUE[i]:
            ivs['Variable range'][i] = ivs.MIN_VALUE[i]
        # For `<` parts
        elif i==0 or ivs.VAR_NAME[i]!=ivs.VAR_NAME[i-1]:
            ivs['Variable range'][i] = '< ' + str(ivs.MIN_VALUE[i+1])
        # For `>` parts
        elif i==len(ivs)-1 or ivs.VAR_NAME[i]!=ivs.VAR_NAME[i+1] or str(ivs.MIN_VALUE[i+1])=='_MISSING_':
            ivs['Variable range'][i] = '>= ' + str(ivs.MIN_VALUE[i])
        # For middle parts
        else:
            ivs['Variable range'][i] = '[ ' + str(ivs.MIN_VALUE[i]) +'; ' + str(ivs.MIN_VALUE[i+1])+' )'
    return ivs

def population_stability(df3, ivs, feat, month_num=0, pic_folder='', date_name='date_requested'):
    # Calculate PSI
    features_of_model = [i for i in list(feat.Feature) if '_INTERCEPT_' not in i]

    try:
        from psi import calculate_psi
    except:
        from autoscoring.psi import calculate_psi
    months = sorted(list(df3.requested_month_year.drop_duplicates()))
    psi = pd.DataFrame()
    psi['Months'] = df3.requested_month_year.drop_duplicates().sort_values().reset_index(drop=True)
    for j in features_of_model:
        psi[j] = 1
        q = list()
        for i in range(len(months)):
            q.append(calculate_psi(df3[df3.requested_month_year==months[i]][j], df3[df3.requested_month_year==months[month_num]][j]))
        psi[j] = q
        q = list()
    psi.set_index('Months', inplace=True)

    # Make Plots
    for col in features_of_model:
        table = pd.pivot_table(
            df3,
            index=['requested_month_year'],
            columns=[col],
            values=['credit_id'],
            aggfunc='count').fillna(0)

        q = ivs[ivs.VAR_NAME==col.replace('WOE_','')]    
        slovar = dict(zip(list(q.WOE), list(q['Variable range'])))    
        table = table.credit_id
        table.columns = table.columns.map(slovar)
        ax1 = table.div(table.sum(1)/100,0).plot(kind='bar', colormap='rainbow', stacked=True, figsize=(12, 8))
        ax2 = ax1.twinx()
        ax1.set_ylabel('Value percent', fontsize=15)
        plt.title(table.columns.names[0]+' '+' - '+'Score Distribution Stability', fontsize=15) 
        plt.plot([0.2]*len(table.index), color='red')
        ax2.plot(list(psi[col]), color='red', marker='o',markersize=10, linewidth=4)
        ax2.set_ylabel('PSI', fontsize=15)

        plt.ylim([0, 0.3])
        plt.savefig(pic_folder + 'Stability_of_{}.png'.format(table.columns.names[0]), dpi=100, bbox_inches = 'tight')
        plt.show()
    return features_of_model

def save_all(filename, X_train, X_test, clf_lr, iv_df_RI, IV, iv_df, df3, else_df, cols_to_drop):
    # Make directory
    import os
    import joblib
    direc = os.getcwd()
    path = direc+"\\" + filename
    if not os.path.exists(path):
        os.makedirs(path)
    # Dump data 
    print ('Saving X_train...')
    joblib.dump(X_train, path+'\\'+'X_train_' + filename + '.pickle')
    print ('Saving X_test...')
    joblib.dump(X_test, path+'\\'+'X_test_' + filename + '.pickle')
    print ('Saving clf_lr...')
    joblib.dump(clf_lr, path+'\\'+'clf_lr_' + filename + '.pickle')
    try:
        print ('Saving iv_df_RI...')
        joblib.dump(iv_df_RI, path+'\\'+'iv_df_RI_' + filename + '.pickle')
    except: pass
    print ('Saving IV...')
    joblib.dump(IV, path+'\\'+'IV_' + filename + '.pickle')
    print ('Saving iv_df...')
    joblib.dump(iv_df, path+'\\'+'iv_df_' + filename + '.pickle')
    print ('Saving df3...')
    joblib.dump(df3, path+'\\'+'df3_' + filename + '.pickle')
    print ('Saving else_df...')
    joblib.dump(else_df, path+'\\'+'else_df_' + filename + '.pickle')
    print ('Saving cols_to_drop...')
    joblib.dump(cols_to_drop, path+'\\'+'cols_to_drop_' + filename + '.pickle')
    print ('All Data Saved!')
    
def load_all(filename):
    # Make directory
    import os
    import joblib
    direc = os.getcwd()
    path = direc+"\\" + filename
    if not os.path.exists(path):
        print ("Can't find the Path!")
    else:
        # Load data 
        print ('Loading X_train...')
        X_train = joblib.load(path+'\\'+'X_train_' + filename + '.pickle')
        print ('Loading X_test...')
        X_test = joblib.load(path+'\\'+'X_test_' + filename + '.pickle')
        print ('Loading clf_lr...')
        clf_lr = joblib.load(path+'\\'+'clf_lr_' + filename + '.pickle')
        print ('Loading iv_df_RI...')
        iv_df_RI = joblib.load(path+'\\'+'iv_df_RI_' + filename + '.pickle')
        print ('Loading IV...')
        IV = joblib.load(path+'\\'+'IV_' + filename + '.pickle')
        print ('Loading iv_df...')
        iv_df = joblib.load(path+'\\'+'iv_df_' + filename + '.pickle')
        print ('Loading df3...')
        df3 = joblib.load(path+'\\'+'df3_' + filename + '.pickle')
        print ('Loading else_df...')
        else_df = joblib.load(path+'\\'+'else_df_' + filename + '.pickle')
        print ('Loading cols_to_drop...')
        cols_to_drop = joblib.load(path+'\\'+'cols_to_drop_' + filename + '.pickle')
        print ('All Data Loaded!')
        return X_train, X_test, clf_lr, iv_df_RI, IV, iv_df, df3, else_df, cols_to_drop

def RepresentsFloat(s):
    try: 
        float(s)
        return True
    except ValueError:
        return False

def generate_DSL(ivs: pd.DataFrame, feat: pd.DataFrame, nan_to_woe: str='max', else_to_woe: str='max') -> str:
    '''
    ivs: таблица для биннинга (iv_df)
    feat: pd.DataFrame, в котором указаны переменные, вошедшие в модель и их коэфф
    nan_to_woe: в какую woe-категорию определять NaN значения, если это не указано явно
        max - максимальное WOE
        max_cat - WOE самой крупной группы
        min - минимальное WOE
        zero - значение WOE становится 0
    else_to_woe: в какую woe-категорию определять иные (Else) значения, если это не указано явно
        max - максимальное WOE
        max_cat - WOE самой крупной группы
        min - минимальное WOE
        zero - значение WOE становится 0

    '''
    iv_dsl = ivs.copy()
    #select num and categorical
    iv_dsl['type']=None
    for i in list(iv_dsl.VAR_NAME.drop_duplicates()):
        if type(iv_dsl[iv_dsl.VAR_NAME==i].MAX_VALUE.any())==float or type(iv_dsl[iv_dsl.VAR_NAME==i].MAX_VALUE.any())==int \
            or type(iv_dsl[iv_dsl.VAR_NAME==i].MAX_VALUE.iloc[0]) in [int, float] \
            or RepresentsFloat(iv_dsl[iv_dsl.VAR_NAME==i].MAX_VALUE.iloc[0]):
            iv_dsl.loc[iv_dsl['VAR_NAME'] == i, ['type']] = 'Numeric'
        else:
            iv_dsl.loc[iv_dsl['VAR_NAME'] == i, ['type']] = 'Categorical'

    cat_vars = list(iv_dsl[iv_dsl.type=='Categorical'].VAR_NAME.drop_duplicates())
    num_vars = list(iv_dsl[iv_dsl.type=='Numeric'].VAR_NAME.drop_duplicates())
    DSL = '''
    package scoring


    spec(scoring) {
        initScore(%s)

        def params = params as Scoring_Name 
    ''' %(feat[feat.Feature=='_INTERCEPT_'].Coefficient.values[0])

    nums = ''
    for i in num_vars:
    # Названия и коэффициент
        nums+='''
        rule("%s") {
            multiplier(%s)
            value(%s) {
    '''%(i, feat[feat.Feature=='WOE_'+i].Coefficient.values[0], i)

        #VALUES
        q = iv_dsl[iv_dsl.VAR_NAME==i].reset_index(drop=True)
        s = ''

        # Missing handling
        if len(q[q.MAX_VALUE=='_MISSING_'].MAX_VALUE.values)==1:
            s+='''            when { missing(x) } then %s
    '''%q[q.MAX_VALUE=='_MISSING_'].WOE.values[0]
        elif nan_to_woe == 'max':
            s+='''            when { missing(x) } then %s
    '''%q.WOE.max()
        elif nan_to_woe == 'min':
            s+='''            when { missing(x) } then %s
    '''%q.WOE.min()
        elif nan_to_woe == 'zero':
            s+='''            when { missing(x) } then 0
    '''
        elif nan_to_woe == 'max_cat':
            s+='''            when { missing(x) } then %s
    '''%q[q['COUNT'] == q['COUNT'].max()].WOE.values[0]
        else:
            s+='''            when { missing(x) } then %s
    '''%q[q['COUNT'] == q['COUNT'].max()].WOE.values[0]


        # Normal values
        for j in range((len(q[q.MIN_VALUE!='_MISSING_']))):
            if j==0:
                s+='''            when { lowerThan(%s) } then %s
    '''%(q.MIN_VALUE[1], q.WOE[0])

            elif j==len((q[q.MIN_VALUE!='_MISSING_']))-1:
                s+='''            when { moreThanOrEq(%s) } then %s
    '''% (q.MIN_VALUE[j], q.WOE[j])
            else:
                s+='''            when { between(%s, %s) } then %s
    '''% (q.MIN_VALUE[j], q.MIN_VALUE[j+1], q.WOE[j])
        nums+=s
        nums+='''        }
        }
    '''
    if len(cat_vars)!=0:
        cats = ''
        for i in cat_vars:
        # Названия и коэффициент
            cats+='''
            rule("%s") {
                multiplier(%s)
                value(%s) {
        '''%(i, feat[feat.Feature=='WOE_'+i].Coefficient.values[0], i)

            #VALUES
            q = iv_dsl[iv_dsl.VAR_NAME==i].reset_index(drop=True)
            s = ''
            # Missing handling
            try:
                if len(q[q.MAX_VALUE.str.contains('_MISSING_')].MAX_VALUE.values)==1:
                    s+='''            when { missing(x) } then %s
            '''%q[q.MAX_VALUE.str.contains('_MISSING_')].WOE.values[0]
                elif nan_to_woe == 'max':
                    s+='''            when { missing(x) } then %s
            '''%q.WOE.max()
                elif nan_to_woe == 'min':
                    s+='''            when { missing(x) } then %s
            '''%q.WOE.min()
                elif nan_to_woe == 'zero':
                    s+='''            when { missing(x) } then 0
            '''
                elif nan_to_woe == 'max_cat':
                    s+='''            when { missing(x) } then %s
            '''%q[q['COUNT'] == q['COUNT'].max()].WOE.values[0]
                else:
                    s+='''            when { missing(x) } then %s
            '''%q[q['COUNT'] == q['COUNT'].max()].WOE.values[0]

            except:
                print(q)
                print(q.MAX_VALUE)
                print(i)
                raise

            # Normal values
            ######################################################
            for j in range(len(q)): #[(q.MIN_VALUE!='_MISSING_') & (q.MIN_VALUE!='_ELSE_')]
                # Много в одной группе
                if q.MIN_VALUE[j].find(' | ')!=-1:
                    m ='            when { [ '
                    split_arr = q.MIN_VALUE[j].split(' | ')
                    for k in range(len(split_arr)):
                        if split_arr[k]!='_ELSE_' and split_arr[k]!='_MISSING_':
                            m+='''"%s", '''%(split_arr[k])
                    # Удаляем лишнюю запятую
                    if m[-2]==',':
                        m = m[:-2]
                    m+='''].contains(x) } then %s
        '''%(q.WOE[j])
                    s+=m
                # Одиночные    
                else:
                    if q.MIN_VALUE[j]!='_ELSE_' and q.MIN_VALUE[j]!='_MISSING_':
                        s+='''            when { [ "%s"].contains(x) } then %s
        '''% (q.MIN_VALUE[j], q.WOE[j])



            ######################################################        
            # ELSE group
            if len(q[q.MAX_VALUE.str.contains('_ELSE_')].MAX_VALUE.values)==1:
                s+='''            otherwise(%s)
        '''%q[q.MAX_VALUE.str.contains('_ELSE_')].WOE.values[0]
            elif else_to_woe == 'max':
                s+='''            otherwise(%s)
        '''%q.WOE.max()
            elif else_to_woe == 'min':
                s+='''            otherwise(%s)
        '''%q.WOE.min()
            elif else_to_woe == 'zero':
                s+='''            otherwise(0)
        '''
            elif else_to_woe == 'max_cat':
                s+='''            otherwise(%s)
        '''%q[q['COUNT'] == q['COUNT'].max()].WOE.values[0]
            else:
                s+='''            otherwise(%s)
        '''%q[q['COUNT'] == q['COUNT'].max()].WOE.values[0]
   
            cats+=s
            cats+='''            }
            }'''  
    DSL+=nums
    try:DSL+=cats
    except: pass
    DSL+='''
        result {
            new ScoringResult((int) Math.round((1.0 / (1.0 + Math.exp(-1 * score.doubleValue())) * 1000)))
        }
    }
    '''
    return DSL

def generate_SQL(ivs: pd.DataFrame, feat: pd.DataFrame, nan_to_woe: str='max', else_to_woe: str='max') -> str:
    '''
    ivs: таблица для биннинга (iv_df)
    feat: pd.DataFrame, в котором указаны переменные, вошедшие в модель и их коэфф
    nan_to_woe: в какую woe-категорию определять NaN значения, если это не указано явно
        max - максимальное WOE
        max_cat - WOE самой крупной группы
        min - минимальное WOE
        zero - значение WOE становится 0
    else_to_woe: в какую woe-категорию определять иные (Else) значения, если это не указано явно
        max - максимальное WOE
        max_cat - WOE самой крупной группы
        min - минимальное WOE
        zero - значение WOE становится 0

    '''
    iv_sql = ivs.copy()
    #select num and categorical
    iv_sql['type']=None
    for i in list(iv_sql.VAR_NAME.drop_duplicates()):
        if type(iv_sql[iv_sql.VAR_NAME==i].MAX_VALUE.any())==float or type(iv_sql[iv_sql.VAR_NAME==i].MAX_VALUE.any())==int \
            or type(iv_sql[iv_sql.VAR_NAME==i].MAX_VALUE.iloc[0]) in [int, float] \
            or RepresentsFloat(iv_sql[iv_sql.VAR_NAME==i].MAX_VALUE.iloc[0]):
            iv_sql.loc[iv_sql['VAR_NAME'] == i, ['type']] = 'Numeric'
        else:
            iv_sql.loc[iv_sql['VAR_NAME'] == i, ['type']] = 'Categorical'

    cat_vars = list(iv_sql[iv_sql.type=='Categorical'].VAR_NAME.drop_duplicates())
    num_vars = list(iv_sql[iv_sql.type=='Numeric'].VAR_NAME.drop_duplicates())

    sql = '''SELECT sc.*,
        round((1 / (1 + exp((-1)*({}
    '''.format(abs(feat[feat.Feature=='_INTERCEPT_'].Coefficient.values[0]))
    for i in range(len(feat)):
        if feat.Feature[i]=='_INTERCEPT_': pass
        else:
            sql+='''+ -{}*{} '''.format(feat.Coefficient[i], feat.Feature[i])

    sql+=''')))) * 1000) as Scoring
    FROM
        (SELECT t.*,

    '''

    nums = ''
    s=''
    for i in num_vars:
        # Start
        q = iv_sql[iv_sql.VAR_NAME==i].reset_index(drop=True)
        nums = ''

        # Missing handling
        if len(q[q.MAX_VALUE=='_MISSING_'].MAX_VALUE.values)==1:
            s+='''            case when {} is null then {}  
    '''.format(i, q[q.MAX_VALUE=='_MISSING_'].WOE.values[0])
        elif nan_to_woe == 'max':
            s+='''            case when {} is null then {}  
    '''.format(i, q.WOE.max())
        elif nan_to_woe == 'min':
                s+='''            case when {} is null then {}  
    '''.format(i, q.WOE.min())
        elif nan_to_woe == 'zero':
                s+='''            case when {} is null then {}  
    '''.format(i, 0)
        elif nan_to_woe == 'max_cat':
                s+='''            case when {} is null then {}  
    '''.format(i, q[q['COUNT'] == q['COUNT'].max()].WOE.values[0])
        else:
             s+='''            case when {} is null then {}  
    '''.format(i, q[q['COUNT'] == q['COUNT'].max()].WOE.values[0])

        # Normal values
        for j in range((len(q[q.MIN_VALUE!='_MISSING_']))):
            # <
            if j==0:
                s+='''                 when {} < {} then {}
    '''.format(i, q.MIN_VALUE[1], q.WOE[0])


            elif j==len((q[q.MIN_VALUE!='_MISSING_']))-1:
                s+='''                 when {} >= {} then {}
    '''.format(i, q.MIN_VALUE[len((q[q.MIN_VALUE!='_MISSING_']))-1], q.WOE[len((q[q.MIN_VALUE!='_MISSING_']))-1])

            else:
                s+='''                 when {} >= {} and {} < {} then {}
    '''.format(i, q.MIN_VALUE[j], i, q.MIN_VALUE[j+1],q.WOE[j])
        s+='''                 end as WOE_{}, 

    '''.format(i)
        nums+=s
    sql+=nums

    if len(cat_vars)!=0:
        cats = ''
        s=''
        for i in cat_vars:
            # Start
            q = iv_sql[iv_sql.VAR_NAME==i].reset_index(drop=True)
            cats = ''

            # Missing handling
            if len(q[q.MAX_VALUE.str.contains('_MISSING_')].MAX_VALUE.values)==1:
                s+='''            case when {} is null then {}  
    '''.format(i, q[q.MAX_VALUE.str.contains('_MISSING_')].WOE.values[0])
            elif nan_to_woe == 'max':
                s+='''            case when {} is null then {}  
    '''.format(i, q.WOE.max())
            elif nan_to_woe == 'min':
                    s+='''            case when {} is null then {}  
    '''.format(i, q.WOE.min())
            elif nan_to_woe == 'zero':
                    s+='''            case when {} is null then {}  
    '''.format(i, 0)
            elif nan_to_woe == 'max_cat':
                    s+='''            case when {} is null then {}  
    '''.format(i, q[q['COUNT'] == q['COUNT'].max()].WOE.values[0])
            else:
                s+='''            case when {} is null then {}  
    '''.format(i, q[q['COUNT'] == q['COUNT'].max()].WOE.values[0])

            # Normal values

            for j in range(len(q)): 
                        # Много в одной группе
                        if q.MIN_VALUE[j].find(' | ')!=-1:
                            m ='                 when {} in ('.format(i)
                            split_arr = q.MIN_VALUE[j].split(' | ')
                            for k in range(len(split_arr)):
                                if split_arr[k]!='_ELSE_' and split_arr[k]!='_MISSING_':
                                    m+='''"{}", '''.format(split_arr[k])
                            # Удаляем лишнюю запятую
                            if m[-2]==',':
                                m = m[:-2]
                            m+=''') then {}
    '''.format(q.WOE[j])
                            s+=m
                        # Одиночные    
                        else:
                            if q.MIN_VALUE[j] != '_ELSE_' and q.MIN_VALUE[j] != '_MISSING_':
                                s+='''                 when {} in ("{}") then {}
    '''.format(i, q.MIN_VALUE[j], q.WOE[j])

            # ELSE handling
            if len(q[q.MAX_VALUE.str.contains('_ELSE_')].MAX_VALUE.values)==1:
                s+='''                 else {}  
    '''.format(q[q.MAX_VALUE.str.contains('_ELSE_')].WOE.values[0])
            elif else_to_woe == 'max':
                s+='''                 else {}  
    '''.format(i, q.WOE.max())
            elif else_to_woe == 'min':
                s+='''                 else {}  
    '''.format(i, q.WOE.min())
            elif else_to_woe == 'zero':
                s+='''                 else {}  
    '''.format(i, 0)
            elif else_to_woe == 'max_cat':
                s+='''                 else {}  
    '''.format(i, q[q['COUNT'] == q['COUNT'].max()].WOE.values[0])
            else:
                s+='''                 else {}  
    '''.format(i, q[q['COUNT'] == q['COUNT'].max()].WOE.values[0])
            

            # END
            s+='''                 end as WOE_{}, 

    '''.format(i)
            cats+=s
        sql+=cats
    # Delete comma
    if sql[-8]==',':
        sql = sql[:-8]
    sql+='''
        FROM'''

    return sql

def export_to_excel(DSL, SQL, X_train, X_test, y_train, y_test, y, df3, iv_df, ivs,
                    Ginis, table, scores, feat, features_of_model, clf_lr, 
                    gini_by_vars=None, df_gini_months=None, PV=None,
                    X_out=None, y_out=None, name='Scoring',
                    pic_folder='', target_description='', model_description=''):
    #WRITING
    writer = pd.ExcelWriter('{}.xlsx'.format(name), engine='xlsxwriter')

    workbook  = writer.book
    worksheet = workbook.add_worksheet('Sample information')
    bold = workbook.add_format({'bold': True})
    percent_fmt = workbook.add_format({'num_format': '0.00%'})

    worksheet.set_column('A:A', 20)
    worksheet.set_column('B:B', 15)
    worksheet.set_column('C:C', 10)

    # Sample
    worksheet.write('A2', 'Sample conditions', bold)
    worksheet.write('A3', 1)
    worksheet.write('A4', 2)
    worksheet.write('A5', 3)
    worksheet.write('A6', 4)

    # Model
    worksheet.write('A8', 'Model development', bold)

    worksheet.write('A9', 1)
    #labels
    worksheet.write('C8', 'Bads')
    worksheet.write('D8', 'Goods')
    worksheet.write('B9', 'Train')
    worksheet.write('B10', 'Valid')
    worksheet.write('B11', 'Out')
    worksheet.write('B12', 'Total')

    # goods and bads
    worksheet.write('C9', y_train.value_counts()[1])
    worksheet.write('C10', y_test.value_counts()[1])
    worksheet.write('D9', y_train.value_counts()[0])
    worksheet.write('D10', y_test.value_counts()[0])
    worksheet.write('C12', y.value_counts()[1])
    worksheet.write('D12', y.value_counts()[0])
    if y_out is not None:
        worksheet.write('C11', y_out.value_counts()[1])
        worksheet.write('D11', y_out.value_counts()[0])

    # NPL
    worksheet.write('A14', 2)
    worksheet.write('B14', 'NPL')
    worksheet.write('C14', (y.value_counts()[1] / (y.value_counts()[1] + y.value_counts()[0])), percent_fmt)

    worksheet.write('A17', 3)
    worksheet.write('C16', 'Gini')
    worksheet.write('B17', 'Train')
    worksheet.write('B18', 'Valid')
    worksheet.write('B19', 'CV Scores')
    try: worksheet.write('C19', str([round(sc, 2) for sc in scores]))
    except: print ('Error! - Cross-Validation')
    try:    
        worksheet.write('C17', round(2*roc_auc_score(y_train, clf_lr.predict_proba(X_train)[:,1]) - 1, 3))
        worksheet.write('C18', round(2*roc_auc_score(y_test, clf_lr.predict_proba(X_test)[:,1]) - 1, 3))
    except:print ('Error! - Gini Train\Test Calculation')
    if X_out is not None and y_out is not None:
        try:
            worksheet.write('B20', 'Out')
            worksheet.write('C20', round(2*roc_auc_score(y_out, clf_lr.predict_proba(X_out)[:,1]) - 1, 3))
        except:
            print('Error! - Gini Out calcualtion')

    worksheet.write('A23', 'Описание таргета')
    worksheet.write('B23', target_description)
    worksheet.write('A24', 'Описание модели')
    worksheet.write('B24', model_description)
    worksheet.write('A25', 'Времменые рамки')
    start_date = df3['date_requested'].min().strftime('%Y-%m-%d')
    end_date = df3['date_requested'].max().strftime('%Y-%m-%d')
    worksheet.write('B25', f'from {start_date} to {end_date}')

    # Sheet for feature description
    feat_names = pd.DataFrame(feat['Feature'].apply(lambda x: x.replace('WOE_', ''))[:-1], columns=['Feature'])
    feat_names.to_excel(writer, sheet_name='Feat description', index=False)
    worksheet2 = writer.sheets['Feat description']
    worksheet2.set_column('A:A', 35)

    # Regression coefs
    feat.to_excel(writer, sheet_name='Regression coefficients', index=False)
    worksheet2 = writer.sheets['Regression coefficients']
    worksheet2.set_column('A:A', 35)
    worksheet2.set_column('B:B', 25)

    # Gini by var
    if gini_by_vars is not None:
        gini_by_vars.to_excel(writer, sheet_name='Gini by var', index=False)
        worksheet2 = writer.sheets['Gini by var']
        worksheet2.set_column('A:A', 35)

    if df_gini_months is not None:
        df_gini_months.to_excel(writer, sheet_name='Month ginis by var', index=False)
        worksheet2 = writer.sheets['Month ginis by var']
        worksheet2.set_column('A:A', 35)

    # P-value
    if PV is not None:
        PV.to_excel(writer, sheet_name='P-values')
        worksheet2 = writer.sheets['P-values']
        worksheet2.set_column('A:A', 35)
        worksheet2.set_column('B:B', 15)

    #WOE
    ivs[['VAR_NAME', 'Variable range', 'WOE', 'COUNT', 'WOE_group']].to_excel(writer, sheet_name='WOE', index=False)
    worksheet3 = writer.sheets['WOE']
    worksheet3.set_column('A:A', 50)
    worksheet3.set_column('B:B', 60)
    worksheet3.set_column('C:C', 30)
    worksheet3.set_column('D:D', 20)
    worksheet3.set_column('E:E', 12)
    try:
        for num, i in enumerate([x.replace('WOE_','') for x in X_train.columns]):
            ev = iv_df[iv_df.VAR_NAME==i]
            ev.reset_index(inplace=True)
            worksheet3.insert_image('G{}'.format(num*34+1), pic_folder + '{}.png'.format(i))
    except: print ('Error! - WOE Plots')

    df3.to_excel(writer, sheet_name='Data', index=False)

    try:
        table.to_excel(writer, sheet_name='Scores by buckets', header = True, index = True)
        worksheet4 = writer.sheets['Scores by buckets']
        worksheet4.set_column('A:A', 20)
        worksheet4.insert_image('J1', pic_folder + 'score_distribution.png')
    except: print ('Error! - Score Distribution')
    try:
        Ginis.to_excel(writer, sheet_name='Gini distribution', header = True, index = True)
        worksheet5 = writer.sheets['Gini distribution']
        worksheet5.insert_image('E1', pic_folder + 'gini_stability.png')
    except: print ('Error! - Gini Stability')
    worksheet6 = workbook.add_worksheet('Variables Stability')

    try:
        for num, i in enumerate(features_of_model): 
            worksheet6.insert_image('A{}'.format(num*34+1), pic_folder + 'Stability_of_{}.png'.format(i))
    except: print ('Error! - Variables Stability')
    try:
        worksheet7 = workbook.add_worksheet('DSL')
        worksheet7.set_column('A:A', 270)
        worksheet7.write('A1', DSL)
    except: print ('Error! - DSL')
    try:
        worksheet8 = workbook.add_worksheet('SQL')
        worksheet8.set_column('A:A', 270)
        worksheet8.write('A1', SQL)
    except: print ('Error! - SQL')
    writer.save()
    print ('Exported!')

def splitDataFrameList(df, target_column, separator): 
    row_accumulator = []
    old_columns = df.columns
    def splitListToRows(row, separator):
        try:
            split_row = row[target_column].split(separator)
        except:
            split_row = [row[target_column]]
        for s in split_row:
            new_row = row.to_dict()
            new_row[target_column] = s
            row_accumulator.append(new_row)

    df.apply(splitListToRows, axis=1, args = (separator, ))
    new_df = pd.DataFrame(row_accumulator)
    new_df = new_df[old_columns]
    for i in range(len(new_df)):
        if type(new_df.MIN_VALUE[i])==str:
            new_df.MAX_VALUE[i] = new_df.MIN_VALUE[i]
    return new_df