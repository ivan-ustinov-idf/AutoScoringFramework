import pymysql.cursors
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import scikitplot as skplt
 
from IPython.display import Markdown, display
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
#from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LassoCV, RidgeClassifierCV
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score, train_test_split, KFold, StratifiedKFold, GridSearchCV 
from sklearn.metrics import roc_auc_score, roc_curve, log_loss, f1_score, confusion_matrix, precision_score, recall_score, classification_report
scaler = StandardScaler()
label = LabelEncoder()
from difflib import SequenceMatcher
def import_all_modules():
    import scikitplot as skplt
    from sklearn.decomposition import TruncatedSVD
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import LabelEncoder
    import matplotlib.pyplot as plt
    from pandas import ExcelWriter
    import scipy
    scaler = StandardScaler()
    label = LabelEncoder()
    #sql
    import pymysql.cursors 
    #NLP
    import nltk
    import re
    import pymorphy2
    from nltk.tokenize import sent_tokenize, RegexpTokenizer
    from nltk.stem.snowball import RussianStemmer
    from nltk.util import ngrams
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from difflib import SequenceMatcher
    #warnings
    import warnings
    warnings.filterwarnings('ignore')
    #visualisation
    
    sns.set(style="white", color_codes=True)        
    #vectorization
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    #model 
    from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
    from sklearn.cluster import KMeans
    from sklearn.cross_validation import cross_val_score
    from sklearn.cross_validation import train_test_split
    from sklearn.cross_validation import KFold
    from sklearn.metrics import roc_auc_score, roc_curve, log_loss, f1_score, confusion_matrix, precision_score, recall_score, classification_report, accuracy_score
    #classificators
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.ensemble import VotingClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from xgboost.sklearn import XGBClassifier 
    from sklearn.svm import SVC
    import datetime
    from sklearn.externals import joblib

            
def find_col(df, word):
    for i in df.columns:
        for j in range(len(df)):
            try:
                if df[i][j].find(word) != -1:
                    print('Column: %s \n Row: %s' % (i, j))
            except:
                print ('There is no such word in dataframe')
    
def read_from_mysql(country, user, password, script):
    if country=='br':
        conn = pymysql.connect(host='192.168.64.1', port=3306,user=user,password=password, db='mysql')
    elif country=='es':
        conn = pymysql.connect(host='10.100.0.100', port=33062, user=user, password=password,db='mysql')
    elif country=='mx':
        conn = pymysql.connect(host='192.168.65.1', port=3306, user=user, password=password,db='mysql')
    elif country=='ge':
        conn = pymysql.connect(host='192.168.250.14', port=3306, user=user, password=password,db='mysql')
    elif country=='kz':
        conn = pymysql.connect(host='192.168.250.15', port=3306, user=user, password=password,db='mysql')
    elif country=='ru':
        conn = pymysql.connect(host='109.234.153.116', port=3306, user=user, password=password,db='mysql')
    elif country=='solva_ru':
        conn = pymysql.connect(host='192.168.250.50', port=3306, user=user, password=password,db='mysql')
    elif country=='solva_kz':
        conn = pymysql.connect(host='192.168.250.17', port=3306, user=user, password=password,db='mysql')
    elif country=='solva_ge':
        conn = pymysql.connect(host='192.168.250.13', port=3306, user=user, password=password,db='mysql')
    elif country=='amp_ru':
        conn = pymysql.connect(host='95.213.187.6', port=3306, user=user, password=password,db='mysql')
    else:
        print ('unknown country')
    return pd.read_sql(script, con=conn)

# function for bytes columns replacing
def bytes_to_string(x):
        try:
            return str(ord(x))
        except:
            return x
        
#Replace_not_frequents_cat_vars
def replace_not_frequent_2(df, cols, num_min=100, value_to_replace = "_ELSE_"):
        else_df = pd.DataFrame(columns=['var', 'list'])
        for i in cols:
            if i != 'date_requested' and i != 'credit_id':
                t = df[i].value_counts()
                q = list(t[t.values < num_min].index)
                if q:
                    else_df = else_df.append(pd.DataFrame([[i, q]], columns=['var', 'list']))
                df.loc[df[i].value_counts()[df[i]].values < num_min, i] = value_to_replace
        else_df = else_df.set_index('var')
        return df, else_df
    
def filling(df):
    cat_vars = df.select_dtypes(include=[object]).columns
    num_vars = df.select_dtypes(exclude=[object]).columns
    df[cat_vars] = df[cat_vars].fillna('MISSING')
    df[num_vars] = df[num_vars].fillna(-1)
    return df
            
def preprocessing(df, labl_dict=False, num_min=150, cat_vars=False):
    if labl_dict==False:
        def filling(df):
            cat_vars = df.select_dtypes(include=[object]).columns
            num_vars = df.select_dtypes(include=[np.number]).columns
            df[cat_vars] = df[cat_vars].fillna('_MISSING_')
            df[num_vars] = df[num_vars].fillna(-1)
            return df 

        def replace_not_frequent(df, cols, num_min=num_min, value_to_replace = "_ELSE_"):
                else_df = pd.DataFrame(columns=['var', 'list'])
                for i in cols:
                    if i != 'date_requested' and i != 'credit_id':
                        t = df[i].value_counts()
                        q = list(t[t.values < num_min].index)
                        if q:
                            else_df = else_df.append(pd.DataFrame([[i, q]], columns=['var', 'list']))
                        df.loc[df[i].value_counts()[df[i]].values < num_min, i] = value_to_replace
                else_df = else_df.set_index('var')
                return df, else_df

        df = filling(df)
        cat_vars = df.select_dtypes(include=[object]).columns
        df, else_df = replace_not_frequent(df, cat_vars)

        def create_label_dict(df):
            cols = list(df.select_dtypes(include=object, exclude='datetime').columns)
            end_labl_dict = dict()
            for col in cols:
                if col!='date_requested':
                    try:
                        df[col] = df[col].map(lambda x: str(x))
                        label.fit(df[col])
                        keys = label.classes_
                        values = label.transform(label.classes_)
                        labl_dict_col = dict(zip(keys, values))
                        end_labl_dict[col] = labl_dict_col
                    except:
                        print (col)
            return end_labl_dict

        end_labl_dict = create_label_dict(df)

        def map_labl_dict(df, end_labl_dict):
            df_new = df.copy()
            for col in cat_vars:
                if col!='date_requested':
                    col_labels = end_labl_dict[col]
                    df_new[col] = df_new[col].map(col_labels)
            return df_new
        

        df = map_labl_dict(df, end_labl_dict)
        def drop_single_value_column(df):
            df2 = df.copy()
            for i in df2.columns:
                if df2[i].value_counts().shape[0]==1: df2.drop(i, axis=1, inplace=True)
            return df2 
        
        df = drop_single_value_column(df)
        return df, end_labl_dict
    
    else:
        end_labl_dict = labl_dict
        
        def filling(df):
            cat_vars = list(set(df.columns) & set(list(end_labl_dict.keys())))
            df[cat_vars] = df[cat_vars].fillna('_MISSING_')
            df = df.fillna(-1)
            return df
        df = filling(df)

        for i in list(end_labl_dict.keys()):
            try:
                values = list(end_labl_dict[i])
                df[i] = df[i].map(lambda x: x if x in values else '_ELSE_')
                df[i] = df[i].map(end_labl_dict[i])
            except: print(i)
        return df
    

def plot_score(clf, X_test, y_test, X_train, y_train, feat_to_show=30, is_normalize=False, cut_off=0.5):
    #cm = confusion_matrix(pd.Series(clf.predict_proba(X_test)[:,1]).apply(lambda x: 1 if x>cut_off else 0), y_test)
    print ('ROC_AUC:  ', round(roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]), 3))
    print ('Gini Train:', round(2*roc_auc_score(y_train, clf.predict_proba(X_train)[:,1]) - 1, 3))
    print ('Gini Test:', round(2*roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]) - 1, 3))
    print ('F1_score: ', round(f1_score(y_test, clf.predict(X_test)), 3))
    print ('Log_loss: ', round(log_loss(y_test, clf.predict(X_test)), 3))
    
    print ('\n')
    print ('Classification_report: \n', classification_report(pd.Series(clf.predict_proba(X_test)[:,1]).apply(lambda x: 1 if x>cut_off else 0), y_test))
    skplt.metrics.plot_confusion_matrix(y_test, pd.Series(clf.predict_proba(X_test)[:,1]).apply(lambda x: 1 if x>cut_off else 0), title="Confusion Matrix",
                    normalize=is_normalize,figsize=(8,8),text_fontsize='large')
    try:
        display(eli5.show_weights(clf, top=20, feature_names = list(X_test.columns)))
    except:
        pass
    if type(clf)==Pipeline:
        imp = pd.DataFrame(list(zip(X_test.columns, clf.steps[1][1].feature_importances_)))
    elif type(clf)==LogisticRegressionCV or type(clf)==LogisticRegression:
        imp = pd.DataFrame(list(zip(X_test.columns, clf.coef_[0])))
    else:
        imp = pd.DataFrame(list(zip(X_test.columns, clf.feature_importances_)))
        
    imp = imp.reindex(imp[1].abs().sort_values().index).set_index(0)
    imp = imp[-feat_to_show:]
    #график_фич
    ax = imp.plot.barh(width = .6, legend = "", figsize = (12, 10))
    ax.set_title("Feature Importances", y = 1.03, fontsize = 16.)
    _ = ax.set(frame_on = False, xlabel = "", xticklabels = "", ylabel = "")
    for i, labl in enumerate(list(imp.index)):
        score = imp.loc[labl][1]
        ax.annotate('%.2f' % score, (score + (-.12 if score < 0 else .02), i - .2), fontsize = 10.5)

def printmd(string):
        display(Markdown(string))
        
def try_all_clfs(X_train, y_train, X_test, y_test, include_CatBst = True):
    import xgboost as xgb
    import lightgbm
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from catboost import CatBoostClassifier
    
    
    lr = LogisticRegressionCV(random_state=1, cv=3)
    cb = CatBoostClassifier(logging_level='Silent')
    lgm = lightgbm.LGBMClassifier()
    xgb = xgb.XGBClassifier()
    rf = RandomForestClassifier(n_estimators=400)
    gb = GradientBoostingClassifier()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    if include_CatBst:
        clf_list = lr, rf, cb, lgm, xgb, gb
    else:
        clf_list = lr, rf, lgm, xgb, gb

    max_roc = 0.5
    for clf in clf_list:
        print ('Now fitting {} .........'.format(type(clf)))
        clf.fit(X_train, y_train)
        printmd('Fitted with ROC: **{}** / **{}**'.format(round(roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]), 3), round(roc_auc_score(y_train, clf.predict_proba(X_train)[:,1]), 3)))
        if roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])>max_roc:
            q = clf
            max_roc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
   # print ('\n')
    print ('___________________________________________________________________________________________________________________________')
    printmd ('**Winner classifier:**')
    print (type(q))
    print ('\n')
    plot_score(q, X_test, y_test, X_train, y_train)

def similar_words(a, b):
    q = []
    for word1 in a.split(' '):
        ma = 0
        for word2 in b.split(' '):            
            if SequenceMatcher(None, word1, word2).ratio()>ma:
                ma = SequenceMatcher(None, word1, word2).ratio()        
        q.append(ma)    
    return np.mean(q)

def similar_sentences(a, b):
    q = []
    for i in range(len(a)):
        q.append(similar_words(str(a[i]), str(b[i])))
    return np.mean(q)

def ren_cols(df):
    for i in range(len(df.columns)):
        for j in range(len(df.columns)):
            if df.iloc[:,i].name==df.iloc[:,j].name and i!=j:
                df.columns.values[i] = df.columns.values[i]+'_1'
                df.columns.values[j] = df.columns.values[j]+'_2'

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
            best_feature = new_pval.argmin()
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
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

def delete_correlated_features(df, cut_off=0.75, exclude = []):
    # Create correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    
    # Plotting All correlations
    f, ax = plt.subplots(figsize=(15, 10))
    plt.title('All correlations', fontsize=20)
    sns.heatmap(X_train.corr(), annot=True)
    
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

def RFE_feature_selection(clf_lr, X, y):
    rfecv = RFECV(estimator=clf_lr, step=1, cv=StratifiedKFold(5), verbose=0, scoring='roc_auc')
    rfecv.fit(X, y)

    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    f, ax = plt.subplots(figsize=(14, 9))
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    mask = rfecv.get_support()
    X = X.loc[:, mask]
    return X