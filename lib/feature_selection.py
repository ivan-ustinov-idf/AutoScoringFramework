import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from itertools import combinations
from tqdm import tqdm


def bacward_elimination(n_features, X_train, y_train, _logreg=None):
    '''
    Отбираем признак обратным отбором. Проходимся по всем переменным и
    на каждом шаге убираем переменную без которой скор будет наибольшим.
    Так до тех пор, пока не останется требуемое количество переменных.

    '''

    try:
        X_train.drop(['normal_score'], axis=1, inplace=True)
    except:
        print('')

    if _logreg is None:
        _logreg = LogisticRegression(penalty = 'l1', C=1, solver='liblinear',
                                     class_weight=None, random_state=42)

    dict_gini_vars = {}
    df_gini_vars = pd.DataFrame(columns=['var_name', 'gini_train', 'gini_test', 'gini_out'])
    cols = list(X_train.columns)

    i = 0
    while len(cols) > n_features:

        # Будем отбирать признак, без которого скор Gini_train наилучший.
        worst_var = {'name': None, 'score': [0, 0, 0]}
        for c in cols:
            __vars_current = np.delete(cols, np.where(cols == col))
            _logreg.fit(X_train[__vars_current], y_train)

            predict_proba_train = _logreg.predict_proba(X_train[__vars_current])[:, 1]
            # predict_proba_test = _logreg.predict_proba(X_test[__vars_current])[:, 1]
            # predict_proba_out = _logreg.predict_proba(X_out[__vars_current])[:, 1]

            roc_train = roc_auc_score(y_train, predict_proba_train)
            # roc_test = roc_auc_score(y_test, predict_proba_test)
            # roc_out = roc_auc_score(y_out, predict_proba_out)
            Gini_train = 2*roc_train - 1
            # Gini_test = 2*roc_test - 1
            # Gini_out = 2*roc_out - 1

            if worst_var['score'][0] < Gini_train:
                worst_var['name'] = c
                worst_var['score'] = [Gini_train] #, Gini_test, Gini_out]
        
        cols = np.delete(cols, np.where(cols == worst_var['name']))

        df_gini_vars.loc[i, 'var_name'] = worst_var['name']
        df_gini_vars.loc[i, 'gini_train'] = worst_var['score'][0]
        # df_gini_vars.loc[i, 'gini_test'] = worst_var['score'][1]
        # df_gini_vars.loc[i, 'gini_out'] = worst_var['score'][2]
        i += 1

    return cols, df_gini_vars


def feature_brute_force(X_all, y_all, base_feature, new_feature, params, n_features=3):
    '''
    Перебираем всевозможные комбинации признаков для обучения.
    Имеет смысл перебирать не больше 4 признаков.
    '''
    X_train, X_test, X_out = X_all[0], X_all[1], X_all[2]
    y_train, y_test, y_out = y_all[0], y_all[1], y_all[2]
    try:
        X_train.drop(['normal_score'], axis=1, inplace=True)
    except:
        print('')

    _logreg = LogisticRegression(**params)
    best_combination = [{'vars': [], 'score': 0} for i in range(10)]  # Храним топ10 комбинаций.

    for combin in tqdm(combinations(new_feature, n_features)):
        combin = list(combin)
        _logreg.fit(X_train[base_feature + combin], y_train)
        predict_proba_train = _logreg.predict_proba(X_train[base_feature + combin])[:, 1]
        roc_train = roc_auc_score(y_train, predict_proba_train)
        Gini_train = 2*roc_train - 1

        # if best_combination['score'] < Gini_train:
        #     best_combination['vars'] = combin
        #     best_combination['score'] = Gini_train
        for i, best_comb in enumerate(best_combination):
            if best_comb['score'] < Gini_train:
                best_combination[i]['score'] = Gini_train
                best_combination[i]['vars'] = combin
                # Отсортируем, чтобы он всегда был в порядке возрастания.
                best_combination.sort(key=lambda x: x['score'])
                break

    return best_combination


def feature_exclude(X_all, y_all, vars_current, iv_df, params, sample_weight=None):
    # Смотрим что будет, если убрать один признак
    df_var_ginis = pd.DataFrame(columns=['var_name', 'gini_train', 'gini_test'])

    X_train, X_test, X_out = X_all[0], X_all[1], X_all[2]
    y_train, y_test, y_out = y_all[0], y_all[1], y_all[2]

    for i, var in enumerate(['with all'] + vars_current):
        __vars_current = np.delete(vars_current, np.where(np.array(vars_current) == var))

        _logreg = LogisticRegression(**params).fit(
            X_train[__vars_current],
            y_train,
            sample_weight=sample_weight
        )

        predict_proba_train = _logreg.predict_proba(X_train[__vars_current])[:, 1]
        predict_proba_test = _logreg.predict_proba(X_test[__vars_current])[:, 1]
        predict_proba_out = _logreg.predict_proba(X_out[__vars_current])[:, 1]

        df_var_ginis.loc[i, 'var_name'] = var
        df_var_ginis.loc[i, 'gini_train'] = 2 * roc_auc_score(y_train, predict_proba_train) - 1
        df_var_ginis.loc[i, 'gini_test'] =  2 * roc_auc_score(y_test, predict_proba_test) - 1
        df_var_ginis.loc[i, 'gini_out'] =  2 * roc_auc_score(y_out, predict_proba_out) - 1

    return df_var_ginis


def feature_include1(X_all, y_all, vars_current, iv_df, params, sample_weight=None):
    # Смотрим, что будет после добавления одного признака дополнительно.
    df_var_ginis = pd.DataFrame(columns=['var_name', 'gini_train', 'gini_test'])
    # clf = IsolationForest(n_estimators=50, max_samples=0.3, max_features=0.75, random_state=123)

    X_train, X_test, X_out = X_all[0], X_all[1], X_all[2]
    y_train, y_test, y_out = y_all[0], y_all[1], y_all[2]

    for i, var in enumerate(list(set(X_train.columns) - set(vars_current + ['normal_score']))):
        __vars_current = list(vars_current) + [var]
        
        _logreg = LogisticRegression(**params).fit(
            X_train[__vars_current],
            y_train,
            sample_weight=sample_weight
        )

        predict_proba_train = _logreg.predict_proba(X_train[__vars_current])[:, 1]
        predict_proba_test = _logreg.predict_proba(X_test[__vars_current])[:, 1]
        predict_proba_out = _logreg.predict_proba(X_out[__vars_current])[:, 1]

        df_var_ginis.loc[i, 'var_name'] = var
        df_var_ginis.loc[i, 'gini_train'] = 2 * roc_auc_score(y_train, predict_proba_train) - 1
        df_var_ginis.loc[i, 'gini_test'] =  2 * roc_auc_score(y_test, predict_proba_test) - 1
        df_var_ginis.loc[i, 'gini_out'] =  2 * roc_auc_score(y_out, predict_proba_out) - 1
        IV_vars = map(lambda x: str(round(x, 4)), iv_df[iv_df['VAR_NAME'].isin([var.replace('WOE_', '')])]['IV'].unique())
        df_var_ginis.loc[i, 'IV'] = ', '.join(IV_vars)

    return df_var_ginis


def feature_include2(X_all, y_all, vars_current, iv_df, params, sample_weight=None):
    # Смотрим, что будет после добавления двух признаков дополнительно.
    df_var_ginis = pd.DataFrame(columns=['var_name', 'gini_train', 'gini_test'])
    # clf = IsolationForest(n_estimators=50, max_samples=0.3, max_features=0.75, random_state=123)


    if len(X_all) == 3:
        X_train, X_test, X_out = X_all[0], X_all[1], X_all[2]
        y_train, y_test, y_out = y_all[0], y_all[1], y_all[2]
        X_train_check, y_train_check = X_train, y_train
    else:
        X_train, X_train_check, X_test, X_out = X_all[0], X_all[1], X_all[2], X_all[3]
        y_train, y_train_check, y_test, y_out = y_all[0], y_all[1], y_all[2], y_all[3]

    new_vars = list(set(X_train.columns) - set(vars_current + ['normal_score']))
    for i, var in enumerate(new_vars):
        for j, var2 in enumerate(new_vars[i:]):
            if var == var2:
                continue
            __vars_current = list(vars_current) + [var, var2]
            
            _logreg = LogisticRegression(**params).fit(
                X_train[__vars_current],
                y_train,
                sample_weight=sample_weight
            )

            predict_proba_train = _logreg.predict_proba(X_train_check[__vars_current])[:, 1]
            predict_proba_test = _logreg.predict_proba(X_test[__vars_current])[:, 1]
            predict_proba_out = _logreg.predict_proba(X_out[__vars_current])[:, 1]

            df_var_ginis.loc[i+j, 'var_name'] = ', '.join([var, var2])
            df_var_ginis.loc[i+j, 'gini_train'] = 2 * roc_auc_score(y_train_check, predict_proba_train) - 1
            df_var_ginis.loc[i+j, 'gini_test'] =  2 * roc_auc_score(y_test, predict_proba_test) - 1
            df_var_ginis.loc[i+j, 'gini_out'] =  2 * roc_auc_score(y_out, predict_proba_out) - 1
            IV_vars = map(lambda x: str(round(x, 4)), iv_df[iv_df['VAR_NAME'].isin([var.replace('WOE_', ''), var2.replace('WOE_', '')])]['IV'].unique())
            df_var_ginis.loc[i+j, 'IV'] = ', '.join(IV_vars)

    return df_var_ginis