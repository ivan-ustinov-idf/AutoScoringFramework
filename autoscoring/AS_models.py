import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats

from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from copy import deepcopy

from catboost import Pool
from catboost import CatBoostClassifier, EShapCalcType, EFeaturesSelectionAlgorithm

from autoscoring.lineartree import LinearTreeClassifier

from .AS_2 import var_name_original


def construct_df3_bbox(vars_woe, model, X_train, X_test, df_train, df_test, X_out=None, df_out=None,
                            date_name='date_requested', target='target', intervals='month'):
    
    vars = var_name_original(vars_woe)

    X_full = pd.concat([X_train, X_test, X_out])[vars_woe]
    df2 = pd.concat([df_train.reset_index(drop=True), df_test.reset_index(drop=True),
                        df_out.reset_index(drop=True)])[['credit_id', date_name, target] + vars]

    df2 = pd.concat([df2.reset_index(drop=True), X_full.reset_index(drop=True)], axis=1)

    df3 = df2.copy()
    # df3['requested_month_year'] = df3[date_name].map(lambda x: str(x)[:7])
    if intervals == 'month':
        df3['requested_month_year'] = df3[date_name].map(lambda x: str(x)[:7])
    elif intervals == 'week':
        df3['requested_month_year'] = df3[date_name].dt.strftime('%Y-%U')
    else:
        df3['requested_month_year'] = df3[date_name].map(lambda x: str(x)[:7])

    df3[target] = df3[target].astype(float)
    df3['PD'] = model.predict_proba(df3[vars_woe])[:,1] 
    df3['Score'] = 1000 - round(1000*(df3['PD']))

    # Score buckets
    df3['Score_bucket'] = df3.Score.map(lambda x: str(int((x//100)*100))+'-'+ str(int((x//100)+1)*100))

    return df3


def feature_exclude_model(X_all, y_all, vars_current, vars_to_exclude, iv_df, base_model):

    # Смотрим что будет, если убрать один признак
    df_var_ginis = pd.DataFrame(columns=['var_name', 'gini_train', 'gini_test'])

    X_train, X_test, X_out = X_all[0], X_all[1], X_all[2]
    y_train, y_test, y_out = y_all[0], y_all[1], y_all[2]

    for i, var in enumerate(['with all'] + vars_to_exclude):
        __vars_current = np.delete(vars_current, np.where(np.array(vars_current) == var))

        model = deepcopy(base_model)
        model.fit(X_train[__vars_current], y_train)

        predict_proba_train = model.predict_proba(X_train[__vars_current])[:, 1]
        predict_proba_test = model.predict_proba(X_test[__vars_current])[:, 1]
        predict_proba_out = model.predict_proba(X_out[__vars_current])[:, 1]

        df_var_ginis.loc[i, 'var_name'] = var
        df_var_ginis.loc[i, 'gini_train'] = 2 * roc_auc_score(y_train, predict_proba_train) - 1
        df_var_ginis.loc[i, 'gini_test'] =  2 * roc_auc_score(y_test, predict_proba_test) - 1
        df_var_ginis.loc[i, 'gini_out'] =  2 * roc_auc_score(y_out, predict_proba_out) - 1

        IV_vars = map(lambda x: str(round(x, 4)), iv_df[iv_df['VAR_NAME'].isin([var.replace('WOE_', '')])]['IV'].unique())
        df_var_ginis.loc[i, 'IV'] = ', '.join(IV_vars)

    return df_var_ginis


def gini_univariate_selection(gini_by_vars, cut_off: float=0.6):
    '''
    Удаляем нестабильные признаки по значению gini на train/test/out.
    Если gini на train выборке сильно больше gini на test,
    или gini на test сильно больше на out, то удаляем признак.
    
    cut_off - если различия больше, чем на 60%, то удалим признак.

    '''

    bad_feats_idx = []
    for i, row in gini_by_vars.iterrows():

        diff_train_test = (row['gini_train'] - row['gini_test']) / row['gini_train']
        diff_test_out = (row['gini_test'] - row['gini_out']) / row['gini_test']


        if diff_train_test >= cut_off or diff_test_out >= cut_off:
            bad_feats_idx.append(i)

    return gini_by_vars.drop(bad_feats_idx)


def calc_woe_target_differences(X_train, X_test, X_out, y_train, y_test, y_out, vars):
    '''
    Считаем среднее WOE каждого признака для класса 0 и класса 1.
    Чем лучше разделяющая способность признака, тем больше разница.

    '''

    woe_differences = []
    for feat in vars:

        woe_for_0_train = X_train[y_train == 0][feat].mean()
        woe_for_1_train = X_train[y_train == 1][feat].mean()

        woe_for_0_test = X_test[y_test == 0][feat].mean()
        woe_for_1_test = X_test[y_test == 1][feat].mean()

        woe_for_0_out = X_out[y_out == 0][feat].mean()
        woe_for_1_out = X_out[y_out == 1][feat].mean()

        woe_differences.append((
            feat,
            woe_for_0_train - woe_for_1_train,
            woe_for_0_test - woe_for_1_test,
            woe_for_0_out - woe_for_1_out,
        ))

    df_woe_diff = pd.DataFrame(woe_differences, columns=['vars', 'woe_diff_train', 'woe_diff_test', 'woe_diff_out'])
    return df_woe_diff


def woe_univariate_selection(df_woe_diff, cut_off: float=0.6):
    '''
    Удаляем нестабильные признаки, если разделяющая способность признака
    сильно упала, то удаляем признак.
    
    cut_off - если различия больше, чем на 60%, то удалим признак.

    '''

    bad_feats_idx = []
    for i, row in df_woe_diff.iterrows():

        diff_train_test = (row['woe_diff_train'] - row['woe_diff_test']) / row['woe_diff_train']
        diff_test_out = (row['woe_diff_test'] - row['woe_diff_out']) / row['woe_diff_test']


        if diff_train_test >= cut_off or diff_test_out >= cut_off:
            bad_feats_idx.append(i)

    return df_woe_diff.drop(bad_feats_idx)


def catboost_feat_selection(X_train, X_test, y_train, y_test,
                            feats_to_select: list, cat_params: dict=None,
                            num_features: int=25, steps: int=5):
    '''
    Отбор признаков на основе алгоритма RFE реализованного в catboost.

    '''
    train_pool = Pool(X_train, y_train)
    test_pool = Pool(X_test, y_test)

    if cat_params is None:
        model = CatBoostClassifier(eval_metric='AUC', iterations=40, depth=4,
                                   min_data_in_leaf=X_train.shape[0]*0.025, random_seed=0)
    else:    
        model = CatBoostClassifier(**cat_params)

    algorithm = EFeaturesSelectionAlgorithm.RecursiveByShapValues

    summary = model.select_features(
        train_pool,
        eval_set=test_pool,
        features_for_select=feats_to_select,     # we will select from all features
        num_features_to_select=num_features,  # we want to select exactly important features
        steps=steps,                                     # more steps - more accurate selection
        algorithm=algorithm,
        shap_calc_type=EShapCalcType.Regular,            # can be Approximate, Regular and Exact
        # train_final_model=True,                          # to train model with selected features
        logging_level='Silent',
        # plot=True
    )

    return summary['selected_features_names']


def calc_permutation_importance(model, X, y, n_repeats=100, n_jobs=4):
    feature_importances_ = permutation_importance(model, X, y, scoring='roc_auc',
                                                  n_repeats=n_repeats, n_jobs=n_jobs, random_state=123)
    perm_imp = pd.DataFrame(
            {
                'mean_imp': feature_importances_['importances_mean'],
                'feature': X.columns
            }
        ).sort_values(by='mean_imp', ascending=False)
    perm_importances = perm_imp['mean_imp'].values
    perm_imp['percentile_imp'] = perm_imp['mean_imp'].apply(lambda x: stats.percentileofscore(perm_importances, x))   

    return perm_imp


def feature_include1_model(X_all, y_all, vars_current, iv_df, base_model):

    # Смотрим, что будет после добавления одного признака дополнительно.
    df_var_ginis = pd.DataFrame(columns=['var_name', 'gini_train', 'gini_test'])

    X_train, X_test, X_out = X_all[0], X_all[1], X_all[2]
    y_train, y_test, y_out = y_all[0], y_all[1], y_all[2]

    for i, var in enumerate(list(set(X_train.columns) - set(vars_current + ['normal_score'])) + ['with_all']):
        if var == 'with_all':
            __vars_current = list(vars_current)
        else:
            __vars_current = list(vars_current) + [var]
        
        model = deepcopy(base_model)
        model.fit(X_train[__vars_current], y_train)

        predict_proba_train = model.predict_proba(X_train[__vars_current])[:, 1]
        predict_proba_test = model.predict_proba(X_test[__vars_current])[:, 1]
        predict_proba_out = model.predict_proba(X_out[__vars_current])[:, 1]

        df_var_ginis.loc[i, 'var_name'] = var
        df_var_ginis.loc[i, 'gini_train'] = 2 * roc_auc_score(y_train, predict_proba_train) - 1
        df_var_ginis.loc[i, 'gini_test'] =  2 * roc_auc_score(y_test, predict_proba_test) - 1
        df_var_ginis.loc[i, 'gini_out'] =  2 * roc_auc_score(y_out, predict_proba_out) - 1

        IV_vars = map(lambda x: str(round(x, 4)), iv_df[iv_df['VAR_NAME'].isin([var.replace('WOE_', '')])]['IV'].unique())
        df_var_ginis.loc[i, 'IV'] = ', '.join(IV_vars)

    return df_var_ginis


def plot_feature_importances(feature_importance: pd.DataFrame, chunk_size: int=15, pic_folder='pic/'):
    '''
    Проходимся по всем признакам и отрисовываем их важность в порядке убывания.

    '''

    plt.figure(figsize=(12, 18))
    sns.barplot(y='feature', x='mean_imp', data=feature_importance)
    plt.title(f'Важность признаков')
    plt.xticks(rotation=30)
    pic_name = 'feature_import'
    plt.savefig(pic_folder + pic_name + '.png', dpi=100)
    plt.show()


# Вывести ROC-кривую, можно по train/test/out, можно по всем вместе
def plot_roc_curve(model, vars, X_all, y_all, X_train, y_train, X_test, y_test,
                     X_out=None, y_out=None, pic_folder='pic/'):

    # X_all = pd.concat([X_train, X_test, X_out], axis=0).reset_index(drop=True)
    # y_all =   # df_all['target']

    preds_train = model.predict_proba(X_train[vars])[:,1]
    preds_test = model.predict_proba(X_test[vars])[:,1]
    preds_out = model.predict_proba(X_out[vars])[:,1]
    preds_all = model.predict_proba(X_all[vars])[:,1]
    fpr_train, tpr_train, _ = metrics.roc_curve(y_train, preds_train)
    fpr_test, tpr_test, _ = metrics.roc_curve(y_test, preds_test)
    fpr_out, tpr_out, _ = metrics.roc_curve(y_out, preds_out)
    fpr_all, tpr_all, _ = metrics.roc_curve(y_all, preds_all)

    roc_auc_train = round(metrics.auc(fpr_train, tpr_train), 3)
    roc_auc_test = round(metrics.auc(fpr_test, tpr_test), 3)
    roc_auc_out = round(metrics.auc(fpr_out, tpr_out), 3)
    roc_auc_all = round(metrics.auc(fpr_all, tpr_all), 3)
    gini_train = round(2 * roc_auc_train - 1, 3)
    gini_test = round(2 * roc_auc_test - 1, 3)
    gini_out = round(2 * roc_auc_out - 1, 3)
    gini_all = round(2 * roc_auc_all - 1, 3)


    fig = plt.figure(figsize=(12, 8))
    fig.suptitle('Receiver Operating Characteristic')

    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    plt.subplot(221)
    plt.title('Train')
    plt.plot(fpr_train, tpr_train, 'b', label = f'AUC = {roc_auc_train}\nGini = {gini_train}')
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.subplot(222)
    plt.title('Test')
    plt.plot(fpr_test, tpr_test, 'b', label = f'AUC = {roc_auc_test}\nGini = {gini_test}')
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.subplot(223)
    plt.title('Out')
    plt.plot(fpr_out, tpr_out, 'b', label = f'AUC = {roc_auc_out}\nGini = {gini_out}')
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])


    plt.subplot(224)
    plt.title('Train + Test + Out')
    plt.plot(fpr_all, tpr_all, 'b', label = f'AUC = {roc_auc_all}\nGini = {gini_all}')
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    # fig.supylabel('True Positive Rate')
    # fig.supxlabel('False Positive Rate')

    # Set common labels
    fig.text(0.5, 0.04, 'False Positive Rate', ha='center', va='center')
    fig.text(0.06, 0.5, 'True Positive Rate', ha='center', va='center', rotation='vertical')

    pic_name = 'roc_curve'
    plt.savefig(pic_folder + pic_name + '.png', dpi=100)
    plt.show()


def export_to_excel_model(X_train, X_test, y_train, y_test, y, df3, iv_df,
                    Ginis, table, scores, feat, clf_lr, ivs,
                    gini_by_vars=None, df_gini_months=None, PV=None, DSL='', SQL='',
                    X_out=None, y_out=None, name='Scoring', date_name='date_requested',
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
    # feat_names = pd.DataFrame(feat['Feature'].apply(lambda x: x.replace('WOE_', ''))[:-1], columns=['Feature'])
    # feat_names.to_excel(writer, sheet_name='Feat description', index=False)
    # worksheet2 = writer.sheets['Feat description']
    # worksheet2.set_column('A:A', 35)

    # # Gini by var
    # if gini_by_vars is not None:
    #     gini_by_vars.to_excel(writer, sheet_name='Gini by var', index=False)
    #     worksheet2 = writer.sheets['Gini by var']
    #     worksheet2.set_column('A:A', 35)

    if df_gini_months is not None:
        df_gini_months.to_excel(writer, sheet_name='Month ginis by var', index=False)
        worksheet2 = writer.sheets['Month ginis by var']
        worksheet2.set_column('A:A', 35)

    worksheet = workbook.add_worksheet('Feature importance')
    worksheet.insert_image('B2', pic_folder + 'feature_import.png')

    worksheet = workbook.add_worksheet('ROC Curve')
    worksheet.insert_image('B2', pic_folder + 'roc_curve.png')

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
        for num, i in enumerate(feat): 
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


# Оптимизация гиперпараметров, библиотекой OPTUNA.
def parameters_optimizaion(X_train, X_test, y_train, y_test, params_borders, vars_woe, model_type,
                           n_trials=1000, n_jobs=2):
    '''
    Оптимизируем и подбираем гипермараметры модели.
    Здесь 3 заранее подготовленных сетапа для каждого типа модели: RandomForest, DecisionTree, LinearTree


    '''
    try:
        import optuna
    except:
        Exception('You dont have optuna library, pip install optuna')

    def objective_tree(trial):
        # Задаем возможные варианты преебора параметров
        params = {
            'max_depth': trial.suggest_int('max_depth', params_borders['max_depth'][0], params_borders['max_depth'][1]),
            'min_samples_leaf': trial.suggest_float('min_samples_leaf', params_borders['min_samples_leaf'][0], params_borders['min_samples_leaf'][1], step=0.015),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', params_borders['min_impurity_decrease'][0], params_borders['min_impurity_decrease'][1], step=0.015),
            'max_features': trial.suggest_categorical('max_features', params_borders['max_features']),
            'class_weight': params_borders['class_weight'],
            'random_state': params_borders['random_state'],
        }

        # Обучаем модель, с учетом разных возможных параметров.
        _model = DecisionTreeClassifier(**params)
        _model.fit(X_train[vars_woe], y_train)

        # Считаем метрику, которую будем оптимизировать.
        predict_proba_test = _model.predict_proba(X_test[vars_woe])[:, 1]
        return roc_auc_score(y_test, predict_proba_test)

    def objective_random_forest(trial):
        # Задаем возможные варианты преебора параметров
        params = {
            'n_estimators': trial.suggest_int('n_estimators', params_borders['n_estimators'][0], params_borders['n_estimators'][1], step=10),
            'criterion': trial.suggest_categorical('criterion', params_borders['criterion']),
            'max_depth': trial.suggest_int('max_depth', params_borders['max_depth'][0], params_borders['max_depth'][1]),
            'min_samples_leaf': trial.suggest_float('min_samples_leaf', params_borders['min_samples_leaf'][0], params_borders['min_samples_leaf'][1], step=0.015),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', params_borders['min_impurity_decrease'][0], params_borders['min_impurity_decrease'][1], step=0.015),
            'max_features': trial.suggest_categorical('max_features', params_borders['max_features']),
            'class_weight': params_borders['class_weight'],
            'random_state': params_borders['random_state'],
            'n_jobs': params_borders['n_jobs'],
        }

        # Обучаем модель, с учетом разных возможных параметров.
        _model = RandomForestClassifier(**params)
        _model.fit(X_train[vars_woe], y_train)

        # Считаем метрику, которую будем оптимизировать.
        predict_proba_test = _model.predict_proba(X_test[vars_woe])[:, 1]
        return roc_auc_score(y_test, predict_proba_test)

    def objective_linear_tree(trial):
        # Задаем возможные варианты преебора параметров
        params = {
            'max_depth': trial.suggest_int('max_depth', params_borders['max_depth'][0], params_borders['max_depth'][1]),
            'min_samples_leaf': trial.suggest_float('min_samples_leaf', params_borders['min_samples_leaf'][0], params_borders['min_samples_leaf'][1], step=0.015),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', params_borders['min_impurity_decrease'][0], params_borders['min_impurity_decrease'][1], step=0.015),
            'base_estimator': params_borders['base_estimator'],
        }

        # Обучаем модель, с учетом разных возможных параметров.
        _model = LinearTreeClassifier(**params)
        _model.fit(X_train[vars_woe], y_train)

        # Считаем метрику, которую будем оптимизировать.
        predict_proba_test = _model.predict_proba(X_test[vars_woe])[:, 1]
        return roc_auc_score(y_test, predict_proba_test)

    
    # Выбираем, какой из сетапов будем использовать при оптимизации.
    if model_type == 'tree':
        objective = objective_tree
    elif model_type == 'random_forest':
        objective = objective_random_forest
    elif model_type == 'linear_tree':
        objective = objective_linear_tree

    # Выключаем логгирование
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Задаем задачу оптимизации и запускаем её для заданного количества итераций.
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    return study.best_trial.params