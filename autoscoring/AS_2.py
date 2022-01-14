import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from scipy.spatial.distance import mahalanobis
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
from .AS import coef_woe_columns


def var_name_original(vars_woe):
    return [var.replace('WOE_', '') for var in vars_woe]


# Преобразовываем название категориальных признаков для df_out и остальных
def change_value_name(name, value_names):
    for var in value_names['MIN_VALUE'].values:
        if name in set(var.split(' | ')):
            return var
    for var in value_names['MIN_VALUE'].values:
        if '_ELSE_' in var:
            return var
    print(name)
    raise


def apply_iv_df_names(df, iv_df):
    '''
    После создания таблицы iv_df требуется заменить категориальные
    переменные на имена группы категорий, которым они соответсвуют

    Пример:
        df_train  = new_functions.apply_iv_df_names(df_train, iv_df)

    '''
    for col in df.select_dtypes(include=object).columns:
        value_names = iv_df[iv_df['VAR_NAME'] == col]
        if value_names.shape[0] == 0:
            continue
        try:
            df[col] = df[col].apply(lambda x: change_value_name(x, value_names))
        except:
            print(F'ERROR with column {col}')
    
    return df


def permutation_two_forest_selection(X: pd.DataFrame, y: pd.Series, top_n: int, n_repeats: int=50,
                 n_jobs: int=-1, rf_params: dict={'n_estimators': 50, 'max_depth': 5}) -> pd.DataFrame:
    '''
    Считаем важность признаков методом two-forest.
    1. Разбиваем выборку на две части, на каждой из них обучаем RF
    2. На выборке, которая не учавствовала в обучении считаем permutation_importance
    3. Для каждого признака считаем персентиль, в котором он лежит
        среди значенией permutation_importance для всей совокупности признаков.
    
    '''
    X_1, X_2, y_1, y_2 = train_test_split(X, y, test_size=0.5, random_state=42)

    rf_params['n_jobs'] = n_jobs
    if rf_params.get('random_state') is None:
        rf_params['random_state'] = 42
    rf_1 = RandomForestClassifier(**rf_params).fit(X_1, y_1)
    rf_2 = RandomForestClassifier(**rf_params).fit(X_2, y_2)

    res_1 = permutation_importance(rf_1, X_2, y_2, scoring='roc_auc',
                                    n_repeats=n_repeats, random_state=42, n_jobs=n_jobs)
    res_2 = permutation_importance(rf_2, X_1, y_1, scoring='roc_auc',
                                    n_repeats=n_repeats, random_state=42, n_jobs=n_jobs)

    perm_imp = pd.DataFrame(
        {
            'mean_imp': (res_1['importances_mean'] + res_2['importances_mean']) / 2,
            'feature': X.columns
        }
    ).sort_values(by='mean_imp', ascending=False)
    perm_importances = perm_imp['mean_imp'].values
    perm_imp['percentile_imp'] = perm_imp['mean_imp'].apply(lambda x: stats.percentileofscore(perm_importances, x))

    return perm_imp['feature'].values[:top_n], perm_imp


def parameter_optimization(X: pd.DataFrame, y: pd.Series, vars_current: list,
         logreg_params: dict, n_trials: int=3000, n_jobs: int=-1) -> dict:
    '''
    Отборо параметров логистической регрессии.

    vars_current: имена переменных
    logreg_params: словарь, с границами для перебора параметров
       пример параметра - example: {
           'class_weight_1': [1, 3.5] - максимум и минимум для параметра class_weight класса 1
           'C': [1e-5, 10000] - максимум и минимум для параметра C
       }

    Пример:
        best_param = parameter_optimization(X_train, y_train, ['WOE_PDL_IL', 'WOE_sex', 'WOE_DTI'],
                                       {'class_weight_1': [1, 3.5],'C': [1e-5, 10000]})

    '''
    try:
        import optuna
    except:
        raise Exception(f'you dont have optuna library, pip install optuna')

    def objective(trial):
        # Задаем возможные варианты преебора параметров
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
        class_weight_1 = trial.suggest_uniform('class_weight_1', logreg_params['class_weight_1'][0], logreg_params['class_weight_1'][1])
        C = trial.suggest_loguniform('C', logreg_params['C'][0], logreg_params['C'][1])
        __vars = vars_current

        # Обучаем модель, с учетом разных возможных параметров.
        params = {
            'penalty': penalty, 'C': C, 'solver': 'liblinear',
            'class_weight': {0: 1, 1: class_weight_1}, 'random_state': 42
        }
        logreg = LogisticRegression(**params)
        logreg.fit(X[__vars], y)

        # Считаем метрику, которую будем оптимизировать.
        predict_proba_train = logreg.predict_proba(X[__vars])[:, 1]
        Gini_train = 2 * roc_auc_score(y, predict_proba_train) - 1

        return Gini_train

    # Выключаем логгирование
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Задаем задачу оптимизации и запускаем её для заданного количества итераций.
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    return study.best_trial.params


def preprocessing_raw_data(X_train: pd.DataFrame, X_test: pd.DataFrame, X_out: pd.DataFrame,
     transform_type_numeric: str, transform_type_cat: str, y_train: pd.Series=None):
    '''
    Производим преобразование непрерывных и категориальных переменных
    (для данных без WOE биннинга)

    transform_type_numeric: тип предобработки непрерывных переменных.
        "scale" - применение шкалировния данных
        "box-cox" - преобразование Бокса-Кокса, только для положительных
        "yeo-johnson" - преобразование Йео-Джонсона
    transform_type_cat: тип предобработки для категориальных переменных
        "target_encoding" - применение TargetEncoding
        "oh_encoding" - применение OneHotEncoding
    y_train: требуется, когда надо выполнить target_encoding

    '''

    num_cols = X_train.select_dtypes(exclude=object)
    cat_cols = X_train.select_dtypes(include=object)

    if transform_type_numeric == 'scale':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        # TODO: надо отдельно тогда сохранять scaler или mean+std
        X_train_num = scaler.fit_transform(X_train[num_cols])
        X_test_num = scaler.transform(X_test[num_cols])
        X_out_num = scaler.transform(X_out[num_cols])

    elif transform_type_numeric == 'box-cox':
        from sklearn.preprocessing import power_transform
        X_train_num = power_transform(X_train, method='box-cox')
        X_test_num = power_transform(X_test, method='box-cox')
        X_out_num = power_transform(X_out, method='box-cox')

    elif transform_type_numeric == 'yeo-johnson':
        from sklearn.preprocessing import power_transform
        X_train_num = power_transform(X_train, method='yeo-johnson')
        X_test_num = power_transform(X_test, method='yeo-johnson')
        X_out_num = power_transform(X_out, method='yeo-johnson')
    else:
        print('ERROR parameter transfor_type_numeric is not correct')
        raise

    if transform_type_cat == 'target_encoding' and y_train is not None:
        from category_encoders.target_encoder import TargetEncoder
        encoder = TargetEncoder()
        X_train_cat = encoder.fit_transform(X_train, y_train)
        X_test_cat = encoder.transform(X_test)
        X_out_cat = encoder.transform(X_out)

    elif transform_type_cat == 'oh_encoding':
        from sklearn.preprocessing import OneHotEncoder
        encoder = OneHotEncoder()
        X_train_cat = encoder.fit_transform(X_train, y_train)
        X_test_cat = encoder.transform(X_test)
        X_out_cat = encoder.transform(X_out)

    return X_train_num, X_test_num, X_out_num, X_train_cat, X_test_cat, X_out_cat


def cramer_correlation(df: pd.DataFrame, cat_columns: list=None,
    max_uniq_cat: int=30, plot: bool=True, figsize=(10, 8)) -> pd.DataFrame:
    '''
    Рассчет коэффициента корреляции Крамера для всех пар категориальных признаков.

    cat_columns: список всех категориальных переменных
    max_uniq_cat: максимальное количество уникальных категорий, для проверки, что признак не количественный
    plot: требуется ли отрисовывать heatmap корреляций
    figsize: размер фигуры для отрисовки графика

    Пример:
        cramer_corr = cramer_correlation(df_train, df_train.select_dtypes(include=object).columns)
        # df_train.select_dtypes(include=object).columns - список всех категориальные признаки

    '''
    if cat_columns is None:
        cat_columns = df.select_dtypes(include=object).columns
    for col in cat_columns:
        if df[col].nunique() > max_uniq_cat:
            raise Exception(f'Column {col} is probably categorical, check it or change "max_uniq_cat" parameter')
        if df[col].dtype != np.dtype('O'):
            raise Exception(f'Change type of column {col} for object')
    
    # Создаем словарь, куда будем складывать попраные корреляции.
    correlations = {
        col1: {} for col1 in cat_columns
    }
    for i, col1 in enumerate(cat_columns):
        for col2 in cat_columns[i+1:]:
            
            df_cross = pd.crosstab(df[col1], df[col2])

            X2 = stats.chi2_contingency(df_cross)[0]
            n = np.sum(df_cross.values)
            minDim = min(df_cross.shape)-1

            # calculate Cramer's V 
            V = np.sqrt((X2/n) / minDim)

            correlations[col1][col2] = V
            correlations[col2][col1] = V
    cramer_corr = pd.DataFrame(correlations)

    if plot:
        plt.figure(figsize=figsize)
        sns.heatmap(cramer_corr)
        plt.show()

    return cramer_corr


def anomaly_detection_forest(X: pd.DataFrame, y: pd.DataFrame, percentile: float=0.05) -> pd.DataFrame:
    '''
    Удаляем аномальные объекты, используя метод IsolationForest.
    1. Рассчитываем score - мера аномальности каждого объекта
    2. Отрезаем хвост рампределения согласно percentile
    3. Возвращаем сэмпл без аномальных объектов

    X: pd.DataFrame с переменными, по которым надо произвести отбор аномалий
    y: pd.DataFrame с таргетом, чтобы получить таргет для итоговых объектов после отбора аномалий
    percentile: доля объектов, которую требуется удалить исходя из оценки аномальности
    
    Пример:
        df_train_new = anomaly_detection_forest(df_train, y_train, 0.05)

    '''
    model = IsolationForest(n_estimators=50, max_samples=0.3, max_features=0.75, random_state=142)
    model.fit(X)
    # Посчитали оценку аномальности для каждого объекта
    scores = model.score_samples(X)
    # Ищем cutoff по которому будем обрезать сэмпл
    cutoff = np.percentile(scores, percentile*100)

    return X[scores >= cutoff], y[scores >= cutoff]


def anomaly_detection_svm(X: pd.DataFrame, y: pd.DataFrame, percentile: float=0.05) -> pd.DataFrame:
    '''
    Удаляем аномальные объекты, используя метод OneClassSVM.
    1. Рассчитываем score - мера аномальности каждого объекта
    2. Отрезаем хвост рампределения согласно percentile
    3. Возвращаем сэмпл без аномальных объектов

    X: pd.Dataframe с переменными, по которым надо произвести отбор аномалий
    percentile: доля объектов, которую требуется удалить исходя из оценки аномальности

    Пример:
        df_train_new = anomaly_detection_svm(df_train, y_train, 0.05)

    '''
    model = OneClassSVM()
    model.fit(X)
    # Посчитали оценку аномальности для каждого объекта
    scores = model.score_samples(X)
    # Ищем cutoff по которому будем обрезать сэмпл
    cutoff = np.percentile(scores, percentile*100)

    return X[scores >= cutoff], y[scores >= cutoff]


def anomaly_detection_distribution(X: pd.DataFrame, y: pd.DataFrame, percentile: int=0.05) -> pd.DataFrame:
    '''
    Удаляем аномальные объекты, используя метод расстояние Махаланобиса от центра сэмпла.
    1. Предполагаем, что наши данные - примерно многомерное нормальное распределение.
    2. Рассчитываем расстояние центра нашего распределения (среднее сэмпла).
    3. Считаем матрицу ковариации и её обратную матрицу.
    4. Считаем расстояние для каждой точки от среднего сэмпла.
    5. Отрезаем хвост рампределения согласно percentile
    6. Возвращаем сэмпл без аномальных объектов

    X: pd.Dataframe с переменными, по которым надо произвести отбор аномалий
    percentile: доля объектов, которую требуется удалить исходя из оценки аномальности

    Пример:
        df_train_new = anomaly_detection_distribution(df_train, y_train, 0.05)

    '''
    # Среднее сэмпла
    sample_mean = X.mean(axis=0)

    covariance  = np.cov(X.to_numpy(), rowvar=False)
    covariance_pm1 = np.linalg.matrix_power(covariance, -1) # Covariance matrix power of -1

    # Считаем расстояние Махолонобиса для каждого объекта относительно центра.
    distances = X.apply(lambda row: mahalanobis(sample_mean, row, covariance_pm1), axis=1)
    # Ищем cutoff для расстояния от центра, по которому будем обрезать сэмпл
    cutoff = np.percentile(distances, percentile*100)

    return X[distances >= cutoff], y[distances >= cutoff]


def find_outliers_z_score(data, technical_cols, treshold = 3):

    cols = list(data.columns)
    
    for i in technical_cols:
        if i in cols:
            cols.remove(i)
            
    attr_data = []
    
    for i in cols:
        scales_right = treshold*data[i].std()+data[i].mean()
        scales_left = (-1)*treshold*data[i].std()+data[i].mean()
        
        attr_data.append([i, treshold, scales_right, scales_left])
        
    attr_columns = ['variable', 'treshold', 'right_border', 'left_border']    
    outlier_data = pd.DataFrame.from_records(attr_data, columns = attr_columns)

    return outlier_data


def mannwhitney_target_test(df: pd.DataFrame, y: pd.Series, num_columns: list=None) -> pd.DataFrame:
    '''
    Проводим тест Манна-Уитни для оценки разделяющей способности количественных признаков.

    '''
    if num_columns is None:
        num_columns = df.select_dtypes(exclude=object).columns

    p_values = {} # Словарь, в котором будем хранить результат.
    for col in num_columns:
        if df[col].isna().sum() > 0:
            print(f'WARNING column {col} contains NaN')
        # Разбиваеем сэмпл на две группы по таргету.
        group0 = df[y == 0][col].fillna(df[col].mean())
        group1 = df[y == 1][col].fillna(df[col].mean())

        # Проверяем тест, с гипотезой об отсутствии различий в признаке col для таргета 0 и 1.
        stat, p_value = stats.mannwhitneyu(group0, group1)

        p_values[col] = p_value

    return pd.DataFrame(p_values.values(), index=p_values.keys(), columns=['p-value'])


def pca_transforamation(df: pd.DataFrame, columns: list, n_components: float=0.95) -> pd.DataFrame:
    '''
    Производим PCA трансформацию и создаем новый датафрейм 
    c новыми переменными взамен старых.
    
    columns: набор переменных, которые требуется учитывать в PCA трансформации.
    n_components: это число от 0 до 1 - процент объясненной дисперсии, которую требуется оставить
        или это число от 1 до len(columns) - количество итоговых компонент

    '''
    model = PCA(n_components=n_components)
    transformed_data = model.fit_transform(df[columns])

    # Удаляем столбцы с преобразованными данными и добавляем вместо них новые,
    # полученные после трансформации.
    df_new = df.copy()
    df_new.drop(columns, axis=1, inplace=True)
    for i in range(transformed_data.shape[1]):
        df_new[f'PCA_{i}'] = transformed_data[:, i]

    return df_new, model


def l1_feature_selection(X: pd.DataFrame, y: pd.Series, C: float=1):
    '''
    C: параметр регуляризации, чем он меньше, тем регуляризация сильнее,
        следует подобрать в зависимости от количества отбираемых признаков

    '''
    model = LogisticRegression(C=C, penalty='l1', solver='liblinear', random_state=142)
    model.fit(X, y)

    good_features = X.columns[model.coef_[0] != 0]
    regularized_features = X.columns[model.coef_[0] == 0]

    return good_features, regularized_features


def rf_feature_selection(df: pd.DataFrame, y: pd.Series, top_n: int=20) -> pd.DataFrame:
    '''
    Считаем feature_importances с использование Случайного леса

    top_n: количество признаков для отбора

    '''
    model = RandomForestClassifier(random_state=142)
    model.fit(df, y)

    feature_imp = pd.DataFrame(model.feature_importances_, index=df.columns, columns=['feature_importance'])
    feature_imp = feature_imp.sort_values(by='feature_importance', ascending=False)

    return list(feature_imp[:top_n].index), feature_imp


def gini_month_selection(X: pd.DataFrame, df: pd.DataFrame, gini_min: float=0.05,
                         num_bad_months: int=2, target_name: str='target',
                        date_name: str='date_requested') -> Tuple[list, pd.DataFrame]:
    '''
    Отбор признаков по однофакторной оценке gini по месяцам.
    Отбираем переменные, для которых для каждого месяца gini выше gini_min,
    допускается если gini ниже, но только если таких месяцев <= num_bad_moths.

    X: pd.DataFrame тренировочный набор преобразованных данных (X_train)
    df: pd.DataFrame тренировочный набор данных, содержащих date_requested и target (df_train)
    gini_min: минимальный порог gini для отбора
    num_bad_months: количество месяцев, в которых gini может быть меньше заданного
    target_name: имя таргета в df

    Пример:
        gini_feats, df_gini_months = new_functions.gini_month_selection(X_train, df_train)

    '''
    df_x_month = pd.concat([X.reset_index(drop=True), df[[date_name, target_name]].reset_index(drop=True)], axis=1)
    df_x_month['requested_month_year'] = df_x_month[date_name].map(lambda x: str(x)[:7])
    vars_woe = X.columns

    requested_month_year = np.sort(df_x_month['requested_month_year'].unique())
    df_gini_months = pd.DataFrame(np.zeros((len(vars_woe), len(requested_month_year))), columns=requested_month_year)
    df_gini_months.index = vars_woe

    # Для каждого месяца и для каждой переменной рассчитываем однофакторный gini
    for month_year in requested_month_year:
        df_tmp = df_x_month[df_x_month['requested_month_year'] == month_year]

        for x in vars_woe:
            vars_t = [x] #vars_current + [x]
            df_train_m = df_tmp[vars_t]
            y_train = df_tmp[target_name]

            if y_train.value_counts().shape[0] < 2:
                # Таргет состоит только из одного класса
                Gini_train = -1
            else:
                _logreg = LogisticRegression().fit(df_train_m, y_train)

                predict_proba_train = _logreg.predict_proba(df_train_m)[:, 1]
                Gini_train = round(2 * roc_auc_score(y_train, predict_proba_train) - 1, 3)
            
            df_gini_months.loc[x, month_year] = Gini_train

    # Отбираем признаки, для которых количество плохо предсказанных месяцев меньше заданного числа.
    good_features = df_gini_months[((df_gini_months < gini_min).sum(axis=1) <= num_bad_months)].index

    df_gini_months.reset_index(inplace=True)
    df_gini_months = df_gini_months.rename(columns={'index': 'vars'})
    return good_features, df_gini_months


def gini_univariate(X_train: pd.DataFrame, X_test: pd.DataFrame, X_out: pd.DataFrame,
                y_train: pd.Series, y_test: pd.Series, y_out: pd.Series,
                vars_woe: list, params: dict={}) -> pd.DataFrame:
    '''
    Анализ изменчивости метрики gini из-за отдельных признаков.
    vars_woe: list[str] название переменных, которые требуется проанализировать
    params: dict набор переменных, которые надо использовать при обучении LogisticRegression

    '''
    df_gf = pd.DataFrame(columns = ['vars', 'gini_train', 'gini_test'])
    # vars_rest = list(set(vars_all_woe) - set(vars_current))
    vars_rest = vars_woe
    j = 0
    for x in vars_rest:
        vars_t = [x] #vars_current + [x]
        df_train_m = X_train[vars_t]
        
        _logreg = LogisticRegression(**params).fit(df_train_m, y_train)
        
        predict_proba_train = _logreg.predict_proba(X_train[vars_t])[:, 1]
        predict_proba_test = _logreg.predict_proba(X_test[vars_t])[:, 1]

        Gini_train = 2 * roc_auc_score(y_train, predict_proba_train) - 1
        Gini_test = 2 * roc_auc_score(y_test, predict_proba_test) - 1

        df_gf.loc[j, 'vars'] = x
        df_gf.loc[j, 'gini_train'] = round(Gini_train, 3)
        df_gf.loc[j, 'gini_test'] = round(Gini_test, 3)
        
        if X_out is not None and y_out is not None:
            predict_proba_out = _logreg.predict_proba(X_out[vars_t])[:, 1]
            Gini_out = 2 * roc_auc_score(y_out, predict_proba_out) - 1
            df_gf.loc[j, 'gini_out'] = round(Gini_out, 3)

        j = j+1

    gini_by_vars = df_gf.sort_values(by='gini_train', ascending=False)

    return gini_by_vars


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

        df_var_ginis.loc[i, 'var_name'] = var
        df_var_ginis.loc[i, 'gini_train'] = 2 * roc_auc_score(y_train, predict_proba_train) - 1
        df_var_ginis.loc[i, 'gini_test'] =  2 * roc_auc_score(y_test, predict_proba_test) - 1

        if X_all[2] is not None and y_all[2] is not None:
            predict_proba_out = _logreg.predict_proba(X_out[__vars_current])[:, 1]
            df_var_ginis.loc[i, 'gini_out'] =  2 * roc_auc_score(y_out, predict_proba_out) - 1

        IV_vars = map(lambda x: str(round(x, 4)), iv_df[iv_df['VAR_NAME'].isin([var.replace('WOE_', '')])]['IV'].unique())
        df_var_ginis.loc[i, 'IV'] = ', '.join(IV_vars)

    return df_var_ginis


def feature_include1(X_all, y_all, vars_current, iv_df, params, sample_weight=None):
    # Смотрим, что будет после добавления одного признака дополнительно.
    df_var_ginis = pd.DataFrame(columns=['var_name', 'gini_train', 'gini_test'])
    # clf = IsolationForest(n_estimators=50, max_samples=0.3, max_features=0.75, random_state=123)

    X_train, X_test, X_out = X_all[0], X_all[1], X_all[2]
    y_train, y_test, y_out = y_all[0], y_all[1], y_all[2]

    for i, var in enumerate(list(set(X_train.columns) - set(vars_current + ['normal_score'])) + ['with_all']):
        if var == 'with_all':
            __vars_current = list(vars_current)
        else:
            __vars_current = list(vars_current) + [var]
        
        _logreg = LogisticRegression(**params).fit(
            X_train[__vars_current],
            y_train,
            sample_weight=sample_weight
        )

        predict_proba_train = _logreg.predict_proba(X_train[__vars_current])[:, 1]
        predict_proba_test = _logreg.predict_proba(X_test[__vars_current])[:, 1]

        df_var_ginis.loc[i, 'var_name'] = var
        df_var_ginis.loc[i, 'gini_train'] = 2 * roc_auc_score(y_train, predict_proba_train) - 1
        df_var_ginis.loc[i, 'gini_test'] =  2 * roc_auc_score(y_test, predict_proba_test) - 1
        
        if X_all[2] is not None and y_all[2] is not None:
            predict_proba_out = _logreg.predict_proba(X_out[__vars_current])[:, 1]
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

            df_var_ginis.loc[i+j, 'var_name'] = ', '.join([var, var2])
            df_var_ginis.loc[i+j, 'gini_train'] = 2 * roc_auc_score(y_train_check, predict_proba_train) - 1
            df_var_ginis.loc[i+j, 'gini_test'] =  2 * roc_auc_score(y_test, predict_proba_test) - 1
                    
            if X_all[2] is not None and y_all[2] is not None:
                predict_proba_out = _logreg.predict_proba(X_out[__vars_current])[:, 1]
                df_var_ginis.loc[i, 'gini_out'] =  2 * roc_auc_score(y_out, predict_proba_out) - 1

            IV_vars = map(lambda x: str(round(x, 4)), iv_df[iv_df['VAR_NAME'].isin([var.replace('WOE_', ''), var2.replace('WOE_', '')])]['IV'].unique())
            df_var_ginis.loc[i+j, 'IV'] = ', '.join(IV_vars)

    return df_var_ginis


def construct_df3(vars_woe, logreg, df_req, X_train, X_test, df_train, df_test, X_out=None, df_out=None,
                 date_name='date_requested', target='target'):
    vars = [var.replace('WOE_', '') for var in vars_woe]

    if X_out is None:
        X_full = pd.concat([X_train, X_test])[vars_woe]
        df2 = pd.concat([df_train.reset_index(drop=True),
                     df_test.reset_index(drop=True)])[['credit_id', date_name, target] + list(vars)]
    else:
        X_full = pd.concat([X_train, X_test, X_out])[vars_woe]
        df2 = pd.concat([df_train.reset_index(drop=True), df_test.reset_index(drop=True),
                         df_out.reset_index(drop=True)])[['credit_id', date_name, target] + list(vars)]
    
    df2 = pd.concat([df2.reset_index(drop=True), X_full.reset_index(drop=True)], axis=1)

    # Features for excel
    feat = pd.DataFrame({'Feature': list(X_full.columns), 'Coefficient': list(logreg.coef_[0])}).sort_values(['Coefficient'], ascending=False)
    feat = feat.append(pd.DataFrame([[logreg.intercept_[0], '_INTERCEPT_']], columns=['Coefficient', 'Feature']), ignore_index=True)

    df3 = df2.copy()
    df3 = pd.merge(df3.drop([date_name], axis=1), df_req, on='credit_id')
    df3['requested_month_year'] = df3[date_name].map(lambda x: str(x)[:7])

    df3[target] = df3[target].astype(float)
    df3['PD'] = logreg.predict_proba(df3[vars_woe])[:,1] 
    df3['Score'] = 1000 - round(1000*(df3['PD']))

    df3 = coef_woe_columns(df3, feat)
    # Score buckets
    df3['Score_bucket'] = df3.Score.map(lambda x: str(int((x//100)*100))+'-'+ str(int((x//100)+1)*100))

    return df3, feat


def vif_feature_selection(X_train: pd.DataFrame, iv_df: pd.DataFrame, vif_threshold: int=15) -> list:
    '''
    Делаем отбор признаков по показателю VIF, считается,
    что значения выше 10 свидетельствуют о высокой мультиколлинеарности признаков.
    В данном случае не удаляем признак, если у него достаточно большой показатель IV.
    
    X_train: массив с переменными, по которым считать VIF
    iv_df: таблица со столбцами VAR_NAME и IV
    vif_threshold: пороговое значение VIF

    '''
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant

    vif_feats = list(X_train.columns)
    # Порог для IV, выше которого признак не будет удаляться.
    iv_threshold = np.percentile(iv_df[iv_df['VAR_NAME'].isin(var_name_original(vif_feats))]['IV'], 40)
    max_vif_value = np.inf

    while max_vif_value > vif_threshold:
        vif_train = add_constant(X_train[vif_feats])
        # Считаем VIF для каждой переменной.
        vif_series = pd.Series([variance_inflation_factor(vif_train.values, i) 
                                for i in range(vif_train.shape[1])], 
                                index=vif_train.columns).drop(['const'])
        max_vif_ind = vif_series.argsort()[::-1]

        for index in max_vif_ind:
            max_vif_value = vif_series.iloc[index]
            max_vif_name = vif_feats[index].replace('WOE_', '')

            # Удаляем признак, если у него низкое IV и выский VIF.
            if iv_df[iv_df['VAR_NAME'] == max_vif_name]['IV'].values[0] < iv_threshold and max_vif_value > vif_threshold:
                print(f'delete {max_vif_name} feature')
                del vif_feats[index]
                break
            # Если не осталось признаков с высоким VIF, то останавливаем весь процесс.
            elif max_vif_value <= vif_threshold:
                break
    print('DONE')
    
    return vif_feats


def grid_search_heatmap(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series,
                         C_arr: list, class_weight_1_arr: list) -> dict:
    '''
    Делаем перебор основных параметров логистической регрессиии.
    Формируем тепловую карту исходя из метрики gini на тестовой выборке.
    
    C_arr: list набор параметров C для перебора
    class_weight_1_arr: list набор значений для class_weight класса 1 для перебора
    
    '''
    # C_arr = [10 ** i for i in range(-2, 4)]
    # class_weight_1_arr = np.linspace(1, 8, 8)
    metrics = pd.DataFrame(columns=['C', 'class_weight_1', 'gini_test'])

    for C in C_arr:
        for class_weight_1 in class_weight_1_arr:
            
            params = {
                'penalty': 'l2', 'C': C, 'solver': 'liblinear',
                'class_weight': {0: 1, 1: class_weight_1}, 'random_state': 142
            }
            logreg = LogisticRegression(**params).fit(X_train, y_train)
            predict_proba_test = logreg.predict_proba(X_test)[:, 1]
            Gini_test = 2 * roc_auc_score(y_test, predict_proba_test) - 1

            metrics = metrics.append({
                'C': C,
                'class_weight_1': class_weight_1,
                'gini_test': Gini_test
            }, ignore_index=True)

    metrics_pivot = metrics.pivot('C', 'class_weight_1', 'gini_test')

    plt.figure(figsize=(10, 6))
    sns.heatmap(data=metrics_pivot)
    plt.show()

    return metrics.sort_values(by='gini_test', ascending=False).iloc[0].to_dict()


def save_encoding_excel(dict_cat_encoding: dict, dict_nan_encoding: dict, name: str='result_rules/encoding_methods'):
    '''
    Данная функция используется во время построения правил
    с помощью дерефьев решений. С помощью её можно сохранить
    методы кодировки категориальных переменных и заполнения NaN
    значений в excel файл.

    '''
    nan_dataframe = pd.DataFrame(
        [[key, val] for key, val in dict_nan_encoding.items()],
        columns=['feature', 'value for NaN']
    )

    # Сохраним значения для кодировки NaN в отдельный excel файл
    writer = pd.ExcelWriter('{}.xlsx'.format(name), engine='xlsxwriter')
    nan_dataframe.to_excel(writer, sheet_name='NaN encoding', index=False)
    worksheet2 = writer.sheets['NaN encoding']
    worksheet2.set_column('A:A', 35)
    worksheet2.set_column('B:B', 25)

    for i, (feat, enc_dataframe) in enumerate(dict_cat_encoding.items()):
        enc_dataframe.to_excel(writer, sheet_name='Category encoding', index=False,
                                startcol=i*3, startrow=0)


    writer.save()
    writer.close()


def save_selection_stages(selection_stages: dict, name: str='result/selection_stages.xlsx'):
    '''
    Сохрянем стадии отбора признаков в excel файл
    selection_stages: dict, словарь, в котором перечислены
        наименование стадии отбора признаков, и набор признаков,
        который после этой стадии остался
        '<имя>': <list массив названий признаков>

    '''
    writer = pd.ExcelWriter(name, engine='xlsxwriter')
    for i, (stage_name, stage_feats) in enumerate(selection_stages.items()):
        if i == 0:
            continue
        
        # Формируем набор признаков которы был до, и который остался после текущего этапа отбора.
        before_feats = sorted(var_name_original(list(selection_stages.values())[i-1]))
        selected_feats = set(var_name_original(list(stage_feats)))
        after_feats = []

        # Формируем набор оставшихся признаков, чтобы он был в той же последовательности.
        for feat in before_feats:
            if feat not in selected_feats:
                after_feats.append('')
            else:
                after_feats.append(feat)

        df_selection = pd.DataFrame([before_feats, after_feats]).T
        df_selection.columns = [f'до {stage_name}', f'после {stage_name}']

        df_selection.to_excel(writer, sheet_name=stage_name[:30])
        worksheet2 = writer.sheets[stage_name[:30]]
        worksheet2.set_column('B:B', 35)
        worksheet2.set_column('C:C', 35)

    writer.save()
    writer.close()