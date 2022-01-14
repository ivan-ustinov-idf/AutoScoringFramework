import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


def RepresentsFloat(s):
    try: 
        float(s)
        return True
    except ValueError:
        return False

def woe_transformation(x, iv_col):
    if isinstance(x, str):
        # ОБработка категориального случая, при этом признак должен быть
        # преобразован к нужному виду, но необязательно.
        tmp = iv_col[iv_col['MIN_VALUE'] == x]
        if tmp.shape[0] == 0:
            for i, value in enumerate(iv_col['MIN_VALUE'].values):
                if x in set(value.split(' | ')):
                    tmp = iv_col.iloc[i]
                    break
        q = float(tmp.WOE)
        return q
    elif pd.isna(x) == True:
        # Обраотка NaN для непрерывных значений
        q = float(iv_col[pd.isna(iv_col.MAX_VALUE) == True].WOE)
        return q

    if RepresentsFloat(x) is False:
        print('ERROR in {col}, {x} is object but should be float, check if you transform feature')
        raise

    if x >= iv_col.MAX_VALUE.max():
        # Значение больше максимального бина
        q = float(iv_col[iv_col.MAX_VALUE == iv_col.MAX_VALUE.max()].WOE)
    elif x < iv_col.MAX_VALUE.min():
        # Значение меньше минимального бина
        q = float(iv_col.head(1).WOE)
    # elif pd.isna(x)==True:
    #     q = float(iv_col[pd.isna(iv_col.MAX_VALUE)==True].WOE)
    else:
        # Ищем бин, внутри которого лежит значение.
        try:
            tmp = iv_col[(x >= iv_col.MIN_VALUE) & (x < iv_col.MAX_VALUE)]
            if tmp.shape[0] == 0:
                # Из-за улсовия x < iv_col.MAX_VALUE в зависимости от биннинга может не найтись бина,
                # поэтому ищем соотествующий бин среди ближайших по значению
                q = float(iv_col[((iv_col.MAX_VALUE - x).abs().argsort()==0)].WOE)
            else:
                q = float(tmp.WOE)
        except:
            print(f'ERROR with transformation of value {x}, VAR_NAME {iv_col["VAR_NAME"].iloc[0]}')
            raise
            # q = float(iv_col[((iv_col.MAX_VALUE - x).abs().argsort()==0)].WOE)
    return q


def woe_transform_apply(col, series, iv_col):
    try:
        res = series.map(lambda x: woe_transformation(x, iv_col))
        return col, res, 'OK'
    except:
        return col, None, 'ERROR'


def transform_df_to_woe(df, y, IV, iv_df, iv_cut_off=None, n_jobs=None):
    
    if iv_cut_off is not None:
        df1 = df[list(IV[IV.IV>iv_cut_off].VAR_NAME)].copy()
    else:
        df1 = df.copy()
    try:
        df1 = df1.drop('credit_id', axis=1)
    except:
        pass
    try:
        df1 = df1.drop('date_requested', axis=1)
    except:
        pass
    print('Features left after IV drop: {}'.format(len(df1.columns)))

    # Набор параметров, для вызова функции применительно к одному столбцу переменной.
    params_gen = (
        (col, df1[col], iv_df[iv_df.VAR_NAME==col]) for col in df1.columns
    )

    with Pool(n_jobs) as pool:
        result = pool.starmap(woe_transform_apply, params_gen)

    for col_res in result:
        col, series, err = col_res
        if err == 'ERROR':
            print (col, f'- ERROR! Column was dropped!, Exception')
            df1.drop(col, axis=1, inplace=True)
            continue
        df1[col] = series
    
    print('DONE!')

    return df1, y


def construction_iv_df_from_autowoe(df, auto_woe, TARGET, features_type, features,
                             nan_to_woe: str='max_cat', else_to_woe: str='max_cat'):
    '''
    Преобразование объектов для WOE трансформации из библиотеки AutoWOE Сбера
    в классическую таблицу iv_df принятую в IDF

    '''
    iv_df = pd.DataFrame(columns=['VAR_NAME', 'MIN_VALUE', 'MAX_VALUE', 'COUNT', 'DR', 'EVENT', 'EVENT_RATE',
                                    'NONEVENT', 'NON_EVENT_RATE', 'WOE', 'IV'])
    idx = -1
    dr = df[TARGET].value_counts().to_dict()

    for feature in features:
        if auto_woe.split_dict.get(feature) is None:
            continue
        split = auto_woe.split_dict[feature]
        if len(split) == 0:
            continue
        feat_idxs = []

        if features_type[feature] == 'real':

            split[np.isclose(split, 1e-35)] = 0.00001

            for i, val in enumerate(split):
                idx += 1
                feat_idxs.append(idx)
                iv_df.loc[idx, 'VAR_NAME'] = feature
                if i == 0:
                    # Обработка первого бина для данной переменной
                    dr_tmp = df[df[feature] < val][TARGET].value_counts().to_dict()
                    iv_df.loc[idx, 'MIN_VALUE'] = df[feature].min() - 0.1
                    iv_df.loc[idx, 'MAX_VALUE'] = val
                else:
                    # Обработка промежуточного бина для переменной
                    dr_tmp = df[(df[feature] >= split[i-1]) & (df[feature] < val)][TARGET].value_counts().to_dict()
                    iv_df.loc[idx, 'MIN_VALUE'] = split[i-1]
                    iv_df.loc[idx, 'MAX_VALUE'] = val

                if dr_tmp.get(0) is None and dr_tmp.get(1) is None:
                    print(f'ERROR {feature} doesnt have information, {split}, {i}')
                    raise
                if dr_tmp.get(1) is None:
                    dr_tmp[1] = 0
                elif dr_tmp.get(0) is None:
                    dr_tmp[0] = 0

                iv_df.loc[idx, 'COUNT'] = dr_tmp[0] + dr_tmp[1]
                iv_df.loc[idx, 'DR'] =  dr_tmp[1] / (dr_tmp[0] + dr_tmp[1])
                iv_df.loc[idx, 'EVENT'] = dr_tmp[1]
                iv_df.loc[idx, 'EVENT_RATE'] = dr_tmp[1] / dr[1]
                iv_df.loc[idx, 'NONEVENT'] = dr_tmp[0]
                iv_df.loc[idx, 'NON_EVENT_RATE'] = dr_tmp[0] / dr[0]
                iv_df.loc[idx, 'WOE'] = auto_woe.woe_dict[feature].cod_dict[i]

                if i == len(split) - 1:
                    # Обработка последнего бина данной переменной
                    dr_tmp = df[(df[feature] >= val)][TARGET].value_counts().to_dict()
                    if dr_tmp.get(0) is None and dr_tmp.get(1) is None:
                        print(f'ERROR in last {feature} doesnt have information, {split}, {i}')
                        continue
                    if dr_tmp.get(1) is None:
                        dr_tmp[1] = 0
                    elif dr_tmp.get(0) is None:
                        dr_tmp[0] = 0

                    idx += 1
                    feat_idxs.append(idx)
                    iv_df.loc[idx, 'VAR_NAME'] = feature
                    iv_df.loc[idx, 'MIN_VALUE'] = val
                    iv_df.loc[idx, 'MAX_VALUE'] = df[feature].max() + 0.1

                    iv_df.loc[idx, 'COUNT'] = dr_tmp[0] + dr_tmp[1]
                    iv_df.loc[idx, 'DR'] =  dr_tmp[1] / (dr_tmp[0] + dr_tmp[1])
                    iv_df.loc[idx, 'EVENT'] = dr_tmp[1]
                    iv_df.loc[idx, 'EVENT_RATE'] = dr_tmp[1] / dr[1]
                    iv_df.loc[idx, 'NONEVENT'] = dr_tmp[0]
                    iv_df.loc[idx, 'NON_EVENT_RATE'] = dr_tmp[0] / dr[0]
                    iv_df.loc[idx, 'WOE'] = auto_woe.woe_dict[feature].cod_dict[i+1]

            if df[feature].isna().sum() > 0:
                # Обработка NaN значений
                dr_tmp = df[(df[feature].isna() == True)][TARGET].value_counts().to_dict()
                if dr_tmp.get(0) is None and dr_tmp.get(1) is None:
                    print(f'ERROR in NaN {feature} doesnt have information, {split}, {i}')
                    raise
                if dr_tmp.get(1) is None:
                    dr_tmp[1] = 0
                elif dr_tmp.get(0) is None:
                    dr_tmp[0] = 0

                idx += 1
                feat_idxs.append(idx)
                iv_df.loc[idx, 'VAR_NAME'] = feature
                # iv_df.loc[idx, 'MIN_VALUE'] = 
                # iv_df.loc[idx, 'MAX_VALUE'] = 

                iv_df.loc[idx, 'COUNT'] = dr_tmp[0] + dr_tmp[1]
                iv_df.loc[idx, 'DR'] =  dr_tmp[1] / (dr_tmp[0] + dr_tmp[1])
                iv_df.loc[idx, 'EVENT'] = dr_tmp[1]
                iv_df.loc[idx, 'EVENT_RATE'] = dr_tmp[1] / dr[1]
                iv_df.loc[idx, 'NONEVENT'] = dr_tmp[0]
                iv_df.loc[idx, 'NON_EVENT_RATE'] = dr_tmp[0] / dr[0]
                try:
                    # Ищем разбиения с NaN которые построил AutoWOE
                    nan_key = [key for key in auto_woe.woe_dict[feature].cod_dict.keys() if 'NaN' in str(key)][0]
                    iv_df.loc[idx, 'WOE'] = auto_woe.woe_dict[feature].cod_dict[nan_key]
                except:
                    print('ERROR in NaN {feature}, doesnt have woe-binning')
                    raise
            else:
                # Если нет значений NaN, то вставим дефолтное для ясности.
                idx += 1
                feat_idxs.append(idx)
                iv_df.loc[idx, 'VAR_NAME'] = feature
                # iv_df.loc[idx, 'MIN_VALUE'] = 
                # iv_df.loc[idx, 'MAX_VALUE'] = 

                iv_df.loc[idx, 'COUNT'] = 0
                iv_df.loc[idx, 'DR'] =  0
                iv_df.loc[idx, 'EVENT'] = 0
                iv_df.loc[idx, 'EVENT_RATE'] = 0
                iv_df.loc[idx, 'NONEVENT'] = 0
                iv_df.loc[idx, 'NON_EVENT_RATE'] = 0

                tmp_iv_df = iv_df.loc[feat_idxs[:-1]]  # Без текущего значения
                if nan_to_woe == 'max':
                    iv_df.loc[idx, 'WOE'] = tmp_iv_df['WOE'].max()
                elif nan_to_woe == 'min':
                    iv_df.loc[idx, 'WOE'] = tmp_iv_df['WOE'].min()
                elif nan_to_woe == 'zero':
                    iv_df.loc[idx, 'WOE'] = 0
                elif nan_to_woe == 'max_cat':
                    iv_df.loc[idx, 'WOE'] = tmp_iv_df[tmp_iv_df['COUNT'] == tmp_iv_df['COUNT'].max()]['WOE'].values[0]

        elif features_type[feature] == 'cat':
            bins = {}
            # Формируем словарь: <номер бина>: <переменные, разделенные ' | '>
            nan_exist = False  # Если в разбиениях отсутствуют _MISSING_ или _ELSE_, то в конце их вставим для ясности.
            else_exist = False
            for key, val in auto_woe.split_dict[feature].items():
                if 'Small' in key:
                    continue
                if '_MISSING_' in key:
                    nan_exist = True
                if '_ELSE_' in key:
                    else_exist = True

                if bins.get(val) is None:
                    bins[val] = key
                else:
                    bins[val] += ' | ' + key

            for bin_num, feats in bins.items():
                idx += 1
                feat_idxs.append(idx)
                iv_df.loc[idx, 'VAR_NAME'] = feature
                iv_df.loc[idx, 'MIN_VALUE'] = feats
                iv_df.loc[idx, 'MAX_VALUE'] = feats

                if 'NaN' in feats:
                    # Обработка NaN избыточна, т.к. по дефолту категориальные переменные заполняются '_MISSING_'
                    dr_tmp = df[(df[feature].isin(feats.split(' | '))) | (df[feature].isna() == True)][TARGET].value_counts().to_dict()
                else:
                    dr_tmp = df[df[feature].isin(feats.split(' | '))][TARGET].value_counts().to_dict()
                if dr_tmp.get(0) is None and dr_tmp.get(1) is None:
                    print(f'ERROR {feature} doesnt have information, {split}')
                    raise
                if dr_tmp.get(1) is None:
                    dr_tmp[1] = 0
                elif dr_tmp.get(0) is None:
                    dr_tmp[0] = 0

                iv_df.loc[idx, 'COUNT'] = dr_tmp[0] + dr_tmp[1]
                iv_df.loc[idx, 'DR'] =  dr_tmp[1] / (dr_tmp[0] + dr_tmp[1])
                iv_df.loc[idx, 'EVENT'] = dr_tmp[1]
                iv_df.loc[idx, 'EVENT_RATE'] = dr_tmp[1] / dr[1]
                iv_df.loc[idx, 'NONEVENT'] = dr_tmp[0]
                iv_df.loc[idx, 'NON_EVENT_RATE'] = dr_tmp[0] / dr[0]
                iv_df.loc[idx, 'WOE'] = auto_woe.woe_dict[feature].cod_dict[bin_num]

            if nan_exist is False:
                # Вставим _MISSING_ в явном виде если его нет.
                # Пока вставка только в категорию с наибольшим количеством объектов.
                tmp_iv_df = iv_df.loc[feat_idxs]
                if nan_to_woe == 'max':
                    max_idx = tmp_iv_df[tmp_iv_df['COUNT'] == tmp_iv_df['WOE'].max()].index[0]
                    iv_df.loc[max_idx, 'MIN_VALUE'] += ' | _MISSING_'
                    iv_df.loc[max_idx, 'MAX_VALUE'] += ' | _MISSING_'
                elif nan_to_woe == 'min':
                    min_idx = tmp_iv_df[tmp_iv_df['COUNT'] == tmp_iv_df['WOE'].min()].index[0]
                    iv_df.loc[min_idx, 'MIN_VALUE'] += ' | _MISSING_'
                    iv_df.loc[min_idx, 'MAX_VALUE'] += ' | _MISSING_'
                else:
                    max_cat_idx = tmp_iv_df[tmp_iv_df['COUNT'] == tmp_iv_df['COUNT'].max()].index[0]
                    iv_df.loc[max_cat_idx, 'MIN_VALUE'] += ' | _MISSING_'
                    iv_df.loc[max_cat_idx, 'MAX_VALUE'] += ' | _MISSING_'

            if else_exist is False:
                # Вставим _ELSE_ в явном виде если его нет.
                # Пока вставка только в категорию с наибольшим количеством объектов.
                tmp_iv_df = iv_df.loc[feat_idxs]
                if else_to_woe == 'max':
                    max_idx = tmp_iv_df[tmp_iv_df['COUNT'] == tmp_iv_df['WOE'].max()].index[0]
                    iv_df.loc[max_idx, 'MIN_VALUE'] += ' | _ELSE_'
                    iv_df.loc[max_idx, 'MAX_VALUE'] += ' | _ELSE_'
                elif else_to_woe == 'min':
                    min_idx = tmp_iv_df[tmp_iv_df['COUNT'] == tmp_iv_df['WOE'].min()].index[0]
                    iv_df.loc[min_idx, 'MIN_VALUE'] += ' | _ELSE_'
                    iv_df.loc[min_idx, 'MAX_VALUE'] += ' | _ELSE_'
                else:
                    max_cat_idx = tmp_iv_df[tmp_iv_df['COUNT'] == tmp_iv_df['COUNT'].max()].index[0]
                    iv_df.loc[max_cat_idx, 'MIN_VALUE'] += ' | _ELSE_'
                    iv_df.loc[max_cat_idx, 'MAX_VALUE'] += ' | _ELSE_'

        # Считаем итоговый IV для признака.
        # Оставляем только те бины, которые имеет смысл учитываеть в расчете IV.
        # Это делается на случай, если есть отдельный бин с NaN, в котором мало значений.
        __feat_idxs = iv_df.loc[feat_idxs][iv_df['COUNT'] >= 20].index
        iv_df.loc[__feat_idxs, 'IV'] = (iv_df.loc[__feat_idxs, 'NON_EVENT_RATE'] - iv_df.loc[__feat_idxs, 'EVENT_RATE']) \
                                        * (iv_df.loc[__feat_idxs, 'NON_EVENT_RATE'] / (iv_df.loc[__feat_idxs, 'EVENT_RATE'] + 0.00001)).apply(lambda x: np.log(x) if x > 0.0001 else 0)
        IV = iv_df.loc[__feat_idxs, 'IV'].sum()
        iv_df.loc[feat_idxs, 'IV'] = IV

    return iv_df


def construction_binning(df: pd.DataFrame, features: list, target: str='target',
        max_bin_count: int=5, min_bin_size: float=0.05, monotonic: bool=False,
        th_const: float=0.005, nan_to_woe: str='max_cat', else_to_woe: str='max_cat',
        n_jobs: int=4) -> pd.DataFrame:
    '''
    Построение бининга за счет внешней открытой библиотеки AutoWOE от Сбера
    https://github.com/sberbank-ai-lab/AutoMLWhitebox

    Parameters:
    df: сэмпла данных, для построения
    features: list[str] список переменных, для которых следует строить бининг
    target: имя таргета
    max_bin_count: максимальное количество бинов для разбиения, по дефолту 5, можно не трогать
    min_bin_size: минимальное количество объектов в бине, по дефолту 0.05, требуется менять
        только в исключительных ситуациях
    monotonic: жесткое условие на монотонность бинов, по дефолту False, можно не трогать
    th_const: порог, по которому признак является константным, если процентное соотношение валидных
        значений меньше этого порога, то признак не учитывается
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
    n_jobs: количество ядер, используемых в построении биннинга

    Returns:
    iv_df: таблица для WOE преобразования
    dropped_feats: признаки, для которых не построилось WOE преобразование
        ВАЖНО: переменные отбрасываются, потому что для них не нашлось сколько-нибудь хорошего
        разбиения, или признак константный исходя из заданного порога
    included_feats: признаки, которые отобрались моделью
    excluded_feats: признаки, которые отфильтровались этой моделью
    auto_woe: модель, если требуется, но в общем случае, она не нужна

    Пример:
        iv_df, dropped_feats, feats, best_features, auto_woe = construction_binning(df_train, features, 'target')

    '''
    from .autowoe.lib.autowoe import AutoWoE

    if nan_to_woe == 'max':
        nan_merge_to = 'to_maxp'
    elif nan_to_woe == 'min':
        nan_merge_to = 'to_minp'
    elif nan_to_woe == 'zero':
        nan_merge_to = 'to_woe_0'
    elif nan_to_woe == 'max_cat':
        nan_merge_to = 'to_maxfreq'
    else:
        nan_merge_to = 'to_maxfreq'
    
    if else_to_woe == 'max':
        cat_merge_to = 'to_maxp'
    elif else_to_woe == 'min':
        cat_merge_to = 'to_minp'
    elif else_to_woe == 'zero':
        cat_merge_to = 'to_woe_0'
    elif else_to_woe == 'max_cat':
        cat_merge_to = 'to_maxfreq'
    else:
        cat_merge_to = 'to_maxfreq'

    num_col = list(df[features].select_dtypes(exclude=object))
    num_feature_type = {x: "real" for x in num_col}

    cat_col = list(df[features].select_dtypes(include=object))
    cat_feature_type = {x: "cat" for x in cat_col}

    features_type = dict(**num_feature_type, **cat_feature_type)
    features = num_col + cat_col

    auto_woe = AutoWoE(interpreted_model=True,
                     monotonic=monotonic, # надо прокидывать
                     max_bin_count=max_bin_count, # надо прокидывать
                     select_type=None,
                     pearson_th=0.9,
                     auc_th=.505,
                     vif_th=10.,
                     imp_th=-99, # 0.00,
                     th_const=th_const, # 100
                     force_single_split=True,
                     th_nan=0.05,
                     th_cat=0.000,
                     woe_diff_th=0.08,
                     min_bin_size=min_bin_size, # надо прокидывать
                     min_bin_mults=(2, 4),
                     min_gains_to_split=(0.0, 0.5, 1.0),
                     auc_tol=1e-4,
                     cat_alpha=100,
                     cat_merge_to=cat_merge_to,  # "to_woe_0",
                     nan_merge_to=nan_merge_to, # "to_woe_0",
                     oof_woe=False,
                     n_folds=4,
                     n_jobs=n_jobs,
                     l1_grid_size=20,
                     l1_exp_scale=6,
                     imp_type="feature_imp",
                     regularized_refit=False,
                     p_val=0.09,
                     debug=False,
                     verbose=0
        )

    best_features = auto_woe.fit(df[features + [target]],
        target_name=target,
        features_type=features_type,
        group_kf=None,
        only_woe_transform=True
    )

    iv_df = construction_iv_df_from_autowoe(df, auto_woe, target, features_type, features, nan_to_woe, else_to_woe)
    dropped_feat_names = list(set(features) - set(iv_df['VAR_NAME'].unique()))
    dropped_feats = pd.DataFrame([[feat, auto_woe.feature_history[feat]] for feat in dropped_feat_names], columns=['VAR_NAME', 'reason'])

    feats = pd.merge(
        pd.DataFrame([[col, val] for col, val in auto_woe.feature_history.items()], columns=['VAR_NAME', 'reason']),
        iv_df[['VAR_NAME', 'IV']].drop_duplicates(),
        on='VAR_NAME'
    )

    return iv_df, dropped_feats, feats, best_features, auto_woe


def change_feature_binning(iv_df, iv_df_old, df_train, col):
    '''
    Значение для NaN берется тем же самым, что и в старом iv_df

    '''
    try:
        # Заменить df на требуемый DataFrame (df_train или другой)
        df1 = df_train[[col, 'target']].copy()
        iv_col = iv_df[iv_df['VAR_NAME'] == col]
    except:
        return iv_df

    if RepresentsFloat(iv_col.iloc[0]['MIN_VALUE']):
        iv_df.loc[iv_df['VAR_NAME'] == col, 'MIN_VALUE'] = iv_df.loc[iv_df['VAR_NAME'] == col, 'MIN_VALUE'].astype(float)
        iv_df.loc[iv_df['VAR_NAME'] == col, 'MAX_VALUE'] = iv_df.loc[iv_df['VAR_NAME'] == col, 'MAX_VALUE'].astype(float)
        iv_col = iv_df[iv_df['VAR_NAME'] == col]

    for i, row in iv_col.iterrows():
        if pd.isna(row['MIN_VALUE']):
            val_count = df1[(df1[col].isna())]['target']
        elif type(row['MIN_VALUE']) in [int, float]:
        # elif RepresentsFloat(row['MIN_VALUE']):
            min_val, max_val = float(row['MIN_VALUE']), float(row['MAX_VALUE'])
            val_count = df1[(df1[col] >= min_val) & (df1[col] <= max_val - 1e-8)]['target']
        else:
            categories = row['MIN_VALUE'].split(' | ')
            val_count = df1[df1[col].isin(categories)]['target']

        iv_df.loc[i, 'COUNT'] = val_count.count()
        iv_df.loc[i, 'EVENT'] = val_count.sum()
        iv_df.loc[i, 'DR'] = float(val_count.sum()) / val_count.count()
        iv_df.loc[i, 'NONEVENT'] = val_count.count() - val_count.sum()

    column_condition = (iv_df['VAR_NAME'] == col) & (iv_df['MIN_VALUE'].isna() == False)
    d3 = iv_df.loc[column_condition]
    iv_df.loc[column_condition, "EVENT_RATE"] = d3.EVENT / d3.EVENT.sum()
    iv_df.loc[column_condition, "NON_EVENT_RATE"] = d3.NONEVENT / d3.NONEVENT.sum()

    d3 = iv_df.loc[column_condition]
    iv_df.loc[column_condition, "WOE"] = np.log((d3.NON_EVENT_RATE / d3.EVENT_RATE).astype(float))

    # Сохраняем тот резульатт для NaN, который предсатвлен в изначальном excel файле.
    iv_df.loc[(iv_df['VAR_NAME'] == col) & (iv_df['MIN_VALUE'].isna())] = iv_df_old[(iv_df['VAR_NAME'] == col) & (iv_df['MIN_VALUE'].isna())]

    iv_df.loc[iv_df['VAR_NAME'] == col, "IV"] = (d3.NON_EVENT_RATE - d3.EVENT_RATE) * np.log((d3.NON_EVENT_RATE / d3.EVENT_RATE).astype(float))

    iv_df.loc[iv_df['VAR_NAME'] == col] = iv_df.loc[iv_df['VAR_NAME'] == col].replace([np.inf, -np.inf], 0)
    iv_df.loc[iv_df['VAR_NAME'] == col, 'IV'] = iv_df.loc[iv_df['VAR_NAME'] == col].IV.sum()

    if iv_df.loc[iv_df['VAR_NAME'] == col]['COUNT'].sum() != df1.shape[0]:
        print(f'ERROR columns "{col}" has less objects in bins than should')

    return iv_df


def correct_binning(iv_df: pd.DataFrame, df_train: pd.DataFrame) -> pd.DataFrame:
    '''
    ВАЖНО! NaN значения заоплняются так же, как это указано в изначальной
    excel таблице, так что их стоит менять там.

    '''
    iv_df_old = iv_df.copy()

    # Пересчитываем WOE и IV для новых, исправленных ручным образом, бинов.
    for col in iv_df['VAR_NAME'].unique():
        iv_df = change_feature_binning(iv_df, iv_df_old, df_train, col)
    
    return iv_df


def print_feature_bins(feature: str, iv_df: pd.DataFrame):
    binning_intervals = []

    if RepresentsFloat(iv_df[iv_df['VAR_NAME'] == feature].iloc[0]['MIN_VALUE']):
        for i, row in iv_df[iv_df['VAR_NAME'] == feature].iterrows():
            if pd.isna(row['MIN_VALUE']):
                binning_intervals.append(('nan', 'nan', row['WOE']))
            else:
                binning_intervals.append(
                    (round(row['MIN_VALUE'], 5), round(row['MAX_VALUE'], 5))
                )
    else:
        for i, row in iv_df[iv_df['VAR_NAME'] == feature].iterrows():
            binning_intervals.append(
                row['MIN_VALUE']
            )

    print(f"'{feature}': {binning_intervals},")


def correct_binning_dict(new_bins: dict, iv_df: pd.DataFrame, df_train: pd.DataFrame) -> pd.DataFrame:
    '''
    ВАЖНО! NaN значения заоплняются так же, как это указано в изначальной
    excel таблице, так что их стоит менять там.

    '''

    for col, bin_values in new_bins.items():
        # Обработка непрерывных переменных
        if RepresentsFloat(iv_df[iv_df['VAR_NAME'] == col].iloc[0]['MIN_VALUE']):
            iv_col_nan = iv_df[(iv_df['VAR_NAME'] == col) & (iv_df['MIN_VALUE'].isna())]
            iv_df = iv_df[iv_df['VAR_NAME'] != col]

            for bin in bin_values:
                if bin[0] != 'nan':
                    iv_df = iv_df.append(
                        {'VAR_NAME': col, 'MIN_VALUE': float(bin[0]), 'MAX_VALUE': float(bin[1])},
                        ignore_index=True
                    )
                else:
                    if len(bin) == 3:
                        iv_col_nan['WOE'] = bin[2]
                    iv_df = iv_df.append(
                        iv_col_nan,
                        ignore_index=True
                    )
        else:
            iv_df = iv_df[iv_df['VAR_NAME'] != col]
            for bin in bin_values:
                iv_df = iv_df.append(
                    {'VAR_NAME': col, 'MIN_VALUE': bin, 'MAX_VALUE': bin},
                    ignore_index=True
                )

    iv_df.reset_index(drop=True, inplace=True)
    iv_df_old = iv_df.copy()

    # Пересчитываем WOE и IV для новых, исправленных ручным образом, бинов.
    for col in new_bins.keys():
        try:
            iv_df = change_feature_binning(iv_df, iv_df_old, df_train, col)
        except Exception as e:
            print(col)
            display(iv_df[iv_df['VAR_NAME'] == col])
            raise(e)
        
    
    return iv_df