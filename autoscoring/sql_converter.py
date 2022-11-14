from skompiler import skompile
import sqlparse
import re
from .AS import RepresentsFloat, ivs_to_excel, iv_groups
import pandas as pd


def _change_probability_to_score_tree(WORD, SQL: str) -> str:
    '''
    Преобразуем значение вероятности (класса 0) в листе в значение скора.

    '''
    # Ищем все матчи, которые удовлетворяют листу дерева
    for match in re.findall(WORD + r' \d\.\d*', SQL):
        probability = match.split(' ')[1]

        # Заменяем значение вероятности на значение скора в этом же месте
        score = int(float(probability) * 1000)
        SQL = SQL.replace(match, WORD + f' {score}')

    return SQL


def _add_woe_transform_to_sql(sql: str, ivs: pd.DataFrame, vars_woe, nan_to_woe: str='max', else_to_woe: str='max') -> str:
    '''
    Добавялем строки в SQL с WOE преобразованием.

    ivs: таблица для биннинга (iv_df)
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
    # vars = var_name_original(vars_woe)

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

    sql+='''
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


def convert_model_to_sql(model, vars_woe, iv_df, nan_to_woe: str='max', else_to_woe: str='max') -> str:
    '''
    Преобразуем sklearn модель в SQL скрипт
    Поддерживается DecisionTreeClassiifier
    
    model: обученный объект sklearn модели, который требуется сконвертировать в SQL
    vars_woe: названия WOE признаков, которые учавтсвуют в модели,
                 ВАЖНО иметь тот же порядок, как и при обучении модели

    '''

    expr = skompile(model.predict_proba, vars_woe)
    sql = expr.to('sqlalchemy/mysql')
    sql = sql.split(',')[0]  # Берем y1 - вероятность принадлежности к классу 0.

    # Форматируем SQL код из однострочного в читаемый вид.
    sql_formated = sqlparse.format(sql, reindent=True, indent_width=1,
                                    indent_tabs=False, reindent_aligned=False)

    # Делаем замену значения вероятности в листе на значение скора.
    SQL = _change_probability_to_score_tree('THEN', sql_formated)
    SQL = _change_probability_to_score_tree('ELSE', SQL)
    SQL = SQL.replace('y1', 'Scoring')
    SQL = SQL.replace('SELECT', 'SELECT sc.*,\n\t')

    # Добавялем SQL код с WOE преобразованием.
    tmp = pd.DataFrame(columns=vars_woe)
    ivs = ivs_to_excel(iv_df, tmp[vars_woe])
    ivs = iv_groups(ivs)

    SQL = _add_woe_transform_to_sql(SQL, ivs, vars_woe, nan_to_woe, else_to_woe)
    SQL = SQL.replace("`", "'")

    return SQL


def _sql_condition_leaf(leaf_thrsh, var_names) -> str:
    '''
    Прописываем условия на выделение листа в дереве
    Пример результата: x1 <= 0.123 AND x2 > 1.2 AND x1 > 0.2
    leaf_thrsh: объект, который содержит
         номер признака, направленеи разделения (L, R), отсечку для разбиения

    '''
    sql_cond = ''
    for feat, direction, split in leaf_thrsh:
        if direction == 'L':
            sign = '<='
        else:
            sign = '>'
        
        sql_cond += f'{var_names[feat]} {sign} {split} AND '
    
    return sql_cond[:-5]  # Не учитываем последний AND


def _create_logreg_at_leaf(model, var_names):
    '''
    Создаем SQL скрипт, для модели в листе дерева решений.
    Пример результата: round( 1000 * 1 / (1 + exp(-0.809 * x0 + 1.659 * x1 + 0.1549)))
    model: модель LogisticRegression, расположенная в соответсвующем листе дерева решения.

    '''

    # Преобразуем логистическую регрессию в SQL с помощью библиотеки skompile.
    expr = skompile(model.predict_proba, var_names)
    sql = expr.to('sqlalchemy/mysql')

    # Оставляем только ту часть скрипта, в которой расположеная сама формула, обрезая лишнее.
    sql_logreg = sql.split('\n SELECT ')[1].split(' AS')[0]

    # Заменяем названия признаков.
    for i, name in enumerate(var_names):
        sql_logreg = sql_logreg.replace(f'_tmp1.f{i+1}', name)

    # Добавялем округление и преобразование вероятности в скор
    # (чем больше, тем выше вероятность класса 0)
    sql_logreg = 'round( 1000 * ' + sql_logreg.replace('exp(-', 'exp')

    return sql_logreg


def convert_linear_tree_to_sql(clf, var_names, iv_df, nan_to_woe: str='max', else_to_woe: str='max') -> str:
    '''
    Преобразуем модель LinearTree в SQL.
    clf: обученная модель LinearTree.

    '''

    SQL = 'SELECT sc.*,\n'
    for i, (name, leaf) in enumerate(clf._leaves.items()):
        tree_leaf_cond = _sql_condition_leaf(leaf.threshold, var_names)
        model = leaf.model
        sql_logreg = _create_logreg_at_leaf(model, var_names)

        if i == 0:
            SQL += f"\tIF {tree_leaf_cond} THEN \n\t\t{sql_logreg}\n"
        else:
            SQL += f"\tELSEIF {tree_leaf_cond} THEN \n\t\t{sql_logreg}\n"

    SQL += '\t' + 'END IF as Scoring\n'

     # Добавялем SQL код с WOE преобразованием.
    tmp = pd.DataFrame(columns=var_names)
    ivs = ivs_to_excel(iv_df, tmp[var_names])
    ivs = iv_groups(ivs)

    SQL = _add_woe_transform_to_sql(SQL, ivs, var_names, nan_to_woe, else_to_woe)

    return SQL