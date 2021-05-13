import pandas as pd
from textblob import TextBlob
from collections import OrderedDict
import catboost
import ast
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pickle


def data_prep(train_df, mode='train', year_bin=False):
    if mode == 'test':
        object_dict = pickle.load(open('data/relevant_data.pkl', "rb"))
    train_df['genres'] = train_df.genres.apply(lambda s: list(ast.literal_eval(s)))
    train_df['production_companies'] = train_df.production_companies.apply(lambda s: list(ast.literal_eval(s)))
    train_df['production_countries'] = train_df.production_countries.apply(lambda s: list(ast.literal_eval(s)))
    train_df['spoken_languages'] = train_df.spoken_languages.apply(lambda s: list(ast.literal_eval(s)))
    train_df['cast'] = train_df.cast.apply(lambda s: list(ast.literal_eval(s)))
    train_df['crew'] = train_df.crew.apply(lambda s: list(ast.literal_eval(s)))

    genres_dict = dict()
    production_companies_dict = dict()
    production_countries_dict = dict()
    spoken_languages_dict = dict()
    actors_dict = dict()
    crew_dict = {'Producer': dict(), 'Director': dict(), 'Executive Producer': dict()}

    for idx, row in train_df.iterrows():
        genres_list = row.genres
        prod_list = row.production_companies
        production_countries_list = row.production_countries
        spoken_languages_list = row.spoken_languages
        actors_list = row.cast
        crew_list = row.crew

        for g in genres_list:
            if g['name'] not in genres_dict.keys():
                genres_dict[g['name']] = 1
            else:
                genres_dict[g['name']] += 1

        for p in prod_list:
            if p['name'] not in production_companies_dict.keys():
                production_companies_dict[p['name']] = 1
            else:
                production_companies_dict[p['name']] += 1

        for c in production_countries_list:
            if c['name'] not in production_countries_dict.keys():
                production_countries_dict[c['name']] = 1
            else:
                production_countries_dict[c['name']] += 1

        for l in spoken_languages_list:
            if l['name'] not in spoken_languages_dict.keys():
                spoken_languages_dict[l['name']] = 1
            else:
                spoken_languages_dict[l['name']] += 1

        for a in actors_list:
            if a['name'] not in actors_dict.keys():
                actors_dict[a['name']] = 1
            else:
                actors_dict[a['name']] += 1

        for c in crew_list:
            if c['job'] not in ['Producer', 'Director', 'Executive Producer']:
                continue
            if c['name'] not in crew_dict[c['job']].keys():
                crew_dict[c['job']][c['name']] = 1
            else:
                crew_dict[c['job']][c['name']] += 1

    for key, value in crew_dict.items():
        crew_dict[key] = OrderedDict(sorted(value.items(), key=lambda x: x[1], reverse=True))
    actors_dict = OrderedDict(sorted(actors_dict.items(), key=lambda x: x[1], reverse=True))

    # split the release_date colum to year, month and is_holiday
    train_df['year'] = pd.DatetimeIndex(train_df['release_date']).year
    train_df['month'] = pd.DatetimeIndex(train_df['release_date']).month
    train_df['is_holiday'] = np.where(train_df['month'].isin([11, 12, 5, 6, 7]), 1, 0)

    # update the budget column with the median value of the train set
    train_df['budget'].replace({0: None}, inplace=True)
    if mode == 'test':
        train_df['update_budget'] = np.where(train_df['budget'].isna(), object_dict['budget_median'],
                                             train_df['budget'])
    else:  # 'train'
        train_df['update_budget'] = np.where(train_df['budget'].isna(), train_df['budget'].median(), train_df['budget'])
        budget_median = train_df['budget'].median()
    train_df['update_budget'] = pd.to_numeric(train_df['update_budget'])

    # attempt to fill the budget with the mean according to different years
    # train_df['update_budget'] = train_df.groupby(['year'])['budget'].transform(lambda x: x.fillna(x.mean()))
    # train_df['update_budget'].fillna((train_df['update_budget'].mean()), inplace=True)

    # create some binary features
    train_df['is_original_language_en'] = np.where(train_df['original_language'] == 'en', 1, 0)
    train_df['has_homepage'] = np.where(train_df['homepage'].isna(), 0, 1)
    train_df['has_collection'] = np.where(train_df['belongs_to_collection'].isna(), 0, 1)
    train_df['has_tagline'] = np.where(train_df['tagline'].isna(), 0, 1)

    # complete the runtime nan value with the median value of the train set
    if mode == 'test':
        train_df['runtime'] = np.where(train_df['runtime'].isna(), object_dict['runtime_median'], train_df['runtime'])
    else:
        train_df['runtime'] = np.where(train_df['runtime'].isna(), train_df['runtime'].median(), train_df['runtime'])
        runtime_median = train_df['runtime'].median()

    # use year+ other correlated features to create new features
    train_df['budget_year_ratio'] = round(train_df['update_budget'] / train_df['year'], 2)
    train_df['popularity_year_ratio'] = round(train_df['popularity'] / train_df['year'], 2)
    train_df['vote_count_year_ratio'] = round(train_df['vote_count'] / train_df['year'], 2)

    # additional manipulation on the budget column
    train_df['inflation_Budget'] = train_df['update_budget'] + (train_df['update_budget'] * 1.8) / (
            100 * (2020 - train_df['year']))
    train_df['log_budget'] = np.log1p(train_df['update_budget'])

    # sentiment analyses on the overview and title
    train_df['overview'] = np.where(train_df['overview'].isna(), '', train_df['overview'])
    train_df['sentiment_overview'] = train_df['overview'].map(lambda text: TextBlob(text).sentiment.polarity)
    train_df['sentiment_title'] = train_df['title'].map(lambda text: TextBlob(text).sentiment.polarity)

    sort_production_companies = sorted(production_companies_dict.items(), key=lambda x: x[1], reverse=True)
    sort_production_companies = dict(sort_production_companies)
    sort_production_companies_df = pd.DataFrame(columns=['name', 'val'])
    sort_production_companies_df['name'] = sort_production_companies.keys()
    sort_production_companies_df['val'] = sort_production_companies.values()

    # creating the columns that relate to genres- "multi_hot" per genre and amount of genres per movie
    g_list = list()
    different_g = list()
    for genres in train_df.genres:
        tmp = []
        for i in genres:
            tmp.append(i['name'])
            if i['name'] not in different_g:
                different_g.append(i['name'])
        g_list.append(tmp)

    train_df['new_genre'] = g_list
    train_df['amount_of_genre'] = [len(i) for i in train_df.new_genre]
    for g in different_g:
        train_df[g] = np.where(train_df['new_genre'].str.contains(g, regex=False), 1, 0)

    # create the 'is_country_us' binary feature
    def is_in_list(x):
        for item in x:
            if item == 'United States of America':
                return 1
        return 0

    c_list = list()
    for country in train_df.production_countries:
        tmp = []
        for i in country:
            tmp.append(i['name'])
        c_list.append(tmp)

    train_df['country'] = c_list
    train_df['is_country_us'] = train_df['country'].apply(is_in_list)
    train_df['is_country_us'].value_counts()

    # create binary feature for popular production company and popular actors, producers,
    # executive producers and directors
    def is_in_list(x, top_p):
        for item in x:
            if item in top_p:
                return 1
        return 0

    p_list, producer_list, e_producer_list, director_list, actors_list = list(), list(), list(), list(), list()
    if mode == 'test':
        top_p = object_dict['top_p']
        top_producer = object_dict['top_producer']
        top_e_producer = object_dict['top_e_producer']
        top_director = object_dict['top_director']
        top_actors = object_dict['top_actors']
    else:
        top_p = list(sort_production_companies_df['name'][0:10])
        top_producer = list(crew_dict['Producer'])[0:10]
        top_e_producer = list(crew_dict['Executive Producer'])[0:2]
        top_director = list(crew_dict['Director'])[0:3]
        top_actors = list(actors_dict)[0:50]

    for production in train_df.production_companies:
        tmp = []
        for i in production:
            tmp.append(i['name'])
        p_list.append(tmp)

    for c in train_df.crew:
        tmp_p, tmp_e_p, tmp_d = [], [], []
        for i in c:
            if i['job'] == 'Producer':
                tmp_p.append(i['name'])
            elif i['job'] == 'Executive Producer':
                tmp_e_p.append(i['name'])
            else:
                tmp_d.append(i['name'])
        producer_list.append(tmp_p)
        e_producer_list.append(tmp_e_p)
        director_list.append(tmp_d)

    for a in train_df.cast:
        tmp = []
        for i in a:
            tmp.append(i['name'])
        actors_list.append(tmp)

    train_df['new_production'] = p_list
    train_df['new_producer'] = producer_list
    train_df['new_e_producer'] = e_producer_list
    train_df['new_director'] = director_list
    train_df['new_actor'] = actors_list

    train_df['is_in_popular_production'] = train_df['new_production'].apply(is_in_list, args=(top_p,))
    train_df['is_in_popular_producer'] = train_df['new_producer'].apply(is_in_list, args=(top_producer,))
    train_df['is_in_popular_executive_producer'] = train_df['new_e_producer'].apply(is_in_list, args=(top_e_producer,))
    train_df['is_in_popular_director'] = train_df['new_director'].apply(is_in_list, args=(top_director,))
    train_df['is_in_popular_actores'] = train_df['new_actor'].apply(is_in_list, args=(top_actors,))

    # if training different model for different year bin
    if year_bin:
        conditions = [
            (train_df['year'] <= 1980),
            (train_df['year'] > 1980) & (train_df['year'] <= 1990),
            (train_df['year'] > 1990) & (train_df['year'] <= 2000),
            (train_df['year'] > 2000) & (train_df['year'] <= 2005),
            (train_df['year'] > 2005) & (train_df['year'] <= 2010),
            (train_df['year'] > 2010) & (train_df['year'] <= 2015),
            (train_df['year'] > 15)
        ]
        values = ['1980', '1980-1990', '1990-2000', '2000-2005', '2005-2010', '2010-2015', '2015', ]
        train_df['year_bins'] = np.select(conditions, values)
        train_df = train_df[train_df['year_bins'] == year_bin]

    #### choosing only the wanted features for out x_train table ####
    ## first model ##
    # x_train = train_df[['popularity', 'vote_count', 'update_budget', 'has_collection', 'Adventure',
    # 'budget_year_ratio', 'popularity_year_ratio', 'vote_count_year_ratio', 'inflation_Budget', 'log_budget']]
    ## second model ##
    # x_train = train_df[
    #     ['popularity', 'runtime', 'vote_count', 'is_original_language_en', 'has_homepage', 'update_budget',
    #      'is_in_popular_production', 'is_country_us', 'has_collection', 'year', 'Comedy', 'Documentary', 'Animation',
    #      'budget_year_ratio', 'vote_count_year_ratio', 'inflation_Budget','is_in_popular_executive_producer',
    #      'log_budget']]
    ## best model ##
    x_train = train_df[
        ['id', 'popularity', 'runtime', 'vote_count', 'is_original_language_en', 'has_homepage', 'update_budget',
         'is_in_popular_production', 'is_country_us', 'has_collection', 'year', 'Comedy', 'Documentary', 'Animation',
         'budget_year_ratio', 'vote_count_year_ratio', 'inflation_Budget', 'log_budget']]
    y_train = train_df['revenue']
    # saving the needed value for the test set
    if mode == 'train':
        object_dict = {'runtime_median': runtime_median, 'budget_median': budget_median, 'top_p': top_p,
                       'top_producer': top_producer, 'top_e_producer': top_e_producer, 'top_director': top_director,
                       'top_actors': top_actors}
        pickle.dump(object_dict, open('relevant_data.pkl', "wb"))
    return x_train, y_train


def run_different_algorithms():
    train_df = pd.read_csv('data/train.tsv', sep="\t")
    test_df = pd.read_csv('data/test.tsv', sep="\t")

    x_train, y_train = data_prep(train_df, mode='train')
    x_test, y_test = data_prep(test_df, mode='test')
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    pickle.dump(scaler, open('data/min_max_scaler.pkl', "wb"))

    new_train_x = scaler.transform(x_train)
    new_test_x = scaler.transform(x_test)

    # random forest
    model = RandomForestRegressor(max_depth=10, random_state=40)
    y_train = np.log1p(y_train)
    model.fit(new_train_x, y_train)
    y_pred = model.predict(new_test_x)
    y_pred = np.expm1(y_pred)

    y_pred_rf = y_pred

    RMSLE = np.sqrt(mean_squared_log_error(y_test, y_pred))
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    print('Random forest RMSLE: ', RMSLE)
    print('Random forest RMSE: ', RMSE)

    ##xgboost
    parameters_xgb = dict()
    parameters_xgb['alpha'] = 0
    parameters_xgb['subsample'] = 0.5
    parameters_xgb['learning_rate'] = 0.01
    parameters_xgb['max_depth'] = 6
    parameters_xgb['min_child_weight'] = 3
    parameters_xgb['colsample_bytree'] = 0.7

    X_train_mini, X_val, y_train_mini, y_val = train_test_split(new_train_x, y_train, test_size=0.2, random_state=42)

    train_set = xgb.DMatrix(X_train_mini, label=y_train_mini)
    val_set = xgb.DMatrix(X_val, label=y_val)
    test_set = xgb.DMatrix(new_test_x, label=y_test)

    clf_xgb = xgb.train(params=parameters_xgb,
                        dtrain=train_set,
                        num_boost_round=1000,
                        evals=[(val_set, "Test")],
                        early_stopping_rounds=100)
    importance = clf_xgb.get_score(importance_type='gain')
    importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print(importance)
    y_pred = clf_xgb.predict(test_set)
    y_pred = np.expm1(y_pred)
    y_pred_xgb = y_pred

    RMSLE = np.sqrt(mean_squared_log_error(y_test, y_pred))
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    print('xgboost RMSLE: ', RMSLE)
    print('xgboost RMSE: ', RMSE)
    pickle.dump(clf_xgb, open('data/xgb_model.pkl', "wb"))

    # catboost
    model = catboost.CatBoostRegressor(iterations=700, early_stopping_rounds=100, rsm=0.8, learning_rate=0.01, depth=6,
                                       random_state=42, eval_metric='MSLE')
    model.fit(X_train_mini, y_train_mini, eval_set=(X_val, y_val))
    y_pred = model.predict(new_test_x)
    y_pred = np.expm1(y_pred)
    y_pred = [max(item, 0) for item in y_pred]
    y_pred_catboost = y_pred
    RMSLE = np.sqrt(mean_squared_log_error(y_test, y_pred))
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    print('catboost RMSLE: ', RMSLE)
    print('catboost RMSE: ', RMSE)

    ##lightgbm
    parameters_lgb = dict()
    parameters_lgb['learning_rate'] = 0.01
    parameters_lgb['max_depth'] = 5
    parameters_lgb['max_bin'] = 255
    parameters_lgb['num_leaves'] = 31
    parameters_lgb['feature_fraction'] = 0.5
    parameters_lgb['lambda_l1'] = 0.2

    X_train_mini, X_val, y_train_mini, y_val = train_test_split(new_train_x, y_train, test_size=0.2, random_state=42)

    clf_lgb = lgb.train(params=parameters_lgb,
                        train_set=lgb.Dataset(X_train_mini, y_train_mini),
                        num_boost_round=10000,
                        valid_sets=[lgb.Dataset(X_val, y_val)],
                        valid_names='Test', )

    y_pred = clf_lgb.predict(new_test_x)
    y_pred = [max(item, 0) for item in y_pred]
    y_pred = np.expm1(y_pred)
    RMSLE = np.sqrt(mean_squared_log_error(y_test, y_pred))
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    print('lightgbm RMSLE: ', RMSLE)
    print('lightgbm RMSE: ', RMSE)

    # LinearRegression
    reg = LinearRegression().fit(new_train_x, y_train)
    y_pred = reg.predict(new_test_x)
    y_pred = np.expm1(y_pred)
    y_pred = [max(item, 0) for item in y_pred]
    RMSLE = np.sqrt(mean_squared_log_error(y_test, y_pred))
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    print('LinearRegression RMSLE: ', RMSLE)
    print('LinearRegression RMSE: ', RMSE)

    # Lasso
    reg = Lasso(alpha=0.0001, precompute=True, max_iter=1000,
                positive=True, random_state=9999, selection='random')
    reg.fit(new_train_x, y_train)
    y_pred = reg.predict(new_test_x)
    y_pred = np.expm1(y_pred)
    y_pred = [max(item, 0) for item in y_pred]
    RMSLE = np.sqrt(mean_squared_log_error(y_test, y_pred))
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    print('Lasso RMSLE: ', RMSLE)
    print('Lasso RMSE: ', RMSE)

    # Ridge
    reg = Ridge(alpha=0.0001, max_iter=1000, random_state=9999)
    reg.fit(new_train_x, y_train)
    y_pred = reg.predict(new_test_x)
    y_pred = np.expm1(y_pred)
    y_pred = [max(item, 0) for item in y_pred]
    RMSLE = np.sqrt(mean_squared_log_error(y_test, y_pred))
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    print('Ridge RMSLE: ', RMSLE)
    print('Ridge RMSE: ', RMSE)

    # ensemble catbost and xgboost
    combine_y_pred = np.mean(np.array([y_pred_catboost, y_pred_xgb, y_pred_rf]), axis=0)
    combine_y_pred = [max(item, 0) for item in combine_y_pred]
    RMSLE = np.sqrt(mean_squared_log_error(y_test, combine_y_pred))
    RMSE = np.sqrt(mean_squared_error(y_test, combine_y_pred))
    print('ensemble-rf,catb,xgb RMSLE: ', RMSLE)
    print('ensemble-rf,catb,xgb RMSE: ', RMSE)
    combine_y_pred = np.mean(np.array([y_pred_catboost, y_pred_xgb]), axis=0)
    combine_y_pred = [max(item, 0) for item in combine_y_pred]
    RMSLE = np.sqrt(mean_squared_log_error(y_test, combine_y_pred))
    RMSE = np.sqrt(mean_squared_error(y_test, combine_y_pred))
    print('ensemble-catb,xgb RMSLE: ', RMSLE)
    print('ensemble-catb,xgb RMSE: ', RMSE)


if __name__ == '__main__':
    run_different_algorithms()
