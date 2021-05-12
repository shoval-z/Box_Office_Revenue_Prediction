import pandas as pd
from textblob import TextBlob
from collections import OrderedDict
import catboost
import ast
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
import numpy as np
import time
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


def wrapper_method(X_train, Y_train, model, num_features):
    classifier_pipeline = make_pipeline(MinMaxScaler(), model)
    sfs1 = SFS(classifier_pipeline,
               k_features=num_features,
               forward=True,
               scoring='neg_mean_squared_log_error'
               )
    sfs1.fit(X_train, Y_train)

    return list(sfs1.k_feature_names_), sfs1.subsets_[len(sfs1.subsets_)]['avg_score']


def data_prep(train_df, budget_val=None, runtime_val=None):
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

    train_df['year'] = pd.DatetimeIndex(train_df['release_date']).year
    train_df['month'] = pd.DatetimeIndex(train_df['release_date']).month
    train_df['budget'].replace({0: None}, inplace=True)
    if budget_val:
        train_df['update_budget'] = np.where(train_df['budget'].isna(), budget_val, train_df['budget'])
    else:
        train_df['update_budget'] = np.where(train_df['budget'].isna(), train_df['budget'].median(), train_df['budget'])
        budget_median = train_df['budget'].median()
    train_df['budget'].replace({None: 0}, inplace=True)
    train_df['update_budget'] = pd.to_numeric(train_df['update_budget'])
    # train_df['update_budget'] = train_df.groupby(['year'])['budget'].transform(lambda x: x.fillna(x.mean()))
    # train_df['update_budget'].fillna((train_df['update_budget'].mean()), inplace=True)
    train_df['is_original_language_en'] = np.where(train_df['original_language'] == 'en', 1, 0)
    train_df['has_homepage'] = np.where(train_df['homepage'].isna(), 0, 1)
    train_df['has_collection'] = np.where(train_df['belongs_to_collection'].isna(), 0, 1)
    train_df['has_tagline'] = np.where(train_df['tagline'].isna(), 0, 1)
    if runtime_val:
        train_df['runtime'] = np.where(train_df['runtime'].isna(), runtime_val, train_df['runtime'])
    else:
        train_df['runtime'] = np.where(train_df['runtime'].isna(), train_df['runtime'].median(), train_df['runtime'])
        runtime_median = train_df['runtime'].median()
    train_df['is_holiday'] = np.where(train_df['month'].isin([11, 12, 5, 6, 7]), 1, 0)
    train_df['budget_year_ratio'] = round(train_df['update_budget'] / train_df['year'], 2)
    train_df['popularity_year_ratio'] = round(train_df['popularity'] / train_df['year'], 2)
    train_df['vote_count_year_ratio'] = round(train_df['vote_count'] / train_df['year'], 2)
    train_df['overview'] = np.where(train_df['overview'].isna(), '', train_df['overview'])
    train_df['sentiment_overview'] = train_df['overview'].map(lambda text: TextBlob(text).sentiment.polarity)
    train_df['sentiment_title'] = train_df['title'].map(lambda text: TextBlob(text).sentiment.polarity)
    train_df['inflation_Budget'] = train_df['update_budget'] + (train_df['update_budget'] * 1.8) / (
                100 * (2020 - train_df['year']))
    train_df['log_budget'] = np.log1p(train_df['update_budget'])
    sort_production_companies = sorted(production_companies_dict.items(), key=lambda x: x[1], reverse=True)
    sort_production_companies = dict(sort_production_companies)
    sort_production_companies_df = pd.DataFrame(columns=['name', 'val'])
    sort_production_companies_df['name'] = sort_production_companies.keys()
    sort_production_companies_df['val'] = sort_production_companies.values()

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

    g_list = list()
    different_g = list()
    for genres in train_df.genres:
        tmp = []
        for i in genres:
            tmp.append(i['name'])
            if i['name'] not in different_g:
                different_g.append(i['name'])
        g_list.append(tmp)

    # df_new = pd.DataFrame({"new_genres" :g_list })
    train_df['new_genre'] = g_list
    train_df['amount_of_genre'] = [len(i) for i in train_df.new_genre]
    for g in different_g:
        train_df[g] = np.where(train_df['new_genre'].str.contains(g, regex=False), 1, 0)

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

    # create binary feature for popular production company
    def is_in_list(x, top_p):
        for item in x:
            if item in top_p:
                return 1
        return 0

    p_list, producer_list, e_producer_list, director_list, actors_list = list(), list(), list(), list(), list()
    top_p = list(sort_production_companies_df['name'][0:10])
    top_producer = list(crew_dict['Producer'])[0:10]
    top_e_producer = list(crew_dict['Executive Producer'])[0:2]
    top_director = list(crew_dict['Director'])[0:3]
    top_actore = list(actors_dict)[0:50]

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
    train_df['is_in_popular_actores'] = train_df['new_actor'].apply(is_in_list, args=(top_actore,))

    x_train = train_df[['has_tagline', 'popularity', 'runtime', 'video', 'vote_average', 'vote_count',
                             'is_original_language_en', 'has_homepage',
                             'update_budget', 'month', 'is_in_popular_production', 'is_country_us', 'has_collection',
                             'amount_of_genre', 'year',
                             'Fantasy', 'Action', 'Adventure', 'Comedy', 'Horror', 'Documentary', 'Mystery', 'Thriller',
                             'Science Fiction', 'Drama',
                             'Romance', 'Crime', 'War', 'History', 'Music', 'Family', 'Animation', 'Western',
                             'budget_year_ratio', 'sentiment_overview', 'sentiment_title',
                             'is_in_popular_producer', 'is_in_popular_executive_producer', 'is_in_popular_director',
                             'is_in_popular_actores', 'is_holiday',
                             'popularity_year_ratio', 'vote_count_year_ratio',
                             'inflation_Budget', 'log_budget']]
    y_train = train_df['revenue']

    if budget_val and runtime_val:
        return x_train, y_train
    else:
        return x_train, y_train, runtime_median, budget_median


S = time.time()
train_df = pd.read_csv('hw1_data/train.tsv', sep="\t")
x_train, y_train, runtime_median, budget_median = data_prep(train_df)
y_train = np.log1p(y_train)
print('selection')
model = catboost.CatBoostRegressor(iterations=700, early_stopping_rounds=100,rsm=0.8,learning_rate=0.01,depth=5, random_state=42,eval_metric='MSLE').fit(x_train, y_train)
num_features = 'best'
feature_list, sfs= wrapper_method(x_train, y_train, model, num_features)
print(feature_list, sfs)