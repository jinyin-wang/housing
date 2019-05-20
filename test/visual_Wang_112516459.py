import os
import sys
import pymysql
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
import matplotlib
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from flask import Flask
from flask import render_template
from flask import request

import statsmodels.formula.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn import manifold
from sklearn.metrics import pairwise_distances
import pickle
import numpy as np

app = Flask(__name__)

imp_ftrs = []
samplesize = 200


def total_room(df):
    plt.figure(figsize=(10, 4))
    plt.hist(df[df['total_bedrooms'].notnull()]['total_bedrooms'], bins=20, color='green')  # histogram of totalbedrooms
    # data has some outliers
    (df['total_bedrooms'] > 4000).sum()
    plt.title('frequency historgram')
    plt.xlabel('total bedrooms')
    plt.ylabel('frequency')
    plt.show()


# we will calculate the median for total_bedrooms based  upon categories of ocean_proximity column
def calc_categorical_median(x):
    """this function fill the missing values of total_bedrooms based upon categories of ocean_proximity"""
    unique_colums_ocean_proximity = x['ocean_proximity'].unique()
    for i in unique_colums_ocean_proximity:
        median = x[x['ocean_proximity'] == i]['total_bedrooms'].median()
        x.loc[x['ocean_proximity'] == i, 'total_bedrooms'] = x[x['ocean_proximity'] == i]['total_bedrooms'].fillna(
            median)


def median_house_value():
    # we can see that area where median price frequencey for >= 500000 is more and could be a outlier or wrong data

    plt.figure(figsize=(10, 6))
    sns.distplot(df['median_house_value'], color='grey')
    plt.show()


def popu_house_value():
    plt.figure(figsize=(10, 6))

    plt.scatter(df['population'], df['median_house_value'], c=df['median_house_value'], s=df['median_income'] * 50,
                edgecolors='red')
    plt.colorbar()
    plt.title('population vs house value')
    plt.xlabel('population')
    plt.ylabel('house value')
    plt.plot()
    plt.show()


def price_geo_coordinates():
    plt.figure(figsize=(15, 10))
    plt.scatter(df['longitude'], df['latitude'], c=df['median_house_value'], s=df['population'] / 10, cmap='viridis')
    plt.colorbar()
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.title('house price on basis of geo-coordinates')
    plt.show()


def corr_matrix():
    # corelation matrix
    plt.figure(figsize=(11, 7))
    sns.heatmap(cbar=False, annot=True, data=df.corr() * 100, cmap='coolwarm')
    plt.title('% Corelation Matrix')
    plt.show()


def ocean_proximity():
    # barplot on ocean_proximity categories
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='ocean_proximity')
    plt.plot()
    plt.show()


def ocean_prox_median_value1():
    # boxplot of house value on ocean_proximity categories
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='ocean_proximity', y='median_house_value', palette='viridis')
    plt.plot()
    plt.show()


def ocean_prox_median_value2():
    plt.figure(figsize=(10, 6))

    sns.stripplot(data=df, x='ocean_proximity', y='median_house_value', jitter=0.3)
    plt.plot()
    plt.show()


# def clean_dataset(x):
#     assert isinstance(x, pd.DataFrame), "x needs to be a pd.DataFrame"
#     x.dropna(inplace=True)
#     indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
#     return x[indices_to_keep].astype(np.float64)


# def pca_random():
#     print ('in pca_random')
#     data_columns = []
#     try:
#         global random_samples
#         global imp_ftrs
#         pca_data = PCA(n_components=2)
#         X = random_samples
#         pca_data.fit(X)
#         X = pca_data.transform(X)
#         data_columns = pd.DataFrame(X)
#         # This is for tool-tip showcasing. Should have taken corresponding samples from random sampling or adaptive sampling
#         # not the first 200 samples. So, needs to be changed similarly in all other functions.
#         for i in range(0, 2):
#             data_columns[ftrs[imp_ftrs[i]]] = data_csv_original[ftrs[imp_ftrs[i]]][:samplesize]
#         # We actually donot use clusterId in random sampling but is sent because otherwise Javascript will brea, because it expects 5 columns.
#         data_columns['clusterid'] = data_csv['kcluster'][:samplesize]
#         print(data_columns)
#         print(":::")
#
#         # data_columns['departure'] = data_csv['DepTime'][:samplesize]
#         # data_columns['arrival'] = data_csv['ArrTime'][:samplesize]
#         # pca_variance = pca_data.explained_variance_ratio_
#         # data_columns['variance'] = pandas.DataFrame(pca_variance)[0]
#     except:
#         e = sys.exc_info()[0]
#         print (e)
#     return pd.DataFrame.to_json(data_columns)


def pca_kmeans(df):
    # converting ocean_proximity to dummies
    # df = pd.concat([pd.get_dummies(df['ocean_proximity'], drop_first=True), df], axis=1).drop('ocean_proximity', axis=1)
    # df['income per working population'] = df['median_income'] / (df['population'] - df['households'])
    # df['bed per house'] = df['total_bedrooms'] / df['total_rooms']
    # df['h/p'] = df['households'] / df['population']

    # df = pd.concat([df, pd.get_dummies(df['housing_median_age'].apply(type_building), drop_first=True)], axis=1)
    x = df.drop('price', axis=1).values
    y = df['price'].values
    from sklearn.model_selection import train_test_split
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=0)

    from sklearn.preprocessing import MinMaxScaler
    ms = MinMaxScaler()
    xtrain = ms.fit_transform(xtrain)

    '''k-means'''
    plot_kmeans_elbow(xtrain)  # we get 3
    draw_kmeans(xtrain, 3)
    print("xtrain_bf_pca", xtrain)

    xtest = ms.transform(xtest)
    x_train_variance = list(map(lambda x: x * 100, c_variance(xtrain)))
    print("xtrain_pca", x_train_variance)

    plt.figure(figsize=(15, 10))
    plt.plot(np.arange(1, xtrain.shape[1] + 1), x_train_variance, marker='o', markerfacecolor='red', lw=6)
    plt.xlabel('number of components')
    plt.ylabel('comulative variance %')
    plt.title('comulative variance ratio of p.c.a components')
    plt.show()

    squared_loadings = plot_intrinsic_dimensionality_pca(xtrain, 3)
    imp_ftrs = sorted(range(len(squared_loadings)), key=lambda k: squared_loadings[k], reverse=True)
    print("imp_ftrs", imp_ftrs)


def draw_kmeans(data, n):
    k = KMeans(n_clusters=n)
    kpred = k.fit_predict(data)
    p_train = PCA().fit_transform(data)
    plt.figure(figsize=(15, 12))
    color = ['red', 'green', 'blue']
    for i in range(3):
        plt.scatter(p_train[kpred == i][:, 0], p_train[kpred == i][:, 1], c=color[i])
        plt.scatter(k.cluster_centers_[i, 0], k.cluster_centers_[i, 1], c='yellow', marker='x')
    plt.xlabel('pc1')
    plt.ylabel('pc2')
    plt.title('k-means clusters')
    plt.show()


def plot_intrinsic_dimensionality_pca(data, k):
    [eigenValues, eigenVectors] = generate_eigen_vectors(data)
    print(eigenValues)
    squaredLoadings = []
    ftrCount = len(eigenVectors)
    for ftrId in range(0, ftrCount):
        loadings = 0
        for compId in range(0, k):
            loadings = loadings + eigenVectors[compId][ftrId] * eigenVectors[compId][ftrId]
        squaredLoadings.append(loadings)

    print('squaredLoadings', squaredLoadings)
    print("eigenValues", eigenValues)
    plt.plot(eigenValues)
    plt.title('eigen_values')

    plt.show()

    pca = PCA(n_components=2)
    pca.fit(data)
    sns.jointplot(data={'pc1': pca.fit_transform(data)[:, 0], 'pc2': pca.fit_transform(data)[:, 1]}, x='pc1',
                  y='pc2', size=12, kind='hex', color='green')
    plt.xlabel('pc1')
    plt.ylabel('pc2')
    plt.title('pc1 vs pc2')
    plt.show()

    return squaredLoadings


def type_building(x):
    if x <= 10:
        return "new"
    elif x <= 30:
        return 'mid old'
    else:
        return 'old'


def plot_kmeans_elbow(data):  # find the suitable k
    SSE = []  # 存放每次结果的误差平方和
    for k in range(1, 9):
        estimator = KMeans(n_clusters=k)  # 构造聚类器
        estimator.fit(data)
        SSE.append(estimator.inertia_)  # estimator.inertia_获取聚类准则的总和
    X = range(1, 9)
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.title('k-means elbow')
    plt.plot(X, SSE, 'o-')
    plt.show()


# def clustering():
#     plot_kmeans_elbow()
#     features = data_csv[ftrs]
#     k = 3
#     kmeans = KMeans(n_clusters=k)
#     kmeans.fit(features)
#     kmeans_centres = kmeans.cluster_centers_
#     labels = kmeans.labels_
#     data_csv['kcluster'] = pd.Series(labels)


def generate_eigen_vectors(matrix):
    cov_mat = np.cov(matrix.T)
    eig_values, eig_vectors = np.linalg.eig(cov_mat)
    a = eig_values.argsort()[::-1]
    eig_values = eig_values[a]
    eigen_vectors = eig_vectors[:, a]
    return eig_values, eigen_vectors


def get_intrinsic_dimens(data):
    PCA().fit(data)
    return PCA().explained_variance_


def plot_intrinsic_dimen(data, k):  # get the sorted_Lloadings
    eigen_vectors = generate_eigen_vectors(data)
    sorted_Lloadings = []
    for i in range(len(eigen_vectors)):
        loading = 0
        for j in range(k):
            loading = loading + eigen_vectors[j][i] * eigen_vectors[j][i]
        sorted_Lloadings.append(loading)
    return sorted_Lloadings


def c_variance(x):
    total = 0
    clist = []

    print(x.shape[1])
    for i in np.arange(0, x.shape[1]):
        p = PCA(n_components=i + 1)
        print(p)
        p.fit(x)
        total = total + p.explained_variance_ratio_[i]
        clist.append(total)

    return clist


def random_sampling():
    random_samples_array = []
    df = data_drop.sample(frac=0.02)

    df_transform = StandardScaler().fit_transform(df)
    rand_samples = np.array(df_transform)
    for x in rand_samples:
        random_samples_array.append(x)
    return random_samples_array


@app.route("/")
def index():
    return render_template('demo.html')


@app.route("/stats")
def stats():
    return render_template('stats.html')


@app.route("/predict")
def predict():
    return render_template('predict.html')


@app.route("/predictprice", methods=["POST"])
def predictprice():
    print("kkkk")
    bedrooms = request.form.get("bedrooms")
    bathrooms = request.form.get("bathrooms")
    sqft_living = request.form.get("sqft_living")
    sqft_lot = request.form.get("sqft_lot")

    floors = request.form.get("floors")
    waterfront = request.form.get("waterfront")
    condition = request.form.get("condition")
    grade = request.form.get("grade")
    sqft_basement = request.form.get("sqft_basement")
    yr_built = request.form.get("yr_built")
    zipcode = request.form.get("zipcode")
    print(bedrooms)
    print(bathrooms)
    print(sqft_living)

    x = [20150513, bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, 1, condition, grade, sqft_basement,
         yr_built]
    x = list(map(float, x))
    data_final = getpredict(x, zipcode)
    # data = request.form.get('t', '')
    # data = request.get_json()
    return str(data_final)


def getpredict(x, zipcode):
    # use all previous month data to predict 201505 according to different zip code(e.g. 98001)
    # show predicted price and original price

    df = pd.read_csv('model/predict.csv', low_memory=False)
    df = df.loc[df['zipcode'] == zipcode]
    #
    # ftrs = ["date_new", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view", "condition",
    #         "grade", "sqft_basement", "yr_built"]
    # features = df[ftrs]
    ytest = df["price"]
    # x = [20150513, bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, 1, condition, grade, sqft_basement,
    #      yr_built]
    Xtest = np.array([x])

    filename = "model/" + str(zipcode) + ".sav"
    regressor = pickle.load(open(filename, 'rb'))
    # print("predict price")
    # print(regressor.predict(Xtest))
    # print("original price")
    # print(ytest)
    return regressor.predict(Xtest)


@app.route('/random')
def pca_random():
    pca = PCA(n_components=2)
    pca.fit(random_samples_array)
    temp = pca.transform(random_samples_array)
    data_final = pd.DataFrame(temp)
    for i in range(2):
        data_final[filters[imp_ftrs[i]]] = df_randomized[filters[imp_ftrs[i]]]
    data_final['clusterid'] = df_randomized['labels']
    return pd.DataFrame.to_json(data_final)


@app.route('/euclidean_random')
def mds_euclidean_random():
    mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    mds_similarity = pairwise_distances(random_samples_array, metric='euclidean')
    temp = mds_data.fit_transform(mds_similarity)
    data_final = pd.DataFrame(temp)
    for i in range(2):
        data_final[filters[imp_ftrs[i]]] = df_randomized[filters[imp_ftrs[i]]]
    data_final['clusterid'] = df_randomized['labels']
    return pd.DataFrame.to_json(data_final)


@app.route('/correlation_random')
def mds_correlation_random():
    mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    mds_similarity = pairwise_distances(random_samples_array, metric='correlation')
    temp = mds_data.fit_transform(mds_similarity)
    data_final = pd.DataFrame(temp)
    for i in range(2):
        data_final[filters[imp_ftrs[i]]] = df_randomized[filters[imp_ftrs[i]]]
    data_final['clusterid'] = df_randomized['labels']

    return pd.DataFrame.to_json(data_final)


@app.route('/geo_price')
def geo_price():
    df = pd.read_csv('kc_house_data.csv', encoding='gbk', usecols=[2, 17, 18])
    df['price2'] = df['price'].map(lambda x: x / 1000)
    df = df.sample(frac=0.02)

    means = df['price2'].groupby([df['lat'], df['long']]).mean()
    print(means)

    return pd.DataFrame.to_json(means)


@app.route('/geo_rooms')
def geo_rooms():
    df = pd.read_csv('kc_house_data.csv', encoding='gbk', usecols=[3, 4, 16])

    df['count'] = 1
    print(df)
    sum_bed = df['bedrooms'].groupby([df['zipcode']]).sum()
    sum_count = df['count'].groupby([df['zipcode']]).sum()

    sumall = pd.merge(sum_bed, sum_count, how='left', left_on=None, right_on=None,
                      left_index=True, right_index=True)

    return pd.DataFrame.to_json(sumall)


@app.route('/buildtime_price')
def buildtime_price():
    df = pd.read_csv('kc_house_data.csv', encoding='gbk', usecols=[1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16])
    df_sample = data_drop.sample(frac=0.02)

    df['price2'] = df['price'].map(lambda x: x / 1000)

    means = df['price2'].groupby(df['yr_built']).mean()
    print(means)

    return pd.DataFrame.to_json(means)


@app.route('/soldtime_price')
def soldtime_price():
    df = pd.read_csv('kc_house_data.csv', encoding='gbk', usecols=[1, 2])

    df['price2'] = df['price'].map(lambda x: x / 1000)

    means = df['price2'].groupby(df['date']).mean()
    print(means)

    return pd.DataFrame.to_json(means)


@app.route("/zipcorr", methods=["POST"])
def zipcorr():
    def db_query_zipcorr():
        db = Database()
        emps = db.get_zipcorr(zipcode)
        print(emps)
        return emps

    zipcode = request.form.get("zipcode")

    res = db_query_zipcorr()
    return str(res)


class Database:
    def __init__(self):
        host = "127.0.0.1"
        user = "root"
        password = "wangjinyin521"
        db = "mysql"
        self.con = pymysql.connect(host=host, user=user, password=password, db=db, cursorclass=pymysql.cursors.
                                   DictCursor)
        self.cur = self.con.cursor()

    def list_employees(self):
        self.cur.execute("SELECT first_name, last_name, gender FROM employees LIMIT 50")
        result = self.cur.fetchall()

        return result

    def get_prediction(self, ame):
        sql = "SELECT first_name, last_name, gender FROM employees where first_name='%s' LIMIT 50" % ame
        print(sql)
        self.cur.execute(sql)
        result = self.cur.fetchall()
        print(result)
        return result

    def get_zipcorr(self, ame):
        sql = "SELECT * FROM zipcorr where zipcode='%s' LIMIT 50" % int(ame)
        print(sql)
        self.cur.execute(sql)
        result = self.cur.fetchall()
        print(result)
        return result


filters = ['date', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition',
           'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode']
df = pd.read_csv('kc_house_data.csv')
df = df.dropna()
data_drop = df.reindex(
    columns=filters)
data_csv = df[filters].values
data_csv_original = df[filters].values
scaler = StandardScaler()
features = scaler.fit_transform(data_csv)

kmeans = KMeans(n_clusters=3, random_state=1)
kmeans.fit(features)
center = kmeans.cluster_centers_
# 标注每个点的聚类结果
labels = kmeans.labels_

df['kcluster'] = pd.Series(labels)
ms = MinMaxScaler()

df_randomized = data_drop.sample(frac=0.02)
k_means_sample = ms.fit_transform(df_randomized)
plot_intrinsic_dimensionality_pca(k_means_sample, 3)

df_randomized['labels'] = pd.Series(labels)

y = df['price'].values

'''noralize x'''
xtrain = ms.fit_transform(data_csv)

'''k-means'''
plot_kmeans_elbow(xtrain)  # we get 3
draw_kmeans(xtrain, 3)
print("xtrain_bf_pca", xtrain)

x_train_variance = list(map(lambda x: x * 100, c_variance(xtrain)))
print("xtrain_pca", x_train_variance)

squared_loadings = plot_intrinsic_dimensionality_pca(xtrain, 3)
imp_ftrs = sorted(range(len(squared_loadings)), key=lambda k: squared_loadings[k], reverse=True)
print("imp_ftrs", imp_ftrs)

'''since that the zipcode has been dealt as a figure, make it less important, we need to use machine learning correlations'''

random_samples_array = random_sampling()
print("sampling", len(random_samples_array))
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=82)

    # print(df['total_bedrooms'].describe())
    # total_room(df)
    # calc_categorical_median(df)
    # median_house_value()
    # popu_house_value()
    # price_geo_coordinates()
    # corr_matrix()
    # ocean_proximity()
    # ocean_prox_median_value1()
    # ocean_prox_median_value2()
    #
    # '''clean the data'''
    # pca_kmeans(df)

    # print(df.head())
    # print('the number of rows and colums are'+str(df.shape))
    #
    # print('\nthe columns are - \n')
    # [print(i,end='.\t\n') for i in df.columns.values]
