import csv
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template, request, redirect, Response, jsonify
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import manifold
from sklearn.metrics import pairwise_distances

# First of all you have to import it from the flask module:
app = Flask(__name__)


@app.route("/scree", methods=['GET', 'POST'])
def scree():
    test_pca = decomposition.PCA()
    test_pca.fit(strat_samp_array_data)
    test_varience = test_pca.explained_variance_
    test_varience_x = pd.DataFrame(test_varience)
    return pd.DataFrame.to_json(test_varience_x)


@app.route("/scree_original", methods=['GET', 'POST'])
def scree_original():
    test_pca = decomposition.PCA()
    test_pca.fit(all_array_data)
    test_varience = test_pca.explained_variance_
    test_varience_x = pd.DataFrame(test_varience)
    return pd.DataFrame.to_json(test_varience_x)


@app.route("/")
def index():
    return render_template('index.html')


@app.route('/random')
def pca_random():
    pca = PCA(n_components=2)
    pca.fit(random_samples_array)
    temp = pca.transform(random_samples_array)
    data_final = pd.DataFrame(temp)
    for i in range(2):
        data_final[filters[sorted_Lloadings[i]]] = df_randomized[filters[sorted_Lloadings[i]]]
    data_final['clusterid'] = df_randomized['labels']
    return pd.DataFrame.to_json(data_final)


@app.route('/stratify')
def pca_stratify():
    pca = PCA(n_components=2)
    pca.fit(stratified_samples[filters])
    temp = pca.transform(stratified_samples[filters])
    data_final = pd.DataFrame(temp)
    for i in range(2):
        data_final[filters[sorted_Lloadings[i]]] = df_stra_samples[filters[sorted_Lloadings[i]]]
    data_final['clusterid'] = np.nan
    x = 0
    for index, row in df_stra_samples.iterrows():
        data_final['clusterid'][x] = row['index']
        x += 1
    return pd.DataFrame.to_json(data_final)


@app.route('/euclidean_random')
def mds_euclidean_random():
    mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    mds_similarity = pairwise_distances(random_samples_array, metric='euclidean')
    temp = mds_data.fit_transform(mds_similarity)
    data_final = pd.DataFrame(temp)
    for i in range(2):
        data_final[filters[sorted_Lloadings[i]]] = df_randomized[filters[sorted_Lloadings[i]]]
    data_final['clusterid'] = df_randomized['labels']
    return pd.DataFrame.to_json(data_final)


@app.route('/correlation_random')
def mds_correlation_random():
    mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    mds_similarity = pairwise_distances(random_samples_array, metric='correlation')
    temp = mds_data.fit_transform(mds_similarity)
    data_final = pd.DataFrame(temp)
    for i in range(2):
        data_final[filters[sorted_Lloadings[i]]] = df_randomized[filters[sorted_Lloadings[i]]]
    data_final['clusterid'] = df_randomized['labels']

    return pd.DataFrame.to_json(data_final)


def plot_kmeans_elbow():  # find the suitable k
    SSE = []  # 存放每次结果的误差平方和
    for k in range(1, 9):
        estimator = KMeans(n_clusters=k)  # 构造聚类器
        estimator.fit(data_drop)
        SSE.append(estimator.inertia_)  # estimator.inertia_获取聚类准则的总和
    X = range(1, 9)
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.plot(X, SSE, 'o-')
    plt.show()


def random_sampling():
    random_samples_array = []
    random_samples = df_original.sample(frac=0.2)
    df = random_samples.reindex(
        columns=filters)
    df_transform = StandardScaler().fit_transform(df)
    rand_samples = np.array(df_transform)
    for x in rand_samples:
        random_samples_array.append(x)
    return random_samples_array


def stratified_sampling():
    sample_cluster0 = df_stra_samples[df_stra_samples['index'] == 1.0]
    sample_cluster1 = df_stra_samples[df_stra_samples['index'] == 2.0]
    sample_cluster2 = df_stra_samples[df_stra_samples['index'] == 3.0]
    return pd.concat([sample_cluster0, sample_cluster1, sample_cluster2])


def get_stratified_sampling():
    df_drop = df_original.reindex(
        columns=filters)
    dataSet = df_drop.as_matrix(columns=None)
    kmeans = KMeans(n_clusters=3, random_state=1)
    kmeans.fit(df_drop.iloc[:, :])
    center = kmeans.cluster_centers_
    df_center = pd.DataFrame(center,
                             columns=filters)
    # 标注每个点的聚类结果
    labels = kmeans.labels_
    # 将原始数据中的索引设置成得到的数据类别，根据索引提取各类数据并保存
    df_neww = pd.DataFrame(dataSet, index=labels,
                           columns=filters)
    df_new = df_neww.reindex(
        columns=filters)
    '''注意这个地方应该是dataSet而不是df_drop'''
    df1_agg = []
    df1_group = []
    df2_agg = []
    df2_group = []
    df3_agg = []
    df3_group = []
    # x = random.randint(1, 5) % 5
    rand = 4
    df1 = df_new[df_new.index == 0]
    df_transform = StandardScaler().fit_transform(df1)
    strat_samples = np.array(df_transform)
    stratified_sampling_array = []
    for i in range(len(strat_samples)):
        if i % 5 == rand:
            stratified_sampling_array.append(strat_samples[i])
    for row in df1.iterrows():
        index, data = row
        df1_agg.append(data.tolist())
    for i in range(len(df1_agg)):
        if i % 5 == rand:
            df1_group.append(df1_agg[i])

    df2 = df_new[df_new.index == 1]
    # df2.loc[:,'index'] = np.array([2]*len(df2))
    df_transform = StandardScaler().fit_transform(df2)
    strat_samples = np.array(df_transform)
    for i in range(len(strat_samples)):
        if i % 5 == rand:
            stratified_sampling_array.append(strat_samples[i])
    for row in df2.iterrows():
        index, data = row
        df2_agg.append(data.tolist())
    for i in range(len(df2_agg)):
        if i % 5 == rand:
            df2_group.append(df2_agg[i])
    df3 = df_new[df_new.index == 2]
    # df3.loc[:,'index'] = np.array([3]*len(df3))
    df_transform = StandardScaler().fit_transform(df3)
    strat_samples = np.array(df_transform)
    for i in range(len(strat_samples)):
        if i % 5 == rand:
            stratified_sampling_array.append(strat_samples[i])

    for row in df3.iterrows():
        index, data = row
        df3_agg.append(data.tolist())
    for i in range(len(df3_agg)):
        if i % 5 == rand:
            df3_group.append(df3_agg[i])
    stratified_sample = df1_group + df2_group + df3_group
    # print(stratified_sample)
    # out = open('output.csv', 'a', newline='')
    # csv_write = csv.writer(out, dialect='excel')
    # for line in df1_group:
    #     csv_write.writerow(line)
    # for line in df2_group:
    #     csv_write.writerow(line)
    # for line in df3_group:
    #     csv_write.writerow(line)
    return stratified_sampling_array


def get_trans_array_data():
    data = []
    with open('output.csv') as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        birth_header = next(csv_reader)  # 读取第一行每一列的标题
        for row in csv_reader:  # 将csv 文件中的数据保存到birth_data中
            rows = []
            for i in range(len(row)):
                if i == 0:
                    pass
                else:
                    rows.append(int(float(row[i])))
            data.append(rows)
    test = []
    for x in data:
        if x:
            test.append(x)
    x = StandardScaler().fit_transform(test)
    strat_samp_array_data = np.array(x)
    return strat_samp_array_data


def generate_eigen_vectors(matrix):
    cov_mat = np.cov(matrix.T)
    eig_values, eig_vectors = np.linalg.eig(cov_mat)
    print("eigen_values before sorted:", eig_values)
    a = eig_values.argsort()[::-1]
    eigen_vectors = eig_vectors[:, a]
    return eigen_vectors


def get_intrinsic_dimens():
    test_pca = decomposition.PCA()
    test_pca.fit(strat_samp_array_data)
    return test_pca.explained_variance_


def plot_intrinsic_dimen(data, k): # get the sorted_Lloadings
    eigen_vectors = generate_eigen_vectors(data)
    sorted_Lloadings = []
    for i in range(len(eigen_vectors)):
        loading = 0
        for j in range(k):
            loading = loading + eigen_vectors[j][i] * eigen_vectors[j][i]
        sorted_Lloadings.append(loading)
    return sorted_Lloadings


data_csv = pd.read_csv('kc_house_data.csv', encoding='gbk')  # 读入数据
del data_csv['id']
del data_csv['lat']
del data_csv['long']
del data_csv['sqft_living15']
del data_csv['sqft_lot15']
del data_csv['price']

filters = ['date', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition',
           'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode']

df_original = pd.read_csv('kc_house_data.csv')
data_drop = df_original.reindex(
    columns=filters)

data_csv[filters] = StandardScaler().fit_transform(data_csv[filters])
features = data_csv[filters]



kmeans = KMeans(n_clusters=3, random_state=1)
kmeans.fit(features)
center = kmeans.cluster_centers_
# 标注每个点的聚类结果
labels = kmeans.labels_


data_csv['kcluster'] = pd.Series(labels)

df_randomized = data_drop.sample(frac=0.2)
dataSet = data_drop.as_matrix(columns=None)
kmeans = KMeans(n_clusters=3, random_state=1)
kmeans.fit(data_drop)
center = kmeans.cluster_centers_
# 标注每个点的聚类结果
data_drop['labels'] = pd.Series(labels)
df_randomized['labels'] = pd.Series(labels)
data_drop[filters] = StandardScaler().fit_transform(data_drop[filters])

all_array_data = np.array(data_drop[filters])
strat_samp_array_data = get_trans_array_data()

random_samples_array = random_sampling()
stratified_sampling_array = get_stratified_sampling()
df_stra_samples = pd.read_csv('output with index.csv')

samples = 500
stratified_samples = stratified_sampling()
plot_kmeans_elbow()
eigenVectors = generate_eigen_vectors(strat_samp_array_data)


eigenValues = get_intrinsic_dimens()
squared_sorted_Lloadings = plot_intrinsic_dimen(strat_samp_array_data, 3)
sorted_Lloadings = sorted(range(len(squared_sorted_Lloadings)), key=lambda k: squared_sorted_Lloadings[k], reverse=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=81)
