from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import matplotlib

import numpy as np
import pylab as pl
import time

def kmeans(k, data):

    kmeans = KMeans(n_clusters=k, init='random', n_init=10, max_iter=300,
    precompute_distances=True, algorithm='full').fit(data)

    print(kmeans.labels_)

    print(kmeans.cluster_centers_)


def hierarchical_clustering(clusters, data):

    cluster = AgglomerativeClustering(n_clusters=clusters, affinity='euclidean', linkage='ward')  
    cluster.fit_predict(data)  
    plt.figure(figsize=(10, 7))  
    plt.scatter(data[:,4], data[:,7], c=cluster.labels_, cmap='rainbow')
    plt.show()

if __name__ == "__main__":
    
    from question2 import get_data, get_clusters_count

    vectors = get_data()
    # number_of_clusters = get_clusters_count()

    # kmeans(number_of_clusters, vectors)
    hierarchical_clustering(2, vectors)

    a = input()
