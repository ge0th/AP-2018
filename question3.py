from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

def kmeans(k, data):

    kmeans = KMeans(n_clusters=k, init='random', n_init=10, max_iter=300,
    precompute_distances=True, algorithm='full').fit(data)
    
    return kmeans


def hierarchical_clustering(clusters, data):

    hierarchical_clustering = AgglomerativeClustering(n_clusters=clusters, affinity='euclidean', linkage='ward').fit(data)
    
    return hierarchical_clustering


if __name__ == "__main__":
    
    from question2 import get_data, get_clusters_count
    vectors = get_data()
    
    kmeans(number_of_clusters, vectors)
    hierarchical_clustering(2, vectors)

    a = input()
