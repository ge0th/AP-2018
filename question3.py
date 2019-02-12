from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

def apply_kmeans(k):
    """Applys KMeans"""

    kmeans = KMeans(n_clusters=k, init='random', n_init=10, max_iter=300,
    precompute_distances=True, algorithm='full')

    return kmeans



def apply_hierarchical_clustering(clusters):
    """Applys Hierarcical Clustering"""

    hierarchical_clustering = AgglomerativeClustering(n_clusters=clusters, affinity='euclidean', linkage='ward')

    return hierarchical_clustering



