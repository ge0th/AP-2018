from question2 import get_clusters_count
from question3 import apply_kmeans, apply_hierarchical_clustering
from helper import get_data

if __name__ == "__main__":
    
    # Initialization
    vectors = get_data()

    # Question 2
    clusters_count = get_clusters_count(vectors, 2, 2)

    print("The number of clusters found by the BSAS Algorithm is: {}".format(clusters_count))

    # Question 3
    kmeans = apply_kmeans(clusters_count, vectors)

    print("KMeans Clustering")

    print("The centroids of each clusters are")
    print(kmeans.cluster_centers_)

    print("Each vector is positioned in")
    print(kmeans.predict(vectors).labels)

    hierarchical_clustering = apply_hierarchical_clustering(clusters_count, vectors)
    
    print("Hierarchical Clustering")

    print("Each vector is positioned in")
    print(hierarchical_clustering.labels)

    print("Each vector is positioned in")
