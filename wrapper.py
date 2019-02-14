from question2 import get_clusters_count
from question3 import apply_kmeans, apply_hierarchical_clustering
from question4 import user_id_scaler, least_squares_classifier, neural_network_classifier
from helper import get_data, get_training_data, movies_to_data
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    
    # Initialization
    vectors = get_data()

    pca = PCA(n_components=2).fit_transform(vectors)
    pca = pd.DataFrame(pca)


    plt.subplot(2, 1, 1)
    plt.scatter(pca[0], pca[1], color='black')
    plt.title('Plots')
    plt.ylabel('PCA Reduction')

    # plt.subplot(2, 1, 2)
    # plt.plot(x2, y2, '.-')
    # plt.xlabel('time (s)')
    # plt.ylabel('Undamped')

    plt.show()

    # Question 2
    # step = 0.3
    # n_init = 5
    # clusters_count = get_clusters_count(vectors, step, n_init)

    # print("The number of clusters found by the BSAS Algorithm is: {} \n".format(clusters_count))


    X_data, Y_data = get_training_data()



    # # Question 3
    # kmeans = apply_kmeans(clusters_count).fit(vectors)


    # print("KMeans Clustering \n")

    # print("The centroids of each cluster are")
    # print(kmeans.cluster_centers_)

    # print("Each vector is positioned in")

    # print(kmeans.predict(vectors))

    # hierarchical_clustering = apply_hierarchical_clustering(clusters_count)
    
    # print("Hierarchical Clustering \n")

    # print("Each vector is positioned in")

    # print(hierarchical_clustering.fit_predict(vectors))

    # print("Each vector is positioned in")

    # 4444444444444444444444444444

    movies = movies_to_data()

    # least_squares_classifier = least_squares_classifier(X_data, Y_data)

    # while True:
    #     user_id = user_id_scaler(int(input("Type user id: ")))
    #     movie_id = int(input("Type Movie id: "))

    #     while (0 > user_id > 943) or ( 0 > movie_id > 1682):
    #         print("Please type the right information")
    #         print("User Range: [1 - 943]")
    #         print("Movie Range: [1 - 1682]")
    #         user_id = user_id_scaler(int(input("Type user id: ")))
    #         movie_id = int(input("Type Movie id: "))

    #     prediction_table = []
    #     prediction_table.append(user_id)

    #     for movie_type in movies[movie_id - 1]:
    #         prediction_table.append(movie_type)
   
    #     prediction = least_squares_classifier.predict([prediction_table])[0]
    #     prediction = round(prediction)

    #     if prediction == 1:
    #         print("User {} has seen the movie".format(user_id))
    #     else:
    #         print("User has not {} see the movie".format(user_id))
    #     print("\n")

        
    neural_network = neural_network_classifier(X_data, Y_data)

    while True:
        user_id = user_id_scaler(int(input("Type user id: ")))
        movie_id = int(input("Type Movie id: "))

        while (0 > user_id > 943) or ( 0 > movie_id > 1682):
            print("Please type the right information")
            print("User Range: [1 - 943]")
            print("Movie Range: [1 - 1682]")
            user_id = user_id_scaler(int(input("Type user id: ")))
            movie_id = int(input("Type Movie id: "))

        prediction_table = []
        prediction_table.append(user_id)

        for movie_type in movies[movie_id - 1]:
            prediction_table.append(movie_type)
        prediction = neural_network.predict([prediction_table])[0]

        if prediction == 1:
            print("Thn eide...")
        else:
            print("Den psithike...")
        print("\n")
