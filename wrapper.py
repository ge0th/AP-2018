from question2 import get_clusters_count
from question3 import apply_kmeans, apply_hierarchical_clustering,cal_vect_in_each_clusters
from question4 import least_squares_classifier, neural_network_classifier
from helper import get_data, get_training_data, movies_to_data
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

kfold_training = ["u{}.base".format(i) for i in range(1,6)]
kfold_testing = ["u{}.test".format(i) for i in range(1,6)]

if __name__ == "__main__":
    
    # Initialization
    vectors = get_data()

    # PCA Representation of the dat to two-dimensional space
    #pca = PCA(n_components=2).fit_transform(vectors)
    #pca = pd.DataFrame(pca)

    #plt.scatter(pca[0], pca[1], color='black')
    #plt.title('Plots')
    #plt.ylabel('PCA Reduction')
    #plt.show()

    # Question 2
    step = 0.3
    n_init = 5
    clusters_count = get_clusters_count(vectors, step, n_init)
    print("The number of clusters found by the BSAS Algorithm is: {} \n".format(clusters_count))






    # Question 3
    
    X_data, Y_data = get_training_data()
    
    # KMeans
    
    kmeans = apply_kmeans(clusters_count).fit(vectors)

    print("KMeans Clustering \n")

    print("The centroids of each cluster are")
    print(kmeans.cluster_centers_)

    print("Each vector is positioned in")

    print(kmeans.predict(vectors))
    
    print("Number of vectors in each cluster")
    
    print(cal_vect_in_each_clusters(clusters_count, kmeans.predict(vectors)))
    
    # Hierarchical Clustering
    
    hierarchical_clustering = apply_hierarchical_clustering(clusters_count)
    
    print("\n\nHierarchical Clustering \n")

    print("Each vector is positioned in")

    print(hierarchical_clustering.fit_predict(vectors))

    print("Number of vectors in each cluster")
    
    print(cal_vect_in_each_clusters(clusters_count, hierarchical_clustering.fit_predict(vectors)))
    
    # # Question 4
    movies = movies_to_data()

    least_squares = least_squares_classifier(X_data, Y_data)
    while True:
        user_id = int(input("Type user id: "))
        movie_id = int(input("Type Movie id: "))
  
        while (0 > user_id > 943) or ( 0 > movie_id > 1682):
            print("Please type the right information")
            print("User Range: [1 - 943]")
            print("Movie Range: [1 - 1682]")
            user_id = int(input("Type user id: "))
            movie_id = int(input("Type Movie id: "))

        prediction_table = []
        prediction_table.append(user_id)
        prediction_table.append(movie_id)
        for movie_type in movies[movie_id - 1]:
            prediction_table.append(movie_type)
   
        prediction = least_squares.predict([prediction_table])[0]
        prediction = round(prediction)

        if prediction == 1:
            print("User {} has seen the movie".format(user_id))
        else:
            print("User has not {} see the movie".format(user_id))
        print("\n")
        a = input("If you want to continue to MLPClassifier press c\n")
        if a in ['c', 'C']:
        	  print("Starting MLPClassifier...")
        	  break
    
    neural_network = neural_network_classifier(X_data, Y_data)

    
    while True:
        user_id = int(input("Type user id: "))
        movie_id = int(input("Type Movie id: "))

        while (0 > user_id > 943) or ( 0 > movie_id > 1682):
            print("Please type the right information")
            print("User Range: [1 - 943]")
            print("Movie Range: [1 - 1682]")
            user_id = int(input("Type user id: "))
            movie_id = int(input("Type Movie id: "))

        prediction_table = []
        prediction_table.append(user_id)
        prediction_table.append(movie_id)
		  
        for movie_type in movies[movie_id - 1]:
            prediction_table.append(movie_type)
         
            
        prediction = neural_network.predict([prediction_table])[0]

        if prediction == 1:
            print("User {} has seen the movie".format(user_id))
        else:
            print("User has not {} see the movie".format(user_id))
        print("\n")
        a = input("If you want to continue to K-Fold validation press c\n")
        if a in ['c', 'C']:
        	  print("Starting K-Fold...")
        	  break

    # K - Fold
    test1 = 0
    test2 = 0
    for i in range(1, 6):
        X_train, Y_train, = get_training_data('{}'.format(kfold_training[i - 1]))
        X_test, Y_test = get_training_data('{}'.format(kfold_testing[i - 1]))
        neural_network = neural_network_classifier(X_train, Y_train)
        test1 = test1 + neural_network.score(X_test, Y_test)
        print("MLPClassifier, For K = {}, Accuracy: {}".format(i, neural_network.score(X_test, Y_test)))
    print("Average of K-Fold: {}".format(test1/5))
    
    for i in range(1, 6):
        X_train, Y_train, = get_training_data('{}'.format(kfold_training[i - 1]))
        X_test, Y_test = get_training_data('{}'.format(kfold_testing[i - 1]))
        least_squares = least_squares_classifier(X_train, Y_train)
        test2 = test2 + neural_network.score(X_test, Y_test)
        print("Least Squares, For K = {}, Accuracy: {}".format(i, least_squares.score(X_test, Y_test)))
    print("Average of K-Fold: {}".format(test2/5))
    

