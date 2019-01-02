import os
import json
import sys
import math
import pandas as pd
import numpy as np
from random import shuffle

NUMBER_OF_USERS = 943
GENRES = ['Action', 'Adventure', 'Animation',
          'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
          'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
          'Thriller', 'War', 'Western']


def file_exists(file_name):


    # return os.path.isfile(os.path.abspath(file_name))
    return False

def euclidian_distance(vector_a, vector_b):

    return math.sqrt(sum([(vector_a - vector_b) ** 2 for vector_a, vector_b in zip(vector_a, vector_b)]))

def binary_search(array, item):

    first = 0
    last = len(array) - 1

    while first <= last:
        mid = (first + last) // 2

        if array[mid] == item:
            return True  # Found at mid
        elif array[mid] > item:
            last = mid - 1
        elif array[mid] < item:
            first = mid + 1
        else:
            return False


def vector_to_group(vector, groups):
    """ vector_to_group finds the group index that a specific vector has been
        placed
    """

    for index, group in enumerate(groups):
        group_index = binary_search(group, vector)

        if group_index == False:
            continue
            print("roufa1")
        else:
            print("roufa2")
            print("index")
            print(index)
            return index

def min_distance_vector(vector_index, vectors):
    """ This function find the minimum distance between a specific vector and all the others

    """

    min_distance = 10000000
    min_index = vector_index

    for index, target in enumerate(vectors):

        if euclidian_distance(vectors[vector_index], target) < min_distance:
            min_distance = euclidian_distance(vectors[vector_index], target)
            min_index = index
    print(vectors[min_index], min_distance)

    return vectors[min_index], min_distance


def min_max_between_all(vectors):

    distances = []

    if file_exists('distances.json'):
        with open('distances.json') as file:

            for line in file:

                line = line.strip()
                distances.append(line)
    else:

        for i in range(0, len(vectors)):
            for j in range(i + 1, len(vectors)):
                distance = euclidian_distance(vectors[i], vectors[j])
                distances.append(distance)

        with open('distances.json', 'w') as file:
            json.dump(distances, file)


    return min(distances), max(distances)





def BSAS(threshold, q, vectors):

    m = 1
    Cm = [[vectors[0]]] # List of groups

    vectors_count = len(vectors)

    for i in range(1, vectors_count):
        print("BSAS IIIII : {}".format(i))
        min_vector, min_distance = min_distance_vector(i, vectors)


        if min_distance > threshold and m < q:
            # Create a new group
            m = m + 1
            new_group = []
            # Adds the used vector to the new group
            new_group.append(vectors[i])

            # Adds the new group to Cm list of groups
            Cm.append(new_group)
        else:
            min_index = vector_to_group(min_vector, Cm)
            print(min_index)
            Cm[min_index].append(vectors[i])
    
    return len(Cm)


def cleaning_data():
    """This method prepares the data for processing by BSAS.

        Especially it drops the unnecessary columns, removes the rows of movies 
        with unknown genre and computes the average rating of each user of each
        movie genre.
    """

    # Getting absolute path to the data files

    udata = os.path.abspath('dataset/u.data')
    uitem = os.path.abspath('dataset/u.item')

    # Creating User Ratings Matrix

    data_cols = ['user_id', 'movie_id', 'rating', 'timestamp']

    ratings = pd.read_csv(udata, sep='\t', names=data_cols,
                          encoding='latin-1')

    ratings.drop(columns='timestamp', inplace=True, axis=1)


    # Creating Movies Matrix

    movie_cols = ['movie_id', 'movie_title', 'release_date', 'video_release date',
                  'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                  'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                  'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                  'Thriller', 'War', 'Western']

    movies = pd.read_csv(uitem, sep='|', names=movie_cols,
                         encoding='latin-1')




    movies = movies[movies.unknown == 0]

    movies.drop(columns=['video_release date', 'release_date',
                         'IMDb_URL', 'unknown', 'movie_title'], inplace=True, axis=1)


    # Merging Ratings and Movies

    final_matrix = pd.merge(ratings, movies, on='movie_id', how='inner')
        
    df = final_matrix.replace(0, np.NaN) #all 0 values transform to NaN

    for genre in GENRES:
        df.loc[(df[genre] == 1), genre] = df['rating']

    
    grouped=df.groupby('user_id').mean() #group by user id and calculate the avg of non-NaN values
    
    grouped.drop(columns=['movie_id','rating'], inplace=True, axis=1) #leave only user id and agv for every genre

    grouped = grouped.replace(0, np.NaN)
    grouped = list(grouped.values.tolist())

    # Replaces nan with 0
    for group in grouped:
        for number in range(len(group)):
            if math.isnan(group[number]):
                group[number] = 0

    with open('data.json', 'w') as file:
        json.dump(grouped, file)

    return grouped

def get_clusters_count(a, b, c, s, vectors, q):

    groups = []
    for theta in range(a, b + 1, c):
        for i in range(0, s):
            shuffle(vectors)
            count = BSAS(theta, q, vectors)
            groups.append(count)
    #         print("Found {} groups".format(count))
    # print(groups)
    return groups

if __name__ == '__main__':
    
    if file_exists('data.json'):
        vectors = []
        with open('data.json') as file:
            for line in file: 
                line = line.strip() 
                vectors.append(line) 
    else:
        vectors = cleaning_data()


    a, b = min_max_between_all(vectors)[0], min_max_between_all(vectors)[1]
    a, b = int(float(a)), int(float(b))

    clusters = get_clusters_count(a, b, 400, 15, vectors, 400)
    print(clusters)

a = input()
