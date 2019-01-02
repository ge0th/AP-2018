import os
import sys
import math
import pandas as pd
import numpy as np

NUMBER_OF_USERS = 943
GENRES = ['Action', 'Adventure', 'Animation',
          'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
          'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
          'Thriller', 'War', 'Western']

def binary_search(array, item):

    first = 0
    last = len(array) - 1

    while first <= last:
        mid = (first + last) // 2

        if array[mid] == item:
            return mid # Found at mid
        elif array[mid] > item:
            last = mid - 1
        elif array[mid] < item:
            first = mid + 1
        else:
            return False


def euclidian_distance(vector_a, vector_b):

    return math.sqrt(sum([(vector_a - vector_b) ** 2 for vector_a, vector_b in zip(vector_a, vector_b)]))

def vector_to_group(vector_index, groups):
    
    for index, group in enumerate(groups):
        group_index = binary_search(group, vector_index)

        if group_index == False:
            continue
        else:
            return group_index
        


    
def min_distance_vector(vector, vectors):

    min_target = 1000000000000
    min_index = 1000000000000

    for index, target in enumerate(vectors):

        if euclidian_distance(vector, target) < min_target:
            min_target = target
            min_index = index

    return min_index, min_target

    

def max_vector(vectors):
    pass


def BSAS(threshold, q, vectors):

    m = 1
    Cm = [[vectors[0]]] # List of groups

    vectors_count = len(vectors)

    for i in range(2, vectors_count + 1):
        
        min_index, min_distance = min_distance_vector(vectors[i], vectors)


        if min_distance > threshold and m < q:
            # Create a new group
            m = m + 1
            new_group = []
            # Adds the used vector to the new group
            new_group.append(vectors[i])

            # Adds the new group to Cm list of groups
            Cm.append(new_group)
        else:
            Cm[min_index].append(vectors[i])

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


    print(movies.query('unknown == 1').head(10))

    movies = movies[movies.unknown == 0]

    movies.drop(columns=['video_release date', 'release_date',
                         'IMDb_URL', 'unknown', 'movie_title'], inplace=True, axis=1)


    # Merging Ratings and Movies

    final_matrix = pd.merge(ratings, movies, on='movie id', how='inner')
    print(final_matrix.sort_values(by=['user id']).head(100))
        
    df=final_matrix.replace(0, np.NaN) #all 0 values transform to NaN

    df.loc[(df['Action']==1),'Action']=df['rating'] #if Action is the genre (value 1) then becomes the value of rating etc...
    df.loc[(df['Adventure']==1),'Adventure']=df['rating']
    df.loc[(df['Animation']==1),'Animation']=df['rating']
    df.loc[(df['Children\'s']==1),'Children\'s']=df['rating']
    df.loc[(df['Comedy']==1),'Comedy']=df['rating'] 
    df.loc[(df['Crime']==1),'Crime']=df['rating']
    df.loc[(df['Documentary']==1),'Documentary']=df['rating']
    df.loc[(df['Drama']==1),'Drama']=df['rating']
    df.loc[(df['Fantasy']==1),'Fantasy']=df['rating']
    df.loc[(df['Film-Noir']==1),'Film-Noir']=df['rating']
    df.loc[(df['Horror']==1),'Horror']=df['rating']
    df.loc[(df['Musical']==1),'Musical']=df['rating']
    df.loc[(df['Mystery']==1),'Mystery']=df['rating']
    df.loc[(df['Romance']==1),'Romance']=df['rating']
    df.loc[(df['Sci-Fi']==1),'Sci-Fi']=df['rating']
    df.loc[(df['Thriller']==1),'Thriller']=df['rating']
    df.loc[(df['War']==1),'War']=df['rating']
    df.loc[(df['Western']==1),'Western']=df['rating']
    
    grouped=df.groupby('user id').mean() #group by user id and calculate the avg of non-NaN values
    
    grouped.drop(columns=['movie id','rating'], inplace=True, axis=1) #leave only user id and agv for every genre
    print(grouped)
