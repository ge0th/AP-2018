import os
import sys
import math
import pandas as pd
import numpy as np

def euclidian_distance(vector_a, vector_b):

    return math.sqrt(sum([(vector_a - vector_b) ** 2 for vector_a, vector_b in zip(x, y)]))

def cleaning_data():
    """This method prepares the data for processing by BSAS.
    
        Especially it drops the unnecessary columns, removes the rows of movies 
        with unknown genre and computes the average rating of each user of each
        movie genre.
    """



    # Getting absolute path to the data files

    udata = os.path.abspath("dataset/u.data")
    uitem = os.path.abspath("dataset/u.item")


    # Creating User Ratings Matrix

    data_cols = ['user id', 'movie id', 'rating', 'timestamp']

    ratings = pd.read_csv(udata, sep='\t', names=data_cols,
        encoding='latin-1')

    ratings.drop(columns="timestamp", inplace=True, axis=1)

    print(ratings.head(25))

    # Creating Movies Matrix

    movie_cols = ['movie id', 'movie title', 'release date', 'video release date',
                'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation',
                'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                                'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                                'Thriller', 'War', 'Western']

    movies = pd.read_csv(uitem, sep='|', names=movie_cols,
                        encoding='latin-1')

    movies = movies[movies.unknown == 0]

    movies.drop(columns=['video release date', 'release date', 'IMDb URL', 'unknown'], inplace=True, axis=1)

    print(movies.head(25))

    # Merging Ratings and Movies 

    final_matrix = pd.merge(ratings, movies, on='movie id', how='inner')
    print(final_matrix.head(25))


cleaning_data()
