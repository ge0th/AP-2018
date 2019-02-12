# Standard Library Imports
import os

# Third Party Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


NUMBER_OF_USERS = 943
GENRES = ['Action', 'Adventure', 'Animation',
          'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
          'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
          'Thriller', 'War', 'Western']


def get_data():
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

    df = final_matrix.replace(0, np.NaN)  # all 0 values transform to NaN

    for genre in GENRES:
        df.loc[(df[genre] == 1), genre] = df['rating']


    grouped = df.groupby('user_id').mean() #group by user id and calculate the avg of non-NaN values

    grouped.drop(columns=['movie_id', 'rating'], inplace=True, axis=1) #leave only user id and agv for every genre

    grouped.fillna(0, inplace=True)      

    grouped = grouped.as_matrix()

    scaler = MinMaxScaler().fit(grouped)

    grouped = scaler.transform(grouped)

    return grouped
