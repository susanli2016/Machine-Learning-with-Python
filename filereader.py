#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from classes import user
from classes import movie
from numpy import random
from util import min_rating, random_vector, num_users
from random import seed
import pandas as pd
import numpy as np
def read_ratings(filename):
    seed(42)
    np.random.seed(42)
    r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv(filename,  sep=',', names=r_cols, encoding='latin-1')

    ratings['user_id'] = ratings['user_id'].astype(int)
    ratings['movie_id'] = ratings['movie_id'].astype(int)
    ratings['rating'] = ratings['rating'].astype(float)

    numusers = num_users()

    msks = ratings['user_id'] < numusers
    ratings = ratings[msks]
    users = dict()
    testcount = 0
    traincount = 0
    trainuserdict = dict()

    for index, row in ratings.iterrows():
        userid = int(row['user_id'])
        movieid = int(row['movie_id'])
        rating1 = float(row['rating'])
        minmovierating = min_rating()
        if rating1 >= minmovierating:
            if random.random() < 0.7:
                traincount = traincount + 1
                if userid in users.keys():
                    user1 = users[userid]
                    user1.movies_train[movieid] = rating1
                else:
                    user1 = user(userid)
                    user1.factor = random_vector()
                    user1.movies_train[movieid] = rating1
                    users[userid] = user1
                    trainuserdict[userid] = 1
            else:
                testcount = testcount + 1
                if userid in users.keys():
                    user1 = users[userid]
                    user1.movies_test[movieid] = rating1
                else:
                    user1 = user(userid)
                    user1.factor = random_vector()
                    user1.movies_test[movieid] = rating1
                    users[userid] = user1

    for index, row in ratings.iterrows():
        userid = int(row['user_id'])
        movieid = int(row['movie_id'])
        rating1 = float(row['rating'])
        if userid in users.keys():
            user1 = users[userid]
            user1.movies_all[movieid] = rating1

    return users

def read_movies(filename):
    r_cols = ['movie_id', 'title', 'genres']
    df = pd.read_csv(filename, sep=",", encoding='latin-1', names=r_cols)
    movies = dict()
    for index, row in df.iterrows():
        movieid = row['movie_id']
        movie1 = movie(movieid, 0)
        movie1.factor = random_vector()
        movies[movieid] = movie1

    return movies

