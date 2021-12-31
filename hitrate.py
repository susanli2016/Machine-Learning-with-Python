#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from classes import usermovie
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

def hit_rate(users, movies):
    hits = 0
    denom = 0
    actual = []
    predicted = []
    actualall = []
    predictedall = []
    for u1 in users:
        u = users[u1]
        userid = u.userid
        usermovies = []
        if userid in users:
            denom = denom + 1
            ufactor = users[userid].factor
            for m1 in movies:
                m = movies[m1]
                mfactor = m.factor
                dotp = np.dot(ufactor, mfactor)
                if m.movieid in u.movies_all:
                    actualall.append(u.movies_all[m.movieid])
                    predictedall.append(float(dotp))

                if m.movieid in u.movies_test:
                    actual.append(u.movies_test[m.movieid])
                    predicted.append(dotp)

                usermovied = usermovie()
                usermovied.userid = userid
                usermovied.movieid = m.movieid
                usermovied.rating = dotp
                usermovies.append(usermovied)

            usermovies.sort(key=lambda x: x.rating, reverse=True)
            count = 0
            for um in usermovies:
                userid = um.userid
                movieid = um.movieid
                #rating = um.rating
                if movieid in users[userid].movies_test:
                    hits = hits + 1
                    break
                count = count + 1
                if count > 9:
                    break

    sortedpredicted = predicted
    least = min(sortedpredicted)
    sortedpredicted = [x + least for x in sortedpredicted]
    sortedpredicted = [x / max(sortedpredicted) for x in sortedpredicted]
    sortedpredicted = [x * 5 for x in sortedpredicted]
    predicted = sortedpredicted

    sortedpredicted = predictedall
    least = min(sortedpredicted)
    sortedpredicted = [x + least for x in sortedpredicted]
    sortedpredicted = [x / max(sortedpredicted) for x in sortedpredicted]
    sortedpredicted = [x * 5 for x in sortedpredicted]
    predictedall = sortedpredicted

    rms = sqrt(mean_squared_error(actual, predicted))
    rmsall = sqrt(mean_squared_error(actualall, predictedall))

    return hits, denom, rms, rmsall

def hit_rate_SVD(users, movies, svd):
    hits = 0
    denom = 0
    actual = []
    predicted = []
    actualall = []
    predictedall = []
    for u1 in users:
        u = users[u1]
        userid = u.userid
        usermovies = []
        if userid in users:
            denom = denom + 1
            for m1 in movies:
                m = movies[m1]
                dotp = float(svd.predict(int(userid), int(m.movieid))[3])

                if m.movieid in u.movies_all:
                    actualall.append(u.movies_all[m.movieid])
                    predictedall.append(float(dotp))

                if (str(m.movieid) in u.movies_test) | (int(m.movieid) in u.movies_test):
                    actual.append(u.movies_test[m.movieid])
                    predicted.append(float(dotp))

                usermovied = usermovie()
                usermovied.userid = userid
                usermovied.movieid = m.movieid
                usermovied.rating = dotp
                usermovies.append(usermovied)

            usermovies.sort(key=lambda x: x.rating, reverse=True)
            count = 0
            for um in usermovies:
                userid = um.userid
                movieid = um.movieid

                if (str(movieid) in users[userid].movies_test) | (int(movieid) in users[userid].movies_test):
                    hits = hits + 1
                    break
                count = count + 1
                if count > 9:
                    break

    rms = sqrt(mean_squared_error(actual, predicted))
    rmsall = sqrt(mean_squared_error(actualall, predictedall))

    return hits, denom, rms, rmsall

