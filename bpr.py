#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from util import m_normal, learning_rate, get_lambda
from classes import ret
import random as random
import numpy as np
import math
def bpr_update(users, movies):
    count = 0
    lr = learning_rate()
    lam = get_lambda()
    for u1 in users:
        u = users[u1]
        userid = u.userid
        Vu = u.factor
        if (len(u.movies_train) > 0):

            rand_pos = random.sample(u.movies_train.keys(), 1)[0]
            rand_neg = random.sample(movies.keys(), 1)[0]

            if rand_neg not in u.movies_train:
                Vi = movies[rand_pos].factor
                Vj = movies[rand_neg].factor
                firstterm = calculate_first_term(Vu, Vi, Vj)

                # USER FACTOR
                diff = Vi - Vj
                d = firstterm * diff
                derivative = d
                Vu = Vu + lr * (derivative + lam * np.linalg.norm(Vu))
                users[u1].factor = Vu

                # ITEM POSITIVE FACTOR
                d = firstterm * Vu
                derivative = d
                Vi = Vi + lr * (derivative + lam * np.linalg.norm(Vi))
                movies[rand_pos].factor = Vi

                #ITEM NEGATIVE FACTOR
                negvu = -1 * Vu
                d = firstterm * negvu
                derivative = d
                Vj = Vj + lr * (derivative + lam * np.linalg.norm(Vj))
                movies[rand_neg].factor = Vj

def calculate_first_term(Vu, Vi, Vj):
    boughtdot = np.dot(Vu, Vi)
    notboughtdot = np.dot(Vu, Vj)
    negxuij = (boughtdot - notboughtdot) * -1
    if negxuij > 500:
        negxuij = 500
    numerator = math.exp(negxuij)
    denominator = 1 + math.exp(negxuij)
    firstterm = numerator / denominator
    return firstterm

