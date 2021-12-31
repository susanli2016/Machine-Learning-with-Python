#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from numpy import random
import numpy as np

def num_users():
    return 1000000

def dimension():
    return 50

def min_rating():
    return 4

def learning_rate():
    return 1

def get_lambda():
    return 0.1

def random_vector():
    dim = dimension()
    cov_mtx = cov_matrix()
    return random.multivariate_normal(np.zeros(dim), cov_mtx)

def cov_matrix():
    dim = dimension()
    cov = np.zeros((dim, dim), dtype=float)
    for i in range(dim):
        cov[i][i] = 0.1
    return cov

def m_normal(mean):
    cov_mtx = cov_matrix()
    return random.multivariate_normal(mean=mean, cov=cov_mtx)

