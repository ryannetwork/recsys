from getpass import getuser
import datetime
from datetime import datetime as dt
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
import sys
import os

from pyspark.sql.types import TimestampType, DoubleType, IntegerType, StringType, DateType
import pyspark.sql.functions as F
from pyspark.sql.window import *

def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

def euclidean(X, value):
    distances = np.linalg.norm(X - value, axis=1)
    return distances

def cosine(cos_matrix, hash_inn, inn_col_name, dataframe):
    distances = cos_matrix[dataframe.loc[dataframe[inn_col_name] == hash_inn].index[0]]
    return -distances

def percentile(p, distances, X):
        return np.round(np.mean(np.sort(distances)[:len(X)*p//100]),5)
    
def top_n(n, distances, X):
        return np.round(np.mean(np.sort(distances)[:n]),5)

# Функцию писал Василенко Аркадий    
def similars(X, dataframe, hash_inn=1005, cols_names=['inn_col_name','okved_col_name'], N=5, treshold=1,
                                                         metric=euclidean, matrix=None):
    
    inn_col_name, okved_col_name = cols_names[0],cols_names[1]
    value = X[dataframe.loc[dataframe[inn_col_name] == hash_inn].index[0]]
    
    if metric == euclidean:
        distances = metric(X, value)
    elif metric == cosine:
        if matrix is None:
            return print('Нужно передать матрицу cos_matrix')
        else:
            distances = metric(matrix, hash_inn, inn_col_name, dataframe)
        
    arg_nearest = distances.argsort()[0:N]
    
    n_dist = np.sort(distances)[0:N]
    arg_nearest = arg_nearest[(n_dist < treshold)]
    
    #отношение дистанции относительно ближайшего элемента
    per_dist = [abs(n_dist[i]/n_dist[0]) if n_dist[0] != 0 else abs(n_dist[i]) for i in range(len(n_dist))]
    
    result = dataframe.iloc[arg_nearest]
    result['target_inn'] = hash_inn
    result['distance'] = abs(n_dist)

    return result[cols_names + ['target_inn', 'distance']]
