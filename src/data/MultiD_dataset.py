import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
import torch
from torch.utils.data import random_split

def prepare_news(standardise = True):
    # fetch dataset 
    online_news_popularity = fetch_ucirepo(id=332) 
    
    # data (as pandas dataframes) 
    X = online_news_popularity.data.features 
    y = online_news_popularity.data.targets 

    if standardise:
        for column in X.columns:
            max = X[column].max()
            min = X[column].min()
            X[column] = 2*(X[column].values - min)/(max-min)-1

    X['shares'] = y.values
