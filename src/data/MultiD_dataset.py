import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
import torch

def load_crime():
    
    return None

def load_news():
    # fetch dataset 
    online_news_popularity = fetch_ucirepo(id=332) 
    
    # data (as pandas dataframes) 
    X = online_news_popularity.data.features 
    y = online_news_popularity.data.targets 