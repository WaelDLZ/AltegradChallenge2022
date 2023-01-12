"""
Load all features from all nodes, and put them in a csv file for statistical analysis
"""
import numpy as np
import pandas as pd
import pickle


adj = pickle.load(open('data/adj.pkl', 'rb'))
features = pickle.load(open('data/features.pkl', 'rb'))
edge_features = pickle.load(open('data/edge_features.pkl', 'rb'))


node_features = list()

for feat in features:
    node_features.append(feat[:, -60:])

data_array = np.concatenate(node_features, axis=0)
df = pd.DataFrame(data_array)

df.to_csv('data/node_database.csv')
