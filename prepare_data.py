import pandas as pd
import numpy as np
from sklearn import preprocessing


def prepare_data():
    dataframe = pd.read_csv('./data/glass.csv')
    class_mapping = {name: index for index, name in enumerate(sorted(list(set(dataframe.Type))))}
    num_classes = len(set(dataframe.Type))
    dataframe = dataframe.replace({'Type': class_mapping})
    np_dataset = np.array(dataframe)
    normalized_features = preprocessing.normalize(np_dataset[:, :-1], axis=0)
    labels = np_dataset[:, -1]
    #print(normalized_features)
    return np_dataset, normalized_features, labels, num_classes
