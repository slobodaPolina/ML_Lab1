import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

from find_best_parameters import find_best_parameters_one_hot
from classification import classify_object
from prepare_data import prepare_data
import parameters

#read normalized data from csv file
dataset, features, labels, num_classes = prepare_data()

#set label not a number of class, but array of 0 and 1 - is object in this class
encoded_labels = np.array(pd.get_dummies(pd.Series(labels)))

#print(find_best_parameters_one_hot(features, labels, encoded_labels, num_classes))

f_scores = []

if parameters.best_combination['window_type'] == 'variable':
    xLabel = 'nearest neighbours'
    window_parameter_range = range(len(features) - 1)
else:
    xLabel = 'window width'
    window_parameter_range = range(0, 1, 0.01)

for window_parameter in window_parameter_range:
    predicted_labels = []

    # count f-score one more time for best combination of distance, kernel and type and every possible parameter (amount of neighbours or window width)
    for index in range(len(features)):
        predicted_label_values = []

        # predict for every class is it 0 or 1 (contains this vector or not)
        for label_index in range(len(encoded_labels[index])):
            predicted_label_value = classify_object(
                features[np.arange(len(features)) != index],
                encoded_labels[np.arange(len(encoded_labels)) != index][:, label_index],
                features[index],
                parameters.best_combination['distance_function'],
                parameters.best_combination['kernel_function'],
                parameters.best_combination['window_type'],
                window_parameter
            )
            predicted_label_values.append(predicted_label_value)

        # parse the class
        predicted_value = [i for i, j in enumerate(predicted_label_values) if j == max(predicted_label_values)][0]
        predicted_label = round(predicted_value)
        predicted_labels.append(predicted_label)

    f_score = f1_score(labels, predicted_labels, labels=[i for i in range(num_classes)], average='weighted')
    f_scores.append({'f_score': f_score, 'window_parameter': window_parameter})

plt.plot([point['window_parameter'] for point in f_scores], [point['f_score'] for point in f_scores])
plt.xlabel(xLabel)
plt.ylabel('f-score')
plt.show()
