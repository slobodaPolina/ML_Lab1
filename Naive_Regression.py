import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from find_best_parameters import find_best_parameters_native
from classification import classify_object
from prepare_data import prepare_data
import parameters

# read normalized data from csv file
dataset, features, labels, num_classes = prepare_data()

# print(find_best_parameters_native(features, labels, num_classes))

f_scores = []

if parameters.best_combination['window_type'] == 'variable':
    xLabel = 'nearest neighbours'
    window_parameter_range = range(len(features) - 1)
else:
    xLabel = 'window width'
    window_parameter_range = range(0, 1, 0.01)

# count f-score one more time for best combination of distance, kernel and type and every possible parameter (amount of neighbours or window width)
for window_parameter in window_parameter_range:
    predicted_labels = []

    for index in range(len(features)):
        predicted_value = classify_object(
            features[np.arange(len(features)) != index],
            labels[np.arange(len(labels)) != index],
            features[index],
            parameters.best_combination['distance_function'],
            parameters.best_combination['kernel_function'],
            parameters.best_combination['window_type'],
            window_parameter
        )

        predicted_label = round(predicted_value)
        predicted_labels.append(predicted_label)

    f_score = f1_score(labels, predicted_labels, labels=[i for i in range(num_classes)], average='weighted')
    f_scores.append({'f_score': f_score, 'window_parameter': window_parameter})

plt.plot([point['window_parameter'] for point in f_scores], [point['f_score'] for point in f_scores])
plt.xlabel(xLabel)
plt.ylabel('f-score')
plt.show()
