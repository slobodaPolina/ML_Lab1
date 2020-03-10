# variation parameters- these data is used only to find best parameters to reduce runtime a little bit
# for the case of fixed width
window_widths = [0.01, 0.05, 0.1, 0.2]
# for the case of neighbours
window_neighbours = [1, 2, 3, 4, 5, 10, 20, 30, 40,  50, 100, 150,  200, 212]

# fixed combination to draw the graphic
best_combination = {
    'distance_function': 'euclidean',
    'kernel_function': 'sigmoid',
    'window_type': 'variable'
}
