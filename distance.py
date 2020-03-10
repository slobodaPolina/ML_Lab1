# distance between 2 vectors
def distance(x, y, distance_function_type):
    if distance_function_type == 'manhattan':
        return sum([abs(xi - yi) for xi, yi in zip(x, y)])

    if distance_function_type == 'euclidean':
        return sum([(xi - yi) ** 2 for xi, yi in zip(x, y)]) ** (1 / 2)

    if distance_function_type == 'chebyshev':
        return max([abs(xi - yi) for xi, yi in zip(x, y)])
