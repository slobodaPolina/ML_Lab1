import math


def kernel(r, kernel_function_type):
    if kernel_function_type == 'uniform':
        return 1 / 2 - (abs(r) // 1) if r < 1 else 0

    if kernel_function_type == 'triangular':
        return (1 - abs(r)) if r < 1 else 0

    if kernel_function_type == 'epanechnikov':
        return (3 / 4 * (1 - r ** 2)) if r < 1 else 0

    if kernel_function_type == 'quartic':
        return (15 / 16 * (1 - r ** 2) ** 2) if r < 1 else 0

    if kernel_function_type == 'triweight':
        return (35 / 32 * (1 - r ** 2) ** 3) if r < 1 else 0

    if kernel_function_type == 'tricube':
        return (70 / 81 * (1 - abs(r ** 3)) ** 3) if r < 1 else 0

    if kernel_function_type == 'gaussian':
        return 1 / (2 * math.pi) ** (1 / 2) * math.e ** (-(1 / 2) * r ** 2)

    if kernel_function_type == 'cosine':
        return (math.pi / 4 * math.cos(math.pi / 2 * r)) if r < 1 else 0

    if kernel_function_type == 'logistic':
        return 1 / (math.e ** r + 2 + math.e ** (-r))

    if kernel_function_type == 'sigmoid':
        return 2 / math.pi * 1 / (math.e ** r + math.e ** (-r))
