import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


def cosine_similarity(matrix):
    return sklearn_cosine(matrix)


def pearson_similarity(matrix):
    matrix_centered = matrix - np.nanmean(matrix, axis=1, keepdims=True)
    matrix_centered = np.nan_to_num(matrix_centered, nan=0.0)
    mask = ~np.isnan(matrix)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.sqrt(np.nansum(matrix_centered ** 2, axis=1, keepdims=True))
        norm[norm == 0] = 1
        matrix_norm = matrix_centered / norm
    return matrix_norm @ matrix_norm.T


def adjusted_cosine_similarity(matrix):
    user_mean = np.nanmean(matrix, axis=1, keepdims=True)
    matrix_centered = matrix - user_mean
    matrix_centered = np.nan_to_num(matrix_centered, nan=0.0)
    return sklearn_cosine(matrix_centered.T)


def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0
