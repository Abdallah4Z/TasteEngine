import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


def cosine_similarity(matrix):
    """Compute pairwise cosine similarity between all rows of a matrix.
    
    Used by User-Based CF to measure user-user similarity.
    Range: -1 to 1 (higher = more similar).
    """
    return sklearn_cosine(matrix)


def pearson_similarity(matrix):
    """Compute Pearson correlation coefficient between all rows.
    
    Centers each row by its mean, then computes cosine similarity on
    centered data. Measures linear correlation between users' rating patterns.
    """
    matrix_centered = matrix - np.nanmean(matrix, axis=1, keepdims=True)
    matrix_centered = np.nan_to_num(matrix_centered, nan=0.0)
    mask = ~np.isnan(matrix)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.sqrt(np.nansum(matrix_centered ** 2, axis=1, keepdims=True))
        norm[norm == 0] = 1
        matrix_norm = matrix_centered / norm
    return matrix_norm @ matrix_norm.T


def adjusted_cosine_similarity(matrix):
    """Adjusted cosine similarity for item-item comparisons.
    
    Subtracts each user's mean rating before computing cosine similarity
    on the transposed (item×item) matrix. Corrects for users who rate
    everything high or low. Used by Item-Based CF.
    """
    user_mean = np.nanmean(matrix, axis=1, keepdims=True)
    matrix_centered = matrix - user_mean
    matrix_centered = np.nan_to_num(matrix_centered, nan=0.0)
    return sklearn_cosine(matrix_centered.T)


def jaccard_similarity(set1, set2):
    """Jaccard similarity = |intersection| / |union|. Measures overlap between two sets.
    
    Range: 0 (no overlap) to 1 (identical sets). Used for binary preference data.
    """
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0
