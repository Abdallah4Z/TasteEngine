import numpy as np
import pandas as pd
from utils.similarity import cosine_similarity, pearson_similarity, adjusted_cosine_similarity
from utils.helpers import build_user_item_matrix


class CollaborativeFiltering:
    def __init__(self, ratings_df):
        self.ratings = ratings_df
        self.matrix = build_user_item_matrix(ratings_df)
        self.user_item_matrix = self.matrix.values
        self.n_users, self.n_items = self.user_item_matrix.shape
        self.user_ids = self.matrix.index.values
        self.item_ids = self.matrix.columns.values

        self.user_means = np.nanmean(self.user_item_matrix, axis=1)
        self.global_mean = np.nanmean(self.user_item_matrix)

    def _get_user_index(self, user_id):
        indices = np.where(self.user_ids == user_id)[0]
        return indices[0] if len(indices) > 0 else None

    def _get_item_index(self, item_id):
        indices = np.where(self.item_ids == item_id)[0]
        return indices[0] if len(indices) > 0 else None

    def user_based_cf(self, user_id, n_recommendations=10, k=20):
        u_idx = self._get_user_index(user_id)
        if u_idx is None:
            return []

        matrix_filled = np.nan_to_num(self.user_item_matrix, nan=self.global_mean)
        sim_matrix = cosine_similarity(matrix_filled)
        user_sim = sim_matrix[u_idx]
        user_sim[u_idx] = 0

        user_ratings = self.user_item_matrix[u_idx]
        unseen = np.where(np.isnan(user_ratings))[0]
        if len(unseen) == 0:
            return []

        predictions = []
        for i_idx in unseen:
            similar_users = np.argsort(user_sim)[::-1][:k]
            valid = []
            for su in similar_users:
                if not np.isnan(self.user_item_matrix[su, i_idx]) and user_sim[su] > 0:
                    valid.append(su)
            if not valid:
                continue
            sim_vals = user_sim[valid]
            ratings_vals = self.user_item_matrix[valid, i_idx]
            pred = np.average(ratings_vals, weights=sim_vals)
            predictions.append((int(self.item_ids[i_idx]), float(pred)))

        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]

    def item_based_cf(self, user_id, n_recommendations=10, k=15):
        u_idx = self._get_user_index(user_id)
        if u_idx is None:
            return []

        item_sim = adjusted_cosine_similarity(self.user_item_matrix)
        user_ratings = self.user_item_matrix[u_idx]
        unseen = np.where(np.isnan(user_ratings))[0]
        rated = np.where(~np.isnan(user_ratings))[0]

        if len(rated) == 0:
            return []

        predictions = []
        for i_idx in unseen:
            sim_to_rated = item_sim[i_idx, rated]
            best = np.argsort(sim_to_rated)[::-1][:k]
            valid = [(r, sim_to_rated[r]) for r in best if sim_to_rated[r] > 0 and r < len(rated)]
            if not valid:
                continue
            neighbor_indices = [rated[r[0]] for r in valid]
            sim_vals = [r[1] for r in valid]
            rating_vals = user_ratings[neighbor_indices]
            pred = np.average(rating_vals, weights=sim_vals)
            predictions.append((int(self.item_ids[i_idx]), float(pred)))

        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]

    def svd(self, user_id, n_recommendations=10, n_factors=20, n_epochs=20, lr=0.005, reg=0.02):
        u_idx = self._get_user_index(user_id)
        if u_idx is None:
            return []

        matrix_imputed = self.user_item_matrix.copy()
        matrix_imputed = np.nan_to_num(matrix_imputed, nan=self.global_mean)

        n_u, n_i = matrix_imputed.shape
        np.random.seed(42)
        P = np.random.normal(0, 0.1, (n_u, n_factors))
        Q = np.random.normal(0, 0.1, (n_i, n_factors))
        bu = np.zeros(n_u)
        bi = np.zeros(n_i)

        observed = []
        for u in range(n_u):
            for i in range(n_i):
                if not np.isnan(self.user_item_matrix[u, i]):
                    observed.append((u, i))

        for epoch in range(n_epochs):
            np.random.shuffle(observed)
            for u, i in observed:
                r = self.user_item_matrix[u, i]
                pred = self.global_mean + bu[u] + bi[i] + np.dot(P[u], Q[i])
                err = r - pred
                bu[u] += lr * (err - reg * bu[u])
                bi[i] += lr * (err - reg * bi[i])
                P[u] += lr * (err * Q[i] - reg * P[u])
                Q[i] += lr * (err * P[u] - reg * Q[i])

        user_ratings = self.user_item_matrix[u_idx]
        unseen = np.where(np.isnan(user_ratings))[0]

        predictions = []
        for i_idx in unseen:
            pred = self.global_mean + bu[u_idx] + bi[i_idx] + np.dot(P[u_idx], Q[i_idx])
            predictions.append((int(self.item_ids[i_idx]), float(pred)))

        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]

    def knn_cf(self, user_id, n_recommendations=10, k=10):
        u_idx = self._get_user_index(user_id)
        if u_idx is None:
            return []

        from sklearn.neighbors import NearestNeighbors
        matrix_imputed = np.nan_to_num(self.user_item_matrix, nan=self.global_mean)
        nn = NearestNeighbors(n_neighbors=min(k + 1, self.n_users), metric="cosine")
        nn.fit(matrix_imputed)
        distances, indices = nn.kneighbors(matrix_imputed[u_idx].reshape(1, -1))
        neighbor_indices = indices[0][1:]

        user_ratings = self.user_item_matrix[u_idx]
        unseen = np.where(np.isnan(user_ratings))[0]

        predictions = []
        for i_idx in unseen:
            neighbor_ratings = []
            neighbor_dists = []
            for ni in neighbor_indices:
                if not np.isnan(self.user_item_matrix[ni, i_idx]):
                    neighbor_ratings.append(self.user_item_matrix[ni, i_idx])
                    neighbor_dists.append(distances[0][list(indices[0]).index(ni)] + 1e-6)
            if not neighbor_ratings:
                continue
            weights = 1.0 / np.array(neighbor_dists)
            pred = np.average(neighbor_ratings, weights=weights)
            predictions.append((int(self.item_ids[i_idx]), float(pred)))

        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]

    def slope_one(self, user_id, n_recommendations=10):
        u_idx = self._get_user_index(user_id)
        if u_idx is None:
            return []

        user_ratings = self.user_item_matrix[u_idx]
        unseen = np.where(np.isnan(user_ratings))[0]
        rated = np.where(~np.isnan(user_ratings))[0]

        if len(rated) == 0:
            return []

        predictions = []
        for i_idx in unseen:
            numerator = 0.0
            denominator = 0.0
            for j_idx in rated:
                dev = 0.0
                count = 0
                for u in range(self.n_users):
                    if not np.isnan(self.user_item_matrix[u, i_idx]) and not np.isnan(self.user_item_matrix[u, j_idx]):
                        dev += self.user_item_matrix[u, i_idx] - self.user_item_matrix[u, j_idx]
                        count += 1
                if count > 0:
                    avg_dev = dev / count
                    numerator += (user_ratings[j_idx] + avg_dev)
                    denominator += 1
            if denominator > 0:
                pred = numerator / denominator
                predictions.append((int(self.item_ids[i_idx]), float(pred)))

        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]

    def recommend(self, method, user_id, n_recommendations=10, **kwargs):
        methods = {
            "user_based": self.user_based_cf,
            "item_based": self.item_based_cf,
            "svd": self.svd,
            "knn": self.knn_cf,
            "slope_one": self.slope_one,
        }
        func = methods.get(method)
        if func is None:
            raise ValueError(f"Unknown method: {method}")
        return func(user_id, n_recommendations=n_recommendations, **kwargs)

    def get_all_methods(self):
        return ["user_based", "item_based", "svd", "knn", "slope_one"]
