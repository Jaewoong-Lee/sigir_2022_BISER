"""Evaluate Implicit Recommendation models."""
from typing import List

from scipy import sparse
import numpy as np
import pandas as pd

from .metrics import average_precision_at_k, dcg_at_k, recall_at_k


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tocsr(data: np.array, num_user: int, num_item: int) -> sparse.csr_matrix:
    """Convert data to csr_matrix."""
    matrix = sparse.lil_matrix((num_user, num_item))
    for (u, i, r) in data[:, :3]:
        matrix[u, i] = r
    return sparse.csr_matrix(matrix)

class PredictRankings:
    """Predict rankings by trained recommendations."""

    def __init__(self, user_embed: np.ndarray, item_embed: np.ndarray) -> None:
        """Initialize Class."""
        # latent embeddings
        self.user_embed = user_embed
        self.item_embed = item_embed

    def predict(self, users: np.array, items: np.array) -> np.ndarray:
        """Predict scores for each user-item pairs."""
        # predict ranking score for each user
        user_emb = self.user_embed[users].reshape(1, self.user_embed.shape[1])
        item_emb = self.item_embed[items]
        scores = (user_emb @ item_emb.T).flatten()

        return scores


class PredictRankings_i_u_AE:
    """Predict rankings by trained recommendations."""

    def __init__(self, train, num_users, num_items, weights, biases) -> None:
        """Initialize Class."""
        self.train_matrix = tocsr(train, num_users, num_items)
        self.num_users = num_users
        self.num_items = num_items
        self.weights_enc = weights[0]
        self.weights_dec = weights[1]
        self.biases_enc = biases[0]
        self.biases_dec = biases[1]

        self.R_u = np.zeros((self.num_users, self.num_items), dtype=np.float32)

        for u_id in range(num_users):
            items_by_user_id = self.train_matrix[u_id].indices

            rating_matrix = np.zeros((1, self.num_items), dtype=np.int32)
            for item_id in items_by_user_id:
                rating_matrix[0, item_id] = 1
            encoder_op = sigmoid((rating_matrix @ self.weights_enc) + self.biases_enc)
            decoder_op = (encoder_op @ self.weights_dec) + self.biases_dec
            scores = sigmoid(decoder_op).flatten()
            self.R_u[u_id] = scores

        train_matrix = tocsr(train, num_users, num_items)
        train_matrix = train_matrix.transpose()
        self.train_matrix = {}
        idx = 0
        for value in train_matrix:
            aaa = value.indices.copy().tolist()

            self.train_matrix[idx] = aaa
            idx += 1

        self.num_users = num_users
        self.num_items = num_items
        self.weights_enc = weights[2]
        self.weights_dec = weights[3]
        self.biases_enc = biases[2]
        self.biases_dec = biases[3]

        self.R_i = np.zeros((self.num_items, self.num_users), dtype=np.float32)

        for i_id in range(num_items):
            users_by_item_id = self.train_matrix[i_id]#.indices

            rating_matrix = np.zeros((1, self.num_users), dtype=np.int32)
            for user_id in users_by_item_id:
                rating_matrix[0, user_id] = 1

            encoder_op = sigmoid((rating_matrix @ self.weights_enc) + self.biases_enc)
            decoder_op = (encoder_op @ self.weights_dec) + self.biases_dec
            scores = sigmoid(decoder_op).flatten()
            self.R_i[i_id] = scores
        
        self.R_i = self.R_i.T

        self.R = (self.R_u + self.R_i)/2


    def predict(self, users: np.array, items: np.array) -> np.ndarray:
        """Predict scores for each user-item pairs."""
        user_predic = self.R[users]
        scores = user_predic[items]
        return scores


class PredictRankings_AE:
    """Predict rankings by trained recommendations."""

    def __init__(self, train, num_users, num_items, weights, biases) -> None:
        """Initialize Class."""
        self.train_matrix = tocsr(train, num_users, num_items)
        self.num_users = num_users
        self.num_items = num_items
        self.weights_enc = weights[0]
        self.weights_dec = weights[1]
        self.biases_enc = biases[0]
        self.biases_dec = biases[1]

        self.R = np.zeros((self.num_users, self.num_items), dtype=np.float32)

        for u_id in range(num_users):
            items_by_user_id = self.train_matrix[u_id].indices

            rating_matrix = np.zeros((1, self.num_items), dtype=np.int32)
            for item_id in items_by_user_id:
                rating_matrix[0, item_id] = 1
            encoder_op = sigmoid((rating_matrix @ self.weights_enc) + self.biases_enc)
            decoder_op = (encoder_op @ self.weights_dec) + self.biases_dec
            scores = sigmoid(decoder_op).flatten()
            self.R[u_id] = scores

    def predict(self, users: np.array, items: np.array) -> np.ndarray:
        """Predict scores for each user-item pairs."""
        user_predic = self.R[users]
        scores = user_predic[items]
        return scores


class PredictRankings_iAE:
    """Predict rankings by trained recommendations."""
    def __init__(self, train, num_users, num_items, weights, biases) -> None:
        """Initialize Class."""
        train_matrix = tocsr(train, num_users, num_items)
        train_matrix = train_matrix.transpose()
        self.train_matrix = {}
        idx = 0
        for value in train_matrix:
            aaa = value.indices.copy().tolist()

            self.train_matrix[idx] = aaa
            idx += 1

        self.num_users = num_users
        self.num_items = num_items
        self.weights_enc = weights[0]
        self.weights_dec = weights[1]
        self.biases_enc = biases[0]
        self.biases_dec = biases[1]

        self.R = np.zeros((self.num_items, self.num_users), dtype=np.float32)

        for i_id in range(num_items):
            users_by_item_id = self.train_matrix[i_id]#.indices

            rating_matrix = np.zeros((1, self.num_users), dtype=np.int32)
            for user_id in users_by_item_id:
                rating_matrix[0, user_id] = 1
            encoder_op = sigmoid((rating_matrix @ self.weights_enc) + self.biases_enc)
            decoder_op = (encoder_op @ self.weights_dec) + self.biases_dec
            scores = sigmoid(decoder_op).flatten()
            self.R[i_id] = scores
        
        self.R = self.R.T

    def predict(self, users: np.array, items: np.array) -> np.ndarray:
        """Predict scores for each user-item pairs."""
        user_predic = self.R[users]
        scores = user_predic[items]

        return scores

class PredictRankings_cjmf:
    """Predict rankings by trained recommendations."""

    def __init__(self, user_embed: list, item_embed: list, num_users:int, num_items:int, pred=None) -> None:
        if pred is None:
            R_all = np.zeros((num_users, num_items))
            C = len(user_embed)
            for m in range(C):
                P = user_embed[m]
                Q = item_embed[m]
                R = sigmoid(np.matmul(P, Q.T)) ## relevacne 예측
                R_all += R ## 예측된 relevacne 모두 합함 (여러 모델 썻기 때문)
            self.R = R_all / C ## 여러 모델 relevacne 평균
        else:
            self.R = pred


    def predict(self, users: int, items: np.array) -> np.ndarray:
        """Predict scores for each user-item pairs."""
        # predict ranking score for each user
        user_predic = self.R[users]
        scores = user_predic[items]

        return scores

class PredictRankings_macr:
    """Predict rankings by trained recommendations."""

    def __init__(self, user_embed: np.ndarray, item_embed: np.ndarray, pred=None) -> None:
        """Initialize Class."""
        # latent embeddings

        if pred is None:
            u_emb_t = user_embed[0]
            i_emb_t = user_embed[1]
            w_user = user_embed[2]

            batch_ratings = u_emb_t @ i_emb_t.T
            user_scores = u_emb_t @ w_user
            pred = batch_ratings*sigmoid(user_scores) 

        self.pred = pred

    # TODO inference time 줄이기
    def predict(self, users: np.array, items: np.array) -> np.ndarray:
        """Predict scores for each user-item pairs."""
        # predict ranking score for each user
        if self.pred is None:
            user_emb = self.user_embed[users].reshape(1, self.user_embed.shape[1])
            item_emb = self.item_embed[items]

            scores = (user_emb @ item_emb.T).flatten()
        else:
            scores = self.pred[users, items].flatten()

        return scores


def aoa_evaluator(user_embed: np.ndarray,
                  item_embed: np.ndarray,
                  train: np.ndarray,
                  test: np.ndarray,
                  num_users: int, 
                  num_items: int,
                  model_name: str,
                  at_k: List[int] = [1, 3, 5],
                  only_dcg: bool = False
                  ) -> pd.DataFrame:
    """Calculate ranking metrics with average-over-all evaluator."""
    
    # test data
    users = test[:, 0]
    items = test[:, 1]
    relevances = test[:, 2]

    # define model
    if model_name in ['uae']:
        dim = user_embed[0].shape[1]
        model = PredictRankings_AE(weights=user_embed, biases=item_embed, num_users=num_users, num_items=num_items, train=train)
    elif model_name in ['iae']:
        dim = user_embed[0].shape[1]
        model = PredictRankings_iAE(weights=user_embed, biases=item_embed, num_users=num_users, num_items=num_items, train=train)
    elif model_name in ['proposed']:
        dim = user_embed[0].shape[1]
        model = PredictRankings_i_u_AE(weights=user_embed, biases=item_embed, num_users=num_users, num_items=num_items, train=train)
    elif model_name in ['cjmf']:
        dim = user_embed[0].shape[1]
        model = PredictRankings_cjmf(user_embed=user_embed, item_embed=item_embed, num_users=num_users, num_items=num_items)
    elif model_name in ['macr']:
        dim = user_embed[0].shape[1]
        model = PredictRankings_macr(user_embed=user_embed, item_embed=item_embed)
    else:
        dim = user_embed.shape[1]
        model = PredictRankings(user_embed=user_embed, item_embed=item_embed)

    # prepare ranking metrics
    if only_dcg:
        metrics = {'NDCG': dcg_at_k}
    else: 
        metrics = {'NDCG': dcg_at_k,
           'Recall': recall_at_k,
           'MAP': average_precision_at_k}

    results = {}
    for k in at_k:
        for metric in metrics:
            results[f'{metric}@{k}'] = []

    # calculate ranking metrics
    for user in set(users):
        indices = users == user
        pos_items = items[indices] # item, relevance
        rel = relevances[indices]
        if len(rel) < max(at_k):
            print("no eval")
            continue
        # predict ranking score for each user
        scores = model.predict(users=user, items=pos_items)
        for k in at_k:
            for metric, metric_func in metrics.items():
                results[f'{metric}@{k}'].append(metric_func(rel, scores, k))

        # aggregate results
        results_df = pd.DataFrame(index=results.keys())
        results_df[f'{model_name}_{dim}'] = list(map(np.mean, list(results.values()))) 

    return results_df.sort_index()


def unbiased_evaluator(user_embed: np.ndarray, 
                  item_embed: np.ndarray,
                  train: np.ndarray,
                  test: np.ndarray,
                  num_users: int, 
                  num_items: int,
                  pscore: np.ndarray,
                  model_name: str,
                  val: np.ndarray,
                  flag_test: bool,
                  flag_unbiased: bool,
                  at_k: List[int] = [1, 3, 5],
                  only_dcg: bool = False,
                  pred = None) -> pd.DataFrame:
                  
    """Calculate ranking metrics by unbiased evaluator."""
    # test data
    users = test[:, 0]
    items = test[:, 1]
    if flag_test:
        train_val = np.r_[train, val, test]
    else:
        train_val = np.r_[train, test]
    positive_pairs = train_val[train_val[:, 2] == 1, :2]

    # define model
    if model_name in ['uae']:
        dim = user_embed[0].shape[1]
        model = PredictRankings_AE(weights=user_embed, biases=item_embed, num_users=num_users, num_items=num_items, train=train)
    elif model_name in ['iae']:
        dim = user_embed[0].shape[1]
        model = PredictRankings_iAE(weights=user_embed, biases=item_embed, num_users=num_users, num_items=num_items, train=train)
    elif model_name in ['proposed']:
        dim = user_embed[0].shape[1]
        model = PredictRankings_i_u_AE(weights=user_embed, biases=item_embed, num_users=num_users, num_items=num_items, train=train)
    elif model_name in ['cjmf']:
        dim = user_embed[0].shape[1]
        model = PredictRankings_cjmf(user_embed=user_embed, item_embed=item_embed, num_users=num_users, num_items=num_items, pred=pred)
    elif model_name in ['macr']:
        dim = user_embed[0].shape[1]
        model = PredictRankings_macr(user_embed=user_embed, item_embed=item_embed, pred=pred)
    else:
        dim = user_embed.shape[1]
        model = PredictRankings(user_embed=user_embed, item_embed=item_embed)

    # prepare ranking metrics
    if only_dcg:
        metrics = {'NDCG': dcg_at_k}
    else: 
        metrics = {'NDCG': dcg_at_k,
           'Recall': recall_at_k,
           'MAP': average_precision_at_k}

    results = {}
    for k in at_k:
        for metric in metrics:
            results[f'{metric}@{k}'] = []

    unique_items = np.asarray( range( num_items ) )
    # calculate ranking metrics

    for user in set(users):
        indices = users == user
        pos_items = items[indices]
        all_pos_items = positive_pairs[positive_pairs[:, 0] == user, 1] 
        neg_items = np.setdiff1d(unique_items, all_pos_items)
        used_items = np.r_[pos_items, neg_items]
        pscore_ = pscore[used_items] 
        relevances = np.r_[np.ones_like(pos_items), np.zeros_like(neg_items)]

        # calculate an unbiased DCG score for a user
        scores = model.predict(users=user, items=used_items)
        for k in at_k:
            for metric, metric_func in metrics.items():
                if flag_unbiased:
                    results[f'{metric}@{k}'].append(metric_func(relevances, scores, k, pscore_)) 
                else:
                    results[f'{metric}@{k}'].append(metric_func(relevances, scores, k, None))

        # aggregate results
        results_df = pd.DataFrame(index=results.keys())
        results_df[f'{model_name}_{dim}'] = list(map(np.mean, list(results.values())))

    return results_df.sort_index()