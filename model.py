import numpy as np
import lightgbm as lgb

from .misc import prepare_lightgbm_sets, remove_docs, rename_dict_key


class SelGB():
    def __init__(self, queries, labels, qs_len, min_neg_rel=0, eval_set=None):
        self.queries = queries
        self.labels = labels
        self.qs_len = qs_len

        self.eval_set = eval_set
        self.eval_results = {}
        self.min_neg_rel = min_neg_rel

    def _update_eval_result(self, tmp_eval_results):
        if not bool(self.eval_results):
            self.eval_set = tmp_eval_results
        else:
            for key in tmp_eval_results:
                self._eval_result[key][self.type_ndcg] += tmp_eval_results[key][self.type_ndcg]

    def _samp_foo(self, y_score, y_true, qs_len, p1, p2):
        n_docs = y_score.shape[0]
        idx_docs = np.arange(n_docs)
        cum = np.cumsum(qs_len)[:-1]

        ids_list = []
        for labels, score, idx in zip(np.array_split(y_true, cum), np.array_split(y_score, cum), np.array_split(idx_docs, cum)):
            idx_pos = np.where(labels > self.min_neg_rel)[0]
            idx_neg = np.where(labels <= self.min_neg_rel)[0]

            score_neg = score[idx_neg]
            rk_neg = len(score_neg) - 1 - np.argsort(score_neg[::-1], kind='stable')[::-1] # stable argsort in descending order of not-relevant documents
            idx_rk_neg = idx_neg[rk_neg]

            # calcola il numero di doc negativi da tenere
            th_neg_p1 = int(len(idx_rk_neg) * p1)
            th_neg_p2 = int(len(idx_rk_neg) * (1 - p2))

            ids_list += [idx[idx_pos], idx[idx_neg[:th_neg_p1]], idx[idx_neg[th_neg_p2:]]]

        return np.concatenate(ids_list)

    def train(self, params, n=10, p1=0.01, p2=0, verbose=1, **kwargs):
        safe_params = rename_dict_key(params, "num_iterations", ["num_iteration", "n_iter", "num_tree", "num_trees", "num_round", "num_rounds", "nrounds", "num_boost_round", "n_estimators", "max_iter"])
        safe_params = rename_dict_key(params, "early_stopping", ["early_stopping_round", "early_stopping_rounds", "n_iter_no_change"])
        safe_params = rename_dict_key(params, "seed", ["random_seed", "random_state"])

        early_stopping = False
        if "early_stopping" in safe_params:
            early_stopping = safe_params["early_stopping"]
            if early_stopping:
                print("[Info] SelGB's early stopping function is used, not LightGBM's")

        if "seed" in safe_params:
            np.random.seed(safe_params["seed"])

        save_num_iter = 100
        if "num_iterations" in kwargs:
            save_num_iter = kwargs.pop("num_iterations")
        if "num_iterations" in safe_params:
            save_num_iter = safe_params.pop("num_iterations")

        callbacks = []
        if "callbacks" in kwargs:
            callbacks = kwargs.pop("callbacks")

        iter_list = [n] * (save_num_iter // n)
        if save_num_iter / n != save_num_iter // n:
            iter_list += [save_num_iter - n * (save_num_iter // n)]

        model = None
        clean_queries, clean_labels, clean_qs_lens = self.queries, self.labels, self.qs_len
        for i, n_iter in enumerate(iter_list):
            train_set, valid_sets, valid_names = prepare_lightgbm_sets((clean_queries, clean_labels, clean_qs_lens), self.eval_set)
            partial_eval_result = {}
            model = lgb.train(params=safe_params, train_set=train_set, num_boost_round=n_iter, valid_sets=valid_sets, valid_names=valid_names, init_model=model,  callbacks = callbacks + [lgb.log_evaluation(verbose), lgb.record_evaluation(partial_eval_result)], **kwargs)
            self.update_eval_result(partial_eval_result)

            if i == len(iter_list) - 1:
                y_score = model.predict(self.queries)
                idx_to_removed = self._samp_foo(y_score, self.labels, self.qs_len, p1, p2)
                clean_queries, clean_labels, clean_qs_lens = remove_docs(self.queries, self.labels, self.qs_len, idx_to_removed)
              
        return model