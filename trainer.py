
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import ops

from evaluate.evaluator import aoa_evaluator, unbiased_evaluator
from models.recommenders import PointwiseRecommender, MACR

from AE_trainer import ae_trainer
from PointWise_trainer import pointwise_trainer, macr_trainer, cjmf_trainer

from tqdm import tqdm


class Trainer:
    """Trainer Class for ImplicitRecommender."""
    def __init__(self, data: str, random_state: list, hidden: int, date_now: str, max_iters: int = 1000, lam: float=1e-4, batch_size: int = 12, wu=0.1, wi=0.1, alpha: float = 0.5, \
                clip: float=0.1, eta: float=0.1, model_name: str='mf', unbiased_eval:bool = True, neg_sample: int=10, C: int=8, alpha_cjmf: float=220000, beta_cjmf: float=0.5, \
                macr_c: float=0.1, macr_alpha: float=0.1, macr_beta: float=0.1, best_model_save: bool=True) -> None:

        """Initialize class."""
        self.data = data
        self.at_k = [1, 3, 5]
        self.dim = hidden
        self.lam = lam
        self.clip = clip if model_name == 'relmf' else 0
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.eta = eta
        self.model_name = model_name
        self.unbiased_eval = unbiased_eval

        self.wu = wu
        self.wi = wi
        self.alpha = alpha
        self.neg_sample = neg_sample

        # cjmf
        self.C = C
        self.alpha_cjmf = alpha_cjmf
        self.beta_cjmf = beta_cjmf

        # macr
        self.macr_c = macr_c
        self.macr_alpha = macr_alpha
        self.macr_beta = macr_beta

        self.best_model_save = best_model_save
        self.date_now = date_now

        self.random_state = [r for r in range(1, int(random_state) + 1)]
        print("======================================================")
        print("random state: ", self.random_state)
        print("======================================================")

    def run(self) -> None:
        print("======================================================")
        print("date: ", self.date_now)
        print("======================================================")

        """Train pointwise implicit recommenders."""
        train = np.load(f'data/{self.data}/point_{self.alpha}/train.npy')
        val = np.load(f'data/{self.data}/point_{self.alpha}/val.npy')
        test = np.load(f'data/{self.data}/point_{self.alpha}/test.npy')
        pscore = np.load(f'data/{self.data}/point_{self.alpha}/pscore.npy')
        item_freq = np.load(f'data/{self.data}/point_{self.alpha}/item_freq.npy')
        num_users = np.int(train[:, 0].max() + 1)
        num_items = np.int(train[:, 1].max() + 1)

        ret_path = Path(f'logs/{self.data}/{self.model_name}/results/')
        ret_path.mkdir(parents=True, exist_ok=True)

        self.log_dir = ret_path / f'{self.date_now}'

        """
        Add methods in AE models 
        """
        sub_results_sum = pd.DataFrame()
        for random_state in tqdm(self.random_state):
            print("random seed now :", random_state)
            tf.reset_default_graph()
            ops.reset_default_graph()
            tf.set_random_seed(random_state)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)

            # train
            if self.model_name in ['uae', 'iae']:
                weights_enc, weights_dec, bias_enc, bias_dec = ae_trainer(sess, data=self.data, train=train, val=val, test=test,
                            num_users=num_users, num_items=num_items, n_components=self.dim, 
                            eta=self.eta, lam=self.lam, max_iters=self.max_iters, batch_size=self.batch_size, 
                            model_name=self.model_name, item_freq=item_freq,
                            unbiased_eval = self.unbiased_eval, random_state=random_state)

            elif self.model_name in ['proposed']:
                weights_enc_u, weights_dec_u, bias_enc_u, bias_dec_u, weights_enc_i, weights_dec_i, bias_enc_i, bias_dec_i = ae_trainer(sess, data=self.data, train=train, val=val, test=test,
                            num_users=num_users, num_items=num_items, n_components=self.dim, wu=self.wu, wi=self.wi,
                            eta=self.eta, lam=self.lam, max_iters=self.max_iters, batch_size=self.batch_size,
                            model_name=self.model_name, item_freq=item_freq,
                            unbiased_eval = self.unbiased_eval, random_state=random_state)

            elif self.model_name in ['mf', 'relmf']:
                model = PointwiseRecommender(model_name=self.model_name,
                    num_users=num_users, num_items=num_items,
                    clip=self.clip, dim=self.dim, lam=self.lam, eta=self.eta)

                u_emb, i_emb = pointwise_trainer(
                    sess, data=self.data, model=model, train=train, val = val, test=test, 
                    num_users=num_users, num_items=num_items, pscore=pscore,
                    max_iters=self.max_iters, batch_size=self.batch_size, item_freq=item_freq, unbiased_eval = self.unbiased_eval,
                    model_name=self.model_name, date_now=self.date_now)

            elif self.model_name in ['cjmf']:
                    u_emb, i_emb = cjmf_trainer(sess=sess, data=self.data, n_components=self.dim, num_users=num_users, num_items=num_items, \
                                                batch_size=self.batch_size, max_iters=self.max_iters, item_freq=item_freq, \
                                                unbiased_eval=self.unbiased_eval, C=self.C,  lr=self.eta, reg=self.lam, \
                                                alpha=self.alpha_cjmf, beta=self.beta_cjmf, train=train, val=val, seed=random_state, model_name=self.model_name)

            elif self.model_name in ['macr']:
                    model = MACR(model_name=self.model_name, 
                        num_users=num_users, num_items=num_items,
                        dim=self.dim, lam=self.lam, eta=self.eta, batch_size=self.batch_size, c=self.macr_c, alpha=self.macr_alpha, beta=self.macr_beta)

                    u_emb, i_emb = macr_trainer(
                        sess, data=self.data, model=model, train=train, val = val, test=test, 
                        num_users=num_users, num_items=num_items, pscore=pscore,
                        max_iters=self.max_iters, batch_size=self.batch_size, item_freq=item_freq, unbiased_eval = self.unbiased_eval,
                        model_name=self.model_name, date_now=self.date_now, neg_sample=self.neg_sample)

            # model parameter save
            if self.best_model_save:
                if self.model_name in ['uae', 'iae']:
                    np.save(file=ret_path / f'{self.date_now}_dim_{self.dim}_random_{random_state}_weights_enc.npy', arr=weights_enc )
                    np.save(file=ret_path / f'{self.date_now}_dim_{self.dim}_random_{random_state}_weights_dec.npy', arr=weights_dec )
                    np.save(file=ret_path / f'{self.date_now}_dim_{self.dim}_random_{random_state}_bias_enc.npy', arr=bias_enc )
                    np.save(file=ret_path / f'{self.date_now}_dim_{self.dim}_random_{random_state}_bias_dec.npy', arr=bias_dec )

                elif self.model_name in ['proposed']:
                    np.save(file=ret_path / f'{self.date_now}_dim_{self.dim}_random_{random_state}_weights_enc_u.npy', arr=weights_enc_u )
                    np.save(file=ret_path / f'{self.date_now}_dim_{self.dim}_random_{random_state}_weights_dec_u.npy', arr=weights_dec_u )
                    np.save(file=ret_path / f'{self.date_now}_dim_{self.dim}_random_{random_state}_bias_enc_u.npy', arr=bias_enc_u )
                    np.save(file=ret_path / f'{self.date_now}_dim_{self.dim}_random_{random_state}_bias_dec_u.npy', arr=bias_dec_u )

                    np.save(file=ret_path / f'{self.date_now}_dim_{self.dim}_random_{random_state}_weights_enc_i.npy', arr=weights_enc_i )
                    np.save(file=ret_path / f'{self.date_now}_dim_{self.dim}_random_{random_state}_weights_dec_i.npy', arr=weights_dec_i )
                    np.save(file=ret_path / f'{self.date_now}_dim_{self.dim}_random_{random_state}_bias_enc_i.npy', arr=bias_enc_i )
                    np.save(file=ret_path / f'{self.date_now}_dim_{self.dim}_random_{random_state}_bias_dec_i.npy', arr=bias_dec_i )
                else:  
                    np.save(file=ret_path / f'{self.date_now}_dim_{self.dim}_random_{random_state}_u_emb.npy', arr=u_emb)
                    np.save(file=ret_path / f'{self.date_now}_dim_{self.dim}_random_{random_state}_i_emb.npy', arr=i_emb)

            # test
            if self.model_name in ['uae', 'iae']:
                results = aoa_evaluator(user_embed=[weights_enc, weights_dec], item_embed=[bias_enc, bias_dec],
                                    train=train, test=test, num_users=num_users, num_items=num_items,
                                    model_name=self.model_name, at_k=self.at_k)

            elif self.model_name in ['proposed']:
                results = aoa_evaluator(user_embed=[weights_enc_u, weights_dec_u, weights_enc_i, weights_dec_i], item_embed=[bias_enc_u, bias_dec_u, bias_enc_i, bias_dec_i],
                                    train=train, test=test, num_users=num_users, num_items=num_items,
                                    model_name=self.model_name, at_k=self.at_k)

            elif self.model_name in ['cjmf', 'macr']:
                results = aoa_evaluator(user_embed=u_emb, item_embed=i_emb,
                                            train=train, test=test, num_users=num_users, num_items=num_items,
                                            model_name=self.model_name, at_k=self.at_k)
            else:
                results = aoa_evaluator(user_embed=u_emb, item_embed=i_emb,
                                    train=train, test=test, num_users=num_users, num_items=num_items,
                                    model_name=self.model_name, at_k=self.at_k)
                                    
            sub_results_sum = sub_results_sum.add(results, fill_value=0) 

        sub_results_mean = sub_results_sum / len(self.random_state)
        sub_results_mean.to_csv(ret_path / f'{self.date_now}_dim_{self.dim}_lr_{self.eta}_reg_{self.lam}_{self.batch_size}_random_{len(self.random_state)}.csv')
