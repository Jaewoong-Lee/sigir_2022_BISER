import tensorflow as tf
import time
import numpy as np

from tqdm import tqdm, trange
from scipy import sparse
from scipy.sparse import coo_matrix
from evaluate.evaluator import aoa_evaluator, unbiased_evaluator


def sigmoid(x):
    sigm = 1. / (1. + np.exp(-x))
    return sigm

def tocsr(data: np.array, num_user: int, num_item: int) -> sparse.csr_matrix:
    """Convert data to csr_matrix."""
    matrix = sparse.lil_matrix((num_user, num_item))
    for (u, i, r) in data[:, :3]:
        matrix[u, i] = r
    return sparse.csr_matrix(matrix)

def csr_to_user_dict(train_matrix):
    """convert a scipy.sparse.csr_matrix to a dict,
    where the key is row number, and value is the
    non-empty index in each row.
    """
    train_dict = {}
    for idx, value in enumerate(train_matrix):
        train_dict[idx] = value.indices.copy().tolist()
    return train_dict

def csr_to_user_dict_neg(train_matrix):
    """convert a scipy.sparse.csr_matrix to a dict,
    where the key is row number, and value is the
    non-empty index in each row.
    """
    train_dict_neg = {}
    unique_items = np.asarray(range(train_matrix.shape[1]))
    for idx, value in enumerate(train_matrix):
        pos_list = value.indices.copy().tolist()
        neg_items = np.setdiff1d(unique_items, pos_list)

        train_dict_neg[idx] = neg_items
    return train_dict_neg


def negative_sampling_MF(num_user, num_item, pos_user_array, pos_item_array, neg_rate):
    train_mat = coo_matrix((np.ones(len(pos_item_array)),
                            (pos_user_array, pos_item_array)),
                           shape=(num_user, num_item)).toarray()
    user_pos = pos_user_array.reshape((-1, 1))
    user_neg = np.tile(pos_user_array, neg_rate).reshape((-1, 1))
    pos = pos_item_array.reshape((-1, 1))
    neg = np.random.choice(np.arange(num_item), size=(neg_rate * pos_user_array.shape[0]), replace=True).reshape((-1, 1))
    label = train_mat[user_neg, neg]
    idx = (label == 0).reshape(-1)
    user_neg = user_neg[idx, :]
    neg = neg[idx, :]
    pos_lable = np.ones(pos.shape)
    neg_lable = np.zeros(neg.shape)
    return np.concatenate([user_pos, user_neg], axis=0), np.concatenate([pos, neg], axis=0), np.concatenate([pos_lable, neg_lable], axis=0)


class CJMF:
    def __init__(self, sess, num_item, num_user, hidden, batch_size, data_name, epoch, item_pop, \
                 C, lr, alpha, beta, train, val, early_stop, unbiased_eval, reg, item_freq, seed):

        self.Pr = np.zeros((num_user, num_item))
        self.R = np.zeros((num_user, num_item))
        self.item_freq = item_freq

        self.seed = seed
        self.data_name = data_name
        self.val = val

        self.unbiased_eval= unbiased_eval
        self.early_stop = early_stop

        self.best_P_list = []
        self.best_Q_list = []

        self.sess = sess

        self.num_item = num_item
        self.num_user = num_user

        self.hidden = hidden

        self.batch_size = batch_size

        self.train = train

###############################
        # neg_num is decided from kinds of dataset
        self.neg_num = 5
        self.item_list_all = train[:,1]
        self.item_list_all = np.unique(self.item_list_all)

        tr_pos_index = np.where(train[:,2] > 0.5)[0]
        val_pos_index = np.where(val[:,2] > 0.5)[0]
        train_pos = train[tr_pos_index]
        val_pos = val[val_pos_index]

        train_val = np.r_[train_pos, val_pos]
        len_train = len(tr_pos_index)

###############################
        self.train_df = [train_pos[:,0], train_pos[:,1], train_pos[:,2]]

        self.C = C
        self.df_list = []

        df_len = int(len_train * 1. / self.C)
        left_idx = range(len_train)

        for i in range(self.C - 1):
            idx = np.random.RandomState(12345).choice(left_idx, int(df_len), replace=False).tolist()
            self.df_list.append([self.train_df[0][idx], self.train_df[1][idx], self.train_df[2][idx]])
            left_idx = list(set(left_idx) - set(idx))
        self.df_list.append([self.train_df[0][left_idx], self.train_df[1][left_idx], self.train_df[2][left_idx]])

        self.epoch = epoch

        self.lr = lr

        self.reg = reg
        self.alpha = alpha
        self.beta = beta

        self.item_pop = item_pop.reshape((-1, 1)) 

        print('******************** CJMF ********************')
        self._prepare_model()
        print('********** CJMF Initialization Done **********')


    def run(self):
        early_stop = 0
        best_score = 0.0
        init = tf.global_variables_initializer()
        self.sess.run(init)

        m_list = np.asarray(range(self.C))
        for epoch_itr in range(1, self.epoch + 1):
            self.train_model(epoch_itr, m_list)
            early_stop += 1

            if early_stop < self.early_stop +1: # early_stop 5
                best_score_tmp = self.test_model(epoch_itr, best_score)

                if best_score < best_score_tmp:
                    best_score = best_score_tmp
                    early_stop = 0
            else:
                break

        self.sess.close()
        return self.best_P_list, self.best_Q_list

    def _prepare_model(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None, 1], name="user_input")
            self.item_input = tf.placeholder(tf.int32, shape=[None, 1], name="item_input")
            self.label_input = tf.placeholder(tf.float32, shape=[None, 1], name="label_input")
            self.pop_input = tf.placeholder(tf.float32, shape=[None, 1], name="pop_input")
            self.rel_input = tf.placeholder(tf.float32, shape=[None, 1], name="rel_input")
            self.exp_input = tf.placeholder(tf.float32, shape=[None, 1], name="exp_input")

        self.P_list = []
        self.Q_list = []
        self.c_list = []
        self.d_list = []
        self.a_list = []
        self.b_list = []
        self.e_list = []
        self.f_list = []

        self.para_rel_list = []
        self.para_exp_list = []

        for m in range(self.C):
            tf.set_random_seed(self.seed) 
            with tf.variable_scope('Relevance_' + str(m), reuse=tf.AUTO_REUSE):
                P = tf.get_variable(name='P_' + str(m),
                                    initializer=tf.truncated_normal(shape=[self.num_user, self.hidden],
                                                                    mean=0, stddev=0.03,
                                                                    dtype=tf.float32), dtype=tf.float32)
                Q = tf.get_variable(name='Q_' + str(m),
                                    initializer=tf.truncated_normal(shape=[self.num_item, self.hidden],
                                                                    mean=0, stddev=0.03,
                                                                    dtype=tf.float32), dtype=tf.float32)
            para_rel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Relevance_' + str(m))
            self.P_list.append(P)
            self.Q_list.append(Q)
            self.para_rel_list.append(para_rel)

            with tf.variable_scope('Exposure_' + str(m), reuse=tf.AUTO_REUSE): 
                c = tf.get_variable(name='c_' + str(m),
                                    initializer=tf.truncated_normal(shape=[self.hidden, 1],
                                                                    mean=0, stddev=0.03,
                                                                    dtype=tf.float32), dtype=tf.float32)
                d = tf.get_variable(name='d_' + str(m),
                                    initializer=tf.truncated_normal(shape=[1, 1],
                                                                    mean=0, stddev=0.03,
                                                                    dtype=tf.float32), dtype=tf.float32)
                a = tf.get_variable(name='a_' + str(m),
                                    initializer=tf.truncated_normal(shape=[self.hidden, 1],
                                                                    mean=0, stddev=0.03,
                                                                    dtype=tf.float32), dtype=tf.float32)
                b = tf.get_variable(name='b_' + str(m),
                                    initializer=tf.truncated_normal(shape=[1, 1],
                                                                    mean=0, stddev=0.03,
                                                                    dtype=tf.float32), dtype=tf.float32)
                e = tf.get_variable(name='e_' + str(m),
                                    initializer=tf.truncated_normal(shape=[self.hidden, 1],
                                                                    mean=0, stddev=0.03,
                                                                    dtype=tf.float32), dtype=tf.float32)
                f = tf.get_variable(name='f_' + str(m),
                                    initializer=tf.truncated_normal(shape=[1, 1],
                                                                    mean=0, stddev=0.03,
                                                                    dtype=tf.float32), dtype=tf.float32)
            para_exp = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Exposure_' + str(m))
            self.c_list.append(c)
            self.d_list.append(d)
            self.a_list.append(a)
            self.b_list.append(b)
            self.e_list.append(e)
            self.f_list.append(f)
            self.para_exp_list.append(para_exp)

        self.rel_cost_list = []
        self.exp_cost_list = []
        self.rel_reg_cost_list = []
        self.exp_reg_cost_list = []
        self.rel_cost_final_list = []
        self.exp_cost_final_list = []

        for m in range(self.C):
            P = self.P_list[m]
            Q = self.Q_list[m]
            c = self.c_list[m]
            d = self.d_list[m]
            a = self.a_list[m]
            b = self.b_list[m]
            e = self.e_list[m]
            f = self.f_list[m]

            p = tf.nn.embedding_lookup(P, tf.reshape(self.user_input, [-1]))
            q = tf.nn.embedding_lookup(Q, tf.reshape(self.item_input, [-1]))

            rel_predict = tf.sigmoid(tf.reduce_sum(p * q, 1, keepdims=True))
            rel_predict = tf.clip_by_value(rel_predict, clip_value_min=0.01, clip_value_max=0.99)

            w = tf.nn.sigmoid(tf.matmul(q, a) + b)
            pop = tf.pow(w * tf.nn.sigmoid(tf.matmul(q, c) + d) + (1 - w) * self.pop_input,
                         tf.nn.sigmoid(tf.matmul(q, e) + f))
            exp_predict = pop
            exp_predict = tf.clip_by_value(exp_predict, clip_value_min=0.01, clip_value_max=0.99)

            rel_cost = -tf.reduce_mean(self.label_input / self.exp_input * tf.log(rel_predict)
                                       + (1 - self.label_input / self.exp_input) * tf.log(1 - rel_predict))
            exp_cost = -tf.reduce_mean(self.label_input / self.rel_input * tf.log(exp_predict)
                                       + (1 - self.label_input / self.rel_input) * tf.log(1 - exp_predict))
            rel_reg_cost = self.reg * 0.5 * (self.l2_norm(P) + self.l2_norm(Q))
            exp_reg_cost = self.alpha * self.reg * 0.5 * (self.l2_norm(c) + self.l2_norm(d)
                                                          + self.l2_norm(a) + self.l2_norm(b)
                                                          + self.l2_norm(e) + self.l2_norm(f))

            rel_cost_final = rel_cost + rel_reg_cost 
            exp_cost_final = exp_cost + exp_reg_cost  
            self.rel_cost_list.append(rel_cost)
            self.exp_cost_list.append(exp_cost) 
            self.rel_reg_cost_list.append(rel_reg_cost)
            self.exp_reg_cost_list.append(exp_reg_cost) 
            self.rel_cost_final_list.append(rel_cost_final)
            self.exp_cost_final_list.append(exp_cost_final)

        self.rP_list = []
        self.rQ_list = []
        self.rc_list = []
        self.rd_list = []
        self.ra_list = []
        self.rb_list = []
        self.re_list = []
        self.rf_list = []
        self.rpara_rel_list = []
        self.rpara_exp_list = []
        for m in range(self.C):
            tf.set_random_seed(self.seed)
            with tf.variable_scope('rRelevance_' + str(m), reuse=tf.AUTO_REUSE):
                P = tf.get_variable(name='rP_' + str(m),
                                    initializer=tf.truncated_normal(shape=[self.num_user, self.hidden],
                                                                    mean=0, stddev=0.01,
                                                                    dtype=tf.float32), dtype=tf.float32)
                Q = tf.get_variable(name='rQ_' + str(m),
                                    initializer=tf.truncated_normal(shape=[self.num_item, self.hidden],
                                                                    mean=0, stddev=0.01,
                                                                    dtype=tf.float32), dtype=tf.float32)
            rpara_rel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='rRelevance_' + str(m))
            self.rP_list.append(P)
            self.rQ_list.append(Q)
            self.rpara_rel_list.append(rpara_rel)
            with tf.variable_scope('rExposure_' + str(m), reuse=tf.AUTO_REUSE):
                c = tf.get_variable(name='rc_' + str(m),
                                    initializer=tf.truncated_normal(shape=[self.hidden, 1],
                                                                    mean=0, stddev=0.01,
                                                                    dtype=tf.float32), dtype=tf.float32)
                d = tf.get_variable(name='rd_' + str(m),
                                    initializer=tf.truncated_normal(shape=[1, 1],
                                                                    mean=0, stddev=0.01,
                                                                    dtype=tf.float32), dtype=tf.float32)
                a = tf.get_variable(name='ra_' + str(m),
                                    initializer=tf.truncated_normal(shape=[self.hidden, 1],
                                                                    mean=0, stddev=0.01,
                                                                    dtype=tf.float32), dtype=tf.float32)
                b = tf.get_variable(name='rb_' + str(m),
                                    initializer=tf.truncated_normal(shape=[1, 1],
                                                                    mean=0, stddev=0.01,
                                                                    dtype=tf.float32), dtype=tf.float32)
                e = tf.get_variable(name='re_' + str(m),
                                    initializer=tf.truncated_normal(shape=[self.hidden, 1],
                                                                    mean=0, stddev=0.01,
                                                                    dtype=tf.float32), dtype=tf.float32)
                f = tf.get_variable(name='rf_' + str(m),
                                    initializer=tf.truncated_normal(shape=[1, 1],
                                                                    mean=0, stddev=0.01,
                                                                    dtype=tf.float32), dtype=tf.float32)
            rpara_exp = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='rExposure_' + str(m))
            self.rc_list.append(c)
            self.rd_list.append(d)
            self.ra_list.append(a)
            self.rb_list.append(b)
            self.re_list.append(e)
            self.rf_list.append(f)
            self.rpara_exp_list.append(rpara_exp)

        self.rrel_cost_list = []
        self.rexp_cost_list = []
        self.rrel_reg_cost_list = []
        self.rexp_reg_cost_list = []
        self.rrel_cost_final_list = []
        self.rexp_cost_final_list = []

        for m in range(self.C):
            rP = self.rP_list[m]
            rQ = self.rQ_list[m]
            rc = self.rc_list[m]
            rd = self.rd_list[m]
            ra = self.ra_list[m]
            rb = self.rb_list[m]
            re = self.re_list[m]
            rf = self.rf_list[m]

            P = self.P_list[m] + rP 
            Q = self.Q_list[m] + rQ
            c = self.c_list[m] + rc
            d = self.d_list[m] + rd
            a = self.a_list[m] + ra
            b = self.b_list[m] + rb
            e = self.e_list[m] + re
            f = self.f_list[m] + rf

            p = tf.nn.embedding_lookup(P, tf.reshape(self.user_input, [-1]))
            q = tf.nn.embedding_lookup(Q, tf.reshape(self.item_input, [-1]))

            rel_predict = tf.sigmoid(tf.reduce_sum(p * q, 1, keepdims=True))
            rel_predict = tf.clip_by_value(rel_predict, clip_value_min=0.01, clip_value_max=0.99)

            w = tf.nn.sigmoid(tf.matmul(q, a) + b)
            pop = tf.pow(w * tf.nn.sigmoid(tf.matmul(q, c) + d) + (1 - w) * self.pop_input,
                         tf.nn.sigmoid(tf.matmul(q, e) + f))
            exp_predict = pop
            exp_predict = tf.clip_by_value(exp_predict, clip_value_min=0.01, clip_value_max=0.99)

            rrel_cost = -tf.reduce_mean(self.label_input * tf.log(rel_predict * self.exp_input)
                                        + (1 - self.label_input) * tf.log(1 - rel_predict * self.exp_input))
            rexp_cost = -tf.reduce_mean(self.label_input * tf.log(self.rel_input * exp_predict)
                                        + (1 - self.label_input) * tf.log(1 - self.rel_input * exp_predict))

            rel_reg_cost = self.beta * self.reg * 0.5 * (self.l2_norm(rP) + self.l2_norm(rQ))
            exp_reg_cost = self.beta * self.alpha * self.reg * 0.5 * (self.l2_norm(rc) + self.l2_norm(rd)
                                                                      + self.l2_norm(ra) + self.l2_norm(rb)
                                                                      + self.l2_norm(re) + self.l2_norm(rf))

            rrel_cost_final = rrel_cost + rel_reg_cost
            rexp_cost_final = rexp_cost + exp_reg_cost

            self.rrel_cost_list.append(rrel_cost)
            self.rexp_cost_list.append(rexp_cost)
            self.rrel_reg_cost_list.append(rel_reg_cost)
            self.rexp_reg_cost_list.append(exp_reg_cost)
            self.rrel_cost_final_list.append(rrel_cost_final) 
            self.rexp_cost_final_list.append(rexp_cost_final) 

        self.rel_optimizer_list = []
        self.exp_optimizer_list = []
        self.rrel_optimizer_list = []
        self.rexp_optimizer_list = []
        with tf.variable_scope("Optimizer", reuse=tf.AUTO_REUSE):
            for m in range(self.C):
                rel_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.rel_cost_final_list[m],
                                                                                       var_list=self.para_rel_list[m])
                exp_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.exp_cost_final_list[m],
                                                                                       var_list=self.para_exp_list[m])
                rrel_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr / 13.).minimize(self.rrel_cost_final_list[m], 
                                                                                        var_list=self.rpara_rel_list[m])
                rexp_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr / 13.).minimize(self.rexp_cost_final_list[m],
                                                                                        var_list=self.rpara_exp_list[m])
                self.rel_optimizer_list.append(rel_optimizer)
                self.exp_optimizer_list.append(exp_optimizer)
                self.rrel_optimizer_list.append(rrel_optimizer)
                self.rexp_optimizer_list.append(rexp_optimizer)


    def generate_R_PR(self):
        self.Pr = np.zeros((self.num_user, self.num_item))
        self.R = np.zeros((self.num_user, self.num_item))
        rel_list = []
        exp_list = []
        for m in range(self.C):
            P, Q, c, d, a, b, e, f = self.sess.run([self.P_list[m], self.Q_list[m],
                                                    self.c_list[m], self.d_list[m],
                                                    self.a_list[m], self.b_list[m],
                                                    self.e_list[m], self.f_list[m]])

            rel = sigmoid(np.matmul(P, Q.T))

            w = sigmoid(np.matmul(Q, a) + b)
            
            pop = np.power(w * sigmoid(np.matmul(Q, c) + d) \
                           + (1 - w) * self.item_pop, sigmoid(np.matmul(Q, e) + f))
            exp = np.zeros((self.num_user, self.num_item)) + pop.T

            user_ids = self.df_list[m][0]
            item_ids = self.df_list[m][1]

            rel_list.append(rel[user_ids, item_ids])
            exp_list.append(exp[user_ids, item_ids])

            self.R += rel
            self.Pr += exp

        self.R /= self.C
        self.Pr /= self.C 
        for m in range(self.C):
            user_ids = self.df_list[m][0]
            item_ids = self.df_list[m][1]

            self.R[user_ids, item_ids] = rel_list[m] 
            self.Pr[user_ids, item_ids] = exp_list[m] 

        self.R[np.where(self.R < 0.01)] = 0.01
        self.R[np.where(self.R > 0.99)] = 0.99
        self.Pr[np.where(self.Pr < 0.01)] = 0.01 
        self.Pr[np.where(self.Pr > 0.99)] = 0.99


    def train_model(self, itr, m_list):
        np.random.RandomState(12345).shuffle(m_list)

        for m in tqdm(m_list, desc=f"Training sub models", position=0):
            self.generate_R_PR() 
            df = self.df_list[m]
################################################################################################################
            self.user_list, self.item_list, self.label_list = negative_sampling_MF(self.num_user,
                                                                                   self.num_item,
                                                                                   df[0], 
                                                                                   df[1],
                                                                                   self.neg_num)


            # self.user_list, self.item_list, self.label_list = df[0], df[1], df[2]

################################################################################################################
            pop_list = self.item_pop[self.item_list.reshape(-1)].reshape((-1, 1))

            num_batch = int(len(self.user_list) / float(self.batch_size)) + 1

            start_time = time.time() * 1000.0
            epoch_rel_cost = 0.
            epoch_exp_cost = 0.
            epoch_rel_reg_cost = 0.
            epoch_exp_reg_cost = 0.

            random_idx = np.random.RandomState(12345+itr).permutation(len(self.user_list)) 
            for i in range(num_batch):

                batch_idx = None
                if i == num_batch - 1:
                    batch_idx = random_idx[i * self.batch_size:]
                elif i < num_batch - 1:
                    batch_idx = random_idx[(i * self.batch_size):((i + 1) * self.batch_size)]

                user_input = self.user_list[batch_idx, :]
                item_input = self.item_list[batch_idx, :]
                label_input = self.label_list[batch_idx, :]
                pop_input = pop_list[batch_idx, :]

                rel_input = self.R[user_input.reshape(-1), item_input.reshape(-1)].reshape((-1, 1))
                exp_input = self.Pr[user_input.reshape(-1), item_input.reshape(-1)].reshape((-1, 1)) 

                for update_id in range(self.C):
                    if update_id != m: 
                        rel_para_list = [self.rel_optimizer_list[update_id], self.rel_cost_list[update_id],
                                         self.rel_reg_cost_list[update_id]]
                        exp_para_list = [self.exp_optimizer_list[update_id], self.exp_cost_list[update_id],
                                         self.exp_reg_cost_list[update_id]]


                        _, tmp_rel_cost, tmp_rel_reg_cost \
                            = self.sess.run(rel_para_list,
                                            feed_dict={self.user_input: user_input,
                                                       self.item_input: item_input,
                                                       self.label_input: label_input,
                                                       self.pop_input: pop_input,
                                                       self.exp_input: exp_input})
                        epoch_rel_cost += tmp_rel_cost
                        epoch_rel_reg_cost += tmp_rel_reg_cost

                        _, tmp_exp_cost, tmp_exp_reg_cost \
                            = self.sess.run(exp_para_list,
                                            feed_dict={self.user_input: user_input,
                                                       self.item_input: item_input,
                                                       self.label_input: label_input,
                                                       self.pop_input: pop_input,
                                                       self.rel_input: rel_input})
                        epoch_exp_cost += tmp_exp_cost
                        epoch_exp_reg_cost += tmp_exp_reg_cost


        for _ in range(1):
            self.generate_R_PR() 
################################################################################################################
            self.user_list, self.item_list, self.label_list = negative_sampling_MF(self.num_user,
                                                                                   self.num_item,
                                                                                   df[0], 
                                                                                   df[1],
                                                                                   self.neg_num)


            # self.user_list, self.item_list, self.label_list = df[0], df[1], df[2]

################################################################################################################
            pop_list = self.item_pop[self.item_list.reshape(-1)].reshape((-1, 1))
            num_batch = int(len(self.user_list) / float(1024)) + 1

            start_time = time.time() * 1000.0
            epoch_rrel_cost = 0.
            epoch_rexp_cost = 0.
            epoch_rrel_reg_cost = 0.
            epoch_rexp_reg_cost = 0.

            random_idx = np.random.RandomState(12345+itr).permutation(len(self.user_list))
            for i in range(num_batch):
                batch_idx = None
                if i == num_batch - 1:
                    batch_idx = random_idx[i * self.batch_size:]
                elif i < num_batch - 1:
                    batch_idx = random_idx[(i * self.batch_size):((i + 1) * self.batch_size)]

                user_input = self.user_list[batch_idx, :]
                item_input = self.item_list[batch_idx, :]
                label_input = self.label_list[batch_idx, :]
                pop_input = pop_list[batch_idx, :]
                rel_input = self.R[user_input.reshape(-1), item_input.reshape(-1)].reshape((-1, 1))
                exp_input = self.Pr[user_input.reshape(-1), item_input.reshape(-1)].reshape((-1, 1))

                for update_id in range(self.C):
                    _, tmp_rrel_cost, tmp_rrel_reg_cost \
                        = self.sess.run([self.rrel_optimizer_list[update_id], self.rrel_cost_list[update_id],
                                         self.rrel_reg_cost_list[update_id]],
                                        feed_dict={self.user_input: user_input,
                                                   self.item_input: item_input,
                                                   self.label_input: label_input,
                                                   self.pop_input: pop_input,
                                                   self.exp_input: exp_input})

                    epoch_rrel_cost += tmp_rrel_cost
                    epoch_rrel_reg_cost += tmp_rrel_reg_cost

                    _, tmp_rexp_cost, tmp_rexp_reg_cost \
                        = self.sess.run([self.rexp_optimizer_list[update_id], self.rexp_cost_list[update_id],  
                                         self.rexp_reg_cost_list[update_id]],
                                        feed_dict={self.user_input: user_input,
                                                   self.item_input: item_input,
                                                   self.label_input: label_input,
                                                   self.pop_input: pop_input,
                                                   self.rel_input: rel_input})
                    epoch_rexp_cost += tmp_rexp_cost
                    epoch_rexp_reg_cost += tmp_rexp_reg_cost


    def test_model(self, itr, best_score):  
        best_P_list = []
        best_Q_list = []

        start_time = time.time() * 1000.0

        R_all = np.zeros((self.num_user, self.num_item))
        for m in trange(self.C, desc="Making embeddings", position=0):
            P, Q = self.sess.run([self.P_list[m], self.Q_list[m]])
            rP, rQ = self.sess.run([self.rP_list[m], self.rQ_list[m]])
            P += rP
            Q += rQ
            R = sigmoid(np.matmul(P, Q.T)) ## relevacne 예측
            
            R_all += R ## 예측된 relevacne 모두 합함 (여러 모델 썻기 때문)

            best_P_list.append(P)
            best_Q_list.append(Q)

        R_predicted = R_all / self.C ## 여러 모델 relevacne 평균

        # validation
        at_k = 3
        val_ret = unbiased_evaluator(user_embed=best_P_list, item_embed=best_Q_list, 
                                train=self.train, val=self.val, test=self.val, num_users=self.num_user, num_items=self.num_item, 
                                pscore=self.item_freq, model_name='cjmf', at_k=[at_k], flag_test=False, flag_unbiased = True,
                                pred=R_predicted)


        max_score = val_ret.loc[f'MAP@{at_k}', f'cjmf_{self.hidden}']

        if best_score < max_score:
            print(f"best_val_MAP@{at_k}: ", max_score)
            self.best_P_list = best_P_list
            self.best_Q_list = best_Q_list

        print("Testing //", "Epoch %d //" % itr,
                "Accuracy Testing time : %d ms" % (time.time() * 1000.0 - start_time))
                
        return max_score

    @staticmethod
    def l2_norm(tensor):
        return tf.reduce_sum(tf.square(tensor))
