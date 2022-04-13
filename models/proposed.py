
import tensorflow as tf
import numpy as np
from scipy import sparse

from models.recommenders import AbstractRecommender
from evaluate.evaluator import aoa_evaluator, unbiased_evaluator

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tocsr(data: np.array, num_user: int, num_item: int) -> sparse.csr_matrix:
    """Convert data to csr_matrix."""
    matrix = sparse.lil_matrix((num_user, num_item))
    for (u, i, r) in data[:, :3]:
        matrix[u, i] = r
    return sparse.csr_matrix(matrix)

def csr_to_user_dict_i(train_matrix):
    """convert a scipy.sparse.csr_matrix to a dict,
    where the key is row number, and value is the
    non-empty index in each row.
    """
    train_matrix = train_matrix.transpose()
    train_dict = {}
    for idx, value in enumerate(train_matrix):
        train_dict[idx] = value.indices.copy().tolist()
    return train_dict

def csr_to_user_dict_u(train_matrix):
    """convert a scipy.sparse.csr_matrix to a dict,
    where the key is row number, and value is the
    non-empty index in each row.
    """
    train_dict = {}
    for idx, value in enumerate(train_matrix):
        train_dict[idx] = value.indices.copy().tolist()
    return train_dict


def l2_loss(*params):
    return tf.add_n([tf.nn.l2_loss(w) for w in params])


class Proposed(AbstractRecommender):
    def __init__(self, sess, data, train, val, test, num_users: np.array, num_items: np.array, \
                 hidden_dim: int, eta: float, reg: float, max_iters: int, batch_size: int, \
                 random_state: int, wu=0.1, wi=0.1) -> None:

        """Initialize Class."""
        self.data = data
        self.num_users = num_users
        self.num_items = num_items

        if data == 'coat':
            self.hidden_dim_u = 50
            self.hidden_dim_i = 50
            self.eta_u = 0.1
            self.eta_i = 0.2
            self.reg_u = 1e-6
            self.reg_i = 1e-7
            self.batch_size_u = 4
            self.batch_size_i = 1
        elif data == 'yahoo':
            self.hidden_dim_u = 200
            self.hidden_dim_i = 200
            self.eta_u = 0.01
            self.eta_i = 0.05
            self.reg_u = 0
            self.reg_i = 0
            self.batch_size_u = 1
            self.batch_size_i = 1

        self.best_weights_enc_u = None
        self.best_weights_dec_u = None
        self.best_bias_enc_u = None
        self.best_bias_dec_u = None

        self.best_weights_enc_i = None
        self.best_weights_dec_i = None
        self.best_bias_enc_i = None
        self.best_bias_dec_i = None

        self.num_epochs = max_iters
        self.train = train
        self.val = val
        self.test = test
        self.wu = wu
        self.wi = wi

        self.train_ui_matrix = tocsr(train, num_users, num_items).toarray()
        self.train_iu_matrix = np.copy( self.train_ui_matrix.T )

        self.sess = sess
        self.random_state = random_state

        self.model_name = 'proposed'

        # Build the graphs
        self.create_placeholders()
        self.build_graph()

        self.create_losses()
        self.add_optimizer()


    def create_placeholders(self):
        with tf.name_scope("input_data"):
            self.input_R_i = tf.placeholder(tf.float32, [None, self.num_users])
            self.pscore_i = tf.placeholder(tf.float32, [None, self.num_users])
            self.iAE_input_i = tf.placeholder(tf.float32, [None, self.num_users])
            self.w_i = tf.placeholder(tf.float32)

            self.input_R_u = tf.placeholder(tf.float32, [None, self.num_items])
            self.pscore_u = tf.placeholder(tf.float32, [None, self.num_items])
            self.iAE_input_u = tf.placeholder(tf.float32, [None, self.num_items])
            self.w_u = tf.placeholder(tf.float32)


    def build_graph(self):
        with tf.name_scope("embedding_layer_i"):  # The embedding initialization is unknown now
            initializer = tf.contrib.layers.xavier_initializer(seed=self.random_state)
            self.weights_i = {'encoder': tf.Variable(initializer([self.num_users, self.hidden_dim_i])),
                            'decoder': tf.Variable(initializer([self.hidden_dim_i, self.num_users]))}
            self.biases_i = {'encoder': tf.Variable(initializer([self.hidden_dim_i])),
                           'decoder': tf.Variable(initializer([self.num_users]))}

        with tf.name_scope("embedding_layer_u"):  # The embedding initialization is unknown now
            initializer = tf.contrib.layers.xavier_initializer(seed=self.random_state)
            self.weights_u = {'encoder': tf.Variable(initializer([self.num_items, self.hidden_dim_u])),
                            'decoder': tf.Variable(initializer([self.hidden_dim_u, self.num_items]))}
            self.biases_u = {'encoder': tf.Variable(initializer([self.hidden_dim_u])),
                           'decoder': tf.Variable(initializer([self.num_items]))}


        with tf.name_scope("prediction_i"):
            corrupted_input_i = self.input_R_i # no mask
            self.encoder_op_i = tf.sigmoid(tf.matmul(corrupted_input_i, self.weights_i['encoder']) +
                                                  self.biases_i['encoder'])
            
            self.decoder_op_i = tf.matmul(self.encoder_op_i, self.weights_i['decoder']) + self.biases_i['decoder']
            self.output_i = tf.sigmoid(self.decoder_op_i)

        with tf.name_scope("prediction_u"):
            corrupted_input_u = self.input_R_u # no mask
            self.encoder_op_u = tf.sigmoid(tf.matmul(corrupted_input_u, self.weights_u['encoder']) +
                                                  self.biases_u['encoder'])
            
            self.decoder_op_u = tf.matmul(self.encoder_op_u, self.weights_u['decoder']) + self.biases_u['decoder']
            self.output_u = tf.sigmoid(self.decoder_op_u)


    def create_losses(self):
        with tf.name_scope("loss"):
            eps = 0.00001

            # iAE update loss
            ppscore_i = tf.clip_by_value(self.output_i, clip_value_min=0.1, clip_value_max = 1 )
            self.loss_self_unbiased_i = tf.reduce_sum( self.input_R_i/ppscore_i * tf.square(1 - self.output_i) + (1 - self.input_R_i/ppscore_i) * tf.square(self.output_i) )
            self.loss_i_u_pos_rel_i = tf.reduce_sum( self.input_R_i * tf.square(self.iAE_input_i - self.output_i)  )
            
            self.reg_loss_i = self.reg_i*l2_loss(self.weights_i['encoder'], self.weights_i['decoder'],
                                             self.biases_i['encoder'], self.biases_i['decoder'])

            self.loss_i = self.reg_loss_i + self.loss_self_unbiased_i + self.w_i*self.loss_i_u_pos_rel_i


            # uAE update loss
            ppscore_u = tf.clip_by_value(self.output_u, clip_value_min=0.1, clip_value_max = 1 )
            self.loss_self_unbiased_u = tf.reduce_sum( self.input_R_u/ppscore_u * tf.square(1. - self.output_u) + (1 - self.input_R_u/ppscore_u) * tf.square(self.output_u) )
            self.loss_i_u_pos_rel_u = tf.reduce_sum( self.input_R_u * tf.square(self.iAE_input_u - self.output_u)  )

            self.reg_loss_u =  self.reg_u*l2_loss(self.weights_u['encoder'], self.weights_u['decoder'], 
                                             self.biases_u['encoder'], self.biases_u['decoder'])

            self.loss_u = self.reg_loss_u + self.loss_self_unbiased_u + self.w_u*self.loss_i_u_pos_rel_u  


    def add_optimizer(self):
        with tf.name_scope("optimizer"):
            self.apply_grads_i = tf.train.AdagradOptimizer(learning_rate=self.eta_i).minimize(self.loss_i)
            self.apply_grads_u = tf.train.AdagradOptimizer(learning_rate=self.eta_u).minimize(self.loss_u)

    def train_model(self, pscore, unbiased_eval):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        max_score = 0
        er_stop_count = 0
        er_stop_flag = False
        early_stop = 5

        all_tr_i = np.arange(self.num_items)
        all_tr_u = np.arange(self.num_users)

        for epoch in range(self.num_epochs):
            weights_enc_u, weights_dec_u, bias_enc_u, bias_dec_u = \
                    self.sess.run([self.weights_u['encoder'], self.weights_u['decoder'], self.biases_u['encoder'], self.biases_u['decoder']])

            uAE_ui_mat = sigmoid( np.matmul( sigmoid(np.matmul(self.train_ui_matrix, weights_enc_u) + bias_enc_u), weights_dec_u) + bias_dec_u)
            uAE_ui_mat = uAE_ui_mat.T
            
            weights_enc_i, weights_dec_i, bias_enc_i, bias_dec_i = \
                    self.sess.run([self.weights_i['encoder'], self.weights_i['decoder'], self.biases_i['encoder'], self.biases_i['decoder']])
            iAE_iu_mat = sigmoid( np.matmul( sigmoid(np.matmul(self.train_iu_matrix, weights_enc_i) + bias_enc_i), weights_dec_i) + bias_dec_i)
            iAE_iu_mat = iAE_iu_mat.T

            np.random.RandomState(12345).shuffle(all_tr_i)
            np.random.RandomState(12345).shuffle(all_tr_u)

            # iAE
            train_loss_i = 0
            batch_num = int(len(all_tr_i) / self.batch_size_i) +1
            for b in range(batch_num):
                batch_set_idx = all_tr_i[b*self.batch_size_i : (b+1)*self.batch_size_i]

                batch_matrix = np.zeros((len(batch_set_idx), self.num_users))
                uAE_bat_mat = np.zeros((len(batch_set_idx), self.num_users))
                for idx, item_id in enumerate(batch_set_idx):
                    batch_matrix[idx] = self.train_iu_matrix[item_id]
                    uAE_bat_mat[idx] = uAE_ui_mat[item_id]

                # pre-training only self bias
                feed_dict = {
                    self.input_R_i: batch_matrix,
                    self.iAE_input_i: uAE_bat_mat,
                    self.w_i: self.wi
                    }

                _, loss_i = self.sess.run([self.apply_grads_i, self.loss_i], feed_dict=feed_dict)
                train_loss_i += loss_i

            # uAE
            train_loss_u = 0
            batch_num = int(len(all_tr_u) / self.batch_size_u) +1
            for b in range(batch_num):
                batch_set_idx = all_tr_u[b*self.batch_size_u : (b+1)*self.batch_size_u]

                batch_matrix = np.zeros((len(batch_set_idx), self.num_items))
                iAE_bat_mat = np.zeros((len(batch_set_idx), self.num_items))

                for idx, user_id in enumerate(batch_set_idx):
                    batch_matrix[idx] = self.train_ui_matrix[user_id]
                    iAE_bat_mat[idx] = iAE_iu_mat[user_id]
    
                # pre-training only self bias
                feed_dict = {
                    self.input_R_u: batch_matrix,
                    self.iAE_input_u: iAE_bat_mat,
                    self.w_u: self.wu
                    }

                _, loss_u = self.sess.run([self.apply_grads_u, self.loss_u], feed_dict=feed_dict)

                train_loss_u += loss_u

            if epoch % 1 == 0 and not er_stop_flag:
                weights_enc_u, weights_dec_u, weights_enc_i, weights_dec_i, bias_enc_u, bias_dec_u, bias_enc_i, bias_dec_i = \
                    self.sess.run([self.weights_u['encoder'], self.weights_u['decoder'], self.weights_i['encoder'], self.weights_i['decoder'], \
                                self.biases_u['encoder'], self.biases_u['decoder'], self.biases_i['encoder'], self.biases_i['decoder']])

                # validation
                val_ret = unbiased_evaluator(user_embed=[weights_enc_u, weights_dec_u, weights_enc_i, weights_dec_i], item_embed=[bias_enc_u, bias_dec_u, bias_enc_i, bias_dec_i],
                                            train=self.train, val=self.val, test=self.val, num_users=self.num_users, num_items=self.num_items, 
                                            pscore=pscore, model_name=self.model_name, at_k=[3], flag_test=False, flag_unbiased=True)

                dim = self.hidden_dim_u

                if max_score < val_ret.loc['MAP@3', f'proposed_{dim}']:
                    max_score = val_ret.loc['MAP@3', f'proposed_{dim}']
                    print("best_val_MAP@3: ", max_score)
                    er_stop_count = 0 

                    self.best_weights_enc_u = weights_enc_u
                    self.best_weights_dec_u = weights_dec_u
                    self.best_bias_enc_u = bias_enc_u
                    self.best_bias_dec_u = bias_dec_u
                    self.best_weights_enc_i = weights_enc_i
                    self.best_weights_dec_i = weights_dec_i
                    self.best_bias_enc_i = bias_enc_i
                    self.best_bias_dec_i = bias_dec_i
                else:
                    er_stop_count += 1
                    if er_stop_count > early_stop:
                        print("stopped!")
                        er_stop_flag = True

            if er_stop_flag:
                break
            else:
                print(epoch," uAE loss: %f   iAE loss: %f"%(train_loss_u, train_loss_i))
                print("---------------------------------------------------------------------")

        self.sess.close()
