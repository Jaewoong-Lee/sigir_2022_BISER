from models.recommenders import AbstractRecommender
import tensorflow as tf
import numpy as np
from scipy import sparse

from evaluate.evaluator import aoa_evaluator, unbiased_evaluator


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

def l2_loss(*params):
    return tf.add_n([tf.nn.l2_loss(w) for w in params])

# sess, dataset, conf
class uAE(AbstractRecommender):
    def __init__(self, sess, data, train, val, test, num_user: np.array, num_item: np.array, \
                 hidden_dim: int, eta: float, reg: float, max_iters: int, batch_size: int, random_state: int) -> None:        
        """Initialize Class."""
        self.data = data
        self.num_users = num_user
        self.num_items = num_item
        self.hidden_dim = hidden_dim
        self.eta = eta
        self.reg = reg
        self.num_epochs = max_iters
        self.batch_size = batch_size
        self.train = train
        self.val = val
        self.test = test
        self.train_dict = csr_to_user_dict(tocsr(train, num_user, num_item))
        self.sess = sess

        self.model_name = 'uae'
        self.random_state = random_state

        # Build the graphs
        self.create_placeholders()
        self.build_graph()
        self.create_losses()
        self.add_optimizer()

        self.best_weights_enc = None
        self.best_weights_dec = None
        self.best_bias_enc = None
        self.best_bias_dec = None

    def create_placeholders(self):
        with tf.name_scope("input_data"):
            self.input_R = tf.placeholder(tf.float32, [None, self.num_items])

            self.tmp1 = tf.placeholder(tf.float32, [None, self.num_items])
            self.tmp2 = tf.placeholder(tf.float32, [None, self.num_items])
            self.tmp3 = tf.placeholder(tf.float32, [None, self.num_items])

    def build_graph(self):
        with tf.name_scope("embedding_layer"):  # The embedding initialization is unknown now
            initializer = tf.contrib.layers.xavier_initializer(seed=self.random_state)
             
            self.weights = {'encoder': tf.Variable(initializer([self.num_items, self.hidden_dim])),
                            'decoder': tf.Variable(initializer([self.hidden_dim, self.num_items]))}
            self.biases = {'encoder': tf.Variable(initializer([self.hidden_dim])),
                           'decoder': tf.Variable(initializer([self.num_items]))}

        with tf.name_scope("prediction"):
            input_R = self.input_R
            self.encoder_op = tf.sigmoid(tf.matmul(input_R, self.weights['encoder']) +
                                                  self.biases['encoder'])
            
            self.decoder_op = tf.matmul(self.encoder_op, self.weights['decoder']) + self.biases['decoder']
            self.output = tf.sigmoid(self.decoder_op)


    def create_losses(self):
        with tf.name_scope("loss"):
            self.loss = tf.reduce_sum( self.input_R * tf.square(1. - self.output) + (1 - self.input_R) * tf.square(self.output) )
            
            self.reg_loss = self.reg*l2_loss(self.weights['encoder'], self.weights['decoder'],
                                             self.biases['encoder'], self.biases['decoder'])

            self.loss = self.loss + self.reg_loss
                

    def add_optimizer(self):
        with tf.name_scope("optimizer"):
            self.apply_grads = tf.train.AdagradOptimizer(learning_rate=self.eta).minimize(self.loss)

    def train_model(self, pscore, unbiased_eval):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        max_score = 0
        er_stop_count = 0
        early_stop = 5

        all_tr = np.arange(self.num_users)
        for epoch in range(self.num_epochs):
            train_loss = 0

            np.random.RandomState(12345).shuffle(all_tr)

            batch_num = int(len(all_tr) / self.batch_size) +1
            for b in range(batch_num):
                batch_set_idx = all_tr[b*self.batch_size : (b+1)*self.batch_size]
                batch_matrix = np.zeros((len(batch_set_idx), self.num_items))
                for idx, user_id in enumerate(batch_set_idx):
                    users_by_user_id = self.train_dict[user_id]
                    batch_matrix[idx, users_by_user_id] = 1
    
                feed_dict = {self.input_R: batch_matrix}
               
                _, loss = self.sess.run([self.apply_grads, self.loss], feed_dict=feed_dict)

                train_loss += loss

            ############### evaluation
            if epoch % 1 == 0:
                print(epoch,":  ", train_loss)
                weights_enc, weights_dec, bias_enc, bias_dec = \
                    self.sess.run([self.weights['encoder'], self.weights['decoder'], self.biases['encoder'], self.biases['decoder']])

                # validation
                val_ret = unbiased_evaluator(user_embed=[weights_enc, weights_dec], item_embed=[bias_enc, bias_dec], 
                                        train=self.train, val=self.val, test=self.val, num_users=self.num_users, num_items=self.num_items, 
                                        pscore=pscore, model_name=self.model_name, at_k=[3], flag_test=False, flag_unbiased = True)

                dim = self.hidden_dim

                if max_score < val_ret.loc['MAP@3', f'{self.model_name}_{dim}']:
                    max_score = val_ret.loc['MAP@3', f'{self.model_name}_{dim}']
                    print("best_val_MAP@3: ", max_score)
                    er_stop_count = 0

                    self.best_weights_enc = weights_enc
                    self.best_weights_dec = weights_dec
                    self.best_bias_enc = bias_enc
                    self.best_bias_dec = bias_dec

                else:
                    er_stop_count += 1
                    if er_stop_count > early_stop:
                        print("stopped!")
                        break

        self.sess.close()
