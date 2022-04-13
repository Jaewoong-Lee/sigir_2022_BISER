"""
Recommender models used for the semi-synthetic experiments
in the paper "Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback".
"""
from __future__ import absolute_import, print_function

from abc import ABCMeta, abstractmethod

import numpy as np
import tensorflow as tf


class AbstractRecommender(metaclass=ABCMeta):
    """Abstract base class for evaluator class."""

    @abstractmethod
    def create_placeholders(self) -> None:
        """Create the placeholders to be used."""
        raise NotImplementedError()

    @abstractmethod
    def build_graph(self) -> None:
        """Build the main tensorflow graph with embedding layers."""
        raise NotImplementedError()

    @abstractmethod
    def add_optimizer(self) -> None:
        """Add the required optimiser to the graph."""
        raise NotImplementedError()


class PointwiseRecommender(AbstractRecommender):
    """Implicit Recommenders based on pointwise approach."""

    def __init__(self, model_name:str, num_users: np.array, num_items: np.array,
                 dim: int, lam: float, eta: float, clip: float = 0) -> None:
        """Initialize Class."""
        self.num_users = num_users
        self.num_items = num_items
        self.dim = dim
        self.lam = lam
        self.eta = eta
        self.clip = clip

        # Build the graphs
        self.create_placeholders()
        self.build_graph()

        if model_name in ["mf"]:
            self.create_mf_losses()
        elif model_name in ["relmf"]:
            self.create_relmf_losses()

        self.add_optimizer()

    def create_placeholders(self) -> None:
        """Create the placeholders to be used."""
        self.users = tf.placeholder(tf.int32, [None], name='user_placeholder')
        self.items = tf.placeholder(tf.int32, [None], name='item_placeholder')
        self.scores = tf.placeholder(
            tf.float32, [None, 1], name='score_placeholder')
        self.labels = tf.placeholder(
            tf.float32, [None, 1], name='label_placeholder')
        
        self.tmp1 = tf.placeholder(
            tf.float32, [None, 1], name='tmp1')
        self.tmp2 = tf.placeholder(
            tf.float32, [None, 1], name='tmp2')
        self.tmp3 = tf.placeholder(
            tf.float32, [None, 1], name='tmp3')


    def build_graph(self) -> None:
        """Build the main tensorflow graph with embedding layers."""
        with tf.name_scope('embedding_layer'):
            # initialize user-item matrices and biases
            self.user_embeddings = tf.get_variable(
                f'user_embeddings', shape=[self.num_users, self.dim],
                initializer=tf.contrib.layers.xavier_initializer())
            self.item_embeddings = tf.get_variable(
                f'item_embeddings', shape=[self.num_items, self.dim],
                initializer=tf.contrib.layers.xavier_initializer())

            # lookup embeddings of current batch
            self.u_embed = tf.nn.embedding_lookup(
                self.user_embeddings, self.users)
            self.i_embed = tf.nn.embedding_lookup(
                self.item_embeddings, self.items)

        with tf.variable_scope('prediction'):
            self.logits = tf.reduce_sum(
                tf.multiply(self.u_embed, self.i_embed), 1)
            self.preds = tf.sigmoid(tf.expand_dims(
                self.logits, 1))

    def create_mf_losses(self) -> None:
        """Create the losses."""
        with tf.name_scope('mflosses'):
            self.weighted_mse = tf.reduce_sum(
                (self.labels) * tf.square(1. - self.preds) +
                (1 - self.labels) * tf.square(self.preds)  ) / \
                tf.reduce_sum(self.labels + (1 - self.labels))

            # add the L2-regularizer terms.
            reg_term_embeds = tf.nn.l2_loss(self.user_embeddings) \
                + tf.nn.l2_loss(self.item_embeddings)
            self.loss = self.weighted_mse + self.lam * reg_term_embeds

    def create_relmf_losses(self) -> None:
        """Create the losses."""
        with tf.name_scope('relmflosses'):
            scores = tf.clip_by_value(
                self.scores, clip_value_min=0.1, clip_value_max=1.0)
            self.weighted_mse = tf.reduce_sum(
                (self.labels / scores) * tf.square(1. - self.preds) +
                (1 - self.labels / scores) * tf.square(self.preds)) / \
                tf.reduce_sum(self.labels + (1 - self.labels))

            # add the L2-regularizer terms.
            reg_term_embeds = tf.nn.l2_loss(self.user_embeddings) \
                + tf.nn.l2_loss(self.item_embeddings)
            self.loss = self.weighted_mse + self.lam * reg_term_embeds


    def add_optimizer(self) -> None:
        """Add the required optimiser to the graph."""
        with tf.name_scope('optimizer'):
            # set Adam Optimizer.
            self.apply_grads = tf.train.AdamOptimizer(
                learning_rate=self.eta).minimize(self.loss)


# https://github.com/weitianxin/MACR/blob/b3b7bf1427d501c8d79ebf2c98c360ac90fdf445/macr_mf/model.py
class MACR:
    def __init__(self, model_name:str, num_users: np.array, num_items: np.array,
                 dim: int, lam: float, eta: float, batch_size: int, c: float, alpha: float, beta: float) -> None:

        self.n_users = num_users 
        self.n_items = num_items 
        self.decay = lam 
        self.emb_dim = dim 
        self.lr = eta

        self.batch_size = batch_size

        self.c = c
        self.alpha = alpha
        self.beta = beta

        #placeholders
        self.users = tf.placeholder(tf.int32, shape = (None,))
        self.pos_items = tf.placeholder(tf.int32, shape = (None,))
        self.neg_items = tf.placeholder(tf.int32, shape = (None,))

        #initiative weights
        self.weights = self.init_weights()

        #neting
        user_embedding = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        pos_item_embedding = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
        neg_item_embedding = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items)
        user_rand_embedding = tf.nn.embedding_lookup(self.weights['user_rand_embedding'], self.users)
        item_rand_embedding = tf.nn.embedding_lookup(self.weights['item_rand_embedding'], self.pos_items)
        

        self.const_embedding = self.weights['c']
        self.user_c = tf.nn.embedding_lookup(self.weights['user_c'], self.users)

        self.batch_ratings = tf.matmul(user_embedding, pos_item_embedding, transpose_a=False, transpose_b = True)    #prediction, shape(user_embedding) != shape(pos_item_embedding)
        self.user_const_ratings = self.batch_ratings - tf.matmul(self.const_embedding, pos_item_embedding, transpose_a=False, transpose_b = True)   #auto tile
        self.item_const_ratings = self.batch_ratings - tf.matmul(user_embedding, self.const_embedding, transpose_a=False, transpose_b = True)       #auto tile
        self.user_rand_ratings = self.batch_ratings - tf.matmul(user_rand_embedding, pos_item_embedding, transpose_a=False, transpose_b = True)
        self.item_rand_ratings = self.batch_ratings - tf.matmul(user_embedding, item_rand_embedding, transpose_a=False, transpose_b = True)


        self.mf_loss, self.reg_loss = self.create_bpr_loss(user_embedding, pos_item_embedding, neg_item_embedding)
        self.loss = self.mf_loss + self.reg_loss

        # self.opt = tf.train.RMSPropOptimizer(learning_rate = self.lr).minimize(self.loss)
        trainable_v1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'parameter')
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, var_list = trainable_v1)
        # two branch
        self.w = tf.Variable(self.initializer([self.emb_dim,1]), name = 'item_branch')
        self.w_user = tf.Variable(self.initializer([self.emb_dim,1]), name = 'user_branch')
        self.sigmoid_yu = tf.squeeze(tf.nn.sigmoid(tf.matmul(self.weights['user_embedding'], self.w_user)))
        self.sigmoid_yi = tf.squeeze(tf.nn.sigmoid(tf.matmul(self.weights['item_embedding'], self.w)))
        # two branch bpr
        self.mf_loss_two, self.reg_loss_two = self.create_bpr_loss_two_brach(user_embedding, pos_item_embedding, neg_item_embedding)
        self.loss_two = self.mf_loss_two + self.reg_loss_two
        self.opt_two = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_two)
        # two branch bce
        self.mf_loss_two_bce, self.reg_loss_two_bce = self.create_bce_loss_two_brach(user_embedding, pos_item_embedding, neg_item_embedding)
        self.loss_two_bce = self.mf_loss_two_bce + self.reg_loss_two_bce
        self.opt_two_bce = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_two_bce)
        # two branch bce user&item
        self.mf_loss_two_bce_both, self.reg_loss_two_bce_both = self.create_bce_loss_two_brach_both(user_embedding, pos_item_embedding, neg_item_embedding)
        self.loss_two_bce_both = self.mf_loss_two_bce_both + self.reg_loss_two_bce_both
        self.opt_two_bce_both = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_two_bce_both)
        # 2-stage training
        self.mf_loss2, self.reg_loss2 = self.create_bpr_loss2(user_embedding, self.const_embedding, pos_item_embedding, neg_item_embedding)
        self.loss2 = self.mf_loss2 + self.reg_loss2
        trainable_v2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'const_embedding')
        self.opt2 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss2, var_list = trainable_v2)


        self.mf_loss2_bce, self.reg_loss2_bce = self.create_bce_loss2(user_embedding, self.const_embedding, pos_item_embedding, neg_item_embedding)
        self.loss2_bce = self.mf_loss2_bce + self.reg_loss2_bce
        self.opt2_bce = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss2_bce, var_list = trainable_v2)
        # self.opt = tf.train.RMSPropOptimizer(learning_rate = self.lr).minimize(self.loss)
        
        self.opt3 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss2, var_list = [self.weights['user_embedding'],self.weights['item_embedding']])
        self.opt3_bce = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss2_bce, var_list = [self.weights['user_embedding'],self.weights['item_embedding']])

        self._statistics_params()

        self.mf_loss_bce, self.reg_loss_bce = self.create_bce_loss(user_embedding, pos_item_embedding, neg_item_embedding)
        self.loss_bce = self.mf_loss_bce + self.reg_loss_bce
        self.opt_bce = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_bce)

        # user wise two branch mf
        self.mf_loss_userc_bce, self.reg_loss_userc_bce = self.create_bce_loss_userc(user_embedding, pos_item_embedding, neg_item_embedding)
        self.loss_userc_bce = self.mf_loss_userc_bce + self.reg_loss_userc_bce
        self.opt_userc_bce = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_userc_bce, var_list = [self.weights['user_c']])
        # self.opt = tf.train.RMSPropOptimizer(learning_rate = self.lr).minimize(self.loss)


    def init_weights(self):
        weights = dict()
        self.initializer = tf.contrib.layers.xavier_initializer()
        initializer = self.initializer
        with tf.variable_scope('parameter'):
            weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name = 'user_embedding')
            weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name = 'item_embedding')
            weights['user_rand_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name = 'user_rand_embedding', trainable = False)
            weights['item_rand_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name = 'item_rand_embedding', trainable = False)
        with tf.variable_scope('const_embedding'):
            self.rubi_c = tf.Variable(tf.zeros([1]), name = 'rubi_c')
            weights['c'] = tf.Variable(tf.zeros([1, self.emb_dim]), name = 'c')
        
        weights['user_c'] = tf.Variable(tf.zeros([self.n_users, 1]), name = 'user_c_v')

        return weights

    def create_bpr_loss_two_brach(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)   #users, pos_items, neg_items have the same shape
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        # item stop


        # pos_items_stop = tf.stop_gradient(pos_items)
        # neg_items_stop = tf.stop_gradient(neg_items)
        pos_items_stop = pos_items
        neg_items_stop = neg_items

        self.pos_item_scores = tf.matmul(pos_items_stop,self.w)
        self.neg_item_scores = tf.matmul(neg_items_stop,self.w)
        # first branch
        pos_scores = pos_scores*tf.nn.sigmoid(self.pos_item_scores)
        neg_scores = neg_scores*tf.nn.sigmoid(self.neg_item_scores)
        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        self.rubi_ratings = (self.batch_ratings-self.rubi_c)*tf.squeeze(tf.nn.sigmoid(self.pos_item_scores))
        self.direct_minus_ratings = self.batch_ratings-self.rubi_c*tf.squeeze(tf.nn.sigmoid(self.pos_item_scores))
        
        self.mf_loss_ori_bce = tf.negative(tf.reduce_mean(maxi))
        # second branch
        maxi_item = tf.log(tf.nn.sigmoid(self.pos_item_scores - self.neg_item_scores))
        self.mf_loss_item_bce = tf.negative(tf.reduce_mean(maxi_item))
        # unify
        mf_loss = self.mf_loss_ori_bce + self.alpha*self.mf_loss_item_bce
        # regular
        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size

        
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss

    def create_bce_loss_two_brach(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)   #users, pos_items, neg_items have the same shape
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        # item score
        # pos_items_stop = tf.stop_gradient(pos_items)
        # neg_items_stop = tf.stop_gradient(neg_items)
        pos_items_stop = pos_items
        neg_items_stop = neg_items
        self.pos_item_scores = tf.matmul(pos_items_stop,self.w)
        self.neg_item_scores = tf.matmul(neg_items_stop,self.w)
        # self.rubi_ratings = (self.batch_ratings-self.rubi_c)*tf.squeeze(tf.nn.sigmoid(self.pos_item_scores))
        # self.direct_minus_ratings = self.batch_ratings-self.rubi_c*tf.squeeze(tf.nn.sigmoid(self.pos_item_scores))
        # first branch
        # fusion
        pos_scores = pos_scores*tf.nn.sigmoid(self.pos_item_scores)
        neg_scores = neg_scores*tf.nn.sigmoid(self.neg_item_scores)
        self.mf_loss_ori = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(pos_scores)+1e-10))+tf.negative(tf.log(1-tf.nn.sigmoid(neg_scores)+1e-10)))
        # second branch
        self.mf_loss_item = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(self.pos_item_scores)+1e-10))+tf.negative(tf.log(1-tf.nn.sigmoid(self.neg_item_scores)+1e-10)))
        # unify
        mf_loss = self.mf_loss_ori + self.alpha*self.mf_loss_item
        # regular
        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss

    def create_bce_loss_two_brach_both(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)   #users, pos_items, neg_items have the same shape
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        # item score
        # pos_items_stop = tf.stop_gradient(pos_items)
        # neg_items_stop = tf.stop_gradient(neg_items)
        pos_items_stop = pos_items
        neg_items_stop = neg_items
        users_stop = users
        self.pos_item_scores = tf.matmul(pos_items_stop,self.w)
        self.neg_item_scores = tf.matmul(neg_items_stop,self.w)
        self.user_scores = tf.matmul(users_stop, self.w_user)
        # self.rubi_ratings_both = (self.batch_ratings-self.rubi_c)*(tf.transpose(tf.nn.sigmoid(self.pos_item_scores))+tf.nn.sigmoid(self.user_scores))
        # self.direct_minus_ratings_both = self.batch_ratings-self.rubi_c*(tf.transpose(tf.nn.sigmoid(self.pos_item_scores))+tf.nn.sigmoid(self.user_scores))
        self.rubi_ratings_both = (self.batch_ratings-self.rubi_c)*tf.transpose(tf.nn.sigmoid(self.pos_item_scores))*tf.nn.sigmoid(self.user_scores)
        self.rubi_ratings_both_poptest = self.batch_ratings*tf.nn.sigmoid(self.user_scores)
        self.direct_minus_ratings_both = self.batch_ratings-self.rubi_c*tf.transpose(tf.nn.sigmoid(self.pos_item_scores))*tf.nn.sigmoid(self.user_scores)
        # first branch
        # fusion
        pos_scores = pos_scores*tf.nn.sigmoid(self.pos_item_scores)*tf.nn.sigmoid(self.user_scores)
        neg_scores = neg_scores*tf.nn.sigmoid(self.neg_item_scores)*tf.nn.sigmoid(self.user_scores)

        # pos_scores = pos_scores*(tf.nn.sigmoid(self.pos_item_scores)+tf.nn.sigmoid(self.user_scores))
        # neg_scores = neg_scores*(tf.nn.sigmoid(self.neg_item_scores)+tf.nn.sigmoid(self.user_scores))


        self.mf_loss_ori = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(pos_scores)+1e-10))+tf.negative(tf.log(1-tf.nn.sigmoid(neg_scores)+1e-10)))
        # second branch
        self.mf_loss_item = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(self.pos_item_scores)+1e-10))+tf.negative(tf.log(1-tf.nn.sigmoid(self.neg_item_scores)+1e-10)))
        # third branch
        self.mf_loss_user = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(self.user_scores)+1e-10))+tf.negative(tf.log(1-tf.nn.sigmoid(self.user_scores)+1e-10)))
        # unify
        mf_loss = self.mf_loss_ori + self.alpha*self.mf_loss_item + self.beta*self.mf_loss_user
        # regular
        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss
    
    def create_bce_loss_userc(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)   #users, pos_items, neg_items have the same shape
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        # item score
        # pos_items_stop = tf.stop_gradient(pos_items)
        # neg_items_stop = tf.stop_gradient(neg_items)
        pos_items_stop = pos_items
        neg_items_stop = neg_items
        self.pos_item_scores = tf.matmul(pos_items_stop,self.w)
        self.neg_item_scores = tf.matmul(neg_items_stop,self.w)
        self.rubi_ratings_userc = (self.batch_ratings-self.user_c)*tf.squeeze(tf.nn.sigmoid(self.pos_item_scores))
        self.direct_minus_ratings_userc = self.batch_ratings-self.user_c*tf.squeeze(tf.nn.sigmoid(self.pos_item_scores))
        # first branch
        # fusion
        pos_scores = (pos_scores-self.user_c)*tf.nn.sigmoid(self.pos_item_scores)
        neg_scores = (pos_scores-self.user_c)*tf.nn.sigmoid(self.neg_item_scores)
        self.mf_loss_ori = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(pos_scores)+1e-10))+tf.negative(tf.log(1-tf.nn.sigmoid(neg_scores)+1e-10)))
        # second branch
        self.mf_loss_item = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(self.pos_item_scores)+1e-10))+tf.negative(tf.log(1-tf.nn.sigmoid(self.neg_item_scores)+1e-10)))
        # unify
        mf_loss = self.mf_loss_ori #+ self.alpha*self.mf_loss_item
        # regular
        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss
        

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)   #users, pos_items, neg_items have the same shape
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size

        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))

        mf_loss = tf.negative(tf.reduce_mean(maxi))
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss

    def create_bce_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)   #users, pos_items, neg_items have the same shape
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        # first branch
        # fusion
        mf_loss = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(pos_scores)+1e-9))+tf.negative(tf.log(1-tf.nn.sigmoid(neg_scores)+1e-9)))
        # regular
        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss

    def create_bpr_loss2(self, users, const_embedding, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1) - tf.matmul(const_embedding, pos_items, transpose_a=False, transpose_b = True)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1) - tf.matmul(const_embedding, neg_items, transpose_a=False, transpose_b = True)

        regularizer = tf.nn.l2_loss(const_embedding)
        regularizer = regularizer/self.batch_size

        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))

        mf_loss = tf.negative(tf.reduce_mean(maxi))
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss
    
    def create_bce_loss2(self, users, const_embedding, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1) - tf.matmul(const_embedding, pos_items, transpose_a=False, transpose_b = True)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1) - tf.matmul(const_embedding, neg_items, transpose_a=False, transpose_b = True)

        regularizer = tf.nn.l2_loss(const_embedding)
        regularizer = regularizer/self.batch_size

        mf_loss = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(pos_scores)+1e-9))+tf.negative(tf.log(1-tf.nn.sigmoid(neg_scores)+1e-9)))
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss

    def update_c(self, sess, c):
        sess.run(tf.assign(self.rubi_c, c*tf.ones([1])))

    def _statistics_params(self):
        # number of params
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()  # shape is an array of tf.Dimension
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print("#params: %d" % total_parameters)
