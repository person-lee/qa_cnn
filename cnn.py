#coding=utf-8
"""
该类通过CNN计算每个句子的特征，然后通过计算问题和相似问题，以及不相似问题的间隔，用于解决qa，目标是使得间隔最大。
https://arxiv.org/abs/1508.01585
"""
import tensorflow as tf

class CNN(object):
    def __init__(self, sequence_len, word_embedding, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        """
        sequence_len: the max length of sentence
        word_embedding: word_embedding
        embedding_size: the dim of embedding
        filter_size: the size of filter, eg:[1,2,3,4,5]
        num_filters: the number of filter, how many filter contain in every layer, eg:128  
        l2_reg_lambda: l2_reg_lambda is use to limit overfit
        """
        # 定义输入变量
        self.org_quest = tf.placeholder(tf.int32, [None, sequence_len], name = "ori_quest")
        self.cand_quest = tf.placeholder(tf.int32, [None, sequence_len], name="cand_quest")
        self.neg_quest = tf.placeholder(tf.int32, [None, sequence_len], name="negative_quest")
        self.keep_dropout = tf.placeholder(tf.float32, name="dropout")

        # 定义词向量
        with tf.name_scope("embedding"):
            W = tf.Variable(tf.to_float(word_embedding), trainable=True, name="W")

            self.org_quest_embedding = tf.nn.embedding_lookup(W, self.org_quest)
            self.cand_quest_embedding = tf.nn.embedding_lookup(W, self.cand_quest)
            self.neg_quest_embedding = tf.nn.embedding_lookup(W, self.neg_quest)

            # 扩充词向量为卷积对应的格式，即(batch_size, sequence_len, embedding_size, in_channels)
            self.org_quest_embeddings = tf.expand_dims(self.org_quest_embedding, -1, name="ori")
            self.cand_quest_embeddings = tf.expand_dims(self.cand_quest_embedding, -1, name="cand")
            self.neg_quest_embeddings = tf.expand_dims(self.neg_quest_embedding, -1, name="neg")

        # 对输入变量进行卷积
        outputs_ori = []
        outputs_cand = []
        outputs_neg = []
        for idx, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s"%   filter_size):
                # filter shape is (weight, width, in_channels, out_channnels)
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                filter_weight = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="filter_weight")
                filter_bias = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="filter_bias")
                
                # convolution (batch_size, sequence_len - filter_size + 1, in_channels, out_channnels)
                conv_ori = tf.nn.conv2d(self.org_quest_embeddings, filter_weight, strides=[1,1,1,1], padding="VALID", name="conv_ori")

                # apply nonlinearity
                relu_output_ori = tf.nn.relu(tf.nn.bias_add(conv_ori, filter_bias), name="relu_ori")

                # maxpool (batch_size, 1, in_channels, out_channnels)
                output_ori = tf.nn.max_pool(relu_output_ori, ksize=[1, sequence_len - filter_size + 1, 1, 1] , strides=[1,1,1,1], padding="VALID", name="maxpool_ori")
                outputs_ori.append(output_ori)


                # convolution (batch_size, sequence_len - filter_size + 1, in_channels, out_channnels)
                conv_cand = tf.nn.conv2d(self.cand_quest_embeddings, filter_weight, strides=[1,1,1,1], padding="VALID", name="conv_cand")

                # apply nonlinearity
                relu_output_cand = tf.nn.relu(tf.nn.bias_add(conv_cand, filter_bias), name="relu_cand")

                # maxpool (batch_size, 1, in_channels, out_channnels)
                output_cand = tf.nn.max_pool(relu_output_cand, ksize=[1, sequence_len - filter_size + 1, 1, 1] , strides=[1,1,1,1], padding="VALID", name="maxpool_cand")
                outputs_cand.append(output_cand)
                
                
                # convolution (batch_size, sequence_len - filter_size + 1, in_channels, out_channnels)
                conv_neg = tf.nn.conv2d(self.neg_quest_embeddings, filter_weight, strides=[1,1,1,1], padding="VALID", name="conv_neg")

                # apply nonlinearity
                relu_output_neg = tf.nn.relu(tf.nn.bias_add(conv_neg, filter_bias), name="relu_neg")

                # maxpool (batch_size, 1, in_channels, out_channnels)
                output_neg = tf.nn.max_pool(relu_output_neg, ksize=[1, sequence_len - filter_size + 1, 1, 1] , strides=[1,1,1,1], padding="VALID", name="maxpool_neg")
                outputs_neg.append(output_neg)

        # concate outputs of all filter_sizes (batch_size, 1, 1, num_filters * len(filter_sizes))
        out_oris = tf.concat(3, outputs_ori, name="out_ori")
        out_cands = tf.concat(3, outputs_cand, name="out_cand")
        out_negs = tf.concat(3, outputs_neg, name="out_neg")

        # reshape out 
        out_ori = tf.squeeze(out_oris, squeeze_dims=[1,2]) 
        out_cand = tf.squeeze(out_cands, squeeze_dims=[1,2])
        out_neg = tf.squeeze(out_negs, squeeze_dims=[1,2])

        # dropout
        out_ori = tf.nn.dropout(out_ori, self.keep_dropout)
        out_cand = tf.nn.dropout(out_cand, self.keep_dropout)
        out_neg = tf.nn.dropout(out_neg, self.keep_dropout)

        # cal cosine simulation
        ori_seq_len = tf.sqrt(tf.reduce_sum(tf.mul(out_ori, out_ori), 1), name="sqrt_ori")
        cand_seq_len = tf.sqrt(tf.reduce_sum(tf.mul(out_cand, out_cand), 1), name="sqrt_cand")
        neg_seq_len = tf.sqrt(tf.reduce_sum(tf.mul(out_neg, out_neg), 1), name="sqrt_neg")

        ori_cand_dist = tf.reduce_sum(tf.mul(out_ori, out_cand), 1, name="ori_cand")
        ori_neg_dist = tf.reduce_sum(tf.mul(out_ori, out_neg), 1, name="ori_neg")

        # cal the score
        with tf.name_scope("score"):
            self.ori_cand_score = tf.div(ori_cand_dist, tf.mul(ori_seq_len, cand_seq_len), name="score_positive")
            self.ori_neg_score = tf.div(ori_neg_dist, tf.mul(ori_seq_len, neg_seq_len), name="score_negative")

        # the target function 
        zero = tf.fill(tf.shape(self.ori_cand_score), 0.0)
        margin = tf.fill(tf.shape(self.ori_cand_score), 0.1)
        l2_loss = tf.constant(0.0)
        with tf.name_scope("loss"):
            self.losses = tf.maximum(zero, tf.sub(margin, tf.sub(self.ori_cand_score, self.ori_neg_score)))
            self.loss = tf.reduce_sum(self.losses) + l2_reg_lambda * l2_loss
        
        # cal accurancy
        with tf.name_scope("acc"):
            self.correct = tf.equal(zero, self.losses)
            self.acc = tf.reduce_mean(tf.cast(self.correct, "float"), name="acc")
