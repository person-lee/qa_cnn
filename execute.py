# coding=utf-8

import logging
import datetime
import time
import operator
import tensorflow as tf


from data_helper import load_train_data, load_test_data, load_embedding, create_valid, batch_iter
from cnn import CNN


#------------------------- define parameter -----------------------------
tf.flags.DEFINE_string("train_file", "../insuranceQA-cnn-lstm/insuranceQA/train", "train corpus file")
tf.flags.DEFINE_string("test_file", "../insuranceQA-cnn-lstm/insuranceQA/test1", "test corpus file")
tf.flags.DEFINE_string("embedding_file", "../insuranceQA-cnn-lstm/insuranceQA/vectors.nobin", "embedding file")
tf.flags.DEFINE_string("filter_sizes", "1,2,3,5", "filter size of cnn")
tf.flags.DEFINE_integer("embedding_size", 100, "embedding size")
tf.flags.DEFINE_integer("sequence_len", 100, "embedding size")
tf.flags.DEFINE_integer("num_filters", 500, "the number of filter in every layer")
tf.flags.DEFINE_float("dropout", 1, "the proportion of dropout")
tf.flags.DEFINE_integer("batch_size", 256, "batch size of each batch")
tf.flags.DEFINE_integer("epoches", 300, "epoches")
tf.flags.DEFINE_integer("evaluate_every", 3000, "run evaluation")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", True, "Log placement of ops on devices")
tf.flags.DEFINE_float("gpu_options", 0.9, "use memory rate")

FLAGS = tf.flags.FLAGS
#----------------------------- define parameter end ----------------------------------

#----------------------------- define a logger -------------------------------
logger = logging.getLogger("execute")
logger.setLevel(logging.INFO)

fh = logging.FileHandler("./run.log")
fh.setLevel(logging.INFO)

fmt = "%(asctime)-15s %(levelname)s %(filename)s %(lineno)d %(process)d %(message)s"
datefmt = "%a %d %b %Y %H:%M:%S"
formatter = logging.Formatter(fmt, datefmt)

fh.setFormatter(formatter)
logger.addHandler(fh)

#----------------------------- define a logger end ----------------------------------

#------------------------------------load data -------------------------------
embedding, word2idx, idx2word = load_embedding(FLAGS.embedding_file, FLAGS.embedding_size)
ori_quests, cand_quests = load_train_data(FLAGS.train_file, word2idx, FLAGS.sequence_len)
#train_quests, valid_quests = create_valid(zip(ori_quests, cand_quests))

test_ori_quests, test_cand_quests, labels, results = load_test_data(FLAGS.test_file, word2idx, FLAGS.sequence_len)
#----------------------------------- load data end ----------------------

#----------------------------------- build model --------------------------------------
filter_sizes = [int(filter_size.strip()) for filter_size in FLAGS.filter_sizes.strip().split(",")]
#----------------------------------- build model end ----------------------------------

#----------------------------------- execute train model ---------------------------------
def run_step(sess, ori_batch, cand_batch, neg_batch, cnn, dropout, is_optimizer=True):
    start_time = time.time()
    feed_dict = {
        cnn.org_quest:ori_batch,
        cnn.cand_quest:cand_batch, 
        cnn.neg_quest:neg_batch,
        cnn.keep_dropout:dropout
    }

    if is_optimizer:
        _, step, loss, acc, ori_cand_score, ori_neg_score = sess.run([train_op, global_step, cnn.loss, cnn.acc, cnn.ori_cand_score, cnn.ori_neg_score], feed_dict)
    else:
        step, loss, acc, ori_cand_score, ori_neg_score = sess.run([global_step, cnn.loss, cnn.acc, cnn.ori_cand_score, cnn.ori_neg_score], feed_dict)


    right, wrong, score = [0.0] * 3
    for i in range(0 ,len(ori_batch)):
        if ori_cand_score[i] > 0.55:
            right += 1.0
        else:
            wrong += 1.0
        score += ori_cand_score[i] - ori_neg_score[i]
    time_elapsed = time.time() - start_time
    logger.info("step %s, loss %s, acc %s, wrong %s, score %s, %6.7f secs/batch"%(step, loss, acc, wrong, score, time_elapsed))
    return loss, ori_cand_score
#---------------------------------- execute train model end --------------------------------------

def cal_acc(labels, results, total_ori_cand):
    if len(labels) == len(results) == len(total_ori_cand):
        retdict = {}
        for label, result, ori_cand in zip(labels, results, total_ori_cand):
            if result not in retdict:
                retdict[result] = []
            retdict[result].append((ori_cand, label))
        
        correct = 0
        for key, value in retdict.items():
            value.sort(key=operator.itemgetter(0), reverse=True)
            score, flag = value[0]
            if flag == 1:
                correct += 1
        return 1. * correct/len(retdict)
    else:
        logger.info("data error")
        return 0

#---------------------------------- execute valid model ------------------------------------------
def valid_model(sess, cnn, valid_ori_quests, valid_cand_quests, labels, results):
    total_loss, idx = 0, 0
    total_ori_cand = []
    #total_right, total_wrong, step = 0, 0, 0, 0
    for ori_valid, cand_valid, neg_valid in batch_iter(valid_ori_quests, valid_cand_quests, FLAGS.batch_size, 1, is_valid=True):
        loss, ori_cand = run_step(sess, ori_valid, cand_valid, cand_valid, cnn, FLAGS.dropout, False)
        total_loss += loss
        total_ori_cand.extend(ori_cand)
        #total_right += right
        #total_wrong += wrong
        idx += 1

    acc = cal_acc(labels, results, total_ori_cand)
    logger.info("evaluation acc:%s"%(acc))
    #logger.info("%s, evaluation loss:%s, acc:%s"%(timestr, total_loss/step, total_right/(total_right + total_wrong)))
#---------------------------------- execute valid model end --------------------------------------

#----------------------------------- begin to train -----------------------------------
with tf.Graph().as_default():
    with tf.device("/gpu:3"):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_options)
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, log_device_placement=FLAGS.log_device_placement, gpu_options=gpu_options)
        with tf.Session(config=session_conf).as_default() as sess:
            cnn = CNN(FLAGS.sequence_len, embedding, FLAGS.embedding_size, filter_sizes, FLAGS.num_filters)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            #optimizer = tf.train.AdamOptimizer(5e-2)
            optimizer = tf.train.GradientDescentOptimizer(1e-1)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            sess.run(tf.initialize_all_variables())

            #ori_quests, cand_quests = zip(*train_quests)
            #valid_ori_quests, valid_cand_quests = zip(*valid_quests)

            for ori_train, cand_train, neg_train in batch_iter(ori_quests, cand_quests, FLAGS.batch_size, FLAGS.epoches):
                run_step(sess, ori_train, cand_train, neg_train, cnn, FLAGS.dropout)
                cur_step = tf.train.global_step(sess, global_step)
                
                if cur_step % FLAGS.evaluate_every == 0 and cur_step != 0:
                    logger.info("start to evaluation model")
                    #valid_model(sess, cnn, valid_ori_quests, valid_cand_quests)
                    valid_model(sess, cnn, test_ori_quests, test_cand_quests, labels, results)
            valid_model(sess, cnn, test_ori_quests, test_cand_quests, labels, results)
            #---------------------------------- end train -----------------------------------
