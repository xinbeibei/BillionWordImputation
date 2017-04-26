# Trains a RNN based Sequence to Sequence model
from __future__ import division
import numpy as np
import json
import os
import argparse
import tensorflow as tf
from tensorflow.contrib import legacy_seq2seq as seq2seq
from tensorflow.contrib import rnn

# Class to translate word vectors to words
import wvModel

parser = argparse.ArgumentParser()
parser.add_argument("sources", help="sources file name", nargs='?')
parser.add_argument("targets", help="targets file name", nargs='?')
parser.add_argument("--weights", help="weights to use for training")
parser.add_argument("--data_shape", help="Shape of the training data", type=int, nargs=3)
parser.add_argument("--batch_size", help="Batch size", type=int, default=128)
parser.add_argument("--print_loss_frequency", help="How often to print the model loss.", type=int, default=10)
parser.add_argument("--print_evaluation_frequency", help="How often to print an evaluation over a batch of sentences.", type=int, default=100)
parser.add_argument("--model_checkpoint_frequency", help="How often to save model checkpoints.", type=int, default=500)
parser.add_argument("--lstm_depth", help="How many LSTM layers to stack.", type=int, default=1)
parser.add_argument("--embedding_dim", help="Size of the word embeddings", type=int, default=100)
parser.add_argument("--predict", help="Run the model in prediction mode at the end of training on supplied data.", default=None)
parser.add_argument("--predict_data_shape", help="Shape of the data used for predictions.", type=int, nargs=3)
parser.add_argument("--predict_batch_size", help="Batch size to evaluate predictions - doens't ", type=int, default=128)
parser.add_argument("--fresh", help="Start training a new model from scratch, ignoring existing checkpoints.", action="store_true", default=False)
parser.add_argument("--training_iterations", help="Number of training iterations to perform", type=int, default=100000)
parser.add_argument("--max_predictions", help="Maximum number of batches to predict", type=int, default=99999)
args = parser.parse_args()

BATCH_SIZE = args.batch_size
LSTM_DEPTH = args.lstm_depth

class DataIterator:
    def __init__(self, dataX, dataY, W):
        self.dataX = dataX
        self.dataY = dataY
        self.W = W
        self.batch_size = BATCH_SIZE
        self.iter = self.make_random_iter()
        
    def next_batch(self):
        try:
            idxs = self.iter.next()
        except StopIteration:
            self.iter = self.make_random_iter()
            idxs = self.iter.next()
        X = np.array([self.dataX[i] for i in idxs])
        Y = np.array([self.dataY[i] for i in idxs])
        W = np.array([self.W[i] for i in idxs])
        return X, Y, W

    def make_random_iter(self):
        splits = np.arange(self.batch_size, self.dataX.shape[0], self.batch_size)
        it = np.split(np.random.permutation(range(self.dataX.shape[0])), splits)[:-1]
        return iter(it)

def get_feed(X, Y, W):
    feed_dict = {}
    for t in xrange(source_seq_length):
        feed_dict[source_inputs[t]] = X[:,t,:]
    for t in xrange(target_seq_length):
        feed_dict[target_labels[t]] = Y[:,t,:]
    feed_dict[weights] = W
    return feed_dict

def train_batch(data_iter):
    X, Y, W = data_iter.next_batch()
    feed_dict = get_feed(X, Y, W)
    _, out = sess.run([train_op, loss], feed_dict)
    return out

def eval_batch(data_iter):
    losses = np.zeros(BATCH_SIZE)
    src = ['']*BATCH_SIZE
    pre = ['']*BATCH_SIZE
    tru = ['']*BATCH_SIZE
    output, X, Y, W = get_eval_batch_data(data_iter)
    for i in range(len(output)):
        for j in range(BATCH_SIZE):
            word = wvmodel.closestWord(X[j][i])
            src[j] += ' ' + word
            if(word == '/MISSING_WORD'):
                losses[j] = wvmodel.distance(Y[j][i], output[i][j])
                pre[j] = wvmodel.closestWord(output[i][j])
                tru[j] = wvmodel.closestWord(Y[j][i])
    
    imax = losses.argmax()
    imin = losses.argmin()
    
    print("\tBest of {} sentences:".format(BATCH_SIZE))
    print(src[imin])
    print("\t{:15s}: {}".format("Prediction", pre[imin]))
    print("\t{:15s}: {}".format("Ground Truth", tru[imin]))
    print("\t{:15s}: {}".format("Distance", losses[imin]))
    print("\tWorst of {} sentences:".format(BATCH_SIZE))
    print(src[imax])
    print("\t{:15s}: {}".format("Prediction", pre[imax]))
    print("\t{:15s}: {}".format("Ground Truth", tru[imax]))
    print("\t{:15s}: {}".format("Distance", losses[imax]))

def get_eval_batch_data(data_iter):
    X, Y, W = data_iter.next_batch()
    feed_dict = get_feed(X, Y, W)
    decode_output = sess.run(target_outputs_train, feed_dict)
    return decode_output, X, Y, W

def main():
    """main(): Loads the data, constructs and trains the model."""
    global source_inputs, target_labels, sess, train_op, wvmodel, weights
    global loss, target_outputs_test, source_seq_length, target_seq_length, target_outputs_train
    
    if(args.sources and args.targets):
        source_data_file = args.sources
        target_data_file = args.targets
        weight_data_file = args.weights
    
        # Load the source input and target input data. These are saved as 
        # numpy .npy files, and are arrays of word vectors.
        source_data = np.memmap(source_data_file, mode='r', dtype=np.float32, shape=tuple(args.data_shape))
        target_data = np.memmap(target_data_file, mode='r', dtype=np.float32, shape=tuple(args.data_shape))
        weight_data = np.memmap(weight_data_file, mode='r', dtype=np.float32, shape=(args.data_shape[0], args.data_shape[1]))
        
        # Set up model parameters
        source_seq_length = source_data.shape[1]
        target_seq_length = target_data.shape[1]
        WORD_SIZE = source_data.shape[2]
    else:
        source_seq_length = 1
        target_seq_length = 1
        WORD_SIZE = args.embedding_dim+3
    
    wvmodel = wvModel.Model(embedding_size=args.embedding_dim)
    
    # Begin building graph
    sess = tf.Session()
    
    source_inputs = [tf.placeholder(tf.float32, shape=(BATCH_SIZE, WORD_SIZE)) for _ in range(source_seq_length)] # sequence of word vectors with one missing word
    target_labels = [tf.placeholder(tf.float32, shape=(BATCH_SIZE, WORD_SIZE)) for _ in range(target_seq_length)] # complete sequence of word vectors
    
    if(args.predict):
        predict_inputs = [tf.placeholder(tf.float32, shape=(None, args.predict_data_shape[2])) for _ in range(args.predict_data_shape[1])]
    
    cells = [rnn.BasicLSTMCell(WORD_SIZE) for _ in range(LSTM_DEPTH)]
    LSTM_cells = rnn.MultiRNNCell(cells)

    with tf.variable_scope("model") as scope:
        # Model to use for training
        target_outputs_train, target_states_train = seq2seq.basic_rnn_seq2seq(
            source_inputs, source_inputs, LSTM_cells)
        
        if(args.predict):
            scope.reuse_variables()
            # Model to use for testing (with shared weights)
            target_outputs_test, target_states_test = seq2seq.basic_rnn_seq2seq(
                predict_inputs, predict_inputs, LSTM_cells)
    
    # Training portion of the graph
    normed_target_labels = [tf.nn.l2_normalize(T, dim=1) for T in target_labels]
    normed_target_outputs_train = [tf.nn.l2_normalize(T, dim=1) for T in target_outputs_train]
    weights = tf.placeholder(tf.float32, shape=(BATCH_SIZE, source_seq_length))
    #loss = tf.losses.cosine_distance(tf.stack(normed_target_labels), tf.stack(normed_target_outputs_train), dim=2, weights=weights)
    cos_dist = 1 - tf.reduce_sum(tf.multiply(tf.stack(normed_target_labels), tf.stack(normed_target_outputs_train)), axis=2, keep_dims=True)
    loss = tf.reduce_sum(tf.multiply(cos_dist, weights))/tf.reduce_sum(weights)
    #loss = tf.reduce_mean(cos_dist)
    optimizer = tf.train.AdamOptimizer(1e-4)
    train_op = optimizer.minimize(loss)
    
    # Get a saver op
    saver = tf.train.Saver()
    if(os.access('RNN_model/checkpoint', os.R_OK) and not args.fresh):
        # Load the model from a checkpoint file
        saver.restore(sess, 'RNN_model/model.ckpt')
        print("\tLoading model from checkpoint file.")
    else:
        # Initialize from scratch
        sess.run(tf.global_variables_initializer())
        print("\tCreating model from scratch.")
    
    if(args.sources and args.targets):
        train_iter = DataIterator(source_data, target_data, weight_data)
        for i in xrange(args.training_iterations):
            try:
                train_loss = train_batch(train_iter)
                if i % args.print_loss_frequency == 0:
                    print "\tTraining on batch %i, train loss: %f" % (i, train_loss)
                if (i+1) % args.print_evaluation_frequency == 0:
                    eval_batch(train_iter)
                if (i+1) % args.model_checkpoint_frequency == 0:
                    saver.save(sess, "RNN_model/model.ckpt")
                    print("\tWrote checkpoint to directory RNN_model.")
            except KeyboardInterrupt:
                print "interrupted by user"
                break
    
    if(args.predict):
        predict_data = np.memmap(args.predict, mode='r', dtype=np.float32, shape=tuple(args.predict_data_shape))
        output = np.memmap('output.npy', dtype=np.float32, shape=(args.predict_data_shape[0], args.predict_data_shape[2]), mode='w+')
        i = 0
        batch_count = 0
        while(i < args.predict_data_shape[0] and batch_count < args.max_predictions):
            if(i < args.predict_data_shape[0] - args.batch_size):
                X = predict_data[i:i+args.predict_batch_size,:,:]
            else:
                X = predict_data[i:,:,:]
            feed_dict = {predict_inputs[t]: X[:,t,:] for t in range(args.predict_data_shape[1])}
            predict_batch = sess.run(target_outputs_test, feed_dict)
            for j in range(len(predict_batch)):
                for k in range(predict_batch[j].shape[0]):
                   if(X[k][j][0] == 1):
                       output[i+k] = predict_batch[j][k]
                       #print("{}: {}".format(wvmodel.closestWord(X[k][j]), wvmodel.closestWord(predict_batch[j][k])))
            i += args.predict_batch_size
            batch_count += 1
        del output
    return 0

if __name__ == '__main__':
    main()

