from __future__ import absolute_import
from __future__ import division

import functools
import math
import random

import numpy as np

from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq as seq2seq_lib
from tensorflow.contrib.rnn.python.ops import core_rnn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import rnn
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adam


cmu_dict_raw = open("/home/bxin/CSCI-544-Project/kernel_tests/cmudict-0.7b").read()

first_line = "A  AH0"
last_line = "ZYWICKI  Z IH0 W IH1 K IY0"

lines = cmu_dict_raw.split("\n")

for i, l in enumerate(lines):
    if l == first_line:
        first_index = i
    if l == last_line:
        last_index = i
        
print "Example line of file : "
print lines[113108]

phonemes = set()

for l in lines[first_index : last_index + 1]:
    word, pronounce = l.split("  ")
    for phoneme in pronounce.split():
        phonemes.add(phoneme)
        
sorted_phonemes = ["_"] + sorted(list(phonemes))

index_to_phoneme = dict(enumerate(sorted_phonemes))
phoneme_to_index = dict((v, k) for k,v in index_to_phoneme.items())

index_to_letter = dict(enumerate("_abcdefghijklmnopqrstuvwxyz"))
letter_to_index = dict((v, k) for k,v in index_to_letter.items())



from collections import defaultdict

pronounce_dict = {}

for l in lines[first_index : last_index + 1]:
    word, phone_list = l.split("  ")
    pronounce_dict[word.lower()] = [phoneme_to_index[p] for p in phone_list.split()]
    

max_k = max([len(k) for k,v in pronounce_dict.items()])
max_v = max([len(v) for k,v in pronounce_dict.items()])
for k,v in pronounce_dict.items():
    if len(k) == max_k or  len(v) == max_v:
        print k
        print v



bad_ct = set()

letters = set("abcdefghijklmnopqrstuvwxyz")
print len(pronounce_dict)
for k, v in pronounce_dict.items():
    if len(k) < 5 or len(k) > 15:
        del pronounce_dict[k]
        continue
    for c in k:
        if c not in letters:
            del pronounce_dict[k]
            break


pairs = np.random.permutation(list(pronounce_dict.keys()))

input_ = np.zeros((len(pairs), 16))
labels_ = np.zeros((len(pairs), 15))

for i, k in enumerate(pairs):
    v = pronounce_dict[k]
    k = k + "_" * (15 - len(k))
    v = v + [0] * (16 - len(v))
    for j, n in enumerate(v):
        input_[i][j] = n
    for j, letter in enumerate(k):
        labels_[i][j] = letter_to_index[letter]
        
input_ = input_.astype(np.int32)
labels_ = labels_.astype(np.int32)

input_test   = input_[:10000]
input_val    = input_[10000:20000]
input_train  = input_[20000:]
labels_test  = labels_[:10000]
labels_val   = labels_[10000:20000]
labels_train = labels_[20000:]

data_test  = zip(input_test, labels_test)
data_val   = zip(input_val, labels_val)
data_train = zip(input_train, labels_train)
import tensorflow as tf

ops.reset_default_graph()
try:
    sess.close()
except:
    
    pass
sess = tf.InteractiveSession()


input_seq_length = 16
output_seq_length = 15
batch_size = 128

input_vocab_size = 70
output_vocab_size = 28
embedding_dim = 256

encode_input = [tf.placeholder(tf.int32, 
                                shape=(None,),
                                name = "ei_%i" %i)
                                for i in range(input_seq_length)]

labels = [tf.placeholder(tf.int32,
                                shape=(None,),
                                name = "l_%i" %i)
                                for i in range(output_seq_length)]

decode_input = [tf.zeros_like(encode_input[0], dtype=np.int32, name="GO")] + labels[:-1]

keep_prob = tf.placeholder("float")
#%%

cell_fn = lambda: core_rnn_cell_impl.BasicLSTMCell(embedding_dim)
cell = cell_fn()
#decode_outputs, decode_state = seq2seq_lib.embedding_rnn_seq2seq(encode_input, decode_input, cell, num_encoder_symbols=input_vocab_size, num_decoder_symbols=output_vocab_size, embedding_size=embedding_dim)

with variable_scope.variable_scope("decoders"):
    decode_outputs, decode_state = seq2seq_lib.embedding_rnn_seq2seq(encode_input, decode_input, cell, num_encoder_symbols=input_vocab_size, num_decoder_symbols=output_vocab_size, embedding_size=embedding_dim)

cell_fn2 = lambda: core_rnn_cell_impl.BasicLSTMCell(embedding_dim)
cell2 = cell_fn2()
#    scope.reuse_variables()
with variable_scope.variable_scope("root"):
    decode_outputs_test, decode_state_test = seq2seq_lib.embedding_rnn_seq2seq(encode_input, decode_input, cell2, input_vocab_size, output_vocab_size, embedding_dim,feed_previous=True)
    
#%%    
loss_weights = [tf.ones_like(l, dtype=tf.float32) for l in labels]
loss = seq2seq_lib.sequence_loss(decode_outputs, labels, loss_weights, output_vocab_size)
optimizer = tf.train.AdamOptimizer(1e-4)
train_op = optimizer.minimize(loss)    

sess.run(tf.initialize_all_variables())

    #%%
class DataIterator:
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.iter = self.make_random_iter()
        
    def next_batch(self):
        try:
            idxs = self.iter.next()
        except StopIteration:
            self.iter = self.make_random_iter()
            idxs = self.iter.next()
        X, Y = zip(*[self.data[i] for i in idxs])
        X = np.array(X).T
        Y = np.array(Y).T
        return X, Y

    def make_random_iter(self):
        splits = np.arange(self.batch_size, len(self.data), self.batch_size)
        it = np.split(np.random.permutation(range(len(self.data))), splits)[:-1]
        return iter(it)
    
train_iter = DataIterator(data_train, 128)
val_iter = DataIterator(data_val, 128)
test_iter = DataIterator(data_test, 128)

#%%
import sys

def get_feed(X, Y):
    feed_dict = {encode_input[t]: X[t] for t in range(input_seq_length)}
    feed_dict.update({labels[t]: Y[t] for t in range(output_seq_length)})
    return feed_dict

def train_batch(data_iter):
    X, Y = data_iter.next_batch()
    feed_dict = get_feed(X, Y)
    feed_dict[keep_prob] = 0.5
    _, out = sess.run([train_op, loss], feed_dict)
    return out

def get_eval_batch_data(data_iter):
    X, Y = data_iter.next_batch()
    feed_dict = get_feed(X, Y)
    feed_dict[keep_prob] = 1.
    all_output = sess.run([loss] + decode_outputs_test, feed_dict)
    eval_loss = all_output[0]
    decode_output = np.array(all_output[1:]).transpose([1,0,2])
    return eval_loss, decode_output, X, Y

def eval_batch(data_iter, num_batches):
    losses = []
    predict_loss = []
    for i in range(num_batches):
        eval_loss, output, X, Y = get_eval_batch_data(data_iter)
        losses.append(eval_loss)
        
        for index in range(len(output)):
            real = Y.T[index]
            predict = np.argmax(output, axis = 2)[index]
            predict_loss.append(all(real==predict))
    return np.mean(losses), np.mean(predict_loss)

#%%
for i in range(100000):
    try:
        train_batch(train_iter)
        if i % 1000 == 0:
            val_loss, val_predict = eval_batch(val_iter, 16)
            train_loss, train_predict = eval_batch(train_iter, 16)
            print "val loss   : %f, val predict   = %.1f%%" %(val_loss, val_predict * 100)
            print "train loss : %f, train predict = %.1f%%" %(train_loss, train_predict * 100)
            print
            sys.stdout.flush()
    except KeyboardInterrupt:
        print "interrupted by user"
        break
    
eval_loss, output, X, Y = get_eval_batch_data(test_iter)

#%%


print "pronunciation".ljust(40),
print "real spelling".ljust(17),
print "model spelling".ljust(17),
print "is correct"
print

for index in range(len(output)):
    phonemes = "-".join([index_to_phoneme[p] for p in X.T[index]]) 
    real = [index_to_letter[l] for l in Y.T[index]] 
    predict = [index_to_letter[l] for l in np.argmax(output, axis = 2)[index]]
   
    print phonemes.split("-_")[0].ljust(40),
    print "".join(real).split("_")[0].ljust(17),
    print "".join(predict).split("_")[0].ljust(17),
    print str(real == predict)
    
#%%
cells = [core_rnn_cell_impl.DropoutWrapper(
        core_rnn_cell_impl.BasicLSTMCell(embedding_dim), output_keep_prob=keep_prob
    ) for i in range(3)]

        
#cells = [rnn_cell.DropoutWrapper(
#        rnn_cell.BasicLSTMCell(embedding_dim), output_keep_prob=keep_prob
#    ) for i in range(3)]

stacked_lstm = core_rnn_cell_impl.MultiRNNCell(cells)

with tf.variable_scope("decoders") as scope:
    decode_outputs, decode_state = seq2seq_lib.embedding_rnn_seq2seq(
        encode_input, decode_input, stacked_lstm, num_encoder_symbols=input_vocab_size, num_decoder_symbols=output_vocab_size, embedding_size=embedding_dim)
    
    scope.reuse_variables()
    
    decode_outputs_test, decode_state_test = seq2seq_lib.embedding_rnn_seq2seq(
        encode_input, decode_input, stacked_lstm, input_vocab_size, output_vocab_size, embedding_dim,
    feed_previous=True)
    