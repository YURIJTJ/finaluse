import tensorflow as tf
import string
import re
import numpy as np
from tensorflow.contrib import learn

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

A_examples = list(open("/home/yuri/article/Atestt/0.npy.txt", "r").readlines())
A_examples = [s.strip() for s in A_examples]

x_text = A_examples
x_text = [clean_str(sent) for sent in x_text]
print (type(x_text))

max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))
print (x.shape)

sequence_length = x.shape[1]
vocab_size=len(vocab_processor.vocabulary_)
filter_size = "3,4,5"
filter_sizes=list(map(int, filter_size.split(",")))
embedding_size = 128
num_filters = 128
dropout_keep_prob = 0.1

sess = tf.Session()
sess.run(tf.initialize_all_variables())

with tf.Session() as sess:
     sess.run(tf.global_variables_initializer())

     WE = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name="WE")
     init = tf.global_variables_initializer()
     sess.run(init)
     #print(sess.run(WE))
     embedded_chars = tf.nn.embedding_lookup(WE,x)
     embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
     pooled_outputs = []

     for i, filter_size in enumerate(filter_sizes):
          with tf.name_scope("conv-maxpool-%s" % filter_size):
             filter_shape = [filter_size, embedding_size, 1, num_filters]
             W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
             init = tf.global_variables_initializer()
             sess.run(init)
             #print(sess.run(W))
             b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
             init = tf.global_variables_initializer()
             sess.run(init)
             #print(sess.run(W))
             conv = tf.nn.conv2d(
                 embedded_chars_expanded,
                 W,
                 strides=[1, 1, 1, 1],
                 padding="VALID", 
                 name="conv")
                
             h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
               
             pooled = tf.nn.max_pool(
                 h,
                 ksize=[1, sequence_length - filter_size + 1, 1, 1],
                 strides=[1, 1, 1, 1],
                 padding='VALID',
                 name="pool")
             pooled_outputs.append(pooled)
        
     num_filters_total = num_filters * len(filter_sizes)
     h_pool = tf.concat(pooled_outputs,3)
     h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
     h_pool_flat_new = tf.reshape(h_pool_flat, [1, -1])
     print (type(h_pool_flat_new))
     print (h_pool_flat_new)
     h_pool_flat_new_numpy=h_pool_flat_new.eval()
     #print (type(h_pool_flat_new_numpy))
     #print (h_pool_flat_new_numpy)
     #a = list(h_pool_flat_new_numpy)
     #print (a)
     #q = open("/home/yuri/article/testt/Atxt.txt",'a')
     #q.write(str(a))
     #q.close
     #h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)
     #print type(h_drop)
     #print (h_drop.eval())
     #h_drop_numpy=h_drop.eval()
     #print type(h_drop_numpy)
     #print (h_drop_numpy)
