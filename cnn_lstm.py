#This open source code is from a project in github https://github.com/dennybritz/cnn-text-classification-tf
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn.python.ops import rnn_cell
#from tensorflow.python.ops import rnn, rnn_cell

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,embedding_size, filter_sizes, num_filters,num_unitlstm,l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID", 
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])


        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
        
        def new_weights(shape):  
            return tf.Variable(tf.truncated_normal(shape, stddev=0.05))  
  
        def new_biases(length):  
            return tf.Variable(tf.constant(0.05, shape=[length]))  

        def flatten_layer(layer): 
            global flatten_layer 
            # Get the shape of the input layer.  
            layer_shape = layer.get_shape()  
  
            num_features = layer_shape[1:4].num_elements()  
  
            layer_flat = tf.reshape(layer, [-1, num_features])  
  
            return layer_flat, num_features  
  
        def new_fc_layer(input,          # The previous layer.  
                         num_inputs,     # Num. inputs from prev. layer.  
                         num_outputs,    # Num. outputs.  
                         use_relu=True): 
            global new_fc_layer# Use Rectified Linear Unit (ReLU)?  
            weights = new_weights(shape=[num_inputs,num_outputs])  
            biases = new_biases(length=num_outputs)  
            layer = tf.matmul(input, weights) + biases
              
  
            # Use ReLU?  
            if use_relu:  
                layer = tf.nn.relu(layer)  
  
            return layer  
        
        #LSTM
        layer_flat, num_features = flatten_layer(self.h_pool_flat)  
        layer_fc1 = new_fc_layer(input=layer_flat, num_inputs=num_features,num_outputs=num_features,use_relu=True)  
        layer_fc1_split=tf.split(1,384,layer_fc1)  
        with tf.variable_scope('lstm'):  
            lstm=tf.contrib.rnn.BasicLSTMCell(num_unitlstm,forget_bias=1.0)  
        with tf.variable_scope('RNN'):  
            output,state=rnn.static_rnn(lstm,layer_fc1_split,dtype=tf.float32)  
        self.rnn_output=tf.concat(output,1)
        layer_fc3 = new_fc_layer(input=self.rnn_output,num_inputs=num_unitlstm*384,num_outputs=num_classes,use_relu=False)    
        self.y_pred = tf.argmax(layer_fc3,1,name="predictions")
        self.max = tf.reduce_max(layer_fc3, name="max")
        self.sum = tf.reduce_sum(layer_fc3, name="sum")
        

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc3, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(tf.argmax(self.input_y, 1),self.y_pred)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

   
        
