#This open source code is from a project in github https://github.com/dennybritz/cnn-text-classification-tf
#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("AA_data_file", "/tmp/guest-dzhXvA/sc/data/rt-polaritydata/rt-polarity (copy).neg", "Data source for the A data.")
tf.flags.DEFINE_string("BB_data_file", "/tmp/guest-dzhXvA/sc/data/rt-polaritydata/rt-polarity (copy).pos", "Data source for the B data.")
tf.flags.DEFINE_string("CC_data_file", "/tmp/guest-dzhXvA/sc/data/rt-polaritydata/rt-polarity (copy).neg", "Data source for the C data.")
tf.flags.DEFINE_string("DD_data_file", "/tmp/guest-dzhXvA/sc/data/rt-polaritydata/rt-polarity (copy).pos", "Data source for the D data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    y_test = np.argmax(y_test, axis=1)
else:
    x_raw = ["a masterpiece four years in the making", "everything is off."]
    y_test = [1, 0]

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        max = graph.get_operation_by_name("output/max").outputs[0]
        sum = graph.get_operation_by_name("output/sum").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            for i in list(batch_predictions):
                if i == 0:
                    batchpre = "computer vision"
                elif i == 1:
                    batchpre = "condensed matter"
                elif i == 2:
                    batchpre = "quantitative biology"
                elif i == 3:
                    batchpre = "pyhsics"
            newmax = sess.run(max, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            newsum = sess.run(sum, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            rate = newmax/newsum
            res = format(rate, '.0%')
            print("{} belongs to label {}".format(res,batchpre))

#all_predictions = np.concatenate([all_predictions, batch_predictions])
## Print accuracy if y_test is defined
#if y_test is not None:
    #correct_predictions = float(sum(all_predictions == y_test))
    #print("Total number of test examples: {}".format(len(y_test)))
    #print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

## Save the evaluation to a csv
#predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
#out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
#print("Saving evaluation to {0}".format(out_path))
#with open(out_path, 'w') as f:
    #csv.writer(f).writerows(predictions_human_readable)
