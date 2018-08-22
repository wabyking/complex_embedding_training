# -*- coding: utf-8 -*-

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Basic word2vec example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import sys
import argparse
import random
from tempfile import gettempdir
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import codecs
from tensorflow.python.framework import dtypes
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.ops import array_ops,candidate_sampling_ops,math_ops
# Give a folder path as an argument with '--log_dir' to save
# TensorBoard summaries. Default is a log folder in current directory.
current_path = os.path.dirname(os.path.realpath(sys.argv[0]))

parser = argparse.ArgumentParser()
parser.add_argument(
    '--log_dir',
    type=str,
    default=os.path.join(current_path, 'log'),
    help='The log directory for TensorBoard summaries.')
FLAGS, unparsed = parser.parse_known_args()

# Create the directory for TensorBoard variables if there is not.
if not os.path.exists(FLAGS.log_dir):
    os.makedirs(FLAGS.log_dir)


num_steps = 100001
batch_size = 128                  #!!!!!!!!!!! did not work now
embedding_size = 50  # Dimension of the embedding vector.
skip_window = 2  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.
num_sampled = 64  # Number of negative examples to sample.
num_true = 1   # waby mention, the size of the target word  
# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.
valid_size = 5  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

from dataHelper import DataHelper
helper = DataHelper("demo.txt")



  # Input data.
with tf.name_scope('inputs'):
    context = tf.placeholder(tf.int32, shape=[None,None],name ="context")
    target = tf.placeholder(tf.int32, shape=[None],name ="target")
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32,name ="valid_dataset")

# Look up embeddings for inputs.
with tf.name_scope('embeddings'):
    embeddings = tf.Variable(tf.random_uniform([helper.vocabulary_size, embedding_size], -1.0, 1.0))
    global_norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    global_normalized_embedding = embeddings / global_norm
embed = tf.nn.embedding_lookup(embeddings, context)
norm = tf.sqrt(tf.reduce_sum(tf.square(embed), 2, keepdims=False))
normalized_embed = embed / tf.tile(tf.expand_dims(norm,2),multiples=[1,1,embedding_size])
normalized_embed_ket = tf.expand_dims(normalized_embed,-1)
normalized_embed_bra = tf.transpose(normalized_embed_ket, perm = [0,1,3,2])

outer_product = tf.matmul(normalized_embed_ket,normalized_embed_bra)
softmax_norm = tf.nn.softmax(norm, 1)
expand_norm= tf.expand_dims(tf.expand_dims(softmax_norm,2),3)
density_matrix = tf.reduce_sum( outer_product * expand_norm,1)

with tf.name_scope('loss'):
    target_expand = tf.expand_dims(math_ops.cast(target, dtypes.int64),1)
    target_flat = array_ops.reshape(target_expand, [-1])
    sampled_values = candidate_sampling_ops.log_uniform_candidate_sampler(
            true_classes=target_expand,
            num_true=num_true,
            num_sampled=num_sampled,
            unique=True,
            range_max=helper.vocabulary_size)
    
        # NOTE: pylint cannot tell that 'sampled_values' is a sequence
        # pylint: disable=unpacking-non-sequence
    sampled, true_expected_count, sampled_expected_count = (
            array_ops.stop_gradient(s) for s in sampled_values)
    sampled = math_ops.cast(sampled, dtypes.int64)
    
    # all_ids = array_ops.concat([target, sampled], 0)
    
    
    sampled_embedding = tf.nn.embedding_lookup(embeddings, sampled)
    sampled_embedding_ket =  tf.expand_dims(sampled_embedding,2)
    sampled_embedding_bra =  tf.transpose(sampled_embedding_ket, perm = [0,2,1])
    sampled_embedding_outer_product = tf.matmul(sampled_embedding_ket,sampled_embedding_bra)
    density_matrix_reshape = tf.reshape( density_matrix, [-1,embedding_size *embedding_size ] )
    sampled_embedding_outer_product_reshape_transpose = tf.transpose( tf.reshape( sampled_embedding_outer_product, [-1,embedding_size *embedding_size ] ), perm=[1,0])
    negative_sample_prob = tf.matmul(density_matrix_reshape,sampled_embedding_outer_product_reshape_transpose)
    
    true_embedding = tf.nn.embedding_lookup(embeddings, target)
    true_embedding_ket =  tf.expand_dims(true_embedding,2)
    true_embedding_bra =  tf.transpose(true_embedding_ket, perm = [0,2,1])
    true_embedding_outer_product = tf.matmul(true_embedding_ket,true_embedding_bra)
    
    true_prob = tf.trace(tf.matmul(true_embedding_outer_product,density_matrix))
    true_prob_expand = tf.expand_dims(true_prob,1)
    
    out_labels = array_ops.concat([array_ops.ones_like(true_prob_expand) / num_true, array_ops.zeros_like(negative_sample_prob)], 1)
    out_logits = array_ops.concat([true_prob_expand,negative_sample_prob],1)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels =out_labels, logits=out_logits ))
      

  
  # Add the loss value as a scalar to summary.
tf.summary.scalar('loss', loss)

  # Construct the SGD optimizer using a learning rate of 1.0.
with tf.name_scope('optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.

valid_embeddings = tf.nn.embedding_lookup(global_normalized_embedding,
                                            valid_dataset)
similarity = tf.matmul(
      valid_embeddings, global_normalized_embedding, transpose_b=True)

  # Merge all summaries.
merged = tf.summary.merge_all()
#  tf.assign(x, tf.clip(x, 0, np.infty))
  # Add variable initializer.
init = tf.global_variables_initializer()

  # Create a saver.
saver = tf.train.Saver()

# Step 5: Begin training.


#with tf.Session(graph=graph) as session:
#if True:
  # Open a writer to write summaries.
session = tf.InteractiveSession()
session.run(init)
writer = tf.summary.FileWriter(FLAGS.log_dir, session.graph)
target_words = np.array([1001,1002,1003])
context_words = np.array([[1001,1,2],[1002,3,4],[1003,5,6]])

y=session.run( loss,feed_dict={target:target_words,context:context_words})
# We must initialize all variables before we use them.
init.run()
print('Initialized')

average_loss = 0
for step in xrange(5):
    for step_i,(batch_inputs, batch_labels) in enumerate( helper.generate_batch_cbow( batch_size,num_skips, skip_window)):
     
        feed_dict = {context: batch_inputs, target: batch_labels}
    
        # Define metadata variable.
  #        run_metadata = tf.RunMetadata()
    
        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        # Also, evaluate the merged op to get all summaries from the returned "summary" variable.
        # Feed metadata variable to session for visualizing the graph in TensorBoard.
        _, summary, loss_val = session.run(
            [optimizer, merged, loss],
            feed_dict=feed_dict)  #,            run_metadata=run_metadata
        average_loss += loss_val

        
  #        print (np.mean(norm_a))
        
  #        # Add returned summaries to writer in each step.
  #        writer.add_summary(summary, step)
  #        # Add metadata to visualize the graph for the last run.
  #        if step == (num_steps - 1):
  #          writer.add_run_metadata(run_metadata, 'step%d' % step)
  #    
        if step_i % 2000 == 0:
          if step_i > 0:
            average_loss /= 2000
          # The average loss is an estimate of the loss over the last 2000 batches.
          print('Average loss at step ', step_i, ': ', average_loss)
          average_loss = 0
    
        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step_i % 10000 == 0:
          sim = similarity.eval()
          for i in xrange(valid_size):
            valid_word = helper.reverse_dictionary[valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in xrange(top_k):
              close_word = helper.reverse_dictionary[nearest[k]]
              log_str = '%s %s,' % (log_str, close_word)
            print(log_str)
    saver.save(session, os.path.join(FLAGS.log_dir, 'model'+str(step)+'.ckpt'))
final_embeddings = global_normalized_embedding.eval()

# Write corresponding labels for the embeddings.
with open(FLAGS.log_dir + '/metadata.tsv', 'w') as f:
    for i in xrange(helper.vocabulary_size):
        f.write(helper.reverse_dictionary[i] + '\n')

# Save the model for checkpoints.
saver.save(session, os.path.join(FLAGS.log_dir, 'model.ckpt'))

# Create a configuration for visualizing embeddings with the labels in TensorBoard.
config = projector.ProjectorConfig()
embedding_conf = config.embeddings.add()
embedding_conf.tensor_name = embeddings.name
embedding_conf.metadata_path = os.path.join(FLAGS.log_dir, 'metadata.tsv')
projector.visualize_embeddings(writer, config)

writer.close()

# Step 6: Visualize the embeddings.


# pylint: disable=missing-docstring
# Function to draw visualization of distance between embeddings.
def plot_with_labels(low_dim_embs, labels, filename):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
      x, y = low_dim_embs[i, :]
      plt.scatter(x, y)
      plt.annotate(
          label,
          xy=(x, y),
          xytext=(5, 2),
          textcoords='offset points',
          ha='right',
          va='bottom')

  plt.savefig(filename)


try:
  # pylint: disable=g-import-not-at-top
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    tsne = TSNE(
        perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    plot_only = len(helper.reverse_dictionary) if len(helper.reverse_dictionary) < 400 else 400
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [helper.reverse_dictionary[i] for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, labels, os.path.join(gettempdir(), 'tsne.png'))

except ImportError as ex:
    print('Please install sklearn, matplotlib, and scipy to show embeddings.')
    print(ex)
