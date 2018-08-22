# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 14:02:32 2018

@author: quartz
"""
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.ops import array_ops,candidate_sampling_ops,math_ops

import numpy as np
from dataHelper import DataHelper
helper = DataHelper("demo.txt")

batch_size = 128                  #!!!!!!!!!!! did not work now
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.
num_sampled = 64  # Number of negative examples to sample.
num_true = 1
# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.
valid_size = 5  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)


with tf.name_scope('inputs'):
    context = tf.placeholder(tf.int32, shape=[None,None])
    target = tf.placeholder(tf.int32, shape=[None])
#    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation

# Look up embeddings for inputs.
with tf.name_scope('embeddings'):
  embeddings = tf.Variable(
      tf.random_uniform([helper.vocabulary_size, embedding_size], -1.0, 1.0))
  embed = tf.nn.embedding_lookup(embeddings, context)
norm = tf.sqrt(tf.reduce_sum(tf.square(embed), 2, keepdims=False))
normalized_embed = embed / tf.tile(tf.expand_dims(norm,2),multiples=[1,1,128])
normalized_embed_ket = tf.expand_dims(normalized_embed,-1)
normalized_embed_bra = tf.transpose(normalized_embed_ket, perm = [0,1,3,2])

outer_product = tf.matmul(normalized_embed_ket,normalized_embed_bra)
softmax_norm = tf.nn.softmax(norm, 1)
expand_norm= tf.expand_dims(tf.expand_dims(softmax_norm,2),3)
density_matrix = tf.reduce_sum( outer_product * expand_norm,1)


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

all_ids = array_ops.concat([target_flat, sampled], 0)


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
loss = tf.nn.softmax_cross_entropy_with_logits(labels =out_labels, logits=out_logits )

session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

    if labels.dtype != dtypes.int64:
      labels = math_ops.cast(labels, dtypes.int64)
    labels_flat = array_ops.reshape(labels, [-1])

    # Sample the negative labels.
    #   sampled shape: [num_sampled] tensor
    #   true_expected_count shape = [batch_size, 1] tensor
    #   sampled_expected_count shape = [num_sampled] tensor
    if sampled_values is None:
      sampled_values = candidate_sampling_ops.log_uniform_candidate_sampler(
          true_classes=labels,
          num_true=num_true,
          num_sampled=num_sampled,
          unique=True,
          range_max=num_classes,
          seed=seed)
    # NOTE: pylint cannot tell that 'sampled_values' is a sequence
    # pylint: disable=unpacking-non-sequence
    sampled, true_expected_count, sampled_expected_count = (
        array_ops.stop_gradient(s) for s in sampled_values)
    # pylint: enable=unpacking-non-sequence
    sampled = math_ops.cast(sampled, dtypes.int64)

    # labels_flat is a [batch_size * num_true] tensor
    # sampled is a [num_sampled] int tensor
    all_ids = array_ops.concat([labels_flat, sampled], 0)

    # Retrieve the true weights and the logits of the sampled weights.

    # weights shape is [num_classes, dim]
    all_w = embedding_ops.embedding_lookup(
        weights, all_ids, partition_strategy=partition_strategy)

    # true_w shape is [batch_size * num_true, dim]
    true_w = array_ops.slice(all_w, [0, 0],
                             array_ops.stack(
                                 [array_ops.shape(labels_flat)[0], -1]))

    sampled_w = array_ops.slice(
        all_w, array_ops.stack([array_ops.shape(labels_flat)[0], 0]), [-1, -1])
    # inputs has shape [batch_size, dim]
    # sampled_w has shape [num_sampled, dim]
    # Apply X*W', which yields [batch_size, num_sampled]
    sampled_logits = math_ops.matmul(inputs, sampled_w, transpose_b=True)

    # Retrieve the true and sampled biases, compute the true logits, and
    # add the biases to the true and sampled logits.
    all_b = embedding_ops.embedding_lookup(
        biases, all_ids, partition_strategy=partition_strategy)
tf.nn.all_candidate_sampler()
print(session.run(xw))