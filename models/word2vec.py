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


import math
import numpy as np
import tensorflow as tf


from models.basicModel import BasicModel

class Word2vec(BasicModel):
    def __init__(self,config):
        print("start building")
        super(Word2vec, self).__init__(config)
        with tf.name_scope('inputs'):
            self.train_inputs = tf.placeholder(tf.int32, shape=[None])
            self.train_labels = tf.placeholder(tf.int32, shape=[None, 1])
            self.valid_examples = np.random.choice(self.config["valid_window"], self.config["valid_size"], replace=False)
            self.valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)
            
        with tf.name_scope('embeddings'):
            self.embeddings = tf.Variable(
                    tf.random_uniform([self.config["vocabulary_size"], self.config["embedding_size"]], -1.0, 1.0))
            # Construct the variables for the NCE loss
            self.norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keepdims=True))
            self.normalized_embeddings = self.embeddings / self.norm
        with tf.name_scope('weights'):
            self.nce_weights = tf.Variable(
                    tf.truncated_normal(
                            [self.config["vocabulary_size"], self.config["embedding_size"]],
                            stddev=1.0 / math.sqrt(self.config["embedding_size"])))
        with tf.name_scope('biases'):
            self.nce_biases = tf.Variable(tf.zeros([self.config["vocabulary_size"]]))
        
        self.build()
        print("build over with " + self.config["network_type"])
          
    def build(self):
        embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
        
        
        
        
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=self.nce_weights,
                    biases=self.nce_biases,
                    labels=self.train_labels,
                    inputs=embed,
                    num_sampled=self.config["num_sampled"],
                    num_classes=self.config["vocabulary_size"]))
        tf.summary.scalar('loss', self.loss)
        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(self.loss)

      # Compute the cosine similarity between minibatch examples and all embeddings.
        valid_embeddings = tf.nn.embedding_lookup(self.embeddings,self.valid_dataset)
        self.similarity = tf.matmul(valid_embeddings, self.embeddings, transpose_b=True)

         # Merge all summaries.
        self.merged = tf.summary.merge_all()
        self.normalization_op = tf.assign(self.embeddings,  self.normalized_embeddings) 
                
