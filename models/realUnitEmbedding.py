# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import math
import numpy as np
import tensorflow as tf


from models.basicModel import BasicModel
from tensorflow.python.ops import array_ops,candidate_sampling_ops,math_ops
from tensorflow.python.framework import dtypes

class RealUnitEmbedding(BasicModel):
    def __init__(self,config):
        print("start building")
        super(RealUnitEmbedding, self).__init__(config)
        with tf.name_scope('inputs'):
            self.context = tf.placeholder(tf.int32, shape=[None,None],name ="context")
            self.target = tf.placeholder(tf.int32, shape=[None],name ="target")
            self.valid_examples = np.random.choice(self.config["valid_window"], self.config["valid_size"], replace=False)
            self.valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)
            
            self.train_inputs= self.context
            self.train_labels= self.target
            
        with tf.name_scope('embeddings'):
            self.embeddings = tf.Variable(
                    tf.random_uniform([self.config["vocabulary_size"], self.config["embedding_size"]], -1.0, 1.0))
            # Construct the variables for the NCE loss
            self.global_norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keepdims=True))
            self.global_normalized_embedding = self.embeddings / self.global_norm
            
            self.embeddings = self.global_normalized_embedding

     
        self.build()
        print("build over with " + self.config["network_type"])
          
    def build(self):
        embed = tf.nn.embedding_lookup(self.embeddings, self.context)
        norm = tf.sqrt(tf.reduce_sum(tf.square(embed), 2, keepdims=False))
        normalized_embed = embed / tf.tile(tf.expand_dims(norm,2),multiples=[1,1,self.config["embedding_size"]])
        normalized_embed_ket = tf.expand_dims(normalized_embed,-1)
        normalized_embed_bra = tf.transpose(normalized_embed_ket, perm = [0,1,3,2])
        
        outer_product = tf.matmul(normalized_embed_ket,normalized_embed_bra)
        softmax_norm = tf.nn.softmax(norm, 1)
        expand_norm= tf.expand_dims(tf.expand_dims(softmax_norm,2),3)
        density_matrix = tf.reduce_sum( outer_product * expand_norm,1)     #[None, dim, dim]
        
        
        
        with tf.name_scope('loss'):
            target_expand = tf.expand_dims(math_ops.cast(self.target, dtypes.int64),1)
#            target_flat = array_ops.reshape(target_expand, [-1])
            sampled_values = candidate_sampling_ops.log_uniform_candidate_sampler(
                    true_classes=target_expand,
                    num_true=self.config["num_true"],
                    num_sampled=self.config["num_sampled"],
                    unique=True,
                    range_max=self.config["vocabulary_size"])
            
                # NOTE: pylint cannot tell that 'sampled_values' is a sequence
                # pylint: disable=unpacking-non-sequence
            sampled, true_expected_count, sampled_expected_count = (
                    array_ops.stop_gradient(s) for s in sampled_values)
            sampled = math_ops.cast(sampled, dtypes.int64)            #[64]
            
            # all_ids = array_ops.concat([target, sampled], 0)
            
            
            sampled_embedding = tf.nn.embedding_lookup(self.embeddings, sampled)
            sampled_embedding_ket =  tf.expand_dims(sampled_embedding,2)
            sampled_embedding_bra =  tf.transpose(sampled_embedding_ket, perm = [0,2,1])
            sampled_embedding_outer_product = tf.matmul(sampled_embedding_ket,sampled_embedding_bra)   #[num_sampled, dim, dim]
            density_matrix_reshape = tf.reshape( density_matrix, [-1,self.config["embedding_size"] *self.config["embedding_size"] ] )
            sampled_embedding_outer_product_reshape_transpose = tf.transpose( tf.reshape( sampled_embedding_outer_product, [-1,self.config["embedding_size"] *self.config["embedding_size"] ] ), perm=[1,0])
            negative_sample_prob = tf.matmul(density_matrix_reshape,sampled_embedding_outer_product_reshape_transpose)  #[None, 64]
            
            true_embedding = tf.nn.embedding_lookup(self.embeddings, self.target)
            true_embedding_ket =  tf.expand_dims(true_embedding,2)
            true_embedding_bra =  tf.transpose(true_embedding_ket, perm = [0,2,1])
            true_embedding_outer_product = tf.matmul(true_embedding_ket,true_embedding_bra)   #[None, dim, dim]
            
            true_prob = tf.trace(tf.matmul(true_embedding_outer_product,density_matrix))
            true_prob_expand = tf.expand_dims(true_prob,1)       #[None, 1]
            
            out_labels = array_ops.concat([array_ops.ones_like(true_prob_expand) / self.config["num_true"], array_ops.zeros_like(negative_sample_prob)], 1)
            out_logits = array_ops.concat([true_prob_expand,negative_sample_prob],1)   #[None, 1+num_sampled]
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels =out_labels, logits=out_logits ))
        tf.summary.scalar('loss', self.loss)
        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(self.loss)

      # Compute the cosine similarity between minibatch examples and all embeddings.
        valid_embeddings = tf.nn.embedding_lookup(self.global_normalized_embedding,self.valid_dataset)
        self.similarity = tf.matmul(valid_embeddings, self.global_normalized_embedding, transpose_b=True)

         # Merge all summaries.
        self.merged = tf.summary.merge_all()
#        self.normalization_op = tf.assign(self.embeddings,  self.global_normalized_embedding) 
        
                
