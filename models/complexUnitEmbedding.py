# -*- coding: utf-8 -*-

import math
import numpy as np
import tensorflow as tf


from models.basicModel import BasicModel
from tensorflow.python.ops import array_ops,candidate_sampling_ops,math_ops
from tensorflow.python.framework import dtypes

class ComplexUnitEmbedding(BasicModel):
    def __init__(self,config):
        print("start building")
        super(ComplexUnitEmbedding, self).__init__(config)
        with tf.name_scope('inputs'):
            self.context = tf.placeholder(tf.int32, shape=[None,None],name ="context")
            self.target = tf.placeholder(tf.int32, shape=[None],name ="target")
            self.valid_examples = np.random.choice(self.config["valid_window"], self.config["valid_size"], replace=False)
            self.valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)
            
            self.train_inputs= self.context
            self.train_labels= self.target
            
        with tf.name_scope('embeddings'):
            self.phase_embedding = tf.Variable(
                    tf.random_uniform([self.config["vocabulary_size"], self.config["embedding_size"]], -1.0, 1.0))
            self.amplitude_embedding = tf.Variable(
                    tf.random_uniform([self.config["vocabulary_size"], self.config["embedding_size"]], -1.0, 1.0))
            # Construct the variables for the NCE loss
            self.global_norm = tf.sqrt(tf.reduce_sum(tf.square(self.amplitude_embedding), 1, keepdims=True))
            self.global_normalized_embedding = self.amplitude_embedding / self.global_norm
     
        self.build()
        print("build over with " + self.config["network_type"])
    
    def getEmbedding(self,words):
        phase_part = tf.nn.embedding_lookup(self.phase_embedding, words)
        amplitude_part = tf.nn.embedding_lookup(self.amplitude_embedding, words)
        return amplitude_part,phase_part
    
    def transform(self,amplitude,phase):
        cos = tf.cos(phase)
        sin = tf.sin(phase)
        return cos*amplitude, sin*amplitude      
    
    def outer_product(self,real,imag,weights=None):     #  batch_size * embedding_dim  
        real_ket, real_bra = tf.expand_dims(real,axis=-1),tf.expand_dims(real,axis=-2)
        imag_ket, imag_bra = tf.expand_dims(imag,axis=-1),tf.expand_dims(imag,axis=-2)
        real_part = tf.matmul(real_bra,real_ket) + tf.matmul(imag_bra,imag_ket)
        imag_part = tf.matmul(real_bra,imag_ket) - tf.matmul(imag_bra,real_ket)         
        return real_part,imag_part  
    
    def getNorm(self,real,imag):
        return  tf.sqrt( tf.square(real)+ tf.square(imag))
    
    def inner_product(self,num1,num2):  # equals tr(\rho * m) = inner_product(vector_\rho ,vector_m)
        real1,imag1 = num1
        real2,imag2 = num2
        dimension = real1.shape[-1]
        real1_reshape = tf.reshape( real1, [-1,dimension *dimension] )
        imag1_reshape = tf.reshape( imag1, [-1,dimension *dimension] )
        real2_reshape_tranpose = tf.transpose( tf.reshape( real2, [-1,dimension *dimension] ), perm=[1,0])
        imag2_reshape_tranpose = tf.transpose( tf.reshape( imag2, [-1,dimension *dimension] ), perm=[1,0])        
        
        real_part = tf.matmul(real1_reshape,real2_reshape_tranpose) - tf.matmul(imag1_reshape,imag2_reshape_tranpose)
        imag_part = tf.matmul(real1_reshape,imag2_reshape_tranpose) + tf.matmul(imag1_reshape,real2_reshape_tranpose)
        return self.getNorm(real_part,imag_part)

        
        
    def build(self,weights = None):

        phase_part,amplitude_part = self.getEmbedding(self.context)
        real,imag = self.transform(amplitude_part,phase_part)
        
        norm = tf.sqrt(tf.reduce_sum(tf.square(amplitude_part), 2, keepdims=False))
        softmax_norm = tf.nn.softmax(norm, 1)
        density_real,density_imag = self.outer_product(real,imag,softmax_norm)
        
        if weights == None:
            density_real_mixture, density_imag_mixture = tf.reduce_mean(density_real,axis=1), tf.reduce_mean(density_imag,axis=1)
        else:
            expand_norm = tf.expand_dims(tf.expand_dims(softmax_norm,2),3)
            density_real_mixture, density_imag_mixture =[ tf.reduce_sum( item * expand_norm,1) for item in (density_real,density_imag)]

        
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
            sampled = math_ops.cast(sampled, dtypes.int64)
            
            # all_ids = array_ops.concat([target, sampled], 0)
            sampled_real_embedding,sampled_imag_embedding = self.getEmbedding(sampled)
            sampled_amplitude_embedding,sampled_phase_embedding = self.transform( sampled_real_embedding,sampled_imag_embedding)
            sampled_real_outer,sampled_imag_outer = self.outer_product(sampled_amplitude_embedding,sampled_phase_embedding)
           
            
            negative_sample_prob = self.inner_product([density_real_mixture, density_imag_mixture],[sampled_real_outer,sampled_imag_outer])
             # projection dirac multipiltion
             
            true_amplitude_embedding, true_phase_embedding = self.getEmbedding(self.target)
            true_real_embedding, true_imag_embedding  = self.transform( true_amplitude_embedding, true_phase_embedding)
            true_real_outer,true_imag_outer = self.outer_product( true_real_embedding, true_imag_embedding)
            
            true_prob = tf.trace(tf.matmul(density_real_mixture,true_real_outer)) -tf.trace(tf.matmul(density_imag_mixture,true_imag_outer))
            # projection-wise multipiltion

            true_prob_expand = tf.expand_dims(true_prob,1)
            
            out_labels = array_ops.concat([array_ops.ones_like(true_prob_expand) / self.config["num_true"], array_ops.zeros_like(negative_sample_prob)], 1)
            out_logits = array_ops.concat([true_prob_expand,negative_sample_prob],1)
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
    
if __name__ == "__main__":        
    class DottableDict(dict):
        def __init__(self, *args, **kwargs):
            dict.__init__(self, *args, **kwargs)
            self.__dict__ = self 
    # -*- coding: utf-8 -*-
    import tensorflow as tf
    from tensorflow.contrib.tensorboard.plugins import projector
    from tempfile import gettempdir
    import os
    
    from dataHelper import DataHelper
    from params import Params
    import models
    
    helper = DataHelper("demo.txt")
    
    params = Params()
    config_file = 'config/complex_cbow_unit.ini'    # define dataset in the config
    config_file = 'config/real_cbow_unit.ini'    # define dataset in the config
    #config_file = 'config/word2vec.ini'    # define dataset in the config
    params.parse_config(config_file)
    self = DottableDict()
    self.config = {}        
    for conf in (params,helper):
        for k,v in conf.__dict__.items():
            self.config[k]=v
        
            
                
