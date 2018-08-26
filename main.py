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
#config_file = 'config/real_cbow_unit.ini'    # define dataset in the config
#config_file = 'config/word2vec.ini'    # define dataset in the config
params.parse_config(config_file)

session = tf.InteractiveSession()#
model = models.setup(params,helper)
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
writer = tf.summary.FileWriter(params.log_dir, session.graph)
    


average_loss = 0
for step in range(5):
    for step_i,(batch_inputs, batch_labels) in enumerate( helper.generate_batch(params.batch_size, params.num_skips, params.skip_window, cbow= params.data_type =="cbow"  )):
     
        feed_dict = {model.train_inputs: batch_inputs, model.train_labels: batch_labels}
  
      # Define metadata variable.
#        run_metadata = tf.RunMetadata()
  
        _, summary, loss_val = session.run([model.optimizer,model.merged, model.loss],
                                           feed_dict=feed_dict)  #,            run_metadata=run_metadata
        average_loss += loss_val
#        _= session.run([model.normalization_op])
        #      norm_a =session.run(norm)
        #        print (np.mean(norm_a))
  
        if step_i % 2000 == 0:
            if step_i > 0:
                average_loss /= 2000
          # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step ', step_i, ': ', average_loss)
            average_loss = 0
        
        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step_i % 10000 == 0:
            sim = model.similarity.eval()
            for i in range(params.valid_size):
                valid_word = helper.reverse_dictionary[model.valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = helper.reverse_dictionary[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
            print(log_str)
    saver.save(session, os.path.join(params.log_dir, 'model'+str(step)+'.ckpt'))
    
    
final_embeddings = model.embeddings.eval()

# Write corresponding labels for the embeddings.
with open(params.log_dir + '/metadata.tsv', 'w') as f:
    for i in range(helper.vocabulary_size):
        f.write(helper.reverse_dictionary[i] + '\n')

# Save the model for checkpoints.
saver.save(session, os.path.join(params.log_dir, 'model.ckpt'))

# Create a configuration for visualizing embeddings with the labels in TensorBoard.
config = projector.ProjectorConfig()
embedding_conf = config.embeddings.add()
embedding_conf.tensor_name = model.embeddings.name
embedding_conf.metadata_path = os.path.join(params.log_dir, 'metadata.tsv')
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
  labels = [helper.reverse_dictionary[i] for i in range(plot_only)]
  plot_with_labels(low_dim_embs, labels, os.path.join(gettempdir(), 'tsne.png'))

except ImportError as ex:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')
  print(ex)
