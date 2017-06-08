# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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


import os
import sys
import time
import numpy as np
import tensorflow as tf
from collections import namedtuple

from data import Vocab
from batcher import Batcher
from model import SummarizationModel
from decode import BeamSearchDecoder

FLAGS = tf.app.flags.FLAGS

# Where to find data
tf.app.flags.DEFINE_string('data_path', 'bin/train*.bin', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('vocab_path', 'bin/vocab', 'Path expression to text vocabulary file.')
tf.app.flags.DEFINE_string('log_dir', 'logs', 'Path for logs.')

# Important settings
tf.app.flags.DEFINE_string('mode', 'eval', 'must be one of train/eval/decode')
tf.app.flags.DEFINE_string('device', '/gpu:0', 'device to run tensorflow')
tf.app.flags.DEFINE_integer('iterations', 0,'max number of iterations, 0 for infinite')
tf.app.flags.DEFINE_integer('random_seed', 123, 'A seed value for randomness.')
tf.app.flags.DEFINE_boolean('onetime', False, 'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint.')
tf.app.flags.DEFINE_boolean('pointer', True, 'If True, use pointer-generator model. If False, use baseline model.')

# Hyperparameters
tf.app.flags.DEFINE_integer('hidden_dim', 160, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('emb_dim', 80, 'dimension of word embeddings')
tf.app.flags.DEFINE_integer('batch_size', 16, 'minibatch size')
tf.app.flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('vocab_size', 40000, 'Size of vocabulary. If this number is set to 0, will read all words in the vocabulary file.')
tf.app.flags.DEFINE_integer('max_enc_steps', 400, 'max timesteps of encoder (max source text tokens)')
tf.app.flags.DEFINE_integer('max_dec_steps', 30, 'max timesteps of decoder (max summary tokens)')
tf.app.flags.DEFINE_integer('min_dec_steps', 5, 'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
tf.app.flags.DEFINE_float('lr', 0.15, 'learning rate')
tf.app.flags.DEFINE_float('adagrad_acc', 0.1, 'initial accumulator value for Adagrad')
tf.app.flags.DEFINE_float('norm_unif', 0.02, 'magnitude for lstm cells random uniform inititalization')
tf.app.flags.DEFINE_float('norm_trunc', 1e-4, 'std of trunc norm init, used for initializing everything else')
tf.app.flags.DEFINE_float('norm_grad', 2.0, 'for gradient clipping')

def train(model, batcher):
  """Repeatedly runs training iterations, logging loss to screen and writing summaries
  """
  train_dir = os.path.join(FLAGS.log_dir, "train")
  if not os.path.exists(train_dir): os.makedirs(train_dir)

  with tf.device(FLAGS.device):
    model._build_graph()
    saver = tf.train.Saver(max_to_keep=1)

  sv = tf.train.Supervisor(logdir=train_dir,
                     is_chief=True,
                     saver=saver,
                     summary_op=None,
                     save_summaries_secs=60,
                     save_model_secs=60,
                     global_step=model.global_step)
  summary_writer = sv.summary_writer
  print('INFO: Preparing or waiting for session...')
  
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth=True
  session = sv.prepare_or_wait_for_session(config=config)
  print('INFO: Created session.')
  
  try:
    print('INFO: starting training')
    train_step = 0
    loss_avg = 0
    t0=time.time()
    with session as sess:
      while (train_step <= FLAGS.iterations or FLAGS.iterations == 0):
        batch = batcher._next_batch()
        if batch is None:
          print('INFO: finish training for {} iterations, all done!'.format(FLAGS.iterations))
          break
        
        results = model._train(sess, batch)

        summaries = results['summaries']
        train_step = results['global_step']
        loss = results['loss']
        loss_avg = avg_loss(loss, loss_avg, summary_writer, train_step, decay=0.99) 
        summary_writer.add_summary(summaries, train_step)
        if train_step % 2 == 0:
          t1=time.time()
          print('INFO: time for training step {}: {:>3}h {:>2}m {:>2}s, loss: {:>8.6}, loss_avg: {:>8.6}'.format(train_step, int(t1-t0)//3600,(int(t1-t0)%3600)//60, int(t1-t0)%60, loss, loss_avg))
         
        if train_step % 100 == 0:
          summary_writer.flush()
  except KeyboardInterrupt:
    print('INFO: Caught keyboard interrupt on worker. Stopping supervisor...')


def cval(model, batcher, vocab):
  """Repeatedly runs eval iterations, logging to screen and writing summaries. Saves the model with the best loss seen so far.
  """
  model._build_graph()
  saver = tf.train.Saver(max_to_keep=3)
  
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth=True
  sess = tf.Session(config=config)
  
  train_dir = os.path.join(FLAGS.log_dir, "train")
  eval_dir = os.path.join(FLAGS.log_dir, "eval")
  model_dir = os.path.join(eval_dir, 'best')
  summary_writer = tf.summary.FileWriter(eval_dir)
  running_avg_loss = 0
  best_loss = None

  while True:
    try:
      ckpt_state = tf.train.get_checkpoint_state(train_dir)
      print('INFO: Loading checkpoint {}'.format(ckpt_state.model_checkpoint_path))
      saver.restore(sess, ckpt_state.model_checkpoint_path)
    except:
      tf.logging.error('Failed to restore checkpoint: {}, sleep for {} secs'.format(train_dir, 10))
      time.sleep(10)
      continue
      
    batch = batcher._next_batch()
    if batch is None: continue

    t0=time.time()
    results = model._eval(sess, batch)
    t1=time.time()
    print('INFO: seconds for eval step: {}'.format(t1-t0))

    loss = results['loss']
    tf.logging.info('loss: %f', loss)

    summaries = results['summaries']
    train_step = results['global_step']
    summary_writer.add_summary(summaries, train_step)

    running_avg_loss = avg_loss(np.asscalar(loss), running_avg_loss, summary_writer, train_step)

    if best_loss is None or running_avg_loss < best_loss:
      print('INFO: Found new best model with {} running_avg_loss. Saving to {}'.format(running_avg_loss, model_dir))
      saver.save(sess, model_dir, global_step=train_step, latest_filename='checkpoint_best')
      best_loss = running_avg_loss

    if train_step % 100 == 0:
      summary_writer.flush()


def avg_loss(loss, running_avg_loss, summary_writer, step, decay=0.99):
  """Calculate the running average loss via exponential decay.
  """
  if running_avg_loss == 0:
    running_avg_loss = loss
  else:
    running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
  running_avg_loss = min(running_avg_loss, 12)
  loss_sum = tf.Summary()
  summary_writer.add_summary(loss_sum, step)
  return running_avg_loss


def main(unused_argv):
  if len(unused_argv) != 1:
    raise Exception("Problem with flags: %s" % unused_argv)
  if FLAGS.mode not in ['train', 'eval', 'decode']:
    raise ValueError("The 'mode' flag must be one of train/eval/decode")

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.set_random_seed(FLAGS.random_seed)
  print('INFO: Starting seq2seq_attention model in {} mode...'.format(FLAGS.mode))
  if not os.path.exists(FLAGS.log_dir): os.makedirs(FLAGS.log_dir)

  if FLAGS.mode == 'decode':
    FLAGS.batch_size = FLAGS.beam_size
    
  hparam_list = ['mode', 'lr', 'adagrad_acc', 'norm_unif', 'norm_trunc', 'norm_grad', 'pointer',
   'hidden_dim', 'emb_dim', 'batch_size', 'max_dec_steps', 'max_enc_steps']
  hps_dict = {}
  for key,val in FLAGS.__flags.iteritems():
    if key in hparam_list:
      hps_dict[key] = val
  hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)

  vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size)
  batcher = Batcher(FLAGS.data_path, vocab, hps, onetime=FLAGS.onetime)

  if hps.mode == 'train':
    print('INFO: creating model...')
    model = SummarizationModel(hps, vocab)
    train(model, batcher)
  elif hps.mode == 'eval':
    model = SummarizationModel(hps, vocab)
    cval(model, batcher, vocab)
  elif hps.mode == 'decode':
    decode_mdl_hps = hps
    decode_mdl_hps = hps._replace(max_dec_steps=1)
    model = SummarizationModel(decode_mdl_hps, vocab)
    decoder = BeamSearchDecoder(model, batcher, vocab)
    decoder._decode()

if __name__ == '__main__':
  tf.app.run()
