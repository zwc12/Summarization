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

"""Script for reading and processing the train/eval/test data and the vocab data.
"""

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops

FLAGS = tf.app.flags.FLAGS

class SummarizationModel(object):
  """A class to represent a sequence-to-sequence model for text summarization. Supports both baseline mode, pointer-generator mode
  """
  def __init__(self, hps, vocab):
    self.hps = hps
    self.vocab = vocab

  def _train(self, sess, batch):
    """Runs one training iteration. Returns a dictionary
    """
    feed_dict={
        self.enc_batch: batch.enc_batch,
        self.enc_lens: batch.enc_lens,
        self.dec_batch: batch.dec_batch,
        self.target_batch: batch.target_batch,
        self.padding_mark: batch.padding_mark
    }
      
    if FLAGS.pointer:
      feed_dict[self.max_oovs] = batch.max_oovs
      feed_dict[self.enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
      
    sess_return = {
        'train_op': self.train_op,
        'summaries': self.summaries,
        'loss': self.loss,
        'global_step': self.global_step,
    }
    return sess.run(sess_return, feed_dict)

  def _eval(self, sess, batch):
    """Runs one evaluation iteration. Returns a dictionary containing summaries, loss, global_step.
    """
    feed_dict={
        self.enc_batch: batch.enc_batch,
        self.enc_lens: batch.enc_lens,
        self.dec_batch: batch.dec_batch,
        self.target_batch: batch.target_batch,
        self.padding_mark: batch.padding_mark
    }
      
    if FLAGS.pointer:
      feed_dict[self.max_oovs] = batch.max_oovs
      feed_dict[self.enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
      
    sess_return = {
        'summaries': self.summaries,
        'loss': self.loss,
        'global_step': self.global_step,
    }
    return sess.run(sess_return, feed_dict)

  def _encode(self, sess, batch):
    """For beam search decoding. Run the encoder on the batch and return the encoder states and decoder initial state.
    """
    feed_dict={
        self.enc_batch: batch.enc_batch,
        self.enc_lens: batch.enc_lens
    }
      
    if FLAGS.pointer:
      feed_dict[self.max_oovs] = batch.max_oovs
      feed_dict[self.enc_batch_extend_vocab] = batch.enc_batch_extend_vocab

    (enc_states, dec_in_state, global_step) = sess.run([self.enc_states, self.dec_state, self.global_step], feed_dict)

    dec_in_state = tf.contrib.rnn.LSTMStateTuple(dec_in_state.c[0], dec_in_state.h[0])
    return enc_states, dec_in_state

  def _decode(self, sess, batch, latest_tokens, enc_states, dec_init_states):
    """For beam search decoding. Run the decoder for one step.
    """
    beam_size = len(dec_init_states)

    cells = [np.expand_dims(state.c, axis=0) for state in dec_init_states]
    hiddens = [np.expand_dims(state.h, axis=0) for state in dec_init_states]
    new_c = np.concatenate(cells, axis=0)
    new_h = np.concatenate(hiddens, axis=0)
    new_dec_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

    feed = {
        self.enc_states: enc_states,
        self.dec_state: new_dec_state,
        self.dec_batch: np.transpose(np.array([latest_tokens])),
    }

    sess_return = {
      "ids": self.topk_ids,
      "probs": self.topk_log_probs,
      "states": self.dec_out_state,
      "attn_dists": self.attn_dists
    }

    if FLAGS.pointer:
      feed[self.enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
      feed[self.max_oovs] = batch.max_oovs
      sess_return['pointers'] = self.pointers

    results = sess.run(sess_return, feed_dict=feed)
    new_states = [tf.contrib.rnn.LSTMStateTuple(results['states'].c[i, :], results['states'].h[i, :]) for i in xrange(beam_size)]

    assert len(results['attn_dists'])==1
    attn_dists = results['attn_dists'][0].tolist()
    if FLAGS.pointer:
      assert len(results['pointers'])==1
      pointers = results['pointers'][0].tolist()
    else:
      pointers = [None for _ in xrange(beam_size)]
    
    return results['ids'], results['probs'], new_states, attn_dists, pointers

  def _build_graph(self):
    """Add the placeholders, model, global step, train_op and summaries to the graph
    """
    print('INFO: Building graph...')
    t0 = time.time()
    
    hps = self.hps
    vsize = self.vocab._size()
    
    self.enc_batch = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch')
    self.enc_lens = tf.placeholder(tf.int32, [hps.batch_size], name='enc_lens')
    if FLAGS.pointer:
      self.enc_batch_extend_vocab = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch_extend_vocab')
      self.max_oovs = tf.placeholder(tf.int32, [], name='max_oovs')

    self.dec_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='dec_batch')
    self.target_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='target_batch')
    self.padding_mark = tf.placeholder(tf.float32, [hps.batch_size, hps.max_dec_steps], name='padding_mark')
    
    with tf.device(FLAGS.device):
      with tf.variable_scope('seq2seq'):
        self.norm_uinf = tf.random_uniform_initializer(-hps.norm_unif, hps.norm_unif, seed=123)
        self.norm_trunc = tf.truncated_normal_initializer(stddev=hps.norm_trunc)

        with tf.variable_scope('embedding'):
          embedding = tf.get_variable('embedding', [vsize, hps.emb_dim], dtype=tf.float32, initializer=self.norm_trunc)
          emb_enc_inputs = tf.nn.embedding_lookup(embedding, self.enc_batch)
          emb_dec_inputs = [tf.nn.embedding_lookup(embedding, x) for x in tf.unstack(self.dec_batch, axis=1)]

        with tf.variable_scope('encoder'):
          cell_fw = tf.contrib.rnn.LSTMCell(self.hps.hidden_dim, initializer=self.norm_uinf, state_is_tuple=True)
          cell_bw = tf.contrib.rnn.LSTMCell(self.hps.hidden_dim, initializer=self.norm_uinf, state_is_tuple=True)
          (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, emb_enc_inputs, dtype=tf.float32, sequence_length=self.enc_lens, swap_memory=True)
          self.enc_states = tf.concat(axis=2, values=encoder_outputs)
        
        with tf.variable_scope('reduce'):
          w_reduce = tf.get_variable('w_reduce', [self.hps.hidden_dim * 2, self.hps.hidden_dim], dtype=tf.float32, initializer=self.norm_trunc)
          v_reduce = tf.get_variable('v_reduce', [self.hps.hidden_dim * 2, self.hps.hidden_dim], dtype=tf.float32, initializer=self.norm_trunc)
          wb_reduce = tf.get_variable('wb_reduce', [self.hps.hidden_dim], dtype=tf.float32, initializer=self.norm_trunc)
          vb_reduce = tf.get_variable('vb_reduce', [self.hps.hidden_dim], dtype=tf.float32, initializer=self.norm_trunc)

          new_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c])
          new_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h])
          new_c = tf.nn.relu(tf.matmul(new_c, w_reduce) + wb_reduce)
          new_h = tf.nn.relu(tf.matmul(new_h, v_reduce) + vb_reduce)
          self.dec_state =  tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

        with tf.variable_scope('decoder'):
          cell = tf.contrib.rnn.LSTMCell(hps.hidden_dim, state_is_tuple=True, initializer=self.norm_uinf)
          dec_out, self.dec_out_state, self.attn_dists, self.pointers = attention_decoder(emb_dec_inputs, self.dec_state, self.enc_states, cell, hps.mode=="decode", hps.pointer)

        with tf.variable_scope('output_projection'):
          w = tf.get_variable('w', [hps.hidden_dim, vsize], dtype=tf.float32, initializer=self.norm_trunc)
          w_t = tf.transpose(w)
          v = tf.get_variable('v', [vsize], dtype=tf.float32, initializer=self.norm_trunc)
          
          vocab_scores = []
          for i,output in enumerate(dec_out):
            if i > 0:
              tf.get_variable_scope().reuse_variables()
            vocab_scores.append(tf.nn.xw_plus_b(output, w, v))

          vocab_dists = [tf.nn.softmax(s) for s in vocab_scores]

        if FLAGS.pointer:
          final_dists = self._final_dist(vocab_dists, self.attn_dists)
          log_vocab_dists = [tf.log(dist + 1e-10) for dist in final_dists]
        else:
          log_vocab_dists = [tf.log(dist + 1e-10) for dist in vocab_dists]

        if hps.mode in ['train', 'eval']:
          with tf.variable_scope('loss'):
            if FLAGS.pointer:
              losses = []
              batch_nums = tf.range(0, limit=hps.batch_size)
              for i, log_dist in enumerate(log_vocab_dists):
                targets = self.target_batch[:,i]
                indices = tf.stack( (batch_nums, targets), axis=1)
                loss = tf.gather_nd(-log_dist, indices)
                losses.append(loss)

              values = [v * self.padding_mark[:,i] for i,v in enumerate(losses)]
              self.loss = tf.reduce_mean(sum(values)/tf.reduce_sum(self.padding_mark, axis=1))
            else:
              self.loss = tf.contrib.seq2seq.sequence_loss(tf.stack(vocab_scores, axis=1), self.target_batch, self.padding_mark)

            tf.summary.scalar('loss', self.loss)

      if hps.mode == "decode":
        assert len(log_vocab_dists)==1
        log_vocab_dists = log_vocab_dists[0]
        self.topk_log_probs, self.topk_ids = tf.nn.top_k(log_vocab_dists, hps.batch_size*2)
      
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    if self.hps.mode == 'train':
      gradients = tf.gradients(self.loss, tf.trainable_variables(), aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
      with tf.device(FLAGS.device):
        grads, global_norm = tf.clip_by_global_norm(gradients, self.hps.norm_grad)
        tf.summary.scalar('global_norm', global_norm)
        optimizer = tf.train.AdagradOptimizer(self.hps.lr, initial_accumulator_value=self.hps.adagrad_acc)
        self.train_op = optimizer.apply_gradients(zip(grads, tf.trainable_variables()), global_step=self.global_step, name='train_step')
     
    self.summaries = tf.summary.merge_all()
    t1 = time.time()
    print('INFO: Time to build graph: {} seconds'.format(t1 - t0))

  def _final_dist(self, vocab_dists, attn_dists):
    """Calculate the final distribution, for the pointer-generator model
    """
    with tf.variable_scope('final_distribution'):
      vocab_dists = [pointer * dist for (pointer,dist) in zip(self.pointers, vocab_dists)]
      attn_dists = [(1-pointer) * dist for (pointer,dist) in zip(self.pointers, attn_dists)]

      extended_vsize = self.vocab._size() + self.max_oovs
      extra_zeros = tf.zeros((self.hps.batch_size, self.max_oovs))
      vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists]

      batch_nums = tf.range(0, limit=self.hps.batch_size)
      batch_nums = tf.expand_dims(batch_nums, 1)
      attn_len = tf.shape(self.enc_batch_extend_vocab)[1]
      batch_nums = tf.tile(batch_nums, [1, attn_len])
      indices = tf.stack( (batch_nums, self.enc_batch_extend_vocab), axis=2)
      shape = [self.hps.batch_size, extended_vsize]
      attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attn_dists]

      final_dists = [vocab_dist + copy_dist for (vocab_dist,copy_dist) in zip(vocab_dists_extended, attn_dists_projected)]

      return final_dists


def attention_decoder(decoder_inputs, initial_state, encoder_states, cell, initial_state_attention=False, pointer=True):
  """
  """
  with variable_scope.variable_scope("attention_decoder") as scope:
    batch_size = encoder_states.get_shape()[0].value 
    attn_size = encoder_states.get_shape()[2].value

    encoder_states = tf.expand_dims(encoder_states, axis=2)

    W_h = variable_scope.get_variable("W_h", [1, 1, attn_size, attn_size])
    encoder_features = nn_ops.conv2d(encoder_states, W_h, [1, 1, 1, 1], "SAME")
    v = variable_scope.get_variable("v", [attn_size])

    def attention(decoder_state):
      """Calculate the context vector and attention distribution from the decoder state.
      """
      with variable_scope.variable_scope("Attention"):
        decoder_features = linear(decoder_state, attn_size, True)
        decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1)
        e = math_ops.reduce_sum(v * math_ops.tanh(encoder_features + decoder_features), [2, 3])
        attn_dist = nn_ops.softmax(e)
        context_vector = math_ops.reduce_sum(array_ops.reshape(attn_dist, [batch_size, -1, 1, 1]) * encoder_states, [1, 2])
        context_vector = array_ops.reshape(context_vector, [-1, attn_size])

      return context_vector, attn_dist

    outputs = []
    attn_dists = []
    pointers = []
    state = initial_state
    context_vector = array_ops.zeros([batch_size, attn_size])
    context_vector.set_shape([None, attn_size])
    if initial_state_attention:
      context_vector, _ = attention(initial_state)
      
    print('INFO: Adding attention_decoder of {} timesteps...'.format(len(decoder_inputs)))
    for i, inp in enumerate(decoder_inputs):
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()

      input_size = inp.get_shape().with_rank(2)[1]
      if input_size.value is None:
        raise ValueError("Could not infer input size from input: %s" % inp.name)
      x = linear([inp] + [context_vector], input_size, True)

      cell_output, state = cell(x, state)

      if i == 0 and initial_state_attention:
        with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=True):
          context_vector, attn_dist= attention(state)
      else:
        context_vector, attn_dist = attention(state)
      attn_dists.append(attn_dist)

      if pointer:
        with tf.variable_scope('calculate_pgen'):
          poi = linear([context_vector, state.c, state.h, x], 1, True)
          poi = tf.sigmoid(poi)
          pointers.append(poi)

      with variable_scope.variable_scope("AttnOutputProjection"):
        output = linear([cell_output] + [context_vector], cell.output_size, True)
      outputs.append(output)

    return outputs, state, attn_dists, pointers

def linear(args, output_size, bias, bias_start=0.0, scope=None):
  """
  """
  if args is None or (isinstance(args, (list, tuple)) and not args):
    raise ValueError("`args` must be specified")
  if not isinstance(args, (list, tuple)):
    args = [args]

  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: {}".format(str(shapes)))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: {}".format(str(shapes)))
    else:
      total_arg_size += shape[1]

  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
    if len(args) == 1:
      res = tf.matmul(args[0], matrix)
    else:
      res = tf.matmul(tf.concat(axis=1, values=args), matrix)
    if not bias:
      return res
    bias_term = tf.get_variable(
        "Bias", [output_size], initializer=tf.constant_initializer(bias_start))
  return res + bias_term
