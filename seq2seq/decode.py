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

"""Script for beam search decoding.
"""

import os
import time
import json
import logging
import numpy as np
import tensorflow as tf

from pyrouge import Rouge155 

import beam_search
import data

FLAGS = tf.app.flags.FLAGS
SECS_UNTIL_NEW_CKPT = 60
class BeamSearchDecoder(object):
  """Beam search decoder.
  """
  def __init__(self, model, batcher, vocab):
    """Initialize decoder.
    """
    self.model = model
    self.model._build_graph()
    self.saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    self.sess = tf.Session(config=config)
    self.batcher = batcher
    self.vocab = vocab
    train_dir = os.path.join(FLAGS.log_dir, "train")
    
    while True:
      try:
        ckpt_state = tf.train.get_checkpoint_state(train_dir)
        self.saver.restore(self.sess, ckpt_state.model_checkpoint_path)
        print('INFO: Loading checkpoint {}'.format(ckpt_state.model_checkpoint_path))
        break
      except:
        print('ERROR: Failed to restore checkpoint: {}, sleep for {} secs'.format(train_dir, 10))
        time.sleep(10)
        continue
        
    self.decode_dir = os.path.join(FLAGS.log_dir, "decode")
    if not os.path.exists(self.decode_dir): os.makedirs(self.decode_dir)
    if FLAGS.onetime:
      self.rouge_ref_dir = os.path.join(FLAGS.log_dir, "decode/ref")
      self.rouge_dec_dir = os.path.join(FLAGS.log_dir, "decode/dec")
      if not os.path.exists(self.rouge_ref_dir): os.makedirs(self.rouge_ref_dir)
      if not os.path.exists(self.rouge_dec_dir): os.makedirs(self.rouge_dec_dir)

  def _decode(self):
    """
    """
    t0 = time.time()
    counter = 0
    while True:
      batch = self.batcher._next_batch()
      if batch is None: 
        assert FLAGS.onetime, "Dataset exhausted, but we are not in onetime mode"
        print('INFO: Decoder has finished reading dataset for onetime.')
        print('INFO: Output has been saved in {} and {}, start ROUGE eval...'.format(self.rouge_ref_dir, self.rouge_dec_dir))
        results_dict = rouge_eval(self.rouge_ref_dir, self.rouge_dec_dir)
        rouge_log(results_dict, self.decode_dir)
        return

      original_article = batch.original_articles[0]
      original_abstract = batch.original_abstracts[0]

      article_withunks = data.show_art_oovs(original_article, self.vocab)
      abstract_withunks = data.show_abs_oovs(original_abstract, self.vocab, (batch.art_oovs[0] if FLAGS.pointer else None))

      best_hyp = beam_search.run_beam_search(self.sess, self.model, self.vocab, batch)

      output_ids = [int(t) for t in best_hyp.tokens[1:]]
      decoded_words = data.outputids2words(output_ids, self.vocab, (batch.art_oovs[0] if FLAGS.pointer else None))

      try:
        fst_stop_idx = decoded_words.index(data.DECODING_END)
        decoded_words = decoded_words[:fst_stop_idx]
      except ValueError:
        decoded_words = decoded_words
      decoded_output = ' '.join(decoded_words)

      if FLAGS.onetime:
        self._write_for_rouge(original_abstract, decoded_words, counter)
        counter += 1
      else:
        print ""
        print('INFO: ARTICLE: {}'.format(article_withunks))
        print('INFO: REFERENCE SUMMARY: {}'.format(abstract_withunks))
        print('INFO: GENERATED SUMMARY: {}'.format(decoded_output))
        print ""
        self._write_for_attnvis(article_withunks, abstract_withunks, decoded_words, best_hyp.attn_dists, best_hyp.pointers)

        t1 = time.time()
        if t1-t0 > SECS_UNTIL_NEW_CKPT:
          print('INFO: Decoding for {} seconds, loading new checkpoint'.format(t1-t0))
          while True:
            try:
              ckpt_state = tf.train.get_checkpoint_state(train_dir)
              print('INFO: Loading checkpoint {}'.format(ckpt_state.model_checkpoint_path))
              self.saver.restore(self.sess, ckpt_state.model_checkpoint_path)
              break
            except:
              print('ERROR: Failed to restore checkpoint: {}, sleep for {} secs'.format(train_dir, 10))
              time.sleep(10)
          t0 = time.time()

  def _write_for_rouge(self, abstract_text, decoded_words, ex_index):
    """
    """
    decoded_sents = (' '.join(decoded_words))
    decoded_sents = rhtml(decoded_sents)
    reference_sents = rhtml(abstract_text)
    ref_file = os.path.join(self.rouge_ref_dir, "%06d_reference.txt" % ex_index)
    decoded_file = os.path.join(self.rouge_dec_dir, "%06d_decoded.txt" % ex_index)
    with open(ref_file, "w") as f:
      f.write(reference_sents) 
    with open(decoded_file, "w") as f:
      f.write(decoded_sents)
    print('INFO: Wrote example {} to file'.format(ex_index))

  def _write_for_attnvis(self, article, abstract, decoded_words, attn_dists, pointers):
    """
    """
    article_lst = article.split()
    decoded_lst = decoded_words
    to_write = {
        'article_str': rhtml(article),
        'decoded_lst': [rhtml(t) for t in decoded_lst],
        'abstract_str': rhtml(abstract),
        'attn_dists': attn_dists
    }
    if FLAGS.pointer:
      to_write['pointers'] = pointers
    output_fname = os.path.join(self.decode_dir, 'attn_vis_data.json')
    with open(output_fname, 'w') as output_file:
      json.dump(to_write, output_file)
    print('INFO: Wrote visualization data to {}'.format(output_fname))


def rhtml(s):
  """
  """
  s.replace("<", "&lt;")
  s.replace(">", "&gt;")
  return s

def rouge_eval(ref_dir, dec_dir):
  """
  """
  r = Rouge155()
  r.model_filename_pattern = '#ID#_reference.txt'
  r.system_filename_pattern = '(\d+)_decoded.txt'
  r.model_dir = ref_dir
  r.system_dir = dec_dir
  logging.getLogger('global').setLevel(logging.WARNING)
  rouge_results = r.convert_and_evaluate()
  return r.output_to_dict(rouge_results)

def rouge_log(results_dict, dir_to_write):
  """
  """
  log_str = ""
  for x in ["1","2","l"]:
    log_str += "\nROUGE-%s:\n" % x
    for y in ["f_score", "recall", "precision"]:
      key = "rouge_%s_%s" % (x,y)
      key_cb = key + "_cb"
      key_ce = key + "_ce"
      val = results_dict[key]
      val_cb = results_dict[key_cb]
      val_ce = results_dict[key_ce]
      log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
  tf.logging.info(log_str)
  results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
  print('INFO: Writing final ROUGE results to {}...'.format(results_file))
  with open(results_file, "w") as f:
    f.write(log_str)
