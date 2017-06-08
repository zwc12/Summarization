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


import glob
import time
import Queue
import struct
import numpy as np
import tensorflow as tf

from random import shuffle
from threading import Thread
from tensorflow.core.example import example_pb2

import data

class Batch(object):
  """Class representing a minibatch of train/val/test inputs for text summarization.
  """
  def __init__(self, tfexamples, hps, vocab):
    """
    """
    self.hps = hps

    self.enc_batch = np.zeros((hps.batch_size, hps.max_enc_steps), dtype=np.int32)
    self.enc_lens = np.zeros((hps.batch_size), dtype=np.int32)
    
    self.dec_batch = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)
    self.target_batch = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)
    self.padding_mark = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.float32)

    for i, ex in enumerate(tfexamples):
      self.enc_batch[i, :] = ex.enc_input[:]
      self.enc_lens[i] = ex.enc_len
      
      self.dec_batch[i, :] = ex.dec_input[:]
      self.target_batch[i, :] = ex.dec_target[:]
      for j in xrange(ex.dec_len):
        self.padding_mark[i][j] = 1

    if hps.pointer:
      self.max_oovs = max([len(ex.article_oovs) for ex in tfexamples])
      self.art_oovs = [ex.article_oovs for ex in tfexamples]
      self.enc_batch_extend_vocab = np.zeros((hps.batch_size, hps.max_enc_steps), dtype=np.int32)
      for i, ex in enumerate(tfexamples):
        self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]
    
    self.original_articles = [ex.original_article for ex in tfexamples]
    self.original_abstracts = [ex.original_abstract for ex in tfexamples]


class Batcher(object):
  """A class to generate minibatches of data.
  """
  BATCH_QUEUE_MAX = 100
  def __init__(self, data_path, vocab, hps, onetime):
    """
    """
    self.data_path = data_path
    self.vocab = vocab
    self.hps = hps
    self.onetime = onetime

    self.batch_queue = Queue.Queue(self.BATCH_QUEUE_MAX)
    self.input_queue = Queue.Queue(self.BATCH_QUEUE_MAX * self.hps.batch_size)

    if onetime:
      self.num_input_threads = 1
      self.num_batch_threads = 1
      self.cache_size = 1 
      self.finished_reading = False
    else:
      self.num_input_threads = 16
      self.num_batch_threads = 4
      self.cache_size = 100

    self.input_threads = []
    for _ in xrange(self.num_input_threads):
      self.input_threads.append(Thread(target=self._fill_input_queue))
      self.input_threads[-1].daemon = True
      self.input_threads[-1].start()
      
    self.batch_threads = []
    for _ in xrange(self.num_batch_threads):
      self.batch_threads.append(Thread(target=self._fill_batch_queue))
      self.batch_threads[-1].daemon = True
      self.batch_threads[-1].start()

    if not onetime:
      self.watch_thread = Thread(target=self._watch_threads)
      self.watch_thread.daemon = True
      self.watch_thread.start()

  def _next_batch(self):
    """Return a Batch from the batch queue.
    """
    if self.batch_queue.qsize() == 0:
      print('INFO: Batch queue is empty.')
      if self.onetime and self.finished_reading:
        print('INFO: Finished reading dataset in onetime mode.')
        return None

    batch = self.batch_queue.get()
    return batch

  def _fill_input_queue(self):
    """Reads data from file and put into input queue
    """
    while True:
      filelist = glob.glob(self.data_path)
      assert filelist, ('ERROR: empty filelist at {}'.format(self.data_path))
      if self.onetime:
        filelist = sorted(filelist)
      else:
        shuffle(filelist)
        
      for f in filelist:
        with open(f, 'rb') as reader:
          while True:
            len_bytes = reader.read(8)
            if not len_bytes: break
            str_len = struct.unpack('q', len_bytes)[0]
            example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
            e = example_pb2.Example.FromString(example_str)
            
            try:
              article_text = e.features.feature['article'].bytes_list.value[0]
              abstract_text = e.features.feature['abstract'].bytes_list.value[0]
            except ValueError:
              print('WARNING: Failed to get article or abstract from example: {}'.format(text_format.MessageToString(e)))
              continue
              
            if len(article_text)==0:
              print('WARNING: Found an example with empty article, skipped.')
              continue
            
            example = Example(article_text, abstract_text, self.vocab, self.hps)
            self.input_queue.put(example)
      
      if self.onetime and self.hps.mode=='decode':
        print('INFO: onetime thread mode is on, we\'ve finished reading dataset, thread stopping...')
        self.finished_reading = True
        break

  def _fill_batch_queue(self):
    """Get data from input queue and put into batch queue
    """
    while True:
      if self.hps.mode == 'decode':
        ex = self.input_queue.get()
        b = [ex for _ in xrange(self.hps.batch_size)]
        self.batch_queue.put(Batch(b, self.hps, self.vocab))

      else:
        inputs = []
        for _ in xrange(self.hps.batch_size * self.cache_size):
          inputs.append(self.input_queue.get())
        inputs.sort(key=lambda e: e.enc_len)

        batches = []
        for i in xrange(0, len(inputs), self.hps.batch_size):
          batches.append(inputs[i:i + self.hps.batch_size])
        if not self.onetime:
          shuffle(batches)
        for b in batches:
          self.batch_queue.put(Batch(b, self.hps, self.vocab))

  def _watch_threads(self):
    """Watch input queue and batch queue threads and restart if dead."""
    while True:
      time.sleep(60)
      for idx,t in enumerate(self.input_threads):
        if not t.is_alive():
          tf.logging.error('Found input queue thread dead. Restarting.')
          new_t = Thread(target=self._fill_input_queue)
          self.input_threads[idx] = new_t
          new_t.daemon = True
          new_t.start()
      for idx,t in enumerate(self.batch_threads):
        if not t.is_alive():
          tf.logging.error('Found batch queue thread dead. Restarting.')
          new_t = Thread(target=self._fill_batch_queue)
          self.batch_threads[idx] = new_t
          new_t.daemon = True
          new_t.start()
         
          
class Example(object):
  """Class representing a train/val/test inputs for text summarization.
  """
  def __init__(self, article, abstract, vocab, hps):
    """
    """
    self.hps = hps

    start_id = vocab._word2id(data.DECODING_START)
    end_id = vocab._word2id(data.DECODING_END)

    article_words = article.split()
    if len(article_words) > hps.max_enc_steps:
      article_words = article_words[:hps.max_enc_steps]
    self.enc_len = len(article_words)
    self.enc_input = [vocab._word2id(w) for w in article_words]

    abstract_words = abstract.split()
    abs_ids = [vocab._word2id(w) for w in abstract_words]

    self.dec_input = [start_id] + abs_ids
    self.dec_target = abs_ids + [end_id]
    
    self.pad_id = vocab._word2id(data.PAD_TOKEN)
    if hps.pointer:
      self.enc_input_extend_vocab, self.article_oovs = data.article2ids(article_words, vocab)
      abs_ids_extend_vocab = data.abstract2ids(abstract_words, vocab, self.article_oovs)
      self.dec_target = abs_ids_extend_vocab + [end_id]
      while len(self.enc_input_extend_vocab) < hps.max_enc_steps:
        self.enc_input_extend_vocab.append(self.pad_id)
      
    while len(self.enc_input) < hps.max_enc_steps:
      self.enc_input.append(self.pad_id)
      
    while len(self.dec_input) < hps.max_dec_steps:
      self.dec_input.append(self.pad_id)
      self.dec_target.append(self.pad_id)
     
    if len(self.dec_input) > hps.max_dec_steps:
      self.dec_input = self.dec_input[:hps.max_dec_steps - 1]
      self.dec_target = self.dec_target[:hps.max_dec_steps - 1]
    
    self.dec_len = len(self.dec_input)
 
    self.original_article = article
    self.original_abstract = abstract
