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

"""Script for reading and processing the train/eval/test data
"""

import glob
import random
import struct
from tensorflow.core.example import example_pb2

PAD_TOKEN = '<PAD>'
UNKNOWN_TOKEN = '<UNK>'
DECODING_START = '<d>'
DECODING_END = '</d>'

class Vocab(object):
  """Vocabulary class for mapping between words and ids (integers)
  """
  def __init__(self, vocab_file, max_size):
    """
    """
    self.word_to_id = {}
    self.id_to_word = {}
    self.count = 0

    for w in [UNKNOWN_TOKEN, PAD_TOKEN, DECODING_START, DECODING_END]:
      self.word_to_id[w] = self.count
      self.id_to_word[self.count] = w
      self.count += 1

    with open(vocab_file, 'r') as vocab_f:
      for line in vocab_f:
        pieces = line.split()
        if len(pieces) != 2:
          print ('WARNING: incorrectly formatted line in vocabulary file: {}'.format(line))
          continue
        if pieces[0] in self.word_to_id:
          raise ValueError('Duplicated word in vocabulary file: {}.'.format(pieces[0]))
        self.word_to_id[pieces[0]] = self.count
        self.id_to_word[self.count] = pieces[0]
        self.count += 1
        if max_size != 0 and self.count >= max_size:
          break
    print ('INFO: Finished reading {} of {} words in vocab, last word added: {}'.format(self.count, max_size, self.id_to_word[self.count-1]))

  def _word2id(self, word):
    """Returns the id (integer) of a word (string). Returns <UNK> id if word is OOV.
    """
    if word not in self.word_to_id:
      return self.word_to_id[UNKNOWN_TOKEN]
    return self.word_to_id[word]

  def _id2word(self, word_id):
    """Returns the word (string) corresponding to an id (integer).
    """
    if word_id not in self.id_to_word:
      raise ValueError('Id not found in vocab: %d' % word_id)
    return self.id_to_word[word_id]

  def _size(self):
    """Returns the total size of the vocabulary
    """
    return self.count


def article2ids(article_words, vocab):
  """Map the article words to their ids. Also return a list of OOVs in the article.
  """
  ids = []
  oovs = []
  unk_id = vocab._word2id(UNKNOWN_TOKEN)
  for w in article_words:
    i = vocab._word2id(w)
    if i == unk_id:
      if w not in oovs:
        oovs.append(w)
      oov_num = oovs.index(w)
      ids.append(vocab._size() + oov_num)
    else:
      ids.append(i)
  return ids, oovs

def abstract2ids(abstract_words, vocab, article_oovs):
  """Map the abstract words to their ids. In-article OOVs are mapped to their temporary OOV numbers.
  """
  ids = []
  unk_id = vocab._word2id(UNKNOWN_TOKEN)
  for w in abstract_words:
    i = vocab._word2id(w)
    if i == unk_id:
      if w in article_oovs:
        vocab_idx = vocab._size() + article_oovs.index(w)
        ids.append(vocab_idx)
      else:
        ids.append(unk_id)
    else:
      ids.append(i)
  return ids

def outputids2words(id_list, vocab, article_oovs):
  """Get words from output ids.
  """
  words = []
  for i in id_list:
    try:
      w = vocab._id2word(i)
    except ValueError as e:
      assert article_oovs is not None, "Error: model produced a word ID {} in article OOVs, but there are no article OOVs" .format(i)
      try:
        w = article_oovs[ i - vocab._size()]
      except ValueError as e:
        raise ValueError('Error: model  produced a word ID {} in article OOVs with id {}, but there are only {} article OOVs'.format(i, i - vocab._size(), len(article_oovs)))
    words.append(w)
  return words
  
def show_art_oovs(article, vocab):
  """Returns the article string, highlighting the OOVs
  """
  unk_id = vocab._word2id(UNKNOWN_TOKEN)
  words = article.split()
  vwords = []
  for w in words:
    if vocab._word2id(w)==unk_id:
      vwords.append("--%s--" % w)
    else:
      vwords.append(w)
  out_str = ' '.join(vwords)
  return out_str

def show_abs_oovs(abstract, vocab, article_oovs):
  """Returns the abstract string, highlighting the OOVs
  """
  unk_id = vocab._word2id(UNKNOWN_TOKEN)
  words = abstract.split()
  vwords = []
  for w in words:
    if vocab._word2id(w) == unk_id:
      if article_oovs is None:
        vwords.append("--%s--" % w)
      else:
        if w in article_oovs:
          vwords.append("__%s__" % w)
        else:
          vwords.append("--%s--" % w)
    else:
      vwords.append(w)
  out_str = ' '.join(vwords)
  return out_str
  
