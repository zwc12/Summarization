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

"""Script for processing summarization stories.
"""

import sys
import os
import struct
import argparse
import collections

from os import listdir
from os.path import isfile, join
from numpy.random import seed as random_seed
from numpy.random import shuffle as random_shuffle

import tensorflow as tf
from tensorflow.core.example import example_pb2

random_seed(123)

# for separating the sentences in the .bin files
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

VOCAB_SIZE = 60000
CHUNK_SIZE = 1000

def ParseStory(story_file):
  lines = []
  with open(story_file, "r") as f:
    for line in f:
      if line.strip()!='':
        lines.append(line.strip())
      
  lines = [line.lower() for line in lines]
  abstract = lines[0]
  lines.pop(0)
  article = ' '.join(["%s %s %s" % (SENTENCE_START, line, SENTENCE_END) for line in lines])

  return article, abstract
  
def WriteBin(stories_directory, bin_directory, outfiles, fraction, makevocab=True):
  
  stories = _get_filenames(stories_directory)
  random_shuffle(stories)
  
  if makevocab:
    vocab_counter = collections.Counter()
    
  print("Writing bin file")
  
  index_start = 0
  progress_bar = ProgressBar(len(stories))
  for index, outfile in enumerate(outfiles):
    counts = int(len(stories) * fraction[index])
    index_stop = min(index_start + counts, len(stories))
    index1 = index_start
    fileindex = 0
    
    while index1< index_stop:
      index1 = min(index_start + CHUNK_SIZE, index_stop)
      story_files = stories[index_start:index1]
      
      with open(join(bin_directory, outfile + '_' + str(fileindex) + '.bin' ), 'wb') as writer:
        for story in story_files:
          article, abstract = ParseStory(join(stories_directory,story))
        
          tf_example = example_pb2.Example()
          tf_example.features.feature['article'].bytes_list.value.extend([article])
          tf_example.features.feature['abstract'].bytes_list.value.extend([abstract])
          tf_example_str = tf_example.SerializeToString()
          str_len = len(tf_example_str)
          writer.write(struct.pack('q', str_len))
          writer.write(struct.pack('%ds' % str_len, tf_example_str))
          progress_bar.Increment()
          if makevocab:
            art_tokens = article.split()
            abs_tokens = abstract.split()
            tokens = art_tokens + abs_tokens
            tokens = [t.strip() for t in tokens if t not in [None, '']]
          vocab_counter.update(tokens)
      fileindex += 1
      index_start = index1
  print("Done writing bin file to directory \"%s\" " % bin_directory)
  
  if makevocab:
    print("Writing vocab file...")
    with open(join(bin_directory, "vocab"), 'w') as writer:
      for word, count in vocab_counter.most_common(VOCAB_SIZE):
        writer.write(word + ' ' + str(count) + '\n')
    print("Done writing vocab file to directory \"%s\" " % bin_directory)

def _get_filenames(input_directories):
  filenames = []
  filenames = [ f for f in listdir(input_directories) if isfile(join(input_directories,f)) ]
  return filenames
  
class ProgressBar(object):
  """Simple progress bar.

  Output example:
    100.00% [2152/2152]
  """

  def __init__(self, total=100, stream=sys.stderr):
    self.total = total
    self.stream = stream
    self.last_len = 0
    self.curr = 0

  def Increment(self):
    self.curr += 1
    self.PrintProgress(self.curr)

    if self.curr == self.total:
      print ''

  def PrintProgress(self, value):
    self.stream.write('\b' * self.last_len)
    pct = 100 * self.curr / float(self.total)
    out = '{:.2f}% [{}/{}]'.format(pct, value, self.total)
    self.last_len = len(out)
    self.stream.write(out)
    self.stream.flush()
    
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Generating tensorflow data files from summarization pairs')
  parser.add_argument('--corpus', choices=['cnn', 'dailymail'], default='cnn')
  parser.add_argument('--outdir', default='bin')
  parser.add_argument('--outfiles', default='train,eval,test')
  parser.add_argument('--splits', default='0.9,0.05,0.05')
  args = parser.parse_args()

  stories_directory = '%s/stories' % args.corpus
  
  outfiles =[s for s in args.outfiles.split(',')]
  fraction = [float(s) for s in args.splits.split(',')]
  assert len(outfiles) == len(fraction)
  for s in fraction: assert s > 0.0 
  assert sum([s for s in fraction]) == 1.0
  
  if not os.path.exists(args.outdir):  os.mkdir(args.outdir)
  
  WriteBin(stories_directory, args.outdir, outfiles, fraction)
  
