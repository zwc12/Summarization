# Copyright 2014 Google Inc. All Rights Reserved.
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

"""Script for generating summarization pairs.
"""

import os
import re
import sys
import time
import math
import argparse
from os import listdir
from os.path import isfile, join
from collections import namedtuple
from itertools import chain
from itertools import izip
from itertools import repeat
from multiprocessing.pool import Pool
import cchardet as chardet
from lxml import html

RawStory = namedtuple('RawStory', 'url html')
ParseStory = namedtuple('ParseStory', 'url content highlights title')

def StoreMode(corpus):
  """
  """
  urls = get_filenames('%s/downloads/' % corpus)
  p = Pool()
  stories = p.imap_unordered(StoreMapper, izip(urls, repeat(corpus)))
  progress_bar = ProgressBar(len(urls))
  for story in stories:
    if story:
      WriteStory(story, corpus)
      
    progress_bar._Increment()

def StoreMapper(t):
  """Reads an URL from disk and returns the parsed news story.

  Args:
    t: a tuple (url, corpus).

  Returns:
    A Story containing the parsed news story.
  """

  url, corpus = t
  story_html = ReadDownloadedUrl(url, corpus)

  if not story_html:
    return None

  raw_story = RawStory(url, story_html)
  return ParseHtml(raw_story, corpus)

def get_filenames(input_directories):
  filenames = []
  filenames = [ f for f in listdir(input_directories) if isfile(join(input_directories,f)) ]
  return filenames
  
def ReadDownloadedUrl(url, corpus):
  """Reads a downloaded URL from disk.
  Args:
    url: The URL to read.
    corpus: The corpus the URL belongs to.
  Returns:
    The content of the URL.
  """
  try:
    with open('%s/downloads/%s' % (corpus, url)) as f:
      return f.read()
  except IOError:
    return None
   
def WriteStory(story, corpus):
  """Writes a news story to disk.
  Args:
    story: The news story to write.
    corpus: The corpus the news story belongs to.
  """
  with open('%s/stories/%s.story' % (corpus, story.url[1:len(story.url)-5]), 'w') as f:
    f.write(story.title + '\n' + story.content)

def ParseHtml(story, corpus):
  """Parses the HTML of a news story.
  Args:
    story: The raw Story to be parsed.
    corpus: Either 'cnn' or 'dailymail'.
  Returns:
    A Story containing URL, paragraphs and highlights.
  """
  parser = html.HTMLParser(encoding=chardet.detect(story.html)['encoding'])
  tree = html.document_fromstring(story.html, parser=parser)

  # Elements to delete.
  delete_selectors = {
      'cnn': [
          '//blockquote[contains(@class, "twitter-tweet")]',
          '//blockquote[contains(@class, "instagram-media")]'
      ],
      'dailymail': [
          '//blockquote[contains(@class, "twitter-tweet")]',
          '//blockquote[contains(@class, "instagram-media")]'
      ]
  }

  # Paragraph exclusions: ads, links, bylines, comments
  cnn_exclude = (
      'not(ancestor::*[contains(@class, "metadata")])'
      ' and not(ancestor::*[contains(@class, "pullquote")])'
      ' and not(ancestor::*[contains(@class, "SandboxRoot")])'
      ' and not(ancestor::*[contains(@class, "twitter-tweet")])'
      ' and not(ancestor::div[contains(@class, "cnnStoryElementBox")])'
      ' and not(contains(@class, "cnnTopics"))'
      ' and not(descendant::*[starts-with(text(), "Read:")])'
      ' and not(descendant::*[starts-with(text(), "READ:")])'
      ' and not(descendant::*[starts-with(text(), "Join us at")])'
      ' and not(descendant::*[starts-with(text(), "Join us on")])'
      ' and not(descendant::*[starts-with(text(), "Read CNNOpinion")])'
      ' and not(descendant::*[contains(text(), "@CNNOpinion")])'
      ' and not(descendant-or-self::*[starts-with(text(), "Follow us")])'
      ' and not(descendant::*[starts-with(text(), "MORE:")])'
      ' and not(descendant::*[starts-with(text(), "SPOILER ALERT:")])')

  dm_exclude = (
      'not(ancestor::*[contains(@id,"reader-comments")])'
      ' and not(contains(@class, "byline-plain"))'
      ' and not(contains(@class, "byline-section"))'
      ' and not(contains(@class, "count-number"))'
      ' and not(contains(@class, "count-text"))'
      ' and not(contains(@class, "video-item-title"))'
      ' and not(ancestor::*[contains(@class, "column-content")])'
      ' and not(ancestor::iframe)')

  paragraph_selectors = {
      'cnn': [
          '//div[contains(@class, "cnnContentContainer")]//p[%s]' % cnn_exclude,
          '//div[contains(@class, "l-container")]//p[%s]' % cnn_exclude,
          '//div[contains(@class, "cnn_strycntntlft")]//p[%s]' % cnn_exclude
      ],
      'dailymail': [
          '//div[contains(@class, "article-text")]//p[%s]' % dm_exclude
      ]
  }
  
  title_selectors = [
          '//title'
      ]

  # Highlight exclusions.
  he = (
      'not(contains(@class, "cnnHiliteHeader"))'
      ' and not(descendant::*[starts-with(text(), "Next Article in")])')
  highlight_selectors = {
      'cnn': [
          '//*[contains(@class, "el__storyhighlights__list")]//li[%s]' % he,
          '//*[contains(@class, "cnnStryHghLght")]//li[%s]' % he,
          '//*[@id="cnnHeaderRightCol"]//li[%s]' % he
      ],
      'dailymail': [
          '//h1/following-sibling::ul//li'
      ]
  }
  
  title_exclusions = [ '- CNN.com', '| Mail Online', '| Daily Mail Online' ]

  def ExtractText(selector):
    """Extracts a list of paragraphs given a XPath selector.
    Args:
      selector: A XPath selector to find the paragraphs.
    Returns:
      A list of raw text paragraphs with leading and trailing whitespace.
    """

    xpaths = map(tree.xpath, selector)
    elements = list(chain.from_iterable(xpaths))
    paragraphs = [e.text_content().encode('utf-8') for e in elements]

    # Remove editorial notes, etc.
    if corpus == 'cnn' and len(paragraphs) >= 2 and '(CNN)' in paragraphs[1]:
      paragraphs.pop(0)

    paragraphs = map(str.strip, paragraphs)
    paragraphs = [s for s in paragraphs if s and not str.isspace(s)]

    return paragraphs

  for selector in delete_selectors[corpus]:
    for bad in tree.xpath(selector):
      bad.getparent().remove(bad)

  paragraphs = ExtractText(paragraph_selectors[corpus])
  highlights = ExtractText(highlight_selectors[corpus])
  titles = ExtractText(title_selectors)
  
  title = titles[0] if len(titles) > 0 else ''
  for title_exclusion in title_exclusions:
    title = title.replace(title_exclusion, '')
  title = title.strip()

  content = '\n\n'.join(paragraphs)

  return ParseStory(story.url, content, highlights, title)
  
  
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

  def _Increment(self):
    self.curr += 1
    self._PrintProgress(self.curr)

    if self.curr == self.total:
      print ''

  def _PrintProgress(self, value):
    self.stream.write('\b' * self.last_len)
    pct = 100 * self.curr / float(self.total)
    out = '{:.2f}% [{}/{}]'.format(pct, value, self.total)
    self.last_len = len(out)
    self.stream.write(out)
    self.stream.flush()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Generates summarization pairs from raw html files')
  parser.add_argument('--corpus', choices=['cnn', 'dailymail'], default='cnn')
  args = parser.parse_args()
  
  downloads_dir = '%s/downloads' % args.corpus
  if not os.path.exists(downloads_dir):
    raise ValueError('Download html directory {} not exist.'.format(downloads_dir))
  
  stories_dir = '%s/stories' % args.corpus
  if not os.path.exists(stories_dir):  os.mkdir(downloads_dir)

  StoreMode(args.corpus)
