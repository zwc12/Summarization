# Sequence to sequence model for abstractive text summarization 



## Introduction

The code is based on Google's [seq2seq model](https://github.com/google/seq2seq/tree/master/seq2seq) and [textsum project](https://github.com/tensorflow/models/tree/master/textsum). Aiming to reproduce the results in https://arxiv.org/abs/1602.06023 . The code model is a traditional sequence-to-sequence model with attention, it's customized for text summarization task, and the pointer mechanism is also added to generate words not in the vocab.



## Dataset

Since Gigaword (a corpus commonly used for this task) is not available due to its license, we used the [CNN/Daily Mail Corpus](http://cs.nyu.edu/~kcho/DMQA/)  (which is used for Q&A systems).

This corpus does not provide examples of articles with headlines as demanded, articles with several highlights (usually 3~4 sentences) are provided instead. Although those highlights can be treated as the summarization of the original article, however, we want a shorter (only one sentence and usually no longer than ten words) summary. Fortunately, the original html files are also available for this corpus, which are articles with both headline and highlights. 

With a small python script (modified from the codes provided [here](https://github.com/deepmind/rc-data)), we are able to obtain our dataset as demanded — files containing the original article and its headline. Then we need to process these files to get datafiles expected for Tensorflow model.



## How to run

####Training

For convience, several preprocessed datafiles are provided for training and testing. And how to obtain these datafiles from raw html files are desribed below. You can download the entire dataset of CNN/Daily Mail from here, and follow the instructions to build your own dataset.

to train：

```python
./seq2seq
# training:
python summary.py --mode=train --data_path=bin/train_*.bin

# eval:
python summary.py --mode=eval --data_path=bin/eval_*.bin

# test and write the generated summaries/rouges to files for single time:
python summary.py --mode=decode --data_path=bin/test_*.bin --onetime=True

# test and print the generated summaries along with original article and headline randomly and infinitely:
python summary.py --mode=decode --data_path=bin/test_*.bin --onetime=False
```



####Obtain Data

First you need to download the raw html data and put it in the directory ./textsum/cnn/downloads for CNN or ./textsum/dailymail/downloads. Then run the code:

```python
./textsum
# generate .story files and put them in ./cnn/stories or ./dailymail/stories
python checksum.py --corpus=cnn
python checksum.py --corpus=dailymail

# generate datafiles and vocab file required for tensorflow model
# this will split the data into train_**.bin/eval_**.bin/test_**.bin with splits
# each file contains at most 1000 examples
python checkbin.py --corpus=cnn --outdir=../seq2seq/bin --outfiles=train,eval.test --splits=0.9,0.05,0.05
```



## Note

####Recommended Parameters:

the configuration for the best trained model on Gigaword:

batch_size: 64

bidirectional encoding layer: 4

article length: first 2 sentences, total words within 120.

summary length: total words within 30.

word embedding size: 128

LSTM hidden units: 256

Sampled softmax: 4096

vocabulary size: Most frequent 200k words from dataset's article and summaries.



However, for this dataset, the articles are much longer, you can set the limit of total words to 400. If you use the whole CNN dataset (about 92k examples, 90% for training, 5% for eval and 5% for test), training would take several days, the Daily Mail dataset is even bigger and would take longer time for training.

Due to limitted time and resources, I did not do enough training. With about 90k iterations for about  3 days' training, the model is able to generate good summaries for some articles, for other articles, the result seems not good enough. I will put the results here later if I get better results with more training.