"""
model training via gensim
"""

from __future__ import division, print_function, unicode_literals

import io
import logging
import os
import random
from gensim.models import Word2Vec
from os import path
from preshed.counter import PreshCounter
from spacy.strings import hash_string

try:
	import ujson as json

except ImportError:
	import json

LOGGER = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)

DATASET_DIRECTORY = 'outputData//preprocessedTexts'
OUTPUT_NAME = 'outputData//model.txt'
OUTPUT_MODEL = 'outputData//model'


class Corpus(object):
	
	def __init__(self, directory, min_freq = 10):
		self.directory = directory
		self.counts = PreshCounter()
		self.strings = {}
		self.min_freq = min_freq
	
	def count_doc(self, words):
		doc_counts = PreshCounter()
		doc_strings = {}
		
		for word in words:
			key = hash_string(word)
			doc_counts.inc(key, 1)
			doc_strings[key] = word
		
		n = 0
		for key, count in doc_counts:
			self.counts.inc(key, count)
			corpus_count = self.counts[key]
			
			if corpus_count >= self.min_freq and (corpus_count - count) < self.min_freq:
				self.strings[key] = doc_strings[key]
			
			n += count
		
		return n
	
	def __iter__(self):
		
		for text_loc in trainModel.iterDir(self.directory):
			
			with io.open(text_loc, 'r', encoding = 'utf8') as file_:
				
				sent_strs = list(file_)
				random.shuffle(sent_strs)
				
				for sent_str in sent_strs:
					yield sent_str.split()


class trainModel:
	
	def __init__(self):
		self.corpus = Corpus(DATASET_DIRECTORY)
		
		self.model = Word2Vec(size = 100,
		                      window = 5,
		                      min_count = 10,
		                      workers = 1,
		                      sample = 1e-3,
		                      negative = 10
		                      )
	
	def run(self):
		totalWords = 0
		totalSents = 0
		
		for textNum, textLoc in enumerate(self.iterDir(self.corpus.directory)):
			with io.open(textLoc, 'r', encoding = 'utf8') as fileOpen:
				text = fileOpen.read()
			
			totalSents += text.count('\n')
			totalWords += self.corpus.count_doc(text.split())
			
			LOGGER.info('PROGRESS: at batch #%i, processed %i words, keeping %i word types',
			            textNum, totalWords, len(self.corpus.strings))
		
		self.model.corpus_count = totalSents
		self.model.iter = 10
		self.model.build_vocab(self.corpus)
		
		self.model.train(self.corpus, total_examples = self.model.corpus_count, epochs = self.model.iter)
		
		self.saveModel(self.model)
	
	@classmethod
	def iterDir(cls, location):
		
		for fileName in os.listdir(location):
			
			if path.isdir(path.join(location, fileName)):
				
				for subFile in os.listdir(path.join(location, fileName)):
					yield path.join(location, fileName, subFile)
			
			else:
				yield path.join(location, fileName)
	
	@classmethod
	def saveModel(cls, model):
		model.save(OUTPUT_MODEL)
		model.wv.save_word2vec_format(OUTPUT_NAME, binary = False)


tm = trainModel()
trainModel.run(tm)
