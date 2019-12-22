"""
custom text processing: spacy + sense2Vec
"""

import io
import pandas as pd
import re
import spacy
import string

DATASET_DIRECTORY = 'outputData//1.1.collectedData.csv'
OUTPUT_NAME = 'outputData//2.1.wordLemma.csv'
OUTPUT_DIRECTORY = 'outputData//preprocessedTexts'

PUNCTUATION = [value for value in string.punctuation if value not in ['%']]
MONTHS = ['january', 'february', 'march', 'april', 'may', 'june',
          'july', 'august', 'september', 'october', 'november', 'december']
YEARS = [value for value in range(1980, 2021)]

nlp = spacy.load('en')


class textPreprocessing:
	
	def __init__(self):
		pass
	
	def run(self):
		data = pd.read_csv(DATASET_DIRECTORY, index_col=0)
		
		bagOfWords = []
		for index, text in enumerate(data['text']):
			
			tempArraySense = []
			tempArrayWord = []
			tempArrayLemma = []
			
			text = self.stripText(text)
			
			###
			# text = text[: 10]# remove
			###
			
			for item in text:
				tokens = nlp(unicode(item, 'utf-8'))
				
				for sentence in tokens.sents:
					sentence = ''.join(token.string for token in sentence).strip()
					
					if len(sentence) <= 1:
						continue
					
					if sentence[-1] in ['!', ',', '.', ':', ';']:
						sentence = sentence[:-1]
					
					itemSense, itemWord, itemLemma = self.transformSpacy(sentence)
					
					if itemSense != '':
						tempArraySense.append(itemSense)
					
					if itemWord != '':
						tempArrayWord.append(itemWord)
					
					if itemLemma != '':
						tempArrayLemma.extend(itemLemma.split('\n'))
			
			bagOfWords.extend(tempArrayLemma)
			
			self.writeText(index, '[sense]', tempArraySense)
			self.writeText(index, '[word]', tempArrayWord)
			
			'''if index >= 10:# remove
				break'''
		
		bagOfWordsClean = self.bagOfWordsConversion(bagOfWords)
		
		self.writeDataSet(bagOfWordsClean, OUTPUT_NAME)
	
	@classmethod
	def stripText(cls, value):
		value = str(value).lower()
		
		value = value.replace('(', ' \n ')
		value = value.replace(')', ' \n ')
		value = value.replace('<', ' \n ')
		value = value.replace('>', ' \n ')
		value = value.replace('[', ' \n ')
		value = value.replace(']', ' \n ')
		value = value.replace('{', ' \n ')
		value = value.replace('}', ' \n ')
		value = value.replace('"', ' \n ')
		
		value = value.replace('\\\'', ' QUOTE ')
		value = value.replace('\'', ' QUOTE ')
		
		value = cls.specialSymbolsType1(value)  # not necessary as I see but let it be
		value = cls.specialSymbolsType2(value)
		value = cls.specialSymbolsType3(value)
		
		# value = value.replace('\'', ' ')
		# value = value.replace('-', ' - ')#. between letters
		
		value = value.replace('\\t?', ' ')
		value = value.replace('\t?', ' ')
		value = value.replace('\\t', ' ')
		value = value.replace('\t', ' ')
		value = value.replace('/', ' ')
		
		value = value.replace('\\', ' ')
		
		while '  ' in value:
			value = value.replace('  ', ' ')
		
		value = value.split('\n')
		
		###
		
		for index, item in enumerate(value):
			item = item.split(' ')
			
			for subIndex, subItem in enumerate(item):
				
				if subItem == 'QUOTE':
					item[subIndex] = subItem
					
					continue
				
				lenValueM1, lenValueConvertedNumericM1, \
				lenValueConvertedPunctuationM1, lenValueConvertedAlphasM1, \
				lenValueConvertedSpacesM1, lenValueConvertedAlphaNumericM1 = cls.countFeatures(subItem[:-1])
				
				lenValue, lenValueConvertedNumeric, \
				lenValueConvertedPunctuation, lenValueConvertedAlphas, \
				lenValueConvertedSpaces, lenValueConvertedAlphaNumeric = cls.countFeatures(subItem)
				
				if len(subItem) >= 2 and cls.defineWord(lenValueConvertedNumericM1,
				                                        lenValueConvertedPunctuationM1,
				                                        lenValueConvertedAlphasM1):
					
					while subItem[-1] in PUNCTUATION and subItem[-1] not in ['.', ',', ':', ';', '!', '?']:
						subItem = subItem[:-1]
						
						if len(subItem) == 0:
							break
					
					for char in PUNCTUATION:
						
						if char in ['-', '/', '.', ',']:
							continue
						
						if char in subItem[: -1]:
							subItem = subItem[: -1].replace(char, '') + subItem[-1]
				
				if len(subItem) == 0 or lenValueConvertedAlphaNumeric == 0:
					item[subIndex] = ''
					
					continue
				
				if subItem[-1] not in PUNCTUATION:
					
					if subItem in MONTHS:
						subItem = 'DATE'
					
					elif '%' in subItem:
						subItem = 'PERCENT'
					
					elif any(str(year) in subItem for year in YEARS):
						subItem = 'DATE'
					
					elif cls.defineNumber(lenValueConvertedNumeric,
					                      lenValueConvertedPunctuation,
					                      lenValueConvertedAlphas):
						subItem = 'NUMBER'
				
				else:
					
					if subItem[:-1] in MONTHS:
						subItem = 'DATE' + subItem[-1]
					
					elif '%' in subItem[:-1]:
						subItem = 'PERCENT' + subItem[-1]
					
					elif any(str(year) in subItem[:-1] for year in YEARS):
						subItem = 'DATE' + subItem[-1]
					
					elif cls.defineNumber(lenValueConvertedNumericM1,
					                      lenValueConvertedPunctuationM1,
					                      lenValueConvertedAlphasM1):
						subItem = 'NUMBER' + subItem[-1]
				
				item[subIndex] = subItem
			
			value[index] = ' '.join(item).strip()
		
		newValue = []
		for item in value:
			
			lenItem, lenItemConvertedNumeric, \
			lenItemConvertedPunctuation, lenItemConvertedAlphas, \
			lenItemConvertedSpaces, lenItemConvertedAlphaNumeric = cls.countFeatures(item)
			
			if lenItem != lenItemConvertedPunctuation + lenItemConvertedSpaces:
				newValue.append(item)
		
		return newValue
	
	@classmethod
	def specialSymbolsType1(cls, value):
		
		newValue = ''
		for index, chr in enumerate(value):
			
			if chr == '\'':
				
				if value[index - 1] != ' ' and value[index + 1] != ' ':
					newValue += value[index]
				
				elif value[index - 1] != ' ' and value[index + 1] == ' ':
					newValue += value[index]
				
				elif value[index - 1] == ' ' and value[index + 1] != ' ':
					newValue += '\n'
				
				elif value[index - 1] == ' ' and value[index + 1] == ' ':
					newValue += '\n'
			
			else:
				newValue += value[index]
		
		return newValue
	
	@classmethod
	def specialSymbolsType2(cls, value):
		
		newValue = ''
		for index, chr in enumerate(value):
			
			if chr == '-':
				
				if value[index - 1] != ' ' and value[index + 1] != ' ':
					newValue += value[index]
				
				elif value[index - 1] != ' ' and value[index + 1] == ' ':
					newValue += ' - '
				
				elif value[index - 1] == ' ' and value[index + 1] != ' ':
					newValue += ' - '
				
				elif value[index - 1] == ' ' and value[index + 1] == ' ':
					newValue += value[index]
			
			else:
				newValue += value[index]
		
		return newValue
	
	@classmethod
	def specialSymbolsType3(cls, value):
		
		newValue = ''
		for index, chr in enumerate(value[:-1]):
			
			if chr == '.':
				
				if value[index - 1].isalpha() and value[index + 1].isalpha():
					newValue += ''
				
				elif value[index - 1].isalpha() and value[index + 1] == ' ':
					newValue += value[index]
				
				elif value[index - 1] == ' ' and value[index + 1].isalpha():
					newValue += value[index]
				
				elif value[index - 1] == ' ' and value[index + 1] == ' ':
					newValue += value[index]
			
			else:
				newValue += value[index]
		
		newValue += value[-1]
		
		return newValue
	
	@classmethod
	def countFeatures(cls, value):
		valueConvertedNumeric = [char for char in unicode(value, 'utf-8') if char.isnumeric()]
		valueConvertedPunctuation = [char for char in value if char in string.punctuation]
		valueConvertedAlphas = [char for char in value if char.isalpha()]
		valueConvertedSpaces = [char for char in value if char.isspace()]
		valueConvertedAlphaNumeric = [char for char in value if char.isalnum()]
		
		return len(value), len(valueConvertedNumeric), \
		       len(valueConvertedPunctuation), len(valueConvertedAlphas), \
		       len(valueConvertedSpaces), len(valueConvertedAlphaNumeric)
	
	@classmethod
	def defineWord(cls, lenValueConvertedNumeric, lenValueConvertedPunctuation, lenValueConvertedAlphas):
		
		if lenValueConvertedNumeric >= 0 and lenValueConvertedPunctuation >= 0 and lenValueConvertedAlphas > 0:
			return True
		
		else:
			return False
	
	@classmethod
	def defineNumber(cls, lenValueConvertedNumeric, lenValueConvertedPunctuation, lenValueConvertedAlphas):
		
		if lenValueConvertedNumeric > 0 and lenValueConvertedPunctuation >= 0 and lenValueConvertedAlphas == 0:
			return True
		
		else:
			return False
	
	@classmethod
	def transformSpacy(cls, item):
		value = nlp(item)
		
		for ent in value.ents:
			ent.merge(tag=ent.root.tag_, lemma=ent.text, label=ent.label_)
		
		wordReturn = cls.returnWord(value)
		lemmaReturn = cls.returnLemma(value)
		senseReturn = cls.returnSense(value)
		
		return senseReturn, wordReturn, lemmaReturn
	
	@classmethod
	def returnWord(cls, value):
		
		string = ''
		for sentence in value.sents:
			
			if sentence.text.strip():
				string = ' '.join(cls.representWord(word)
				                  for word in sentence
				                  if not word.is_space)
		
		return string.strip()
	
	@classmethod
	def representWord(cls, word):
		
		if 'DATE' in word.text:
			return word.text + '|DATE'
		
		elif 'PERCENT' in word.text:
			return word.text + '|PERCENT'
		
		elif 'NUMBER' in word.text:
			return word.text + '|NUMBER'
		
		elif 'QUOTE' in word.text:
			return word.text + '|QUOTE'
		
		else:
			text = re.sub(r'\s', '_', word.text)
			tag = word.pos_
			
			if not tag:
				tag = '?'
			
			return text + '|' + tag
	
	@classmethod
	def returnLemma(cls, value):
		
		string = ''
		for sentence in value.sents:
			
			if sentence.text.strip():
				string = '\n'.join(cls.representWordLemma(word) for word in sentence
				                   if not word.is_space and
				                   not any(item in word.text for item in ['DATE', 'PERCENT', 'NUMBER', 'QUOTE']))
		
		return string.strip()
	
	@classmethod
	def representWordLemma(cls, word):
		text = re.sub(r'\s', '_', word.text)
		lemma = word.lemma_
		
		if lemma != '-PRON-':
			return text + '|' + lemma
		
		else:
			return ''
	
	@classmethod
	def returnSense(cls, value):
		
		for np in value.noun_chunks:
			
			while len(np) > 1 and np[0].dep_ not in ('advmod', 'amod', 'compound'):
				np = np[1:]
			
			if not any(item in np.text for item in ['DATE', 'PERCENT', 'NUMBER', 'QUOTE']):
				np.merge(tag=np.root.tag_, lemma=np.text, label=np.root.ent_type_)
		
		return cls.returnWord(value)
	
	@classmethod
	def writeText(cls, index, tag, array):
		
		with io.open(OUTPUT_DIRECTORY + '//' + str(index) + tag, 'w', encoding='utf8') as fileOpen:
			for item in array:
				fileOpen.write(item + '\n')
	
	@classmethod
	def bagOfWordsConversion(cls, bagOfWords):
		
		bagOfWordsClean = []
		for item in sorted(list(set(bagOfWords))):
			
			if item == '':
				continue
			
			tempItem = item.split('|')
			
			if tempItem[0] != tempItem[1] and '_' not in tempItem[0]:
				bagOfWordsClean.append((tempItem[0], tempItem[1]))
		
		return bagOfWordsClean
	
	@classmethod
	def createDataFrame(cls, array):
		dataSet = pd.DataFrame()
		dataSet['word'] = [value[0] for value in array]
		dataSet['lemma'] = [value[1] for value in array]
		
		return dataSet
	
	@classmethod
	def writeDataSet(cls, array, fileName):
		cls.createDataFrame(array).to_csv(fileName)


tp = textPreprocessing()
textPreprocessing.run(tp)
