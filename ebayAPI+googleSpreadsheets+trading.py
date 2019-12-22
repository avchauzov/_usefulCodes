"""
code describes connections between google spreadsheets and eBay Trading API: ads update option
"""

import cgi
import datetime
import gspread
import logging
import numpy as np
import operator as op
import pandas as pd
import random
import re
import string
from copy import deepcopy
from ebaysdk.trading import Connection as Trading
from logging import handlers
from nltk.corpus import stopwords
from oauth2client.service_account import ServiceAccountCredentials
from time import sleep

stopWords = set(stopwords.words('english'))
import functools
import itertools


def setUpLogging(logFile = None, level = logging.INFO, source = __name__):
	open(logFile, 'w').close()
	
	logger = logging.getLogger(source)
	logger.setLevel(level)
	
	handler = logging.handlers.RotatingFileHandler(
			filename = logFile,
			maxBytes = 10485760,
			backupCount = 10,
			encoding = 'utf8')
	
	logger.addHandler(handler)
	
	return logger


logger = setUpLogging('trainingModule.log')


def ncr(n, r):
	r = min(r, n - r)
	
	coeff1 = functools.reduce(op.mul, range(n, n - r, -1), 1)
	coeff2 = functools.reduce(op.mul, range(1, r + 1), 1)
	
	return coeff1 // coeff2


def removePunctuation(value):
	return ''.join([char for char in value if char not in string.punctuation])


def removeSpaces(value):
	while '  ' in value:
		value = value.replace('  ', ' ')
	
	return value


def removeDuplicates(value):
	seen = set()
	seenAdd = seen.add
	
	return [item for item in value if not (item in seen or seenAdd(item))]


pattern = re.compile('[0-9][0-9] [0-9][0-9] [0-9] [0-9][0-9][0-9] [0-9][0-9][0-9]$')

if __name__ == '__main__':
	
	while True:
		scope = ['https://spreadsheets.google.com/feeds']
		
		try:
			credentials = ServiceAccountCredentials.from_json_keyfile_name('json', scope)  # replace
		
		except:
			credentials = ServiceAccountCredentials.from_json_keyfile_name('json', scope)  # replace
		
		try:
			client = gspread.authorize(credentials)
			sheet = client.open_by_url(
					'url').worksheet(  # replace
					'prices')
			settings = client.open_by_url(
					'url').worksheet(  # replace
					'settings')
		
		except:
			print('[mainAlgo] gs delay: ', datetime.datetime.now())
			sleep(60)
			
			continue
		
		settings = pd.DataFrame(settings.get_all_records(head = 0))
		settings.columns = ['value', 'name']
		
		oemLevel = float(settings['value'][0])
		conditionLevel = float(settings['value'][1])
		manufacturerLevel = float(settings['value'][2])
		mainModelLevel = float(settings['value'][3])
		modelsMaximum = int(settings['value'][4])
		setLevel = float(settings['value'][5])
		numberLevel = float(settings['value'][6])
		numberMax = float(settings['value'][7])
		numberType2 = float(settings['value'][8])
		numberType4 = float(settings['value'][9])
		
		storedData = pd.DataFrame(sheet.get_all_records())
		storedData['id'] = [str(value) for value in storedData['id']]
		
		for shopName in ['shopName',  # replace
		                 'shopName']:  # replace
			tempStoredData = storedData[storedData['shop'] == shopName]
			tempStoredData = tempStoredData.loc[~tempStoredData['buyPrice'].isin(['auction'])]
			
			oemGroup = [[subValue for subValue in str(value).split(';') if subValue != 'none'] for value in
			            tempStoredData['oem']]
			conditionGroup = [[subValue for subValue in str(value).split(';') if subValue != 'none'] for value in
			                  tempStoredData['condition']]
			manufacturerGroup = [[subValue for subValue in str(value).split(';') if subValue != 'none'] for value in
			                     tempStoredData['manufacturer']]
			mainModelGroup = [[subValue for subValue in str(value).split(';') if subValue != 'none'] for value in
			                  tempStoredData['mainModel']]
			setGroup = [[subValue for subValue in str(value).split(';') if subValue != 'none'] for value in
			            tempStoredData['setType']]
			numberGroup = [[subValue for subValue in str(value).split(';') if subValue != 'none'] for value in
			               tempStoredData['number']]
			
			if shopName == 'shopName':  # replace
				api = Trading(debug = False, config_file = None,
				              appid = 'appid',  # replace
				              certid = 'certid',  # replace
				              devid = 'devid',  # replace
				              token = 'token')  # replace
			
			else:
				api = Trading(debug = False, config_file = None,
				              appid = 'appid',
				              certid = 'certid',
				              devid = 'devid',
				              token = 'token')
			
			for id, buyPrice, shippingPrice, \
			    title, oem, condition, yearStart, yearEnd, \
			    manufacturer, mainModel, setType, number, \
			    itemName in zip(tempStoredData['id'],
			                    tempStoredData['buyPrice'],
			                    tempStoredData['shippingPrice'],
			                    tempStoredData['title'],
			                    oemGroup,
			                    conditionGroup,
			                    tempStoredData['yearStart'],
			                    tempStoredData['yearEnd'],
			                    manufacturerGroup,
			                    mainModelGroup,
			                    setGroup,
			                    numberGroup,
			                    tempStoredData['itemName']):
				
				if str(buyPrice) == '':
					continue
				
				#
				oem = [value.upper() if len(value) <= 3 else value.title() for value in oem]
				condition = [value.title() for value in condition]
				manufacturer = [value.upper() for value in manufacturer]
				mainModel = [value.upper() for value in mainModel]
				setType = [value.title() for value in setType]
				
				tempNumber = deepcopy(number)
				for item in number:
					tempNumber.append(''.join(item.split()))
					
					if pattern.match(str(item)):
						tempNumber.append(''.join(item[6:].split()))
						tempNumber.append(item[6:])
				
				tempNumber = sorted(list(set(tempNumber)))
				
				newTitle = [itemName]
				itemTitleFull = ''
				
				listTitle = []
				
				listsUnion = [condition, oem, manufacturer]
				for item in list(itertools.product(*listsUnion)):
					tempValue = deepcopy(newTitle)
					
					if random.uniform(0.0, 1.0) >= oemLevel:
						tempValue.append(item[1])
					
					if random.uniform(0.0, 1.0) >= manufacturerLevel:
						tempValue.insert(0, item[2])
					
					if random.uniform(0.0, 1.0) >= conditionLevel or item[0].lower() == 'new':
						
						if item[0].lower() != 'new':
							tempValue.insert(0, item[0] + ' Condition')
						
						else:
							tempValue.insert(0, str(item[0]).upper())
					
					tempValue = [removeSpaces(value).strip() for value in tempValue if value.strip() != '']
					
					if random.uniform(0.0, 1.0) >= numberLevel and len(number) > 0:
						totalNumbers = min(random.randint(1, numberMax), len(number))
						
						numberValue = number[0]
						
						if random.uniform(0.0, 1.0) >= numberType2:
							tempValue.append(''.join(numberValue.split()))
						
						elif random.uniform(0.0, 1.0) >= numberType4:
							
							if pattern.match(str(numberValue)):
								tempValue.append(''.join(numberValue[6:].split()))
							
							else:
								tempValue.append('-'.join(numberValue.split()))
							
							if len(number) > 1:
								
								for index in range(totalNumbers):
									tempPValue = np.random.geometric(0.5, len(number[1:]))
									tempPValue = tempPValue / sum(tempPValue)
									
									numberValue = np.random.choice(number[1:], p = tempPValue)
									
									if random.uniform(0.0, 1.0) >= numberType2:
										tempValue.append(''.join(numberValue.split()))
									
									elif random.uniform(0.0, 1.0) >= numberType4:
										
										if pattern.match(str(numberValue)):
											tempValue.append(''.join(numberValue[6:].split()))
										
										else:
											tempValue.append('-'.join(numberValue.split()))
					
					tempValue = removeDuplicates(tempValue)
					
					tempLength = len(' '.join(tempValue))
					if tempLength >= 80:
						continue
					
					check = False
					if random.uniform(0.0, 1.0) >= mainModelLevel and len(mainModel) > 0:
						totalModels = min(modelsMaximum, len(mainModel))
						
						for length in range(1, totalModels + 1):
							
							tempCount = ncr(len(mainModel), length)
							if tempCount >= 25000:
								continue
							
							check = False
							tempScore = 0
							for item in list(set(itertools.combinations(mainModel, length))):
								
								if len(' '.join(item)) + tempLength >= 100:
									continue
								
								item = list(item)
								tempScore = 0
								
								item = removeDuplicates(' '.join(item).split())
								
								if len(' '.join(item)) + tempLength >= 85:
									continue
								
								tempValueCategory = deepcopy(tempValue)
								tempValueCategory.append(' '.join(item))
								
								tempValueCategory = ' '.join(tempValueCategory)
								tempValueCategory = [value for value in tempValueCategory.split() if value != '']
								tempValueCategory = ' '.join(tempValueCategory).strip()
								
								if len(tempValueCategory) <= 80:
									listTitle.append((tempValueCategory, tempScore))
									check = True
							
							if not check:
								break
					
					else:
						tempValueCategory = deepcopy(tempValue)
						
						tempValueCategory = ' '.join(tempValueCategory)
						tempValueCategory = [value for value in tempValueCategory.split() if value != '']
						tempValueCategory = ' '.join(tempValueCategory).strip()
						
						if len(tempValueCategory) <= 80:
							listTitle.append((tempValueCategory, 0))
					
					if not check:
						tempValueCategory = deepcopy(tempValue)
						
						tempValueCategory = ' '.join(tempValueCategory)
						tempValueCategory = [value for value in tempValueCategory.split() if value != '']
						tempValueCategory = ' '.join(tempValueCategory).strip()
						
						if len(tempValueCategory) <= 80:
							listTitle.append((tempValueCategory, 0))
						
						continue
				
				if len(listTitle) == 0:
					newTitle = title
				
				else:
					listTitle = sorted(list(set(listTitle)))
					
					tempWeights = [value[1] for value in listTitle if value[1] != 0]
					
					if len(tempWeights) != 0:
						percentile = np.percentile(tempWeights, 50)
						listTitle = [value[0] for value in listTitle if value[1] == 0 or value[1] >= percentile]
					
					else:
						listTitle = [value[0] for value in listTitle]
					
					listTitleLengths = [len(value.split()) for value in listTitle]
					listTitle = zip(listTitle, listTitleLengths)
					percentile = np.percentile(listTitleLengths, 75)
					listTitle = [value[0] for value in listTitle if value[1] >= percentile]
					
					tempList = [[len(subValue) for subValue in value.split()] for value in listTitle]
					listTitleLengths = [np.mean(value) for value in tempList]
					listTitle = zip(listTitle, listTitleLengths)
					percentile = np.percentile(listTitleLengths, 25)
					listTitle = [value[0] for value in listTitle if value[1] <= percentile]
					
					listTitleLengths = [len(value) for value in listTitle]
					listTitle = zip(listTitle, listTitleLengths)
					percentile = np.percentile(listTitleLengths, 75)
					listTitle = [value[0] for value in listTitle if value[1] >= percentile]
					
					newTitle = random.choice(listTitle)
				
				#
				try:
					data = api.execute('GetItem', {
							'ItemID'                      : id, 'DetailLevel': 'ReturnAll',
							'IncludeItemCompatibilityList': True,
							'IncludeItemSpecifics'        : True
							})
				
				except:
					continue
				
				pictureList = data._dict.get('Item').get('PictureDetails').get('PictureURL')
				
				if isinstance(pictureList, list):
					
					if len(pictureList) > 1:
						random.shuffle(pictureList)
				
				else:
					pictureListTemp = []
					pictureListTemp.append(pictureList)
					
					pictureList = deepcopy(pictureListTemp)
				
				descriptionText = ''
				try:
					description = {}
					
					if itemName != '':
						description['<b>Item type: </b>'] = itemName
					
					if len(oem) > 0:
						
						if oem != ['Handmade']:
							description['<b>OEM Type: </b>'] = random.choice(
									oem) + '. Purchased from the official dealer.'
						
						else:
							description['<b>Type: </b>'] = random.choice(oem)
					
					if len(condition) > 0:
						
						if len(condition) == 1:
							condition = random.choice(condition)
							description['<b>Condition: </b>'] = condition
							data._dict['Item']['ConditionDescription'] = condition
						
						else:
							conditionValue = random.choice(condition)
							description['<b>Condition: </b>'] = condition + '. Tested and guaranteed.'
							data._dict['Item']['ConditionDescription'] = condition + '. Tested and guaranteed.'
					
					if str(yearStart) not in ['', 'none'] and str(yearEnd) not in ['', 'none']:
						description['<b>Years: </b>'] = str(yearStart) + ' - ' + str(yearEnd)
					
					elif str(yearStart) not in ['', 'none'] and str(yearEnd) in ['', 'none']:
						description['<b>Years: </b>'] = str(yearStart) + ' - ...'
					
					if len(manufacturer) > 0:
						description['<b>Manufacturer: </b>'] = str(random.choice(manufacturer)).upper()
					
					if len(setType) > 0:
						description['<b>Set Type: </b>'] = random.choice(setType)
					
					if all(value not in ['manufacturer1', 'manufacturer2'] for value in manufacturer):  # replace
						description[
							'<b>Comments: </b>'] = '<p>Please send us your car\'s VIN and we will check the compatibility!</p>'
						description['<b>Shipping: </b>'] = '<p>Shipping withing 24 hours!</p>' \
						                                   '<p>Usually, we send with original postal service (tracking number is provided).</p>' \
						                                   '<p>But for big items (or by request) we use EMS. ' \
						                                   'Shipping to USA with EMS may take even less than 2 weeks!' \
						                                   '<p>Packaging is perfect - no complaints ever!</p>'
						description['<b>Returns: </b>'] = '<p>14 days to return or replace the item!</p>'
					
					if len(mainModel) > 0:
						description['<b>Models: </b>'] = '<ul><li>' + '<li>'.join(
								[str(value) + '</li>' for value in mainModel]) + '</ul>'
					
					if len(number) > 0:
						description['<b>Numbers: </b>'] = '<ul><li>' + '<li>'.join(
								[str(value) + '</li>' for value in number]) + '</ul>'
					
					tempColumns = ['<b>Item type: </b>', '<b>OEM Type: </b>', '<b>Condition: </b>',
					               '<b>Years: </b>', '<b>Manufacturer: </b>',
					               '<b>Set Type: </b>', '<b>Comments: </b>',
					               '<b>Shipping: </b>', '<b>Returns: </b>', '<b>Models: </b>',
					               '<b>Numbers: </b>']
					
					tempColumns = [value for value in tempColumns if value in description.keys()]
					
					descriptionText = '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
					descriptionText += '<p style=""><font face="Georgia" style="" size="4">'
					
					for key in tempColumns:
						value = description.get(key)
						
						if key not in ['<b>Models: </b>', '<b>Numbers: </b>']:
							descriptionText += '<p>' + key + value + '</p>'
						
						else:
							descriptionText += '<p>' + key + '</p>' + value
					
					descriptionText += '</meta>'
					
					descriptionText = cgi.escape(descriptionText)
				
				except:
					descriptionText = data._dict.get('Item').get('Description')
					descriptionText = cgi.escape(descriptionText)
				
				#
				'''mainNumberList = []
				otherNumberList = []

				numberExtended = []
				numberExtended.extend(number)
				for numberValue in number:
					numberExtended.append('-'.join(numberValue.split()))
					numberExtended.append(''.join(numberValue.split()))

					mainNumberList.append(''.join(numberValue.split()))
					otherNumberList.append(''.join(numberValue.split()))

					if pattern.match(str(numberValue)):
						numberExtended.append(' '.join(numberValue[6:].split()))
						numberExtended.append('-'.join(numberValue[6:].split()))

					else:
						numberExtended.append('-'.join(numberValue.split()))

					if pattern.match(str(numberValue)):
						numberExtended.append(''.join(numberValue[6:].split()))
						mainNumberList.append(''.join(numberValue[6:].split()))
						otherNumberList.append(''.join(numberValue[6:].split()))

					else:
						numberExtended.append('-'.join(numberValue.split()))

				numberExtended = list(set(numberExtended))
				random.shuffle(numberExtended)

				if len(mainNumberList) > 0:
					mainNumber = random.choice(mainNumberList)
					otherNumberList = [value for value in otherNumberList if value != mainNumber]

				else:
					mainNumber = random.choice(numberExtended)

				numberExtended = [value for value in numberExtended if value != mainNumber]

				if len(otherNumberList) > 0:
					otherNumber = random.choice(otherNumberList)
					otherNumberList = [value for value in otherNumberList if value != mainNumber]

				else:
					otherNumber = random.choice(numberExtended)

				numberExtended = [value for value in numberExtended if value != mainNumber]

				supersededNumber = []
				interchangeNumber = []
				if len(numberExtended) <= 12:
					supersededNumber = [str(value) for value in numberExtended]

				elif len(numberExtended) > 12:
					supersededNumber = [str(value) for value in numberExtended[: 12]]
					interchangeNumber = [str(value) for value in numberExtended[12: 24]]

				#

				brandIndex = None
				for index, item in enumerate(data._dict.get('Item').get('ItemSpecifics').get('NameValueList')):

					if item.get('Name') == 'Superseded Part Number':

						try:
							data._dict['Item']['ItemSpecifics']['NameValueList'][index]['Value'] = supersededNumber

						except:
							pass

					if item.get('Name') == 'Interchange Part Number':

						try:
							data._dict['Item']['ItemSpecifics']['NameValueList'][index]['Value'] = interchangeNumber

						except:
							pass

					if item.get('Name') == 'Modified Item':

						try:
							data._dict['Item']['ItemSpecifics']['NameValueList'][index]['Value'] = 'No'

						except:
							pass

					if item.get('Name') == 'Custom Bundle':

						try:
							data._dict['Item']['ItemSpecifics']['NameValueList'][index]['Value'] = 'No'

						except:
							pass

					if item.get('Name') == 'Manufacturer Part Number':

						try:
							data._dict['Item']['ItemSpecifics']['NameValueList'][index]['Value'] = str(mainNumber)

						except:
							pass

					if item.get('Name') == 'Other Part Number':

						try:
							data._dict['Item']['ItemSpecifics']['NameValueList'][index]['Value'] = str(otherNumber)

						except:
							pass

					if item.get('Name') == 'Brand':

						try:
							data._dict['Item']['ItemSpecifics']['NameValueList'][index]['Value'] = random.choice(
									manufacturerValue)

						except:
							pass

					if item.get('Name') == 'Fitment Type':

						try:
							data._dict['Item']['ItemSpecifics']['NameValueList'][index]['Value'] = 'Direct Replacement'

						except:
							pass

					if item.get('Name') == 'Non-Domestic Product':

						try:
							data._dict['Item']['ItemSpecifics']['NameValueList'][index]['Value'] = 'No'

						except:
							pass

					if item.get('Name') == 'Warranty':

						try:
							data._dict['Item']['ItemSpecifics']['NameValueList'][index]['Value'] = '14 days to return!'

						except:
							pass

					if item.get('Name') == 'Mounting Hardware Included':

						try:
							data._dict['Item']['ItemSpecifics']['NameValueList'][index]['Value'] = 'Optional'

						except:
							pass

					if item.get('Name') == 'UPC':

						try:
							data._dict['Item']['ItemSpecifics']['NameValueList'][index]['Value'] = 'Does not apply'

						except:
							pass'''
				
				try:
					api.execute('ReviseFixedPriceItem', {
							'Item': {
									'ItemID'          : id,
									'StartPrice'      : buyPrice,
									'BuyItNowPrice'   : buyPrice,
									'PictureDetails'  : {
											'GalleryURL' : pictureList[0],
											'PictureURL' : pictureList,
											'GalleryType': 'Gallery'
											},
									'Title'           : newTitle,
									'BestOfferDetails': {
											'BestOfferEnabled': True
											},
									# 'Description'          : descriptionText,
									# 'DescriptionReviseMode': 'Replace'
									}
							})
				
				except ConnectionError as e:
					print('[mainAlgo] ', e)
					logger.info(' > '.join([id, e]))
				
				except Exception as e:
					print(e)
					print('[mainAlgo] itemID ', id, ' from ', shopName, ' is passed')
					
					print(len(newTitle), newTitle)
					print(descriptionText)
					
					logger.info(' > '.join([id, str(e)]))
				
				except:
					logger.info(' > '.join([id, 'error']))
					continue
		
		print('[mainAlgo] last update: ', datetime.datetime.now())
		sleep(12 * 60 * 60)
