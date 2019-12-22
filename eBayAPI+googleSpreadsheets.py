"""
code describes connections between google spreadsheets and eBay Trading API
"""

import datetime
import gspread
import gspread_dataframe as gd
import logging
import pandas as pd
import string
import sys
from ebaysdk.trading import Connection as Trading
from logging import handlers
from nltk.corpus import stopwords
from oauth2client.service_account import ServiceAccountCredentials
from time import sleep

stopWords = set(stopwords.words('english'))


def setUpLogging(logFile=None, level=logging.INFO, source=__name__):
	open(logFile, 'w').close()
	
	logger = logging.getLogger(source)
	logger.setLevel(level)
	
	handler = logging.handlers.RotatingFileHandler(
			filename=logFile,
			maxBytes=10485760,
			backupCount=10,
			encoding='utf8')
	
	logger.addHandler(handler)
	
	return logger


logger = setUpLogging('trainingModule.log')


def removePunctuation(value):
	return ''.join([char for char in value if char not in string.punctuation])


def tryToAdd(dictionary, keyName, value):
	try:
		value = value
	
	except:
		value = ''
	
	try:
		
		if keyName in dictionary.keys():
			dictionary[keyName].append(value)
		
		else:
			dictionary[keyName] = [value]
	
	except:
		sys.exit(0)
	
	return data


if __name__ == '__main__':
	
	while True:
		data = {}
		
		try:
			api = Trading(debug=False, config_file=None,
			              appid='appid',  # replace
			              certid='certid',  # replace
			              devid='devid',  # replace
			              token='token'  # replace
			              )
			
			activeList = api.execute('GetMyeBaySelling', {
					'ActiveList' : True,
					'DetailLevel': 'ReturnAll'
			})
			
			if activeList.reply.Ack == 'Success':
				
				for index, item in enumerate(activeList.reply.ActiveList.ItemArray.Item):
					data = tryToAdd(data, 'shop', 'shopName')  # replace
					data = tryToAdd(data, 'title', item.Title)
					
					if 'BuyItNowPrice' in activeList._dict.get('ActiveList').get('ItemArray').get('Item')[index].keys():
						data = tryToAdd(data, 'buyPrice', item.BuyItNowPrice.get('value'))
					
					else:
						data = tryToAdd(data, 'buyPrice', 'auction')
					
					data = tryToAdd(data, 'shippingPrice',
					                item.ShippingDetails.get('ShippingServiceOptions').get('ShippingServiceCost').get(
							                'value'))
					
					data = tryToAdd(data, 'url', item.ListingDetails.get('ViewItemURL'))
					data = tryToAdd(data, 'startTIme', str(item.ListingDetails.get('StartTime')))
					data = tryToAdd(data, 'id', str(item.ItemID))
			
			api = Trading(debug=False, config_file=None,
			              appid='appid',  # replace
			              certid='certid',  # replace
			              devid='devid',  # replace
			              token='token'  # replace
			              )
			
			activeList = api.execute('GetMyeBaySelling', {
					'ActiveList' : True,
					'DetailLevel': 'ReturnAll'
			})
			
			if activeList.reply.Ack == 'Success':
				
				for index, item in enumerate(activeList.reply.ActiveList.ItemArray.Item):
					data = tryToAdd(data, 'shop', 'shopName')  # replace
					data = tryToAdd(data, 'title', item.Title)
					
					if 'BuyItNowPrice' in activeList._dict.get('ActiveList').get('ItemArray').get('Item')[index].keys():
						data = tryToAdd(data, 'buyPrice', item.BuyItNowPrice.get('value'))
					
					else:
						data = tryToAdd(data, 'buyPrice', 'auction')
					
					data = tryToAdd(data, 'shippingPrice',
					                item.ShippingDetails.get('ShippingServiceOptions').get('ShippingServiceCost').get(
							                'value'))
					
					data = tryToAdd(data, 'url', item.ListingDetails.get('ViewItemURL'))
					data = tryToAdd(data, 'startTIme', str(item.ListingDetails.get('StartTime')))
					data = tryToAdd(data, 'id', str(item.ItemID))
		
		except Exception as e:
			logger.info(e)
			
			continue
		
		columnsOrder = ['shop', 'title', 'buyPrice', 'shippingPrice', 'url', 'startTIme', 'id']
		
		data = pd.DataFrame(data)
		data = data[columnsOrder]
		
		scope = ['https://spreadsheets.google.com/feeds']
		
		try:
			credentials = ServiceAccountCredentials.from_json_keyfile_name('/home/eBay-b495e607050c.json', scope)
		
		except:
			credentials = ServiceAccountCredentials.from_json_keyfile_name('home/eBay-b495e607050c.json', scope)
		
		try:
			client = gspread.authorize(credentials)
			sheet = client.open_by_url(
					'url').worksheet(  # replace
					'prices')
		
		except Exception as e:
			logger.info(e)
			sleep(60)
			
			continue
		
		try:
			storedData = pd.DataFrame(sheet.get_all_records())
			
			storedData = storedData.loc[storedData['id'].isin(data['id'])]
			storedData.reset_index(drop=True, inplace=True)
			
			storedData['id'] = [str(value) for value in storedData['id']]
		
		except:
			storedData = pd.DataFrame(columns=columnsOrder)
		
		for index, row in data.iterrows():
			
			if row['id'] not in storedData['id'].values:
				storedData.loc[len(storedData.index)] = row
		
		storedData = storedData.sort_values(['startTime', 'shop', 'id'],
		                                    ascending=[False, True, False])
		
		storedData['buyPrice'] = [int(float(value)) if str(value) != '' else ''
		                          for value in storedData['buyPrice']]
		storedData['shippingPrice'] = [int(float(value)) if str(value) != '' else ''
		                               for value in storedData['shippingPrice']]
		
		columnsOrder = ['title', 'buyPrice', 'shippingPrice',
		                'itemName', 'oem', 'number', 'condition', 'yearStart', 'yearEnd',
		                'manufacturer', 'mainModel', 'setType', 'amount', 'needToFix',
		                'url', 'startTime', 'id', 'shop']
		
		storedData = storedData[columnsOrder]
		
		try:
			gd.set_with_dataframe(sheet, storedData, include_column_header=True, resize=True)
		
		except:
			
			try:
				credentials = ServiceAccountCredentials.from_json_keyfile_name('json', scope)  # replace
			
			except:
				credentials = ServiceAccountCredentials.from_json_keyfile_name('json', scope)  # replace
			
			client = gspread.authorize(credentials)
			sheet = client.open_by_url(
					'url').worksheet(  # replace
					'prices')
			
			client = gspread.authorize(credentials)
		
		print('[pricesDB] last update: ', datetime.datetime.now())
		
		sleep(12 * 60 * 60)
