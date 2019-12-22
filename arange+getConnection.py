"""
arange: connection detection function between tokens
"""

import json
from arango import ArangoClient

DATABASE_NAME = 'resulting'


class distance:
	
	def __init__(self):
		self.db = self.getArangoConnection()
	
	def getDistance(self, token1, token2):
		id1 = self.getCursor(self.db, token1)
		id2 = self.getCursor(self.db, token2)
		
		###
		
		cursorType1 = self.getCursorType(self.db, id1, id2)
		cursorType2 = self.getCursorType(self.db, id2, id1)
		
		if len(cursorType1) > 1 or len(cursorType2) > 1:
			return 'multiple connections for ' + str(token1) + ' and ' + str(token2)
		
		if len(cursorType1) == 0 and len(cursorType2) == 1:
			return cursorType2[0]
		
		elif len(cursorType1) == 1 and len(cursorType2) == 0:
			return cursorType1[0]
		
		elif len(cursorType1) == 0 and len(cursorType2) == 0:
			return 'no direct connection'
		
		elif len(cursorType1) == 1 and len(cursorType2) == 1:
			return [cursorType1[0], cursorType2[0]]
		
		else:
			return 'results look very strange for ' + str(token1) + ' and ' + str(token2)
		
		return [value for value in cursor]
	
	@classmethod
	def getArangoConnection(cls):
		client = ArangoClient()
		db = client.db(DATABASE_NAME)
		
		return db
	
	@classmethod
	def getCursor(cls, db, token):
		cursorId = db.aql.execute(
				'for item in token ' + \
				'filter item.name == "' + str(token) + '" ' + \
				'return item')
		
		id = [value['_id'] for value in cursorId]
		
		if len(id) > 1 or len(id) == 0:
			print('error in token ' + str(token))
			
			import sys
			sys.exit(0)
		
		else:
			return id[0]
	
	@classmethod
	def getCursorType(cls, db, id1, id2):
		cursorType = db.aql.execute(
				'for token in token2token ' + \
				'filter token._from == "' + str(id1) + '" ' + \
				'filter token._to == "' + str(id2) + '" ' + \
				'return [token.similarity, token.type]'
				)
		
		return [value for value in cursorType]


gd = distance()
print(distance.getDistance(gd, 'decreased', 'increased'))
