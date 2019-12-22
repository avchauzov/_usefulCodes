"""
arange: common detection function between tokens
"""

import json
import numpy as np
from arango import ArangoClient
from copy import deepcopy

DATABASE_NAME = 'resulting'


class getCommons:
	
	def __init__(self):
		self.db = self.getArangoConnection()
	
	def getCommonPoints(self, token1, token2, maxDepth):
		id1 = self.getCursor(self.db, token1)
		id2 = self.getCursor(self.db, token2)
		
		destinations1 = self.getDestinations(self.db, id1, maxDepth)
		destinations2 = self.getDestinations(self.db, id2, maxDepth)
		
		commonPoints = [value for value in destinations1 if value in destinations2]
		commonPoints = [value for value in commonPoints if value not in [id1, id2]]
		return sorted(list(set(commonPoints)))
	
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
	def getDestinations(cls, db, id, maxDepth):
		destinations = {}
		idToCheck = [id]
		depth = 1
		
		while True:
			idToCheckNew = []
			newAdded = 0
			
			for id in idToCheck:
				tempCursor = db.aql.execute(
						'for token in token2token ' + \
						'filter token._from == "' + str(id) + '" ' + \
						'return [token._to, token.similarity]'
				)
				
				tempCursor = [(value[0], value[1]) for value in tempCursor]
				
				for value in tempCursor:
					
					if value[0] not in destinations.keys():
						destinations[value[0]] = (depth, value[1])
						idToCheckNew.append(value[0])
						
						newAdded += 1
			
			if newAdded == 0 or depth == maxDepth:
				break
			
			depth += 1
			idToCheck = deepcopy(sorted(list(set(idToCheckNew))))
		
		return list(destinations.keys())


gc = getCommons()
print(getCommons.getCommonPoints(gc, 'materially_ad1verse_effect', 'significant_adverse_impact', 7))
