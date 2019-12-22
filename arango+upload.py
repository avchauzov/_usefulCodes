"""
arange: upload function
"""

import pandas as pd
from arango import ArangoClient

DATABASE_NAME = 'resulting'
INPUT_DATA = 'outputData/2.3.1.tokenToken.csv'

GRAPH_NAME = 'graph'
TOKEN_COLLECTION = 'token'
TOKEN2TOKEN_COLLECTION = 'token2token'


class uploadData:
	
	def __init__(self):
		self.db = self.getArangoConnection()
		self.data = pd.read_csv(INPUT_DATA, index_col = 0)
		
		self.dictionaryToken = {}
	
	def run(self, startIndex):
		
		if startIndex is False:
			graph, token, token2token = self.initialize(False)
		
		else:
			graph, token, token2token = self.initialize(True)
		
		if not startIndex:
			self.uploadDictionary(self.listOfTokens, token)
		
		for index in range(len(self.data.index)):
			
			if startIndex is False:
				pass
			
			else:
				
				if index < startIndex:
					continue
			
			fromIndex = str(self.data['originalToken'].values[index])
			toIndex = str(self.data['connectedToken'].values[index])
			similarity = self.data['similarity'].values[index]
			type = str(self.data['type'].values[index])
			
			try:
				token2token.insert({
						'_from'     : TOKEN_COLLECTION + '/' + str(self.dictionaryToken.get(fromIndex)),
						'_to'       : TOKEN_COLLECTION + '/' + str(self.dictionaryToken.get(toIndex)),
						'similarity': round(similarity, 3),
						'type'      : type
						})
			
			except Exception:
				return 'unsuccessful token2token.insert at ' + str(index)
		
		traversal_results = graph.traverse(
				start_vertex = TOKEN_COLLECTION + '/' + str(self.dictionaryToken.get(fromIndex)),
				max_depth = 2,
				direction = 'any',
				# strategy='bfs',
				edge_uniqueness = 'global',
				vertex_uniqueness = 'global',
				)
		
		return 'finished'
	
	@classmethod
	def getArangoConnection(cls):
		client = ArangoClient()
		db = client.db(DATABASE_NAME)
		
		return db
	
	def initialize(self, type):
		
		if type:
			self.graph = self.db.graph(GRAPH_NAME)
			self.token = self.graph.vertex_collection(TOKEN_COLLECTION)
			self.token2token = self.graph.edge_collection(TOKEN2TOKEN_COLLECTION)
		
		else:
			
			try:
				self.graph = self.db.create_graph(GRAPH_NAME)
				self.token = self.graph.create_vertex_collection(TOKEN_COLLECTION)
				
				self.token2token = self.graph.create_edge_definition(
						name = TOKEN2TOKEN_COLLECTION,
						from_collections = [TOKEN_COLLECTION],
						to_collections = [TOKEN_COLLECTION]
						)
			
			except:
				self.graph = self.db.graph(GRAPH_NAME)
				
				try:
					self.graph.delete_vertex_collection(TOKEN_COLLECTION, purge = True)
				
				except:
					pass
				
				try:
					self.token = self.db.collection(TOKEN_COLLECTION)
					self.token.truncate()
				
				except:
					pass
				
				try:
					self.graph.delete_edge_definition(TOKEN2TOKEN_COLLECTION, purge = True)
				
				except:
					pass
				
				self.db.delete_graph(GRAPH_NAME)
				
				###
				
				self.graph = self.db.create_graph(GRAPH_NAME)
				self.token = self.graph.create_vertex_collection(TOKEN_COLLECTION)
				
				self.token2token = self.graph.create_edge_definition(
						name = TOKEN2TOKEN_COLLECTION,
						from_collections = [TOKEN_COLLECTION],
						to_collections = [TOKEN_COLLECTION]
						)
		
		self.listOfTokens = list(self.data['originalToken']) + list(self.data['connectedToken'])
		
		for index, item in enumerate(sorted(list(set(self.listOfTokens)))):
			self.dictionaryToken[item] = index
		
		return self.graph, self.token, self.token2token
	
	def uploadDictionary(self, tempList, token):
		listOfTokens = list(self.data['originalToken']) + list(self.data['connectedToken'])
		
		for index, item in enumerate(sorted(list(set(tempList)))):
			self.dictionaryToken[item] = index
			
			token.insert({'_key': str(index), 'name': item})


ud = uploadData()
print(uploadData.run(ud, False))  # False or number
