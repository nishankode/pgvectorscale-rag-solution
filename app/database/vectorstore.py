import pandas as pd
from openai import OpenAI
from timescale_vector import client

from typing import Any, List, Optional, Union, Tuple
from datetime import datetime
from config.settings import get_settings


class VectorStore:
	
	def __init__(self):
		"""Initializing all parameters."""

		# Collecting parameters from settings
		self.settings = get_settings()
		self.service_url = self.settings.database.service_url
		self.openai_client = OpenAI(api_key=self.settings.openai.api_key)
		self.embedding_model = self.settings.openai.embedding_model
		self.vector_settings = self.settings.vector_store

		# Creating Vectorstore client
		self.vec_client = client.Sync(
			service_url=self.service_url,
			table_name=self.vector_settings.table_name,
			num_dimensions=self.vector_settings.embedding_dimensions,
			time_partition_interval=self.vector_settings.time_partition_levels
		)

	def get_embedding(self, text: str) -> List[float]:
		"""Function to create embeddings from plain text."""

		# Cleaning input text to remove \n
		text = text.replace('\n', ' ')

		# Creting embedding for input text using openai embedding model
		embedding = (self.openai_client.embeddings.create(input=[text], model=self.embedding_model).data[0].embedding)

		return embedding

	def create_tables(self) -> None:
		"""Function to create table to store embeddings."""

		# Creating table using vec_client
		self.vec_client.create_tables()

	def create_index(self) -> None:
		"""Function to create index using DiskAnnIndex."""

		# Creating DiskANNIndex using vec_client
		self.vec_client.create_embedding_index(client.DiskAnnIndex())

	def drop_index(self) -> None:
		"""Function to drop indices."""

		# Dropping the Embedding Index
		self.vec_client.drop_embedding_index()

	def upsert(self, df: pd.DataFrame) -> None:
		"""Function to upload/insert embeddings and data to psostgres table."""

		# Converting DataFrame to records
		records = df.to_records(index=False)

		# Inserting records to table
		self.vec_client.upsert(list(records))

	def search(
			self, 
			query_text: str, 
			limit: int = 5, 
			metadata_filter: Union[dict, List[dict]] = None, 
			predicates: Optional[client.Predicates] = None, 
			time_range: Optional[Tuple[datetime, datetime]] = None, 
			return_dataframe: bool = True
			) -> Union[List[Tuple[Any, ...]], pd.DataFrame]:
		"""Function to search for relavant embeddings similar to input query."""

		# Creating embedding for input query_text
		query_embedding = self.get_embeddings(query_text)

		# Creating a map to store search parameters
		search_params = {
			'limit': limit
		}

		# Adding metadata filter to search_params if available
		if metadata_filter:
			search_params['metadata_filter'] = metadata_filter

		# Adding predicates to search_params if available
		if predicates:
			search_params['predicates'] = predicates

		# Adding time_range to search_params if available
		if time_range:
			start_date, end_date = time_range
			search_params['uuid_time_filter'] = client.UUIDTimeRange(start_date, end_date)

		# Searching for embeddings similar to input_text in the vector database
		results = self.vec_client.search(query_embedding, **search_params)

		# Retrning df if return_dataframe is set True else records
		if return_dataframe:
			return self.create_dataframe_from_result(results)
		else:
			return results

	def create_dataframe_from_result(self, results: List[Tuple[Any, ...]]) -> pd.DataFrame:
		"""Function to create dataframe from fetched search results."""

		# Creating table with necessary column names
		results_df = pd.DataFrame(results, columns=['id', 'metadata', 'contents', 'embedding', 'distance'])

		# Exploding metadata column and combining
		results_df = pd.concat([results_df.drop('metadata', axis=1), results_df['metadata'].apply(pd.Series)], axis=1)

		# Typecasting id column as string
		results_df['id'] = results_df['id'].astype(str)

		return results_df

	def delete(self, ids: List[str], metadata_filter: dict = None, delete_all: bool = False) -> None:
		"""Function to delete contents from a table."""

		# Checking if morethan or lessthal 1 of delete type is passed and raising value error
		if sum(bool(x) for x in (ids, metadata_filter, delete_all)) != 1:
			raise ValueError("Please enter only one from: ids, metadata_filter, or delete_all")
		
		# Deleting reccords accordint to input method
		if delete_all:
			self.vec_client.delete_all()
		elif ids:
			self.vec_client.delete_by_ids(ids)
		elif metadata_filter:
			self.vec_client.delete_by_metadata(metadata_filter)