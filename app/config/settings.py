import logging
import os
from datetime import timedelta
from dotenv import load_dotenv

from typing import Optional
from pydantic import Field, BaseModel
from functools import lru_cache

load_dotenv(dotenv_path="./.env")

def setup_logging():
	"Configure basic logging for the application"

	logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s %(message)s")



class LLMSettings(BaseModel):
	"""LLM Configuration Settings"""

	temperature: float = 0.0
	max_tokens: Optional[int] = None
	max_retries: int = 3

class OpenAISettings(LLMSettings):
	"OpenAI Specific Settings - LLMSettings combined"

	api_key: str = Field(default_factory=lambda: os.getenv('OPENAI_API_KEY'))
	default_model: str = Field(default='gpt-4o-mini')
	embedding_model: str = Field(default='text-embedding-3-small')

class DatabaseSettings(BaseModel):
	"""Database Connection Settings"""

	service_url: str = Field(default_factory=lambda: os.getenv('TIMESCALE_SERVICE_URL'))

class VectorStoreSettings(BaseModel):
	"""Settings for the Vectorstore"""

	table_name: str = 'embeddings'
	embedding_dimensions: int = 1536
	time_partition_levels: timedelta = timedelta(days=7)

class Settings(BaseModel):
	"""Combining All Base Settings"""

	openai: OpenAISettings = Field(default_factory=OpenAISettings)
	database: DatabaseSettings = Field(default_factory=DatabaseSettings)
	vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)

@lru_cache
def get_settings() -> Settings:
	"""Creating and returning cached instance of settings"""
	settings = Settings()
	setup_logging()
	return settings