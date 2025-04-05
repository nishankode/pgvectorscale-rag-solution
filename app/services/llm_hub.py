from typing import List, Dict, Any, Type
import instructor
from openai import OpenAI
from pydantic import BaseModel

from config.settings import get_settings

class LLMHub:
    
	def __init__(self, provider: str):
		"""Constructor to Initialize LLMHub."""

		self.provider = provider
		self.settings = getattr(get_settings(), self.provider)
		self.client = self._initialize_client()

	def _initialize_client(self) -> Any:
		"""Function to initialize client."""

		# Creating a dict of multiple LLM initializers
		client_initializers = {
			'openai': lambda x: instructor.from_openai(OpenAI(api_key=x.api_key))
		}

		# Initializing selected initializer
		initializer = client_initializers.get(self.provider)

		# Checking if initializer valid and returning client
		if initializer:
			return initializer(self.settings)
		else:
			raise ValueError("Selected Provider not Available.")
		
	def create_completion(self, response_model: Type[BaseModel], messages: List[Dict[str, str]], **kwargs) -> Any:
		"""Function to get completion from LLM for given input."""

		# Creating completion params from kwargs and inputs
		completion_params = {
			'model': kwargs.get('model', self.settings.default_model),
			'temperature': kwargs.get('temperature', self.settings.temperature),
			'max_retries': kwargs.get('max_retries', self.settings.max_retries),
			'max_tokens': kwargs.get('max_tokens', self.settings.max_tokens),
			'response_model': response_model,
			'messages': messages
		}

		# Creating completion
		return self.client.chat.completions.create(**completion_params)