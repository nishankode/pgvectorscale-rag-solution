import pandas as pd
from typing import List
from pydantic import BaseModel, Field
from services.llm_hub import LLMHub


class SynthesizedResponse(BaseModel):
	thought_process: List[str] = Field(description="List of thoughts that the AI assistant had while synthesizing the answer")
	answer: str = Field(description="The synthesized answer to the user's question")
	enough_context: str = Field(description="Whether the assistant has enough context to answer the question")

class Responder:
	
	SYSTEM_PROMPT = """
	# Role and Purpose
	You are an AI assistant for an e-commerce FAQ system. Your task is to synthesize a coherent and helpful answer 
	based on the given question and relevant context retrieved from a knowledge database.

	# Guidelines:
	1. Provide a clear and concise answer to the question.
	2. Use only the information from the relevant context to support your answer.
	3. The context is retrieved based on cosine similarity, so some information might be missing or irrelevant.
	4. Be transparent when there is insufficient information to fully answer the question.
	5. Do not make up or infer information not present in the provided context.
	6. If you cannot answer the question based on the given context, clearly state that.
	7. Maintain a helpful and professional tone appropriate for customer service.
	8. Adhere strictly to company guidelines and policies by using only the provided knowledge base.
	
	Review the question from the user:
	"""
	
	@staticmethod
	def dataframe_to_json(context: pd.DataFrame, columns_to_keep: List[str]) -> str:
		"""Function to convert pandas dataframe to json."""

		# Converting pandas dataframe to json
		return context[columns_to_keep].to_json(orient='records', indent=4)

	def generate_response(question: str, context: pd.DataFrame) -> SynthesizedResponse:
		"""Function to get answer for question from LLM using the collected context."""

		# Converting context dataframe to json
		context_str = Responder.dataframe_to_json(context=context, columns_to_keep=['content', 'category'])

		# Creating chat message template
		messages = [
			{"role": "system", "content": Responder.SYSTEM_PROMPT},
			{"role": "user", "content": f"# User question:\n{question}"},
			{
				"role": "assistant",
				"content": f"# Retrieved information:\n{context_str}",
			},
		]

		# Creating LLM Instance
		llm = LLMHub('openai')
		# Getting Response (Answer to user question) form LLM
		response = llm.create_completion(response_model=SynthesizedResponse, messages=messages)
		
		return response