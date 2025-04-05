from datetime import datetime
from database.vectorstore import VectorStore
from services.responder import Responder
from timescale_vector import client

# Creating vectorstore instance
vec = VectorStore()

relevant_question = "What are your shipping options?"
results = vec.search(relevant_question, limit=3)

response = Responder.generate_response(question=relevant_question, context=results)

print(f"\n{response.answer}")
print("\nThought process:")
for thought in response.thought_process:
    print(f"- {thought}")
print(f"\nContext: {response.enough_context}")
