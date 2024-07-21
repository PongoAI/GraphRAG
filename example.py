from interfaces.vector_dbs.datastax_db import DatastaxDB
from interfaces.rerankers.pongo import PongoReranker
from traverser import GraphRAGTraversal
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
# Initialize DatastaxDB instance
db = DatastaxDB(api_endpoint=os.getenv('DATASTAX_API_ENDPOINT'), token=os.getenv('DATASTAX_TOKEN'))
reranker = PongoReranker(os.getenv('PONGO_API_KEY'))
openai_api_key = os.environ.get("OPENAI_API_KEY")

llm_client =  OpenAI(api_key=openai_api_key)

# Assuming we have a collection named 'hotpot_qa' and it's already created
collection_name = 'hotpot_qa'

# Initialize GraphRAGTraversal instance
traverser = GraphRAGTraversal(reranker, db, collection_name, llm_client, 'gpt-4o')

# Call the traverser with the question
question = "The song Arizona was recorded by Paul Revere and Mark Lindsay but who wrote the song?"
result = traverser.do_traversal(question, max_recursion_depth=3, top_k_per_query=2, queries_per_step=2, should_generate_answer=True)
print(result)
