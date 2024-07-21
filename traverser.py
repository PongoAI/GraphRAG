import json
from typing import List, Dict, Any

class GraphRAGTraversal:
    def __init__(self, reranker, vector_db, collection, llm_client, llm_model):
        self.reranker = reranker
        self.vector_db = vector_db
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.colelction = collection

    def do_traversal(self, query: str, max_recursion_depth: int = 3, top_k_per_query: int = 2, queries_per_step: int = 2, should_generate_answer: bool = False) -> Dict[str, Any]:
        return self._recursive_traversal(query, [], max_recursion_depth, top_k_per_query, queries_per_step, should_generate_answer)

    def _recursive_traversal(self, query: str, docs: List[str], max_recursion_depth: int, top_k_per_query: int, queries_per_step: int, should_generate_answer: bool) -> Dict[str, Any]:
        if len(docs) > 0:
            reranked_docs = self.reranker.rerank(query, docs, top_k=len(docs))
            docs = [doc['text'] for doc in reranked_docs]

        if max_recursion_depth > 0:
            should_return = self._can_answer_query(query, docs)
        else:
            should_return = True

        if should_return:
            if should_generate_answer:
                return {
                    'answer_docs': docs,
                    'answer': self._generate_answer(query, docs)
                }
            else:
                return {'answer_docs': docs}

        expand_query_result = self._expand_query(query, docs, queries_per_step)
        expanded_queries = expand_query_result['queries'] if expand_query_result['status'] == 'success' else []        
        if len(docs) == 0:
            expanded_queries.append(query)

        new_docs = set()
        for expanded_query in expanded_queries:
            query_docs = self._fetch_relevant_docs(expanded_query, top_k_per_query)
            new_docs.update(query_docs)

        docs = list(set(docs) | new_docs)

        return self._recursive_traversal(query, docs, max_recursion_depth-1, top_k_per_query, queries_per_step, should_generate_answer)

    def _fetch_relevant_docs(self, query: str, top_k: int) -> List[str]:
        db_results = self.vector_db.search(self.colelction, query, 200)
        db_docs = [res['text'] for res in db_results]
        reranked_docs = self.reranker.rerank(query, db_docs, top_k=top_k)

        return [doc['text'] for doc in reranked_docs]

    def _expand_query(self, query: str, docs: List[str], queries_per_step: int) -> List[str]:
        expansion_prompt = f'''**Query Expansion Task**

You are a system that only responds in valid a JSON array of strings with no other text. I will provide you with a query string and some supporting documents as input. Your task is to break down the query into smaller, specific queries that can be entered into a search system to find the answer. Each query will be searched upon without context of the others, so ensure each will provide valuable information on its own. You will return these component queries in a JSON array. 

Generate up to {queries_per_step} queries in your output list.

**Examples:**

Example 1:
* Query: "Were Scott Derrickson and Ed Wood of the same nationality?"
* Supporting documents: []
* Reasoning: We have two pieces of information this input query needs, so we make specific queries for each
* Output: ["What nationality is Scott Derrickson", "What nationality is Ed Wood"]

Example 2:
* Query: "What is the Pongo CEO's favorite color?"
* Supporting documents: []
* Reasoning: We do not know who the CEO of pongo is, so we first need to find that information before looking into other queries.
* Output: ["Who is the CEO of Pongo?"]

Example 3 (second step to example 2)
* Query: "What is the Pongo CEO's favorite color?"
* Supporting documents: ["The CEO of Pongo is Caleb John"]
* Reasoning: We know that the CEO of Pongo is Caleb John from Document 0, so we fill that information into the output query
* Output: ["What is Caleb John's favorite color?"]

**Input:**

*Query: "{query}"*
*Supporting documents: {docs}*
Output:'''
        llm_response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": expansion_prompt}],
                stream=False,
                temperature=0.2,
            ).choices[0].message.content
        try:
            
            if llm_response.startswith("```json"):
                llm_response = '\n'.join(llm_response.split('\n')[1:-1])
            expansion_response = json.loads(llm_response)
            return {'status': 'success', 'queries': expansion_response}
        except:
            print(f'LLM hallucinated, skipping this question: "{query}"')
            print('\n\n')
            print(llm_response)
            print('\n\n')
            return {'status': 'error', 'queries': []}

    def _can_answer_query(self, query: str, docs: List[str]) -> bool:
        if len(docs) == 0:
            return False

        llm_can_answer_prompt = f'''**Document Answer Assessment task**

You are a system that only responds with a "True" or "False" and no other text. I will provide you with a query string and some supporting documents as input. Your task is to determine wether or not the query can be completely answered using the information in the documents.  Return "True" if it can, or "False" if it cannot.

**Examples:**

Example 1:
* Query: "Who is the CEO of Pongo?"
Supporting documents: []
Output: False

Example 2:
* Query: "Were Scott Derrickson and Ed Wood of the same nationality?"
Supporting documents: ["Scott Derrickson is American.", "Ed Wood currently lives in Albania"]
Output: False

Example 2:
* Query: "What is the Pongo CEO's favorite color?"
Supporting documents: ["The CEO of Pongo is Caleb John", "Caleb John's favorite color is red."]
Output: True

*Query:* {query}

*Documents:* {docs}'''

        try:
            can_answer_response = self.llm_client.chat.completions.create(
                model=self.llm_model,            
                messages=[{"role": "user", "content": llm_can_answer_prompt}],
                stream=False,
                temperature=0.2,
            ).choices[0].message.content
            return can_answer_response.lower().strip() == 'true'
        except:
            print(f'LLM hallucinated, skipping this question: "{query}"')
            return False

    def _generate_answer(self, query: str, docs: List[str]) -> str:
        llm_generate_answer_prompt = f'''**Q&A Task**

You are a helpful assistant, please answer the below question based on the provided documents. Make your answers as brief as possible. If the question cannot be answered using the provided documents, say exactly "The question cannot be answered using the provided documents."

*Question:* {query}

*Documents:* {docs}'''

        try:
            answer_response = self.llm_client.chat.completions.create(
                model=self.llm_model,            
                messages=[{"role": "user", "content": llm_generate_answer_prompt}],
                stream=False,
                temperature=0.2,
            ).choices[0].message.content
            return answer_response
        except:
            print(f'LLM hallucinated, skipping this question: "{query}"')
            return ''
