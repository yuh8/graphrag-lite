import json
from graphrag import GraphRAG

graph_rag = GraphRAG(enable_local=True)


with open("./test_data/contract_03.txt") as f:
    string = f.read()

doc_key = graph_rag.insert(string)


# local query with vector search on embedded entities
example_query_1 = """
    extract values of parties, effective date/start date, contract term, payment term, termination condition from the contract in json format. Out only the json data.
    """
response = graph_rag.query(example_query_1, doc_key=doc_key)
print(response)


# global query on all community reports without vector search
example_query_2 = "Provide a comprehensive summary of the contract so that a legal personnel does not need to read through it to understand all important information of the contract. Be specific about details that relate to dates, amounts, liabilities, locations, names, etc."
response = graph_rag.query(example_query_2, doc_key=doc_key, mode="global")
print(response)
