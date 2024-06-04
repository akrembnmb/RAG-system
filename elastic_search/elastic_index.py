from elastic_client import client
from index_mapping import index_mapping
from dotenv import load_dotenv
#from data.data_processing import import_data
import os

load_dotenv()
index_name = os.getenv("INDEX_NAME")

print(f"The index name is: {index_name}")

#Create index

#client.indices.create(index="question_answering_index",mappings=index_mapping)
#indices = client.indices.exists(index=index_name)
#print(indices)

#Create documents in the index 
#data = import_data('data\questions_with_embeddings.json')

#def generate_documents():
#    for element in data:
#        yield {
#            "_index": index_name,
#            "_source": element
#        }
#
#try:
#    success, _ = client.bulk( generate_documents())
#    print(f"Indexed {success} documents successfully.")
#except Exception as e:
#    print(e)
        
        
        
print(client.count(index = index_name))   
