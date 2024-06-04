import json
from sentence_transformers import SentenceTransformer
from models.model_loader import EMBEDDING_MODEL


#Format original data to Q&As pairs
"""with open('data\dev-v1.1.json', 'r', encoding="utf-8") as file:
    data = json.load(file)

qa_pairs = []
for item in data['data']:
    for paragraph in item['paragraphs']:
        for qa in paragraph['qas']:
            answers = [ans['text'] for ans in qa['answers']]
            qa_pairs.append({'question': qa['question'], 'answers': answers})

output_file = 'qa_pairs.json'
with open(output_file, 'w') as f:
    json.dump(qa_pairs, f, indent=2)

print("success ! ")"""

def import_data(path):
    
    with open(path, 'r', encoding="utf-8") as file:
        data = json.load(file)
        print("Data imported sucessfully !")
        return data
    
    

def import_data(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data
#Only run this once 
#def embed_question(question, model):
#    return model.encode(question)
#
#def embed_data_and_save(data, model):
#    for item in data:
#        question = item["question"]
#        embeddings = embed_question(question, model)
#        item["question_embeddings"] = embeddings.tolist() 
#    with open('questions_with_embeddings.json', 'w') as json_file:
#        json.dump(data, json_file, indent=4)
#
#model = SentenceTransformer(EMBEDDING_MODEL)
#data = import_data('qa_pairs.json')
#embed_data_and_save(data, model)
