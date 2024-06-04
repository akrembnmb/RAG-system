from sentence_transformers import SentenceTransformer
from models.model_loader import EMBEDDING_MODEL,embedding_model
from elastic_search.elastic_client import client 
from fastapi import FastAPI,HTTPException,status
from openai_api.openai_client import create_completion
import gradio as gr
CUSTOM_PATH = "/gradio"

app = FastAPI()
def read_main():
    return {"message": "This is your main app"}
@app.get("/search")

def search(question : str):
    
    asked_question = question
    if asked_question:
        vectorise_question = embedding_model.encode(asked_question )

        query = {
            "field" : "question_embeddings",
            "query_vector" : vectorise_question,
            "k" : 5,
            "num_candidates" : 100, 
        }

        result = client.knn_search(index="question_answering_index", knn=query , source=["question","answers"])
        res = result["hits"]["hits"]
        output_qa = list()
        for e in res:
            if e['_score'] >= 0.1:
                output_qa.append((e['_source']['question'],e['_source']["answers"]))
        if len(output_qa) >=1 :
            print (output_qa)
            return create_completion(str(output_qa))
        else :
            return "Sorry there is no match for your request !"             
    else : return "Please enter a question !"
io = gr.Interface(search, "textbox", "textbox")
app = gr.mount_gradio_app(app, io, path=CUSTOM_PATH)