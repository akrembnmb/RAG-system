
from models.model_loader import embedding_model

asked_question = "newton law"
vectorise_question = embedding_model.encode(asked_question)

query = {
    "field" : "question_embeddings",
    "query_vector" : vectorise_question,
    "k" : 2,
    "num_candidates" : 100, 
}