index_mapping = {
    "properties" : {
        "question" : {
            "type" :"text"
        },
        "answers" : {
            "type" :"keyword"
        },
        "question_embeddings" : {
            "type" :"dense_vector",
            "dims" : 768,
            "index":True,
            "similarity" : "cosine"
        }
        
    }
    
}