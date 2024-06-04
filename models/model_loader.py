import yaml
from sentence_transformers import SentenceTransformer

with open ("data\config.yml","r") as f:
    config = yaml.safe_load(f)
with open ("models\model_config.yml","r") as f:
    model_params = yaml.safe_load(f)



EMBEDDING_MODEL = config["embedding_model"]
TEMPERATURE =model_params["temperature"]
MAX_TOKENS = model_params["max_tokens"]
MODEL = model_params["model"]
PROMPT = model_params["prompt"]


def load_embedding_model():
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model
embedding_model = load_embedding_model()