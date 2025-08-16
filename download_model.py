# download_model.py
from langchain_huggingface import HuggingFaceEmbeddings

print("Downloading and caching HuggingFace embeddings model...")
HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Model download complete.")