# 📦 Imports
import os
from pathlib import Path
from fastapi import FastAPI
from langchain_community.document_loaders import GitLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from pydantic import BaseModel
import shutil
from git import Repo
import git
from tools import get_latest_commit_info, get_repository_size, get_lines_of_code
from fastapi.middleware.cors import CORSMiddleware

import stat

from sqlalchemy import func


# 📁 Repo Configuration


class RepoRequest(BaseModel):
    repo_url: str


# 🕵️ Prompt Template
detective_prompt_template = """
You are an expert software architect. A README.md file is not available.
Your mission is to determine the purpose of the repository based on the provided context.
Analyze the following clues in this order:
1. Look for a 'description' or 'summary' field in configuration files like package.json or pyproject.toml.
2. Examine the main source code files (check the backend folder, app folder in the frontend) for high-level comments, docstrings, and function names.
3. Infer the project's architecture from the directory structure.
3. How can I run the code?

Based on your analysis, provide a concise, high-level summary in 200 words of what this repository does.

Context: {context}
Question: {question}

Inferred Summary:
"""

DETECTIVE_PROMPT = PromptTemplate(
    template=detective_prompt_template,
    input_variables=["context", "question"]
)

# 🔍 QA Chain
# qa_chain_detective = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=db.as_retriever(),
#     chain_type_kwargs={"prompt": DETECTIVE_PROMPT}
# )

# 🚀 FastAPI App
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify domains like ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/analyze")
def analyze_repo():
    response = qa_chain_detective.invoke("what is this repo about")
    return response


@app.post("/analyze")
def analyze_repo(request: RepoRequest):
    TEMP_DIR = "./temp_repo"


    
    def onexc(func, path, exc_info):
    # Try to make the file writable and retry
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception as e:
            print(f"Failed to delete {path}: {e}")


    # shutil.rmtree(TEMP_DIR, onerror=handle_remove_readonly)

    # Clean up previous repo
    if os.path.exists(TEMP_DIR):
        # shutil.rmtree(TEMP_DIR)
        shutil.rmtree(TEMP_DIR, onerror=onexc)


    

    # Clone the new repo
    try:
        Repo.clone_from(request.repo_url, TEMP_DIR)
        repo =git.Repo(TEMP_DIR)
    except Exception as e:
        return {"error": f"Failed to clone repo: {str(e)}"}
    

    # Filter and load documents
    def file_filter(file_path: str) -> bool:
        return (
            file_path.endswith((
                ".ipynb", ".kt", ".c", ".cpp", ".py", ".html", ".css", ".js",
                ".ts", ".tsx", ".json", ".md", ".yaml", ".yml"
            ))
            and "package-lock.json" not in file_path
        )

    documents = []
    for root, _, files in os.walk(TEMP_DIR):
        for file in files:
            full_path = os.path.join(root, file)
            if file_filter(full_path):
                try:
                    loader = TextLoader(full_path)
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"Error loading {full_path}: {e}")

    if not documents:
        return {"error": "No valid documents found in repo."}

    # Split, embed, and store
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0)


    # Run the detective chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": DETECTIVE_PROMPT}
    )

    latest_commit_info = get_latest_commit_info(repo)
    size = get_repository_size(TEMP_DIR)
    loc = get_lines_of_code(TEMP_DIR)

    result = qa_chain.invoke("what is this repo about")
    return {"summary": result["result"], "latest_commit": latest_commit_info, "size": size, "loc": loc}


# import git 




