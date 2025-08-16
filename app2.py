# 📦 Imports
import os
import shutil
import stat
import asyncio
from pathlib import Path
from fastapi import FastAPI
from git import Repo
import git

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS

from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# your helper utils
from tools import get_latest_commit_info, get_repository_size, get_lines_of_code


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
4. How can I run the code?

Provide a concise, high-level summary (≈200 words).

Context: {context}
Question: {question}

Inferred Summary:
"""

DETECTIVE_PROMPT = PromptTemplate(
    template=detective_prompt_template,
    input_variables=["context", "question"]
)


# 🚀 FastAPI App
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze")
async def analyze_repo_post(request: RepoRequest):
    TEMP_DIR = "./temp_repo"

    # --- Cleanup old repo ---
    if os.path.exists(TEMP_DIR):
        def handle_remove_readonly(func, path, _):
            os.chmod(path, stat.S_IWRITE)
            func(path)
        shutil.rmtree(TEMP_DIR, onerror=handle_remove_readonly)

    # --- Clone repo (shallow) ---
    try:
        Repo.clone_from(request.repo_url, TEMP_DIR, depth=1)
        repo = git.Repo(TEMP_DIR)
    except Exception as e:
        return {"error": f"Failed to clone repo: {str(e)}"}

    # --- Load documents (skip big/binary files) ---
    MAX_FILE_SIZE_MB = 2
    def file_filter(file_path: str) -> bool:
        if os.path.getsize(file_path) > MAX_FILE_SIZE_MB * 1024 * 1024:
            return False
        return file_path.endswith((
            ".py", ".js", ".ts", ".tsx", ".html", ".css",
            ".json", ".md", ".yaml", ".yml"
        )) and "package-lock.json" not in file_path

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
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        return {"error": "No valid documents found in repo."}

    # --- Split text ---
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # --- Google Embeddings ---
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # --- Gemini LLM ---
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": DETECTIVE_PROMPT}
    )

    # --- Metadata ---
    latest_commit_info = get_latest_commit_info(repo)
    size = get_repository_size(TEMP_DIR)
    loc = get_lines_of_code(TEMP_DIR)

    # --- Run inference (async) ---
    try:
        result = await qa_chain.ainvoke("what is this repo about")
    except Exception as e:
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        return {"error": f"LLM failed: {str(e)}"}

    # --- Cleanup to save Render space ---
    shutil.rmtree(TEMP_DIR, ignore_errors=True)

    return {
        "summary": result["result"],
        "latest_commit": latest_commit_info,
        "size": size,
        "loc": loc
    }
