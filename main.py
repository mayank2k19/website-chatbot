import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

app = FastAPI()

db = None  # global memory (MVP)

class TrainRequest(BaseModel):
    url: str

class ChatRequest(BaseModel):
    query: str


# 🔹 TRAIN BOT FROM WEBSITE
@app.post("/train")
def train(req: TrainRequest):
    global db

    loader = WebBaseLoader(req.url)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    db = Chroma.from_documents(
        chunks,
        OpenAIEmbeddings(),
        persist_directory="./db"
    )

    db.persist()

    return {"status": "trained", "chunks": len(chunks)}


# 🔹 CHAT WITH BOT
@app.post("/chat")
def chat(req: ChatRequest):
    global db

    if db is None:
        return {"answer": "Please train the bot first."}

    retriever = db.as_retriever(search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(req.query)

    context = "\n".join([d.page_content for d in docs])

    llm = ChatOpenAI()

    prompt = f"""
    Answer ONLY from the context.
    If not found, say "I don't know".

    Context:
    {context}

    Question:
    {req.query}
    """

    response = llm.invoke(prompt)

    return {"answer": response.content}