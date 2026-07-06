import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA

from langchain.tools import Tool    
from langchain.agents import initialize_agent, AgentType

from tools.calculator import calculator
from tools.web_search import web_search

# Load .env file
load_dotenv()

# Load LLM
llm = ChatOpenAI(
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Load profile data
loader = TextLoader("data/profile.txt", encoding="utf-8")
docs = loader.load()

# Split text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

# Embeddings + Vector DB
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY")
)

db = FAISS.from_documents(chunks, embeddings)

retriever = db.as_retriever()

# RAG chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# Personal knowledge tool
def personal_info(query):
    return qa.run(query)

# Tools
tools = [
    Tool(
        name="Calculator",
        func=calculator,
        description="Useful for mathematical calculations"
    ),
    Tool(
        name="WebSearch",
        func=web_search,
        description="Useful for latest information from internet"
    ),
    Tool(
        name="PersonalInfo",
        func=personal_info,
        description="Useful for questions about Krutika's profile, skills and projects"
    )
]

# Agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Chat loop
print("Krutika AI Twin Ready! Type exit to stop.")

while True:
    user = input("You: ")

    if user.lower() == "exit":
        break

    response = agent.run(user)
    print("AI:", response)