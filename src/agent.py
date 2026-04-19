import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import faiss
import pickle
import numpy as np
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. Framework: LangGraph State Management
class AgentState(TypedDict):
    farm_data: str
    query: str
    context: str
    final_report: str
    
model = None
index = None
docs = []

def load_rag():
    """Lazily load the index and documents to avoid blocking the app on import"""
    global model, index, docs
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    if index is None and os.path.exists('models/faiss_index.bin'):
        index = faiss.read_index('models/faiss_index.bin')
    if not docs and os.path.exists('models/docs.pkl'):
        with open('models/docs.pkl', 'rb') as f:
            docs = pickle.load(f)

# Node 1: Retrieve context based on query & conditions
def retrieve_node(state: AgentState):
    load_rag()
    query = state['query']
    
    if index is None or not docs:
        state['context'] = "No agricultural manuals available in the system. Rely on general knowledge."
        return state
        
    # Retrieve top 6 chunks (increased from 4)
    query_vector = model.encode([query]).astype('float32')
    _, indices = index.search(query_vector, k=6)
    
    # Filter valid indices in case there are less than 6 docs
    valid_indices = [i for i in indices[0] if 0 <= i < len(docs)]
    context = "\n---\n".join([docs[i] for i in valid_indices])
    
    state['context'] = context
    return state

# Node 2: Generate the structured response via Groq
def generate_node(state: AgentState):
    farm_data = state['farm_data']
    context = state['context']
    query = state['query']
    
    # Use the User's provided API key via secrets/environment
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        state['final_report'] = "### Error\nGROQ_API_KEY is not set. Please add it to your Streamlit Secrets."
        return state
    client = Groq(api_key=api_key)
    
    # Prompt Strategies to reduce hallucination & Enforce output structure
    prompt = f"""
    SYSTEM: You are a Senior Agronomist Consultant. Your goal is to provide practical, expert advice based ONLY on the provided agricultural manuals.

    INSTRUCTIONS:
    - Avoid mentioning specific study locations (e.g., 'Mali') or technical statistics (e.g., 'R2 values', 'Table 5').
    - Translate scientific data into clear, actionable advice for a farmer.
    - If the user's current conditions are high-risk (e.g., wrong soil for a crop), be direct about the risks.
    - If the manual truly doesn't cover the topic, say: "I cannot find specific evidence for this in my current manuals."

    You MUST format your output EXACTLY with these four headers:
    
    ### STATUS
    [Your Crop & Field summary + Yield Risk Assessment based on the given FARM DATA]
    
    ### ADVICE
    [Your Recommended farming actions, Step-by-Step, based on the CONTEXT manuals to improve the situation]
    
    ### SOURCES
    [List any Agronomic references or general topics from the retrieved text used for advice]
    
    ### DISCLAIMER
    [Include a brief Agricultural/Safety notice acknowledging risks]

    ---
    CONTEXT FROM MANUALS:
    {context}
    
    FARM DATA:
    {farm_data}

    USER QUESTION:
    {query}
    """
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.1
        )
        state['final_report'] = chat_completion.choices[0].message.content
    except Exception as e:
        state['final_report'] = f"### Error\nFailed to generate report from LLM: {str(e)}"
        
    return state

# 2. Framework: LangGraph Workflow
workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile into a runnable application
app = workflow.compile()

def run_agentic_workflow(farm_data_str: str, query: str = "How do I maximize crop yield based on my parameters?"):
    """Entrypoint for Streamlit to call the LangGraph workflow"""
    inputs = {
        "farm_data": farm_data_str,
        "query": query,
        "context": "",
        "final_report": ""
    }
    result = app.invoke(inputs)
    return result['final_report']
