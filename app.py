import streamlit as st
import os
import json
import uuid
import logging
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from transformers import pipeline

# ==========================================
# UI CONFIGURATION
# ==========================================
st.set_page_config(page_title="Mental Health Companion", page_icon="💙", layout="centered")
st.title("Sereni AI - Agentic Mental Health Assistant")
st.caption("A safe, private space to talk about how you're feeling.")

# ==========================================
# SECRETS & AUTHENTICATION
# ==========================================
# Streamlit uses st.secrets instead of hardcoded strings for security
if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["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"]
else:
    st.error("🚨 Hugging Face Token is missing! Please add it to your Streamlit secrets.")
    st.stop()

# ==========================================
# SYSTEM CACHING (The Expert Touch)
# ==========================================
# @st.cache_resource ensures these heavy models load only ONCE when the server starts
@st.cache_resource
def initialize_ai_system():
    # 1. LLM
    hf_endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct", 
        task="text-generation", 
        temperature=0.7,
        max_new_tokens=1024,
        return_full_text=False,
        do_sample=True
    )
    llm = ChatHuggingFace(llm=hf_endpoint)

    # 2. Vector Store (In-Memory for Streamlit Cloud to avoid disk write errors)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    with open("mental_health_knowledge.json", "r") as f:
        raw_data = json.load(f)
        
    docs = []
    for topic_data in raw_data.get("knowledge_base", []):
        topic_name = topic_data.get("topic", "")
        for subtopic_name, content in topic_data.get("subtopics", {}).items():
            clean_subtopic = subtopic_name.replace("_", " ").title()
            content_str = "\n- " + "\n- ".join(content) if isinstance(content, list) else str(content)
            chunk = f"Topic: {topic_name}\nSubtopic: {clean_subtopic}\nInformation: {content_str}"
            docs.append(chunk)

    vectorstore = Chroma.from_texts(texts=docs, embedding=embeddings, collection_name="mental_health_kb")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 3. Emotion Pipeline
    emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=False)
    
    return llm, retriever, emotion_pipeline

with st.spinner("Initializing AI Models... (This takes a moment on startup)"):
    llm, retriever, emotion_pipeline = initialize_ai_system()

# ==========================================
# LANGGRAPH STATE & NODES
# ==========================================
class ConversationState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_input: str
    intent: str  
    sentiment: str  
    is_crisis: bool
    retrieved_knowledge: str
    coping_strategies: str
    final_response: str

def router_agent(state: ConversationState):
    user_input = state["user_input"]
    prompt = ChatPromptTemplate.from_template(
        "Classify the user's input intent into ONE of: greeting, mental_health_query, crisis, off_topic\n"
        "User message: {user_input}\nRespond with JUST the category word."
    )
    response = (prompt | llm).invoke({"user_input": user_input})
    intent_raw = response.content.strip().lower()
    
    intent = "mental_health_query"
    for valid in ["greeting", "mental_health_query", "crisis", "off_topic"]:
        if valid in intent_raw: intent = valid; break
    return {"intent": intent}

def off_topic_handler_agent(state: ConversationState):
    response = "I am a specialized mental health assistant. I'm not equipped to answer general trivia, but I'm here if you want to talk about how you're feeling!"
    return {"final_response": response, "messages": [AIMessage(content=response)]}

def sentiment_analyzer_agent(state: ConversationState):
    user_input = state["user_input"]
    result = emotion_pipeline(user_input)[0]
    # Map raw emotions to simple sentiment
    sentiment_map = {"joy": "positive", "surprise": "neutral", "neutral": "neutral", "sadness": "distressed", "fear": "distressed", "anger": "negative", "disgust": "negative"}
    return {"sentiment": sentiment_map.get(result['label'], "neutral")}

def crisis_detector_agent(state: ConversationState):
    user_input = state["user_input"]
    prompt = ChatPromptTemplate.from_template("Does this message indicate a severe mental health crisis (suicide, self-harm, urgent danger)?\nMessage: {user_input}\nRespond with: yes or no")
    response = (prompt | llm).invoke({"user_input": user_input})
    return {"is_crisis": "yes" in response.content.strip().lower()}

def knowledge_retrieval_agent(state: ConversationState):
    try:
        relevant_docs = retriever.invoke(state["user_input"]) 
        return {"retrieved_knowledge": "\n\n".join([doc.page_content[:500] for doc in relevant_docs])}
    except:
        return {"retrieved_knowledge": "General support available."}

def counselor_agent(state: ConversationState):
    recent_history = "\n".join([f"{m.type}: {m.content}" for m in state["messages"][-4:-1]])
    prompt = ChatPromptTemplate.from_template(
        "You are an empathetic mental health counselor.\nHistory:\n{history}\n\nUser: {user_input}\nKnowledge: {retrieved_knowledge}\n"
        "Provide warm, supportive guidance (3-4 sentences max). No diagnosing."
    )
    response = (prompt | llm).invoke({"history": recent_history, "user_input": state["user_input"], "retrieved_knowledge": state.get("retrieved_knowledge", "")})
    return {"final_response": response.content.strip()}

def coping_strategy_agent(state: ConversationState):
    prompt = ChatPromptTemplate.from_template("Based on this, suggest 2 brief coping strategies as bullet points:\n{retrieved_knowledge}")
    response = (prompt | llm).invoke({"retrieved_knowledge": state.get("retrieved_knowledge", "")})
    return {"coping_strategies": response.content.strip()}

def formatter_agent(state: ConversationState):
    base_resp = state.get("final_response", "")
    coping = state.get("coping_strategies", "")
    final_text = f"{base_resp}\n\n**Things to try:**\n{coping}" if coping and state.get("intent") != "greeting" else base_resp
    return {"final_response": final_text, "messages": [AIMessage(content=final_text)]}

def crisis_handler_agent(state: ConversationState):
    crisis_text = "🚨 **CRISIS SUPPORT NEEDED** 🚨\nPlease reach out for immediate help:\n• Lifeline: **988**\n• iCall (India): **9152987821**\n• Crisis Text: Text **HOME** to **741741**"
    return {"final_response": crisis_text, "messages": [AIMessage(content=crisis_text)]}

# ==========================================
# BUILD & CACHE GRAPH
# ==========================================
@st.cache_resource
def build_graph():
    workflow = StateGraph(ConversationState)
    
    workflow.add_node("router", router_agent)
    workflow.add_node("off_topic", off_topic_handler_agent)
    workflow.add_node("sentiment", sentiment_analyzer_agent)
    workflow.add_node("crisis_detector", crisis_detector_agent)
    workflow.add_node("crisis_handler", crisis_handler_agent)
    workflow.add_node("retrieve", knowledge_retrieval_agent)
    workflow.add_node("counselor", counselor_agent)
    workflow.add_node("coping", coping_strategy_agent)
    workflow.add_node("formatter", formatter_agent)

    workflow.add_edge(START, "router")
    workflow.add_conditional_edges("router", lambda s: "off_topic" if s.get("intent") == "off_topic" else "sentiment")
    workflow.add_edge("off_topic", END)
    workflow.add_edge("sentiment", "crisis_detector")
    workflow.add_conditional_edges("crisis_detector", lambda s: "crisis_handler" if s.get("is_crisis") else "retrieve")
    workflow.add_edge("crisis_handler", END)
    workflow.add_edge("retrieve", "counselor")
    workflow.add_edge("counselor", "coping")
    workflow.add_edge("coping", "formatter")
    workflow.add_edge("formatter", END)

    return workflow.compile(checkpointer=MemorySaver())

app = build_graph()

# ==========================================
# STREAMLIT CHAT UI & SESSION STATE
# ==========================================
# Generate a unique thread ID for this user's browser session
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
# Store messages for the UI
if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = [{"role": "assistant", "content": "Hi there. I'm here to listen. How are you feeling today?"}]

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# Render Chat History
for msg in st.session_state.ui_messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat Input
if prompt := st.chat_input("Type your message here..."):
    # 1. Add user message to UI state and display it
    st.session_state.ui_messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 2. Run LangGraph Engine
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            initial_state = {"user_input": prompt, "messages": [HumanMessage(content=prompt)]}
            result = app.invoke(initial_state, config=config)
            
            # 3. Extract and display final response
            bot_response = result['final_response']
            st.write(bot_response)
            
            # 4. Save to UI state
            st.session_state.ui_messages.append({"role": "assistant", "content": bot_response})