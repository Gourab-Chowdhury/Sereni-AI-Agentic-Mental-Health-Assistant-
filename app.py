import streamlit as st
import os
import json
import uuid
import logging
from typing import Annotated, TypedDict

# LangChain & LangGraph Imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Reranking Imports
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Transformers Import
from transformers import pipeline

# ==========================================
# 1. UI SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Sereni AI", layout="centered")
st.title("Sereni AI - Agentic Mental Health Assistant")
st.caption("A safe, private space to talk about how you're feeling.")

# Handle API Token Safely
if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
else:
    st.error("🚨 Hugging Face Token is missing! Please add it to your Streamlit secrets.")
    st.stop()
    
logging.basicConfig(level=logging.WARNING)

# ==========================================
# 2. CACHE HEAVY AI MODELS (Crucial for Streamlit)
# ==========================================
@st.cache_resource(show_spinner="Loading AI Models (This takes a minute on startup)...")
def load_ai_systems():
    # LLM Initialization
    hf_endpoint = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-20b", # Your chosen model
        task="text-generation", 
        temperature=0.7,
        max_new_tokens=1024,
        return_full_text=False,
        do_sample=True
    )
    llm = ChatHuggingFace(llm=hf_endpoint)

    # Embeddings & Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    file_path = "mental_health_knowledge.json"
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            json.dump({
                "knowledge_base": [{"topic": "Anxiety", "subtopics": {"coping_techniques": ["Deep breathing exercises: Try 4-7-8 breathing.", "Grounding techniques: 5-4-3-2-1 sensory exercise."]}}]
            }, f)

    persist_dir = "./chroma_health_db"
    with open(file_path, "r") as f:
        raw_data = json.load(f)

    docs = []
    for topic_data in raw_data.get("knowledge_base", []):
        topic_name = topic_data.get("topic", "")
        for subtopic_name, content in topic_data.get("subtopics", {}).items():
            clean_subtopic = subtopic_name.replace("_", " ").title()
            content_str = "\n- " + "\n- ".join(content) if isinstance(content, list) else str(content)
            chunk = f"Topic: {topic_name}\nSubtopic: {clean_subtopic}\nInformation: {content_str}"
            docs.append(chunk)

    vectorstore = Chroma.from_texts(texts=docs, embedding=embeddings, collection_name="mental_health_kb", persist_directory=persist_dir)

    # Reranker Setup
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    cross_encoder_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=cross_encoder_model, top_n=2)
    retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)

    # Emotion Pipeline
    emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=False)
    
    return llm, retriever, emotion_pipeline

# Extract the cached models
llm, retriever, emotion_pipeline = load_ai_systems()

# CACHE THE MEMORY: If we don't cache this, LangGraph forgets everything on every keystroke!
@st.cache_resource
def get_memory_saver():
    return MemorySaver()

# ==========================================
# 3. LANGGRAPH STATE & NODES (Your exact logic)
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

def router_agent(state: ConversationState) -> ConversationState:
    user_input = state["user_input"]
    context = ""
    if len(state["messages"]) > 1:
        context = f"\nPrevious Bot Message: {state['messages'][-2].content}\n"
        
    prompt = ChatPromptTemplate.from_template(
        "Classify the user's input intent into EXACTLY ONE of the following categories:\n"
        "- 'greeting': Simple hellos, how are you.\n"
        "- 'mental_health_query': Questions about anxiety, stress, therapy, or FOLLOW-UPS to the previous message.\n"
        "- 'crisis': Severe distress, self-harm, emergencies.\n"
        "- 'off_topic': Coding, math, trivia, general non-health topics.\n\n"
        "If the user is asking a follow-up question to the Previous Bot Message, classify it as 'mental_health_query'.\n"
        "{context}"
        "User message: {user_input}\n\nRespond with JUST the category word."
    )
    response = (prompt | llm).invoke({"context": context, "user_input": user_input})
    intent_raw = response.content.strip().lower()
    
    intent = "mental_health_query"
    for valid in ["greeting", "mental_health_query", "crisis", "off_topic"]:
        if valid in intent_raw: intent = valid; break
    return {"intent": intent}

def off_topic_handler_agent(state: ConversationState) -> ConversationState:
    response = "I am a specialized mental health assistant. I'm not equipped to answer general trivia or off-topic questions. However, if you'd like to talk about how you're feeling or manage stress, I'm here for you!"
    return {"final_response": response, "messages": [AIMessage(content=response)]}

def sentiment_analyzer_agent(state: ConversationState) -> ConversationState:
    try:
        result = emotion_pipeline(state["user_input"])[0]
        sentiment_map = {"joy": "positive", "surprise": "neutral", "neutral": "neutral", "sadness": "distressed", "fear": "distressed", "anger": "negative", "disgust": "negative"}
        sentiment = sentiment_map.get(result['label'], "neutral")
    except:
        sentiment = "neutral"
    return {"sentiment": sentiment}

def crisis_detector_agent(state: ConversationState) -> ConversationState:
    prompt = ChatPromptTemplate.from_template("Detect if this message indicates a mental health crisis requiring immediate help (suicide, self-harm, urgent danger):\nUser message: {user_input}\nRespond with: yes or no")
    response = (prompt | llm).invoke({"user_input": state["user_input"]})
    return {"is_crisis": "yes" in response.content.strip().lower()}

def knowledge_retrieval_agent(state: ConversationState) -> ConversationState:
    user_input = state["user_input"]
    search_query = user_input
    if len(state["messages"]) > 1 and len(user_input.split()) < 8:
        search_query = f"Context: {state['messages'][-2].content[:200]}... Query: {user_input}"
    try:
        relevant_docs = retriever.invoke(search_query) 
        retrieved_knowledge = "\n\n".join([doc.page_content[:500] for doc in relevant_docs])
        return {"retrieved_knowledge": retrieved_knowledge}
    except Exception:
        return {"retrieved_knowledge": "General mental health support information available."}

def counselor_agent(state: ConversationState) -> ConversationState:
    history_text = ""
    for msg in state["messages"][:-1][-6:]: 
        role = "👤 User" if msg.type == "human" else "🤖 Counselor"
        history_text += f"{role}: {msg.content}\n"
        
    prompt = ChatPromptTemplate.from_template(
        "You are an empathetic mental health counselor.\n"
        "--- Conversation History ---\n{history}\n--------------------------\n\n"
        "Current User: {user_input}\nRelevant Knowledge: {retrieved_knowledge}\n\n"
        "Provide warm, supportive guidance (3-4 sentences max) directly answering the current user prompt while using the history for context. Do NOT diagnose."
    )
    response = (prompt | llm).invoke({"history": history_text, "user_input": state["user_input"], "retrieved_knowledge": state.get("retrieved_knowledge", "")})
    return {"final_response": response.content.strip()}

def coping_strategy_agent(state: ConversationState) -> ConversationState:
    prompt = ChatPromptTemplate.from_template("Based on this information, suggest 2-3 practical coping strategies:\n{retrieved_knowledge}\nFormat as bullet points. Keep each strategy to one sentence.")
    response = (prompt | llm).invoke({"retrieved_knowledge": state.get("retrieved_knowledge", "")})
    return {"coping_strategies": response.content.strip()}

def response_formatter_agent(state: ConversationState) -> ConversationState:
    counselor_response = state.get("final_response", "")
    coping_strategies = state.get("coping_strategies", "")
    
    if coping_strategies and state.get("intent") != "greeting":
        formatted_response = f"{counselor_response}\n\n**Practical strategies you might try:**\n{coping_strategies}\n"
    else:
        formatted_response = counselor_response
        
    return {"final_response": formatted_response, "messages": [AIMessage(content=formatted_response)]}

def crisis_handler_agent(state: ConversationState) -> ConversationState:
    crisis_response = (
        "🚨 **CRISIS SUPPORT - IMMEDIATE HELP NEEDED** 🚨\n\n"
        "I'm concerned about what you're experiencing. Please reach out for immediate professional help:\n\n"
        "📞 **Emergency Resources:**\n"
        "• National Suicide Prevention Lifeline: **988** (24/7, free, confidential)\n"
        "• India: iCall **9152987821** or dial **112**\n"
        "• Crisis Text Line: Text **HOME** to **741741**\n"
        "• International: https://findahelpline.com\n\n"
        "You deserve support, and help is available right now. Please reach out immediately."
    )
    return {"final_response": crisis_response, "messages": [AIMessage(content=crisis_response)]}

# ==========================================
# 4. BUILD THE GRAPH 
# ==========================================
workflow = StateGraph(ConversationState)

workflow.add_node("router", router_agent)
workflow.add_node("off_topic_handler", off_topic_handler_agent)
workflow.add_node("sentiment", sentiment_analyzer_agent)
workflow.add_node("crisis_detector", crisis_detector_agent)
workflow.add_node("crisis_handler", crisis_handler_agent)
workflow.add_node("retrieve_knowledge", knowledge_retrieval_agent)
workflow.add_node("counselor", counselor_agent)
workflow.add_node("coping", coping_strategy_agent)
workflow.add_node("formatter", response_formatter_agent)

workflow.add_edge(START, "router")
workflow.add_conditional_edges("router", lambda s: "off_topic_handler" if s.get("intent") == "off_topic" else "sentiment")
workflow.add_edge("off_topic_handler", END)
workflow.add_edge("sentiment", "crisis_detector")
workflow.add_conditional_edges("crisis_detector", lambda s: "crisis_handler" if s.get("is_crisis") else "retrieve_knowledge")
workflow.add_edge("crisis_handler", END)
workflow.add_edge("retrieve_knowledge", "counselor")
workflow.add_edge("counselor", "coping")
workflow.add_edge("coping", "formatter")
workflow.add_edge("formatter", END)

# Attach the cached memory so history persists across Streamlit reruns
app = workflow.compile(checkpointer=get_memory_saver())

# ==========================================
# 5. STREAMLIT CHAT UI
# ==========================================
# Initialize Session State Variables
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi there. I'm here to listen. How are you feeling today?"}]

# Sidebar with a Clear Chat button
with st.sidebar:
    st.header("Settings")
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = [{"role": "assistant", "content": "Hi there. I'm here to listen. How are you feeling today?"}]
        st.session_state.thread_id = str(uuid.uuid4()) # Changing thread_id resets LangGraph memory completely
        st.rerun()

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Type your message here..."):
    # 1. Add user message to UI state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Process with LangGraph
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                initial_state = {
                    "user_input": prompt,
                    "messages": [HumanMessage(content=prompt)]
                }
                
                result = app.invoke(initial_state, config=config)
                bot_response = result['final_response']
                
                st.markdown(bot_response)
                # 3. Save assistant message to UI state
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}\n\n*Remember: You're not alone, and it's okay to ask for help. 💙*"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

