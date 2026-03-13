# Sereni AI - Agentic Mental Health Assistant

## Description

Sereni AI is an agentic mental health assistant designed to provide a safe, private space for users to discuss their feelings. Built using advanced AI technologies, it offers empathetic support, relevant information, and coping strategies for mental health concerns.

## Functionality

The system leverages several AI components to deliver personalized assistance:

- **Intent Classification**: Analyzes user inputs to categorize them as greetings, mental health queries, crises, or off-topic discussions.
- **Emotion Detection**: Uses a pre-trained model to gauge the user's emotional state.
- **Crisis Detection**: Identifies messages indicating severe distress and provides immediate emergency resources.
- **Knowledge Retrieval**: Searches a curated knowledge base (stored in `mental_health_knowledge.json`) for relevant mental health information using vector embeddings and reranking.
- **Conversational AI**: Generates empathetic responses using a large language model (LLM) from Hugging Face.
- **Coping Strategies**: Suggests practical coping techniques based on retrieved knowledge.
- **Memory Management**: Maintains conversation history using LangGraph's state management for coherent interactions.
- **User Interface**: A clean, centered web interface built with Streamlit, allowing users to chat naturally.

The chatbot is not a substitute for professional therapy but aims to offer supportive guidance and encourage seeking help when needed.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd mental-health-chatbot
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up Hugging Face API token:
   - Obtain a token from [Hugging Face](https://huggingface.co/settings/tokens).
   - Add it to Streamlit secrets or set as environment variable `HUGGINGFACEHUB_API_TOKEN`.

4. Run the application:
   ```
   streamlit run app.py
   ```

## Usage

- Open the app in your browser.
- Start chatting by typing your message.
- The chatbot will respond empathetically and provide relevant support.
- Use the sidebar to clear chat history if needed.

## Live Link

[Access the live demo here](https://sereniai.streamlit.app/) 

## Technologies Used

- **Streamlit**: For the web UI.
- **LangChain & LangGraph**: For building the conversational agent workflow.
- **Hugging Face**: For LLM, embeddings, and emotion detection models.
- **Chroma**: Vector database for knowledge retrieval.
- **Transformers**: For emotion classification pipeline.

## Disclaimer

This application is for informational purposes only and does not provide medical advice. In case of a mental health crisis, please contact professional help immediately.
[Go Live]()
