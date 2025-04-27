import os
import json
import logging
from typing import Dict, List, Optional, Any, TypedDict
from datetime import datetime
from pathlib import Path

from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.chat_models import ChatOllama
from langgraph.graph import StateGraph, END

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ChatbotService:
    def __init__(self, data_dir: str = "data"):
        """Initialize the chatbot service"""
        self.data_dir = Path(data_dir)
        self.models_dir = self.data_dir / "models"
        self.langchain_dir = self.data_dir / "langchain"
        
        # Load responses
        with open(self.langchain_dir / "training_data.json", 'r') as f:
            self.responses = json.load(f)
        
        # Initialize components
        self.initialize_components()
        
        logger.info("Chatbot service initialized")
    
    def initialize_components(self):
        """Initialize LangChain components"""
        # Load or create vector store
        self.vectorstore = self._initialize_vectorstore()
        
        # Initialize language model
        self.llm = self._initialize_llm()
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _initialize_vectorstore(self):
        """Initialize the vector store for retrieval"""
        try:
            # Try to load existing vector store
            vectorstore_path = self.models_dir / "vectorstore"
            if vectorstore_path.exists():
                logger.info("Loading existing vector store")
                return FAISS.load_local(
                    vectorstore_path,
                    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
                    allow_dangerous_deserialization=True
                )
            
            # Create new vector store
            logger.info("Creating new vector store")
            documents = []
            
            # Load training data
            with open(self.langchain_dir / "training_data.json", 'r') as f:
                training_data = json.load(f)
            
            # Convert to documents
            for intent, examples in training_data.items():
                for example in examples:
                    doc = Document(
                        page_content=example,
                        metadata={"intent": intent}
                    )
                    documents.append(doc)
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            split_documents = text_splitter.split_documents(documents)
            
            # Create vector store
            vectorstore = FAISS.from_documents(
                split_documents,
                HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            )
            
            # Save vector store
            vectorstore.save_local(vectorstore_path)
            
            return vectorstore
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    def _initialize_llm(self):
        """Initialize the language model"""
        try:
            # Use Ollama with llama3.2 model
            logger.info("Initializing Ollama model with llama3.2")
            return ChatOllama(model="llama3.2")
            
        except Exception as e:
            logger.error(f"Error initializing language model: {e}")
            raise
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph for conversation flow"""
        # Define state types
        class ChatState(TypedDict):
            messages: List[Dict[str, str]]
            context: Optional[List[Document]]
            next_steps: Optional[List[str]]
        
        # Define nodes
        def retrieve_context(state: ChatState) -> ChatState:
            """Retrieve relevant context for the user's message"""
            user_message = state["messages"][-1]["content"]
            
            # Search for relevant documents
            docs = self.vectorstore.similarity_search(user_message, k=3)
            state["context"] = docs
            
            return state
        
        def generate_response(state: ChatState) -> ChatState:
            """Generate a response based on context and intent"""
            user_message = state["messages"][-1]["content"]
            context = state["context"]
            
            # Create prompt
            prompt = PromptTemplate.from_template(
                """You are a helpful customer support chatbot. Use the following context to answer the user's question.
                
                Context:
                {context}
                
                User message: {message}
                
                Response:"""
            )
            
            # Generate response
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({
                "context": "\n".join(doc.page_content for doc in context),
                "message": user_message
            })
            
            # Add response to messages
            state["messages"].append({
                "role": "assistant",
                "content": response
            })
            
            return state
        
        def suggest_next_steps(state: ChatState) -> ChatState:
            """Suggest next steps based on the conversation"""
            # Add next steps based on context and intent
            state["next_steps"] = [
                "Would you like to know more about this?",
                "Is there anything else I can help you with?",
                "Would you like to speak with a human agent?"
            ]
            
            return state
        
        # Build the graph
        workflow = StateGraph(ChatState)
        
        # Add nodes
        workflow.add_node("retrieve_context", retrieve_context)
        workflow.add_node("generate_response", generate_response)
        workflow.add_node("suggest_next_steps", suggest_next_steps)
        
        # Add edges
        workflow.add_edge("retrieve_context", "generate_response")
        workflow.add_edge("generate_response", "suggest_next_steps")
        workflow.add_edge("suggest_next_steps", END)
        
        # Set entry point
        workflow.set_entry_point("retrieve_context")
        
        return workflow.compile()
    
    def process_message(self, message: str, conversation_history: Optional[List[Dict[str, str]]] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a user message and return a response"""
        try:
            # Log incoming request
            logger.info(f"Received message: {message}")
            logger.info(f"Session ID: {session_id if session_id else 'New session'}")
            
            # Initialize conversation history if not provided
            if conversation_history is None:
                conversation_history = []
            
            # Add user message to history with session_id
            conversation_history.append({
                "role": "user",
                "content": message,
                "session_id": session_id
            })
            
            # Create initial state
            state = {
                "messages": conversation_history,
                "context": None,
                "next_steps": None
            }
            
            # Run the graph
            result = self.graph.invoke(state)
            
            # Log the response
            logger.info(f"Generated response: {result['messages'][-1]['content']}")
            
            # Safely log intent
            intent = None
            if result.get('context') and len(result['context']) > 0:
                intent = result['context'][0].metadata.get('intent')
            logger.info(f"Intent: {intent if intent else 'No intent detected'}")
            
            # Log next steps
            next_steps = result.get('next_steps', [])
            logger.info(f"Next steps: {next_steps}")
            
            return {
                "response": result["messages"][-1]["content"],
                "conversation_history": result["messages"],
                "language": "en",  # Always return English
                "intent": intent,
                "next_steps": next_steps
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
            raise 