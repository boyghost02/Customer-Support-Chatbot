import os
import json
import logging
from typing import Dict, List, Optional, Union, Any, Literal, TypedDict, Annotated
from datetime import datetime
import random

# LangChain and LangGraph imports
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.chat_models import ChatOllama
from langgraph.graph import StateGraph, END

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define state types
class ChatMessage(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str

class ChatState(TypedDict):
    messages: List[ChatMessage]
    language: Optional[str]
    intent: Optional[str]
    sentiment: Optional[str]
    context: Optional[List[Document]]
    next_steps: Optional[List[str]]
    response: Optional[str]

class LangGraphModel:
    """
    LangGraph conversation model for the customer support chatbot.
    """
    
    def __init__(self, data_dir: str = "data", model_type: str = "openai"):
        """
        Initialize the LangGraph model.
        
        Args:
            data_dir: Directory containing the data
            model_type: Type of model to use ("openai", "huggingface", or "ollama")
        """
        self.data_dir = data_dir
        self.processed_dir = os.path.join(data_dir, "processed")
        self.models_dir = os.path.join(data_dir, "models")
        
        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize model based on type
        self.model_type = model_type
        self.llm = self._initialize_llm()
        
        # Load data
        self.qa_data = self._load_qa_data()
        
        # Initialize vector store
        self.vectorstore = self._initialize_vectorstore()
        
        # Build the graph
        self.graph = self._build_graph()
        
        logger.info(f"LangGraph model initialized with model type: {model_type}")
    
    def _initialize_llm(self):
        """
        Initialize the language model based on the specified type.
        
        Returns:
            Initialized language model
        """
        if self.model_type == "openai":
            return ChatOpenAI(temperature=0.7)
        elif self.model_type == "huggingface":
            return HuggingFaceEndpoint(
                repo_id="google/flan-t5-xxl",
                max_length=512
            )
        elif self.model_type == "ollama":
            return ChatOllama(model="llama2")
        else:
            logger.warning(f"Unknown model type: {self.model_type}, defaulting to OpenAI")
            return ChatOpenAI(temperature=0.7)
    
    def _load_qa_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load Q&A data from the processed directory.
        
        Returns:
            Dictionary mapping categories to lists of Q&A pairs
        """
        try:
            file_path = os.path.join(self.processed_dir, "langchain_data.json")
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            logger.info(f"Loaded {sum(len(pairs) for pairs in data.values())} Q&A pairs from {len(data)} categories")
            return data
        except FileNotFoundError:
            logger.warning("No Q&A data found. Using empty data.")
            return {}
    
    def _initialize_vectorstore(self):
        """
        Initialize the vector store for retrieval.
        
        Returns:
            Initialized vector store
        """
        # Prepare documents for the vector store
        documents = []
        
        for category, qa_pairs in self.qa_data.items():
            for qa_pair in qa_pairs:
                doc = Document(
                    page_content=f"Question: {qa_pair['question']}\nAnswer: {qa_pair['answer']}",
                    metadata={
                        "category": category,
                        "source": qa_pair["metadata"].get("source", "unknown"),
                        "entities": qa_pair["metadata"].get("entities", [])
                    }
                )
                documents.append(doc)
        
        if not documents:
            logger.warning("No documents to index. Creating empty vector store.")
            # Create a dummy document to initialize the vector store
            documents = [Document(page_content="Empty document", metadata={"category": "general"})]
        
        # Split documents if they're too long
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_documents = text_splitter.split_documents(documents)
        
        # Always use HuggingFace embeddings to avoid OpenAI rate limits
        try:
            logger.info("Using HuggingFace embeddings")
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except Exception as e:
            logger.error(f"Error initializing HuggingFace embeddings: {e}")
            # If HuggingFace fails, try OpenAI as a last resort
            if self.model_type == "openai":
                try:
                    logger.info("Falling back to OpenAI embeddings")
                    embeddings = OpenAIEmbeddings()
                except Exception as e:
                    logger.error(f"Error initializing OpenAI embeddings: {e}")
                    raise ValueError("Failed to initialize embeddings. Please check your API keys and network connection.")
            else:
                raise ValueError("Failed to initialize embeddings. Please check your network connection.")
        
        # Create and save the vector store
        vectorstore = FAISS.from_documents(split_documents, embeddings)
        
        # Save the vector store
        vectorstore_path = os.path.join(self.models_dir, "vectorstore")
        vectorstore.save_local(vectorstore_path)
        
        logger.info(f"Initialized vector store with {len(split_documents)} documents")
        return vectorstore
    
    def _detect_language(self, state: ChatState) -> ChatState:
        """
        Detect the language of the user's message.
        
        Args:
            state: Current chat state
            
        Returns:
            Updated chat state with detected language
        """
        user_message = state["messages"][-1]["content"]
        
        # Simple language detection (Vietnamese vs. English)
        vietnamese_chars = set('àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ')
        
        for char in user_message.lower():
            if char in vietnamese_chars:
                state["language"] = "vi"
                logger.info("Detected language: Vietnamese")
                return state
        
        state["language"] = "en"
        logger.info("Detected language: English")
        return state
    
    def _classify_intent(self, state: ChatState) -> ChatState:
        """
        Classify the intent of the user's message.
        
        Args:
            state: Current chat state
            
        Returns:
            Updated chat state with classified intent
        """
        user_message = state["messages"][-1]["content"]
        
        # Define the prompt for intent classification
        prompt = PromptTemplate.from_template(
            """You are an intent classifier for a customer support chatbot.
            Classify the following message into one of these categories:
            - order_tracking: Questions about order status, tracking, or delivery
            - product_info: Questions about product details, specifications, or availability
            - returns: Questions about returning products, refunds, or exchanges
            - shipping: Questions about shipping methods, costs, or times
            - payment: Questions about payment methods, issues, or billing
            - account: Questions about user accounts, login, or registration
            - product_recommendation: Requests for product recommendations or comparisons
            - general: General inquiries or greetings
            
            User message: {message}
            
            Intent:"""
        )
        
        # Run the intent classification
        chain = prompt | self.llm | StrOutputParser()
        intent = chain.invoke({"message": user_message})
        
        # Clean up the intent
        intent = intent.strip().lower()
        if ":" in intent:
            intent = intent.split(":", 1)[1].strip()
        
        # Validate the intent
        valid_intents = [
            "order_tracking", 
            "product_info", 
            "returns", 
            "shipping", 
            "payment", 
            "account", 
            "product_recommendation",
            "general"
        ]
        
        if intent not in valid_intents:
            # Find the closest match
            for valid_intent in valid_intents:
                if valid_intent in intent:
                    intent = valid_intent
                    break
            else:
                intent = "general"
        
        state["intent"] = intent
        logger.info(f"Classified intent: {intent}")
        return state
    
    def _analyze_sentiment(self, state: ChatState) -> ChatState:
        """
        Analyze the sentiment of the user's message.
        
        Args:
            state: Current chat state
            
        Returns:
            Updated chat state with sentiment analysis
        """
        user_message = state["messages"][-1]["content"]
        
        # Define the prompt for sentiment analysis
        prompt = PromptTemplate.from_template(
            """You are a sentiment analyzer for a customer support chatbot.
            Analyze the sentiment of the following message and classify it as:
            - positive: The user is happy, satisfied, or expressing gratitude
            - neutral: The user is asking a question or making a neutral statement
            - negative: The user is frustrated, angry, or expressing dissatisfaction
            
            User message: {message}
            
            Sentiment:"""
        )
        
        # Run the sentiment analysis
        chain = prompt | self.llm | StrOutputParser()
        sentiment = chain.invoke({"message": user_message})
        
        # Clean up the sentiment
        sentiment = sentiment.strip().lower()
        if ":" in sentiment:
            sentiment = sentiment.split(":", 1)[1].strip()
        
        # Validate the sentiment
        valid_sentiments = ["positive", "neutral", "negative"]
        if sentiment not in valid_sentiments:
            # Find the closest match
            for valid_sentiment in valid_sentiments:
                if valid_sentiment in sentiment:
                    sentiment = valid_sentiment
                    break
            else:
                sentiment = "neutral"
        
        state["sentiment"] = sentiment
        logger.info(f"Analyzed sentiment: {sentiment}")
        return state
    
    def _retrieve_context(self, state: ChatState) -> ChatState:
        """
        Retrieve relevant context for the user's message.
        
        Args:
            state: Current chat state
            
        Returns:
            Updated chat state with retrieved context
        """
        user_message = state["messages"][-1]["content"]
        intent = state["intent"]
        
        # Retrieve relevant documents from the vector store
        docs = self.vectorstore.similarity_search(
            user_message,
            k=3,
            filter={"category": intent} if intent != "general" else None
        )
        
        state["context"] = docs
        logger.info(f"Retrieved {len(docs)} documents for context")
        return state
    
    def _generate_response(self, state: ChatState) -> ChatState:
        """
        Generate a response based on the user's message and context.
        
        Args:
            state: Current chat state
            
        Returns:
            Updated chat state with generated response
        """
        user_message = state["messages"][-1]["content"]
        language = state["language"]
        intent = state["intent"]
        sentiment = state["sentiment"]
        context = state["context"]
        
        # Extract context text
        context_text = "\n\n".join([doc.page_content for doc in context]) if context else ""
        
        # Define the prompt for response generation
        system_prompt = """You are a helpful customer support assistant for an e-commerce company.
        You help customers with their orders, product information, returns, shipping, payments, and account issues.
        You should be polite, helpful, and concise in your responses.
        
        If the user is speaking in Vietnamese, respond in Vietnamese. Otherwise, respond in English.
        
        If the user seems frustrated or angry, be extra empathetic and helpful.
        
        Use the provided context to answer the user's question if relevant.
        If the context doesn't contain the information needed, use your general knowledge but be honest about limitations.
        
        Do not make up specific order details, product information, or policies that aren't in the context.
        """
        
        # Add sentiment-specific instructions
        if sentiment == "negative":
            system_prompt += "\nThe user seems frustrated or upset. Be extra empathetic and helpful in your response."
        elif sentiment == "positive":
            system_prompt += "\nThe user seems happy or satisfied. Maintain the positive tone in your response."
        
        # Add language-specific instructions
        if language == "vi":
            system_prompt += "\nThe user is speaking in Vietnamese. Respond in Vietnamese."
        
        # Create the prompt
        prompt = PromptTemplate.from_template(
            """System: {system}
            
            Context:
            {context}
            
            User intent: {intent}
            
            User: {message}
            
            Assistant:"""
        )
        
        # Run the response generation
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({
            "system": system_prompt,
            "context": context_text,
            "intent": intent,
            "message": user_message
        })
        
        state["response"] = response
        logger.info("Generated response")
        return state
    
    def _suggest_next_steps(self, state: ChatState) -> ChatState:
        """
        Suggest next steps or follow-up questions.
        
        Args:
            state: Current chat state
            
        Returns:
            Updated chat state with suggested next steps
        """
        intent = state["intent"]
        language = state["language"]
        
        # Define follow-up suggestions based on intent
        suggestions = {
            "order_tracking": [
                "When will my order arrive?",
                "Can I change my shipping address?",
                "Why hasn't my order shipped yet?"
            ],
            "product_info": [
                "What are the specifications of this product?",
                "Is this product in stock?",
                "Does this product come with a warranty?"
            ],
            "returns": [
                "How do I return a product?",
                "What's your refund policy?",
                "Can I exchange instead of returning?"
            ],
            "shipping": [
                "What shipping methods do you offer?",
                "How much does shipping cost?",
                "Do you ship internationally?"
            ],
            "payment": [
                "What payment methods do you accept?",
                "Is there a problem with my payment?",
                "Can I pay in installments?"
            ],
            "account": [
                "How do I reset my password?",
                "How do I update my account information?",
                "Why can't I log in to my account?"
            ],
            "product_recommendation": [
                "What's the best laptop for students?",
                "Can you recommend a gaming laptop?",
                "Which laptop has the best battery life?"
            ],
            "general": [
                "How can I contact customer support?",
                "What are your business hours?",
                "Do you have any ongoing promotions?"
            ]
        }
        
        # Vietnamese translations of suggestions
        vi_suggestions = {
            "order_tracking": [
                "Khi nào đơn hàng của tôi sẽ đến?",
                "Tôi có thể thay đổi địa chỉ giao hàng không?",
                "Tại sao đơn hàng của tôi chưa được gửi đi?"
            ],
            "product_info": [
                "Thông số kỹ thuật của sản phẩm này là gì?",
                "Sản phẩm này còn hàng không?",
                "Sản phẩm này có bảo hành không?"
            ],
            "returns": [
                "Làm thế nào để trả lại sản phẩm?",
                "Chính sách hoàn tiền của bạn là gì?",
                "Tôi có thể đổi hàng thay vì trả lại không?"
            ],
            "shipping": [
                "Bạn cung cấp những phương thức vận chuyển nào?",
                "Chi phí vận chuyển là bao nhiêu?",
                "Bạn có giao hàng quốc tế không?"
            ],
            "payment": [
                "Bạn chấp nhận những phương thức thanh toán nào?",
                "Có vấn đề với thanh toán của tôi không?",
                "Tôi có thể trả góp không?"
            ],
            "account": [
                "Làm thế nào để đặt lại mật khẩu?",
                "Làm thế nào để cập nhật thông tin tài khoản?",
                "Tại sao tôi không thể đăng nhập vào tài khoản?"
            ],
            "product_recommendation": [
                "Laptop nào tốt nhất cho sinh viên?",
                "Bạn có thể giới thiệu laptop chơi game không?",
                "Laptop nào có thời lượng pin tốt nhất?"
            ],
            "general": [
                "Làm thế nào để liên hệ với bộ phận hỗ trợ khách hàng?",
                "Giờ làm việc của bạn là gì?",
                "Bạn có chương trình khuyến mãi nào đang diễn ra không?"
            ]
        }
        
        # Select suggestions based on intent and language
        if intent in suggestions:
            if language == "vi" and intent in vi_suggestions:
                next_steps = random.sample(vi_suggestions[intent], min(3, len(vi_suggestions[intent])))
            else:
                next_steps = random.sample(suggestions[intent], min(3, len(suggestions[intent])))
        else:
            if language == "vi":
                next_steps = random.sample(vi_suggestions["general"], min(3, len(vi_suggestions["general"])))
            else:
                next_steps = random.sample(suggestions["general"], min(3, len(suggestions["general"])))
        
        state["next_steps"] = next_steps
        logger.info(f"Suggested {len(next_steps)} next steps")
        return state
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow.
        
        Returns:
            StateGraph for the conversation workflow
        """
        # Create the graph
        graph = StateGraph(ChatState)
        
        # Add nodes
        graph.add_node("detect_language", self._detect_language)
        graph.add_node("classify_intent", self._classify_intent)
        graph.add_node("analyze_sentiment", self._analyze_sentiment)
        graph.add_node("retrieve_context", self._retrieve_context)
        graph.add_node("generate_response", self._generate_response)
        graph.add_node("suggest_next_steps", self._suggest_next_steps)
        
        # Define the workflow
        graph.set_entry_point("detect_language")
        graph.add_edge("detect_language", "classify_intent")
        graph.add_edge("classify_intent", "analyze_sentiment")
        graph.add_edge("analyze_sentiment", "retrieve_context")
        graph.add_edge("retrieve_context", "generate_response")
        graph.add_edge("generate_response", "suggest_next_steps")
        graph.add_edge("suggest_next_steps", END)
        
        # Compile the graph
        return graph.compile()
    
    def process_message(self, message: str, conversation_history: Optional[List[ChatMessage]] = None) -> Dict[str, Any]:
        """
        Process a user message and generate a response.
        
        Args:
            message: User message
            conversation_history: Optional conversation history
            
        Returns:
            Dictionary containing the response and other information
        """
        # Initialize conversation history if not provided
        if conversation_history is None:
            conversation_history = []
        
        # Add the user message to the conversation history
        conversation_history.append({"role": "user", "content": message})
        
        # Initialize the state
        state = ChatState(
            messages=conversation_history,
            language=None,
            intent=None,
            sentiment=None,
            context=None,
            next_steps=None,
            response=None
        )
        
        # Run the graph
        result = self.graph.invoke(state)
        
        # Add the assistant's response to the conversation history
        conversation_history.append({"role": "assistant", "content": result["response"]})
        
        # Return the result
        return {
            "response": result["response"],
            "language": result["language"],
            "intent": result["intent"],
            "sentiment": result["sentiment"],
            "next_steps": result["next_steps"],
            "conversation_history": conversation_history
        }
    
    def save(self, filename: Optional[str] = None) -> str:
        """
        Save the model.
        
        Args:
            filename: Optional filename to save to
            
        Returns:
            Path to the saved model
        """
        if filename is None:
            filename = f"langgraph_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Save model metadata
        model_path = os.path.join(self.models_dir, filename)
        
        with open(model_path, "w", encoding="utf-8") as f:
            json.dump({
                "model_type": self.model_type,
                "timestamp": datetime.now().isoformat(),
                "vectorstore_path": os.path.join(self.models_dir, "vectorstore")
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved model metadata to {model_path}")
        return model_path
    
    @classmethod
    def load(cls, model_path: str, data_dir: str = "data") -> "LangGraphModel":
        """
        Load a saved model.
        
        Args:
            model_path: Path to the saved model
            data_dir: Directory containing the data
            
        Returns:
            Loaded LangGraphModel
        """
        with open(model_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        # Create a new model with the same configuration
        model = cls(data_dir=data_dir, model_type=metadata["model_type"])
        
        logger.info(f"Loaded model from {model_path}")
        return model

# Example usage
if __name__ == "__main__":
    model = LangGraphModel()
    
    # Process a sample message
    result = model.process_message("When will my order arrive?")
    print(f"Response: {result['response']}")
    print(f"Intent: {result['intent']}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Next steps: {result['next_steps']}")
    
    # Save the model
    model_path = model.save()
    print(f"Saved model to {model_path}")
