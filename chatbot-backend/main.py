import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import custom modules
from data_collector import DataCollector
from data_analyzer import DataAnalyzer
from src.services.chatbot_service import ChatbotService

# Configure logging
logging.basicConfig(
    level=logging.getLevelName(os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("chatbot_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Customer Support Chatbot API",
    description="API for the multilingual customer support chatbot with advanced NLU capabilities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global variables
DATA_DIR = os.path.join(os.getcwd(), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize components
data_collector = DataCollector(data_dir=DATA_DIR)
data_analyzer = DataAnalyzer(data_dir=DATA_DIR)
chatbot_service = ChatbotService(data_dir=DATA_DIR)

# Store conversation histories
conversation_histories = {}

# Define Pydantic models for request/response validation
class MessageRequest(BaseModel):
    message: str
    session_id: Optional[str] = Field(default_factory=lambda: f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}_{id(datetime.now())}")

class OrderTrackingRequest(BaseModel):
    order_id: str
    session_id: Optional[str] = None

class ProductInfoRequest(BaseModel):
    product_id: Optional[str] = None
    product_name: Optional[str] = None
    session_id: Optional[str] = None

class ProductRecommendationRequest(BaseModel):
    criteria: str
    session_id: Optional[str] = None

class FeedbackRequest(BaseModel):
    session_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None

class TransferRequest(BaseModel):
    session_id: str
    reason: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    language: str = "en"  # Always English
    intent: Optional[str] = None
    next_steps: Optional[List[str]] = None

class AnalyticsResponse(BaseModel):
    total_conversations: int
    average_rating: Optional[float] = None
    language_distribution: Dict[str, int]
    intent_distribution: Dict[str, int]
    sentiment_distribution: Dict[str, int]

# Helper functions
def get_conversation_history(session_id: str) -> List[Dict[str, str]]:
    """
    Get the conversation history for a session.
    
    Args:
        session_id: Session ID
        
    Returns:
        Conversation history
    """
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []
    
    return conversation_histories[session_id]

async def collect_data_task():
    """
    Background task to collect data.
    """
    try:
        logger.info("Starting data collection task")
        data = data_collector.collect_all_data()
        logger.info(f"Collected {len(data)} records")
        return {"status": "success", "records_collected": len(data)}
    except Exception as e:
        logger.error(f"Error collecting data: {e}")
        return {"status": "error", "message": str(e)}

async def analyze_data_task():
    """
    Background task to analyze data.
    """
    try:
        logger.info("Starting data analysis task")
        df = data_collector.collect_all_data()
        df_analyzed = data_analyzer.analyze_data(df)
        data_analyzer.visualize_data(df_analyzed)
        data_analyzer.prepare_for_langchain(df_analyzed)
        logger.info(f"Analyzed {len(df_analyzed)} records")
        return {"status": "success", "records_analyzed": len(df_analyzed)}
    except Exception as e:
        logger.error(f"Error analyzing data: {e}")
        return {"status": "error", "message": str(e)}

async def reload_model_task():
    """
    Background task to reload the model.
    """
    global chatbot_service
    try:
        logger.info("Reloading model")
        chatbot_service = ChatbotService(data_dir=DATA_DIR)
        logger.info("Model reloaded successfully")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error reloading model: {e}")
        return {"status": "error", "message": str(e)}

# API endpoints
@app.get("/")
async def root():
    """
    Root endpoint.
    """
    return {
        "message": "Customer Support Chatbot API",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs"
    }

@app.post("/message", response_model=ChatResponse)
async def process_message(request: MessageRequest):
    """
    Process a user message and return a response.
    """
    try:
        # Get the conversation history
        conversation_history = get_conversation_history(request.session_id)
        
        # Process the message
        result = chatbot_service.process_message(request.message, conversation_history)
        
        # Update the conversation history
        conversation_histories[request.session_id] = result["conversation_history"]
        
        # Return the response
        return ChatResponse(
            response=result["response"],
            session_id=request.session_id,
            language=result["language"],
            intent=result["intent"],
            next_steps=result["next_steps"]
        )
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "message": "An error occurred while processing your message"
            }
        )

@app.post("/order/track", response_model=ChatResponse)
async def track_order(request: OrderTrackingRequest):
    """
    Track an order by ID.
    """
    try:
        # Generate a session ID if not provided
        session_id = request.session_id or f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}_{id(datetime.now())}"
        
        # Create a message about tracking the order
        message = f"I want to track my order #{request.order_id}"
        
        # Process the message
        conversation_history = get_conversation_history(session_id)
        result = chatbot_service.process_message(message, conversation_history)
        
        # Update the conversation history
        conversation_histories[session_id] = result["conversation_history"]
        
        # Return the response
        return ChatResponse(
            response=result["response"],
            session_id=session_id,
            language=result["language"],
            intent=result["intent"],
            next_steps=result["next_steps"]
        )
    except Exception as e:
        logger.error(f"Error tracking order: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/product/info", response_model=ChatResponse)
async def get_product_info(request: ProductInfoRequest):
    """
    Get information about a product.
    """
    try:
        # Generate a session ID if not provided
        session_id = request.session_id or f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}_{id(datetime.now())}"
        
        # Create a message about the product
        if request.product_id:
            message = f"Can you tell me about product #{request.product_id}?"
        elif request.product_name:
            message = f"Can you tell me about {request.product_name}?"
        else:
            raise HTTPException(status_code=400, detail="Either product_id or product_name must be provided")
        
        # Process the message
        conversation_history = get_conversation_history(session_id)
        result = chatbot_service.process_message(message, conversation_history)
        
        # Update the conversation history
        conversation_histories[session_id] = result["conversation_history"]
        
        # Return the response
        return ChatResponse(
            response=result["response"],
            session_id=session_id,
            language=result["language"],
            intent=result["intent"],
            next_steps=result["next_steps"]
        )
    except Exception as e:
        logger.error(f"Error getting product info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/product/recommend", response_model=ChatResponse)
async def recommend_product(request: ProductRecommendationRequest):
    """
    Get product recommendations based on criteria.
    """
    try:
        # Generate a session ID if not provided
        session_id = request.session_id or f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}_{id(datetime.now())}"
        
        # Create a message about product recommendations
        message = f"Can you recommend a product that {request.criteria}?"
        
        # Process the message
        conversation_history = get_conversation_history(session_id)
        result = chatbot_service.process_message(message, conversation_history)
        
        # Update the conversation history
        conversation_histories[session_id] = result["conversation_history"]
        
        # Return the response
        return ChatResponse(
            response=result["response"],
            session_id=session_id,
            language=result["language"],
            intent=result["intent"],
            next_steps=result["next_steps"]
        )
    except Exception as e:
        logger.error(f"Error recommending product: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest, background_tasks: BackgroundTasks):
    """
    Submit feedback for a conversation.
    """
    try:
        # Check if the session exists
        if request.session_id not in conversation_histories:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Save the feedback
        feedback_dir = os.path.join(DATA_DIR, "feedback")
        os.makedirs(feedback_dir, exist_ok=True)
        
        feedback_data = {
            "session_id": request.session_id,
            "rating": request.rating,
            "comment": request.comment,
            "timestamp": datetime.now().isoformat(),
            "conversation": conversation_histories[request.session_id]
        }
        
        feedback_path = os.path.join(feedback_dir, f"feedback_{request.session_id}.json")
        with open(feedback_path, "w", encoding="utf-8") as f:
            json.dump(feedback_data, f, ensure_ascii=False, indent=2)
        
        # Schedule a background task to analyze feedback and improve the model
        background_tasks.add_task(analyze_data_task)
        
        return {"status": "success", "message": "Feedback submitted successfully"}
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transfer")
async def transfer_to_human(request: TransferRequest):
    """
    Transfer the conversation to a human agent.
    """
    try:
        # Check if the session exists
        if request.session_id not in conversation_histories:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # In a real implementation, this would integrate with a CRM or support ticket system
        # For now, we'll just log the transfer request
        
        transfer_dir = os.path.join(DATA_DIR, "transfers")
        os.makedirs(transfer_dir, exist_ok=True)
        
        transfer_data = {
            "session_id": request.session_id,
            "reason": request.reason,
            "timestamp": datetime.now().isoformat(),
            "conversation": conversation_histories[request.session_id]
        }
        
        transfer_path = os.path.join(transfer_dir, f"transfer_{request.session_id}.json")
        with open(transfer_path, "w", encoding="utf-8") as f:
            json.dump(transfer_data, f, ensure_ascii=False, indent=2)
        
        return {
            "status": "success",
            "message": "Transfer request submitted successfully",
            "transfer_id": f"transfer_{request.session_id}"
        }
    except Exception as e:
        logger.error(f"Error transferring to human: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/conversations", response_model=AnalyticsResponse)
async def get_conversation_analytics():
    """
    Get analytics about conversations.
    """
    try:
        # In a real implementation, this would query a database
        # For now, we'll just analyze the conversation histories
        
        total_conversations = len(conversation_histories)
        
        # Analyze feedback
        feedback_dir = os.path.join(DATA_DIR, "feedback")
        if os.path.exists(feedback_dir):
            feedback_files = [f for f in os.listdir(feedback_dir) if f.endswith(".json")]
            ratings = []
            
            for file in feedback_files:
                try:
                    with open(os.path.join(feedback_dir, file), "r", encoding="utf-8") as f:
                        feedback = json.load(f)
                        if "rating" in feedback:
                            ratings.append(feedback["rating"])
                except Exception as e:
                    logger.error(f"Error reading feedback file {file}: {e}")
            
            average_rating = sum(ratings) / len(ratings) if ratings else None
        else:
            average_rating = None
        
        # Analyze languages, intents, and sentiments
        language_distribution = {}
        intent_distribution = {}
        sentiment_distribution = {}
        
        for session_id, history in conversation_histories.items():
            # Get the most recent user message
            user_messages = [msg for msg in history if msg["role"] == "user"]
            if not user_messages:
                continue
            
            # Process the message to get language, intent, and sentiment
            result = chatbot_service.process_message(user_messages[-1]["content"], history[:-1])
            
            # Update distributions
            language = result["language"]
            intent = result["intent"]
            sentiment = result["sentiment"]
            
            language_distribution[language] = language_distribution.get(language, 0) + 1
            intent_distribution[intent] = intent_distribution.get(intent, 0) + 1
            sentiment_distribution[sentiment] = sentiment_distribution.get(sentiment, 0) + 1
        
        return AnalyticsResponse(
            total_conversations=total_conversations,
            average_rating=average_rating,
            language_distribution=language_distribution,
            intent_distribution=intent_distribution,
            sentiment_distribution=sentiment_distribution
        )
    except Exception as e:
        logger.error(f"Error getting conversation analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/conversations")
async def download_conversations():
    """
    Download all conversation data.
    """
    try:
        return conversation_histories
    except Exception as e:
        logger.error(f"Error downloading conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/training/feedback")
async def submit_training_feedback(request: Request):
    """
    Submit feedback for training purposes.
    """
    try:
        data = await request.json()
        
        # Save the training feedback
        training_dir = os.path.join(DATA_DIR, "training")
        os.makedirs(training_dir, exist_ok=True)
        
        feedback_path = os.path.join(training_dir, f"training_feedback_{datetime.now().strftime('%Y%m%d%H%M%S')}.json")
        with open(feedback_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return {"status": "success", "message": "Training feedback submitted successfully"}
    except Exception as e:
        logger.error(f"Error submitting training feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/collect-data")
async def collect_data(background_tasks: BackgroundTasks):
    """
    Trigger data collection.
    """
    background_tasks.add_task(collect_data_task)
    return {"status": "success", "message": "Data collection started"}

@app.post("/analyze-data")
async def analyze_data(background_tasks: BackgroundTasks):
    """
    Trigger data analysis.
    """
    background_tasks.add_task(analyze_data_task)
    return {"status": "success", "message": "Data analysis started"}

@app.post("/reload-model")
async def reload_model(background_tasks: BackgroundTasks):
    """
    Reload the chatbot model.
    """
    background_tasks.add_task(reload_model_task)
    return {"status": "success", "message": "Model reload started"}

# Error handlers
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """
    Generic exception handler.
    """
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc)}
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """
    Startup event handler.
    """
    logger.info("Starting up the chatbot API")
    
    # Create necessary directories
    os.makedirs(os.path.join(DATA_DIR, "raw"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "processed"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "analyzed"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "models"), exist_ok=True)
    
    # Initialize the model
    chatbot_service = ChatbotService(data_dir=DATA_DIR)

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """
    Shutdown event handler.
    """
    logger.info("Shutting down the chatbot API")
    
    # Save the model if it exists
    global chatbot_service
    if chatbot_service is not None:
        try:
            chatbot_service.save()
        except Exception as e:
            logger.error(f"Error saving model: {e}")

# Run the application
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    debug = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
    
    uvicorn.run("main:app", host=host, port=port, reload=debug)
