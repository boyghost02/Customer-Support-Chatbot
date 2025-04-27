from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import logging

from ..core.config import settings
from ..models.chat_models import ChatMessage, ChatResponse, OrderStatus, ProductInfo, Feedback
from ..services.chat_service import ChatService

# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chat service
chat_service = ChatService()

@app.post("/api/chat/message", response_model=ChatResponse)
async def process_message(message: ChatMessage):
    """Process a user message and return a response"""
    try:
        response = await chat_service.process_message(message.content)
        return response
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/order/track", response_model=OrderStatus)
async def track_order(order_id: str):
    """Track an order by ID"""
    try:
        status = await chat_service.track_order(order_id)
        return status
    except Exception as e:
        logger.error(f"Error tracking order: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/product/info", response_model=ProductInfo)
async def get_product_info(product_id: str):
    """Get product information"""
    try:
        info = await chat_service.get_product_info(product_id)
        return info
    except Exception as e:
        logger.error(f"Error getting product info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/product/recommend")
async def get_product_recommendations(user_id: str):
    """Get personalized product recommendations"""
    try:
        recommendations = await chat_service.get_product_recommendations(user_id)
        return recommendations
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/feedback", response_model=Dict[str, Any])
async def submit_feedback(feedback: Feedback):
    """Submit user feedback"""
    try:
        result = await chat_service.submit_feedback(feedback.dict())
        return result
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/transfer")
async def transfer_to_human(reason: str):
    """Transfer conversation to human agent"""
    try:
        result = await chat_service.transfer_to_human(reason)
        return result
    except Exception as e:
        logger.error(f"Error transferring to human: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"} 