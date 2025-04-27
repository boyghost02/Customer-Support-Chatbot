from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from ..models.chat_models import (
    ChatMessage, ChatResponse, OrderInfo,
    ProductInfo, Feedback
)
from ..services.chat_service import ChatService

router = APIRouter()
chat_service = ChatService()

@router.post("/message", response_model=ChatResponse)
async def process_message(message: ChatMessage) -> ChatResponse:
    """Process a user message and return a response"""
    try:
        response = await chat_service.process_message(message.message)
        return ChatResponse(**response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/order/track", response_model=OrderInfo)
async def track_order(order_id: str) -> OrderInfo:
    """Track an order by ID"""
    try:
        order_info = await chat_service.track_order(order_id)
        return OrderInfo(**order_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/product/info", response_model=ProductInfo)
async def get_product_info(product_id: str) -> ProductInfo:
    """Get product information"""
    try:
        product_info = await chat_service.get_product_info(product_id)
        return ProductInfo(**product_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/product/recommendations", response_model=List[ProductInfo])
async def get_product_recommendations() -> List[ProductInfo]:
    """Get product recommendations"""
    try:
        recommendations = await chat_service.get_product_recommendations()
        return [ProductInfo(**rec) for rec in recommendations]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback", response_model=Dict[str, Any])
async def submit_feedback(feedback: Feedback) -> Dict[str, Any]:
    """Submit user feedback"""
    try:
        result = await chat_service.submit_feedback(feedback.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/transfer", response_model=Dict[str, Any])
async def transfer_to_human() -> Dict[str, Any]:
    """Transfer conversation to human agent"""
    try:
        result = await chat_service.transfer_to_human()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 