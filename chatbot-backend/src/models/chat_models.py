from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class MessageType(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class Intent(str, Enum):
    """Chat intents"""
    GREETING = "greeting"
    ORDER_STATUS = "order_status"
    PRODUCT_INFO = "product_info"
    SHIPPING = "shipping"
    RETURNS = "returns"
    COMPLAINT = "complaint"
    FEEDBACK = "feedback"
    UNKNOWN = "unknown"

class Sentiment(str, Enum):
    """Message sentiment"""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"

class ChatMessage(BaseModel):
    """Chat message model"""
    message: str
    user_id: Optional[str] = None
    timestamp: Optional[str] = None
    language: Optional[str] = "en"

class ChatResponse(BaseModel):
    """Chat response model"""
    message: str
    intent: Intent
    sentiment: Sentiment
    confidence: float
    suggested_actions: List[str]
    language: str = "en"

class OrderInfo(BaseModel):
    """Order information model"""
    order_id: str
    status: str
    estimated_delivery: str
    tracking_number: Optional[str] = None

class ProductInfo(BaseModel):
    """Product information model"""
    product_id: str
    name: str
    price: float
    description: str
    availability: str
    category: Optional[str] = None

class Feedback(BaseModel):
    """User feedback model"""
    user_id: str
    rating: int
    comment: str
    timestamp: str
    category: Optional[str] = None

class AnalyticsData(BaseModel):
    """Analytics data model"""
    total_conversations: int
    average_response_time: float
    intent_distribution: dict
    sentiment_distribution: dict
    common_issues: List[str]
    user_satisfaction: float

class UserPreferences(BaseModel):
    """Model for user preferences"""
    language: str = Field(default="en", description="Preferred language")
    notification_enabled: bool = Field(default=True)
    theme: str = Field(default="light", description="UI theme preference")
    timezone: str = Field(default="UTC", description="User's timezone")

class OrderStatus(BaseModel):
    """Model for order status"""
    order_id: str = Field(..., description="The order ID")
    status: str = Field(..., description="Current status of the order")
    estimated_delivery: Optional[datetime] = None
    tracking_number: Optional[str] = None
    items: List[Dict[str, Any]] = Field(default_factory=list)

class ProductInfo(BaseModel):
    """Model for product information"""
    product_id: str = Field(..., description="The product ID")
    name: str = Field(..., description="Product name")
    description: str = Field(..., description="Product description")
    price: float = Field(..., description="Product price")
    stock: int = Field(..., description="Current stock level")
    specifications: Dict[str, Any] = Field(default_factory=dict) 