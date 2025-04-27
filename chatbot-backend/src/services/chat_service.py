import logging
import json
import torch
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from pathlib import Path
from langchain.llms import OpenAI
from langchain.chains import ConversationChain

from ..models.chat_models import ChatMessage, ChatResponse, Intent, Sentiment
from ..core.config import settings

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self):
        """Initialize chat service"""
        self.model_dir = Path("../models")
        self.data_dir = Path("../data")
        
        # Load models and tokenizers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load intent classifier
        self.intent_tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir / "intent_classifier"
        )
        self.intent_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir / "intent_classifier"
        ).to(self.device)
        
        # Load sentiment analyzer
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(f"{settings.MODEL_SAVE_PATH}/sentiment_analyzer")
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(f"{settings.MODEL_SAVE_PATH}/sentiment_analyzer")
        self.sentiment_model.to(self.device)
        
        # Initialize LangChain components
        self.llm = OpenAI(temperature=0.7)
        self.memory = ConversationBufferMemory()
        
        # Load LangChain training data
        with open(self.data_dir / "langchain" / "training_data.json", 'r') as f:
            self.langchain_data = json.load(f)
        
        # Define response prompt template
        self.response_prompt = """
        Based on the following context and user message, generate a helpful response:
        
        Context: {history}
        User Message: {input}
        Detected Intent: {intent}
        
        Response:
        """
        
        # Initialize response chain
        self.response_chain = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=self.response_prompt
        )
    
    def detect_intent(self, message: str) -> Dict[str, Any]:
        """Detect intent from user message"""
        try:
            # Tokenize input
            inputs = self.intent_tokenizer(
                message,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.intent_model(**inputs)
                predictions = torch.softmax(outputs.logits, dim=1)
                intent_idx = torch.argmax(predictions).item()
                confidence = predictions[0][intent_idx].item()
            
            # Get intent label
            intent_label = self.intent_model.config.id2label[intent_idx]
            
            return {
                "intent": intent_label,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Error detecting intent: {str(e)}")
            raise
    
    def generate_response(self, message: str, intent: str) -> str:
        """Generate response using LangChain"""
        try:
            # Get relevant examples for the intent
            examples = self.langchain_data.get(intent, [])
            
            # Generate response
            response = self.response_chain.predict(
                input=message,
                intent=intent,
                examples=examples
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def get_suggested_actions(self, intent: str) -> List[str]:
        """Get suggested actions based on intent"""
        try:
            # Define intent to actions mapping
            intent_actions = {
                "order_status": ["Track Order", "View Order History"],
                "product_info": ["View Product Details", "Check Availability"],
                "shipping": ["Check Shipping Status", "Update Shipping Address"],
                "returns": ["Start Return", "View Return Policy"],
                "general": ["Contact Support", "View FAQ"]
            }
            
            return intent_actions.get(intent, ["Contact Support"])
            
        except Exception as e:
            logger.error(f"Error getting suggested actions: {str(e)}")
            raise
    
    async def process_message(self, message: str) -> Dict[str, Any]:
        """Process user message and generate response"""
        try:
            # Detect intent
            intent_result = self.detect_intent(message)
            
            # Generate response
            response = self.generate_response(
                message,
                intent_result["intent"]
            )
            
            # Get suggested actions
            suggested_actions = self.get_suggested_actions(
                intent_result["intent"]
            )
            
            return {
                "message": response,
                "intent": intent_result["intent"],
                "confidence": intent_result["confidence"],
                "suggested_actions": suggested_actions
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            raise
    
    async def track_order(self, order_id: str) -> Dict[str, Any]:
        """Track order status"""
        try:
            # TODO: Implement order tracking logic
            return {
                "status": "processing",
                "estimated_delivery": "2024-03-20",
                "tracking_number": "TRK123456"
            }
        except Exception as e:
            logger.error(f"Error tracking order: {str(e)}")
            raise
    
    async def get_product_info(self, product_id: str) -> Dict[str, Any]:
        """Get product information"""
        try:
            # TODO: Implement product info retrieval
            return {
                "name": "Sample Product",
                "price": 99.99,
                "description": "Product description",
                "availability": "in_stock"
            }
        except Exception as e:
            logger.error(f"Error getting product info: {str(e)}")
            raise
    
    async def get_product_recommendations(self) -> List[Dict[str, Any]]:
        """Get product recommendations"""
        try:
            # TODO: Implement product recommendations
            return [
                {
                    "id": "1",
                    "name": "Recommended Product 1",
                    "price": 49.99
                },
                {
                    "id": "2",
                    "name": "Recommended Product 2",
                    "price": 79.99
                }
            ]
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            raise
    
    async def submit_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Submit user feedback"""
        try:
            # TODO: Implement feedback submission
            return {
                "status": "success",
                "message": "Thank you for your feedback!"
            }
        except Exception as e:
            logger.error(f"Error submitting feedback: {str(e)}")
            raise
    
    async def transfer_to_human(self) -> Dict[str, Any]:
        """Transfer conversation to human agent"""
        try:
            # TODO: Implement human transfer logic
            return {
                "status": "transferring",
                "estimated_wait": "2 minutes",
                "agent_id": "AGENT123"
            }
        except Exception as e:
            logger.error(f"Error transferring to human: {str(e)}")
            raise

def main():
    """Main function to run chat service"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize chat service
    chat_service = ChatService()
    
    # Example usage
    async def test_chat():
        response = await chat_service.process_message(
            "I want to check my order status"
        )
        print(response)
    
    import asyncio
    asyncio.run(test_chat())

if __name__ == "__main__":
    main() 