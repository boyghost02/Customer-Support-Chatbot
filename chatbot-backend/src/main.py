from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import logging
from services.chatbot_service import ChatbotService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()
chatbot = ChatbotService()

class MessageRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    conversation_history: Optional[List[Dict]] = None

@app.post("/message")
async def process_message(request: MessageRequest):
    try:
        response = chatbot.process_message(
            message=request.message,
            conversation_history=request.conversation_history,
            session_id=request.session_id
        )
        return response
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 