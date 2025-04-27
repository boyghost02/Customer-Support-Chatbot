from fastapi import APIRouter
from src.models.chat_models import AnalyticsData
from src.services.analytics_service import AnalyticsService

router = APIRouter()
analytics_service = AnalyticsService()

@router.get("/analytics", response_model=AnalyticsData)
async def get_analytics():
    return analytics_service.get_analytics() 