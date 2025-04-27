import json
from datetime import datetime
from pathlib import Path
import sys
from pathlib import Path

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.chat_models import AnalyticsData
from src.services.analytics_service import AnalyticsService

def main():
    # Khởi tạo service
    analytics_service = AnalyticsService()
    
    # Lấy dữ liệu analytics
    analytics_data = analytics_service.get_analytics()
    
    # In kết quả
    print("\n=== Analytics Results ===")
    print(f"Total Conversations: {analytics_data.total_conversations}")
    print(f"Average Response Time: {analytics_data.avg_response_time:.2f} seconds")
    print(f"User Satisfaction Rate: {analytics_data.user_satisfaction_rate:.2f}%")
    
    print("\nTop Intents:")
    for intent, count in analytics_data.top_intents:
        print(f"- {intent}: {count}")
    
    print("\nTop Sentiments:")
    for sentiment, count in analytics_data.top_sentiments:
        print(f"- {sentiment}: {count}")
    
    print("\nHourly Distribution:")
    for hour, count in analytics_data.hourly_distribution:
        print(f"- {hour:02d}:00: {count} conversations")

if __name__ == "__main__":
    main() 