from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings"""
    # Base paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODEL_DIR: Path = BASE_DIR / "models"
    
    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Customer Support Chatbot"
    
    # Model settings
    MODEL_NAME: str = "bert-base-uncased"
    MAX_LENGTH: int = 128
    BATCH_SIZE: int = 16
    EPOCHS: int = 3
    LEARNING_RATE: float = 2e-5
    
    # OpenAI settings
    OPENAI_API_KEY: Optional[str] = None
    
    # Database settings
    DATABASE_URL: Optional[str] = None
    
    # Security settings
    SECRET_KEY: str = "your-secret-key-here"  # Change in production
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # CORS
    BACKEND_CORS_ORIGINS: list[str] = ["*"]
    
    class Config:
        case_sensitive = True
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

# Create global settings instance
settings = get_settings() 