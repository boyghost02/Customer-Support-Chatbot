import os
import argparse
import uvicorn
from dotenv import load_dotenv

def setup_environment():
    """Load environment variables and create necessary directories"""
    # Load environment variables
    load_dotenv()
    
    # Check for required environment variables
    required_vars = [
        'OPENAI_API_KEY',
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            "Please create a .env file with these variables."
        )
    
    # Create necessary directories
    directories = [
        'data',
        'data/visualizations',
        'models',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Ensured directory exists: {directory}")

def main():
    """Main entry point for running the FastAPI application"""
    parser = argparse.ArgumentParser(description='Run the Customer Support Chatbot API')
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to run the server on'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to run the server on'
    )
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable auto-reload for development'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of worker processes'
    )
    
    args = parser.parse_args()
    
    # Setup environment
    try:
        setup_environment()
        print("Environment setup completed successfully")
    except Exception as e:
        print(f"Error setting up environment: {str(e)}")
        return
    
    # Configure uvicorn logging
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
    
    # Run the FastAPI application
    print(f"\nStarting server on {args.host}:{args.port}")
    print(f"Workers: {args.workers}")
    print(f"Auto-reload: {args.reload}")
    print("\nAPI Documentation will be available at:")
    print(f"http://{args.host}:{args.port}/docs")
    print(f"http://{args.host}:{args.port}/redoc")
    
    uvicorn.run(
        "src.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_config=log_config
    )

if __name__ == "__main__":
    main()
