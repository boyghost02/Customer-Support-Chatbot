import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTester:
    def __init__(self):
        """Initialize model tester"""
        self.data_dir = Path(__file__).parent.parent / "data"
        self.model_dir = self.data_dir / "models" / "intent_classifier"
        
        # Load model and tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load label mapping
        with open(self.model_dir / "label_mapping.json", 'r') as f:
            self.label_mapping = json.load(f)
            self.id2label = self.label_mapping['id2label']
        
        # Load responses
        with open(self.data_dir / "langchain" / "training_data.json", 'r') as f:
            self.responses = json.load(f)
        
        # Load model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def get_response(self, intent: str) -> str:
        """Get response for an intent"""
        if intent in self.responses:
            # Get a random response from the list
            import random
            return random.choice(self.responses[intent])
        return "Xin lỗi, tôi không có thông tin về vấn đề này."
    
    def predict(self, text: str) -> dict:
        """Predict intent for a given text"""
        # Tokenize input
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            
            # Get top 3 predictions
            top3_probs, top3_indices = torch.topk(probs, 3)
            
            predictions = []
            for prob, idx in zip(top3_probs[0], top3_indices[0]):
                intent = self.id2label[str(idx.item())]
                predictions.append({
                    "intent": intent,
                    "confidence": prob.item(),
                    "response": self.get_response(intent)
                })
        
        return {
            "text": text,
            "predictions": predictions
        }

def main():
    """Main function to test the model"""
    # Initialize tester
    tester = ModelTester()
    
    # Test questions
    test_questions = [
        # Order tracking (English)
        "When will my order #12345 arrive?",
        "I want to track my order",
        "Has my order been delivered yet?",
        
        # Product information (English)
        "What colors does this product come in?",
        "How much RAM does this laptop have?",
        "Does this product come with warranty?",
        
        # Returns and refunds (English)
        "How can I return this product?",
        "I want a refund for this order",
        "The product is defective, I want to exchange it",
        
        # Shipping (English)
        "What is the shipping cost?",
        "How long will it take to receive my order?",
        "Do you deliver to Hanoi?",
        
        # Payment (English)
        "Can I pay with credit card?",
        "Do you accept MoMo payment?",
        "How can I pay in installments?",
        
        # Account (English)
        "I forgot my password, how can I reset it?",
        "How do I create a new account?",
        "I want to update my personal information",
        
        # Product recommendations (English)
        "Can you recommend a good laptop for students?",
        "What's the best phone under 10 million VND?",
        "I need a tablet for drawing, what do you suggest?",
        
        # General (English)
        "Hello",
        "Thank you",
        "Goodbye",
        
        # Order tracking (Vietnamese)
        "Khi nào đơn hàng #12345 của tôi sẽ đến?",
        "Tôi muốn theo dõi đơn hàng của tôi",
        "Đơn hàng của tôi đã được giao chưa?",
        
        # Product information (Vietnamese)
        "Sản phẩm này có những màu gì?",
        "Laptop này có RAM bao nhiêu?",
        "Sản phẩm này có bảo hành không?",
        
        # Returns and refunds (Vietnamese)
        "Làm thế nào để trả lại sản phẩm?",
        "Tôi muốn hoàn tiền cho đơn hàng này",
        "Sản phẩm bị lỗi, tôi muốn đổi cái khác",
        
        # Shipping (Vietnamese)
        "Phí vận chuyển là bao nhiêu?",
        "Bao lâu thì tôi nhận được hàng?",
        "Bạn có giao hàng đến Hà Nội không?",
        
        # Payment (Vietnamese)
        "Tôi có thể thanh toán bằng thẻ tín dụng không?",
        "Bạn có chấp nhận thanh toán qua MoMo không?",
        "Làm sao để thanh toán trả góp?",
        
        # Account (Vietnamese)
        "Tôi quên mật khẩu, làm sao để đặt lại?",
        "Làm thế nào để đăng ký tài khoản mới?",
        "Tôi muốn thay đổi thông tin cá nhân",
        
        # Product recommendations (Vietnamese)
        "Bạn có thể gợi ý laptop tốt cho sinh viên không?",
        "Đâu là điện thoại tốt nhất trong tầm giá 10 triệu?",
        "Tôi cần một máy tính bảng để vẽ, bạn gợi ý gì?",
        
        # General (Vietnamese)
        "Xin chào",
        "Cảm ơn bạn",
        "Tạm biệt"
    ]
    
    # Test each question
    print("\n=== Model Testing Results ===\n")
    for question in test_questions:
        result = tester.predict(question)
        print(f"\nQuestion: {result['text']}")
        print("Top 3 predictions:")
        for pred in result['predictions']:
            print(f"- {pred['intent']}: {pred['confidence']:.2%}")
            print(f"  Response: {pred['response']}")

if __name__ == "__main__":
    main() 