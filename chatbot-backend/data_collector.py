import os
import json
import pandas as pd
import requests
from bs4 import BeautifulSoup
import logging
import random
from datetime import datetime
from typing import Dict, List, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataCollector:
    """
    Collects customer support data from various sources for the chatbot.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the DataCollector.
        
        Args:
            data_dir: Directory to store collected data
        """
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        
        # Create directories if they don't exist
        os.makedirs(self.raw_dir, exist_ok=True)
        
        # Define predefined categories
        self.categories = [
            "order_tracking", 
            "product_info", 
            "returns", 
            "shipping", 
            "payment", 
            "account", 
            "product_recommendation",
            "general"
        ]
        
        logger.info(f"DataCollector initialized with data directory: {data_dir}")
    
    def collect_all_data(self) -> pd.DataFrame:
        """
        Collect data from all available sources and combine into a single DataFrame.
        
        Returns:
            DataFrame containing all collected data
        """
        logger.info("Starting data collection from all sources")
        
        # Collect data from each source
        twitter_data = self.collect_twitter_data()
        ecommerce_data = self.collect_ecommerce_faq_data()
        product_data = self.collect_product_data()
        
        # Combine all data
        all_data = pd.concat([twitter_data, ecommerce_data, product_data], ignore_index=True)
        
        # Save the combined data
        output_path = os.path.join(self.raw_dir, f"combined_data_{datetime.now().strftime('%Y%m%d')}.csv")
        all_data.to_csv(output_path, index=False)
        logger.info(f"Saved combined data to {output_path}")
        
        return all_data
    
    def collect_twitter_data(self) -> pd.DataFrame:
        """
        Collect customer support data from Twitter dataset.
        
        Returns:
            DataFrame containing Twitter customer support data
        """
        logger.info("Collecting Twitter customer support data")
        
        # In a real implementation, this would download or access the Twitter Customer Support dataset
        # For this example, we'll generate synthetic data
        
        # Sample customer support questions and answers
        questions = [
            "When will my order #12345 arrive?",
            "I haven't received my order yet, it's been 5 days",
            "How do I track my package?",
            "My order status hasn't updated in 3 days",
            "Is there a delay in shipping to Vietnam?",
            "Can I change my shipping address?",
            "How long does shipping take to Ho Chi Minh City?",
            "My package says delivered but I didn't receive it",
            "Do you ship internationally?",
            "What's the status of my return?",
            "How do I return a damaged product?",
            "Can I get a refund without returning the item?",
            "What's your return policy?",
            "How long do refunds take to process?",
            "Can I exchange instead of returning?",
            "Do you cover return shipping costs?",
            "I received the wrong item, what should I do?",
            "Can I pay with PayPal?",
            "My payment was declined, what should I do?",
            "Do you offer installment payments?",
            "Is there a discount for first-time customers?",
            "Can I use multiple discount codes?",
            "When will item X be back in stock?",
            "Do you have this in a different color?",
            "What's the difference between model A and B?",
            "Is this product compatible with X?",
            "Can you recommend a good laptop for gaming?",
            "What's the best laptop for students?",
            "I need a laptop with good battery life",
            "Which laptop is better for video editing?"
        ]
        
        answers = [
            "Your order #12345 is currently in transit and expected to arrive on March 15th. You can track it using the link in your confirmation email.",
            "I'm sorry for the delay. Let me check the status of your order. Could you please provide your order number?",
            "You can track your package by clicking the 'Track Order' link in your confirmation email or by logging into your account on our website.",
            "I apologize for the lack of updates. Sometimes tracking information can be delayed. Let me check the status for you.",
            "We're currently experiencing some delays in shipping to Vietnam due to customs processing. Deliveries may take an additional 3-5 business days.",
            "You can change your shipping address if your order hasn't been processed yet. Please log into your account and update it in the order details.",
            "Standard shipping to Ho Chi Minh City typically takes 7-10 business days from the date of shipment.",
            "I'm sorry to hear that. Let's verify the delivery address and check with the carrier for more information.",
            "Yes, we ship to over 100 countries worldwide. Shipping costs and delivery times vary by location.",
            "I'd be happy to check on your return status. Could you please provide your order number?",
            "To return a damaged product, please take a photo of the damage and contact our support team with your order number and the photos.",
            "For damaged or defective items, we can process a refund without requiring a return in some cases. Please provide details about the issue.",
            "Our return policy allows returns within 30 days of delivery for unused items in original packaging. Some products have specific exceptions.",
            "Refunds are typically processed within 3-5 business days after we receive your return. It may take an additional 5-7 days to appear on your statement.",
            "Yes, you can request an exchange instead of a return. Please contact our support team with your order number and the item you'd like instead.",
            "Yes, we provide a prepaid return shipping label for defective items or if the return is due to our error. For other returns, shipping costs are the customer's responsibility.",
            "I apologize for the mix-up. Please take a photo of the item you received and contact our support team with your order number.",
            "Yes, we accept PayPal as a payment method, along with major credit cards and bank transfers.",
            "Payment declines can happen for various reasons. Please verify your payment details, ensure you have sufficient funds, or try a different payment method.",
            "Yes, we offer installment payments through Klarna and Afterpay for orders over $50.",
            "Yes, new customers can use the code WELCOME10 for 10% off their first purchase.",
            "We allow only one discount code per order. Please use the code that offers the best value for your purchase.",
            "We expect item X to be back in stock within 2 weeks. You can sign up for stock notifications on the product page.",
            "Currently, this item is available in black, white, and navy blue. The red version is sold out.",
            "Model A has 8GB RAM and 256GB storage, while Model B offers 16GB RAM and 512GB storage. Model B also has a faster processor.",
            "To determine compatibility, please provide the specific model of your device, and I can check for you.",
            "For gaming, I'd recommend our XYZ Gaming Laptop with an NVIDIA RTX 3070 GPU, 16GB RAM, and a high-refresh-rate display.",
            "For students, our ABC Ultrabook is popular due to its lightweight design, 12-hour battery life, and affordable price.",
            "Our DEF Ultrabook offers up to 15 hours of battery life and is perfect for users who need all-day productivity.",
            "For video editing, I'd recommend our GHI Pro with an 8-core processor, 32GB RAM, and dedicated graphics card."
        ]
        
        # Generate synthetic data
        num_samples = 200
        data = []
        
        for _ in range(num_samples):
            idx = random.randint(0, len(questions) - 1)
            question = questions[idx]
            answer = answers[idx]
            
            # Determine category based on content
            if "order" in question.lower() or "track" in question.lower():
                category = "order_tracking"
            elif "return" in question.lower() or "refund" in question.lower():
                category = "returns"
            elif "ship" in question.lower() or "deliver" in question.lower():
                category = "shipping"
            elif "pay" in question.lower():
                category = "payment"
            elif "recommend" in question.lower() or "best" in question.lower() or "good" in question.lower():
                category = "product_recommendation"
            elif "laptop" in question.lower() or "product" in question.lower() or "item" in question.lower():
                category = "product_info"
            else:
                category = "general"
            
            # Add some Vietnamese questions and answers
            if random.random() < 0.2:  # 20% chance of Vietnamese
                if category == "order_tracking":
                    question = f"Đơn hàng #{random.randint(10000, 99999)} của tôi khi nào sẽ đến?"
                    answer = f"Đơn hàng của bạn hiện đang được vận chuyển và dự kiến sẽ đến vào ngày {random.randint(1, 30)} tháng {random.randint(1, 12)}."
                elif category == "returns":
                    question = "Làm thế nào để trả lại sản phẩm bị lỗi?"
                    answer = "Để trả lại sản phẩm bị lỗi, vui lòng chụp ảnh sản phẩm và liên hệ với đội ngũ hỗ trợ của chúng tôi kèm theo số đơn hàng và hình ảnh."
            
            data.append({
                "question": question,
                "answer": answer,
                "category": category,
                "source": "twitter_customer_support",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        df = pd.DataFrame(data)
        
        # Save to file
        output_path = os.path.join(self.raw_dir, "twitter_data.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"Saved Twitter data to {output_path} ({len(df)} records)")
        
        return df
    
    def collect_ecommerce_faq_data(self) -> pd.DataFrame:
        """
        Collect FAQ data from e-commerce websites.
        
        Returns:
            DataFrame containing e-commerce FAQ data
        """
        logger.info("Collecting e-commerce FAQ data")
        
        # In a real implementation, this would scrape FAQ pages from e-commerce sites
        # For this example, we'll generate synthetic data
        
        # Sample FAQ questions and answers
        faqs = [
            {
                "question": "How do I create an account?",
                "answer": "To create an account, click on the 'Sign Up' button in the top right corner of our website. Fill in your email address, create a password, and provide your name and shipping address.",
                "category": "account"
            },
            {
                "question": "I forgot my password. How do I reset it?",
                "answer": "Click on the 'Login' button, then select 'Forgot Password'. Enter your email address, and we'll send you a link to reset your password.",
                "category": "account"
            },
            {
                "question": "How can I track my order?",
                "answer": "You can track your order by logging into your account and viewing your order history. Click on the specific order to see its current status and tracking information.",
                "category": "order_tracking"
            },
            {
                "question": "What payment methods do you accept?",
                "answer": "We accept Visa, Mastercard, American Express, PayPal, and bank transfers. For orders within Vietnam, we also accept MoMo and VNPay.",
                "category": "payment"
            },
            {
                "question": "Do you ship internationally?",
                "answer": "Yes, we ship to over 100 countries worldwide. Shipping costs and delivery times vary by location. You can see the shipping options available to your country during checkout.",
                "category": "shipping"
            },
            {
                "question": "What is your return policy?",
                "answer": "We accept returns within 30 days of delivery. Items must be unused and in their original packaging. Some products like electronics and personal care items may have specific return restrictions.",
                "category": "returns"
            },
            {
                "question": "How long will shipping take?",
                "answer": "Domestic shipping within Vietnam typically takes 1-3 business days. International shipping can take 7-21 business days depending on the destination country and shipping method selected.",
                "category": "shipping"
            },
            {
                "question": "Can I change or cancel my order?",
                "answer": "You can change or cancel your order within 1 hour of placing it. After that, if the order has not been shipped, you can contact our customer support team to request changes or cancellation.",
                "category": "order_tracking"
            },
            {
                "question": "Do you offer warranty on your products?",
                "answer": "Yes, all our products come with a minimum 12-month warranty against manufacturing defects. Some products have extended warranty options available for purchase.",
                "category": "product_info"
            },
            {
                "question": "How do I contact customer support?",
                "answer": "You can contact our customer support team via email at support@example.com, by phone at +84 123 456 789 (9 AM - 6 PM, Monday to Friday), or through the live chat feature on our website.",
                "category": "general"
            },
            {
                "question": "Làm thế nào để theo dõi đơn hàng của tôi?",
                "answer": "Bạn có thể theo dõi đơn hàng bằng cách đăng nhập vào tài khoản và xem lịch sử đơn hàng. Nhấp vào đơn hàng cụ thể để xem trạng thái hiện tại và thông tin theo dõi.",
                "category": "order_tracking"
            },
            {
                "question": "Các phương thức thanh toán nào được chấp nhận?",
                "answer": "Chúng tôi chấp nhận Visa, Mastercard, American Express, PayPal và chuyển khoản ngân hàng. Đối với đơn đặt hàng trong Việt Nam, chúng tôi cũng chấp nhận MoMo và VNPay.",
                "category": "payment"
            }
        ]
        
        # Generate synthetic data
        num_samples = 150
        data = []
        
        for _ in range(num_samples):
            faq = random.choice(faqs)
            
            data.append({
                "question": faq["question"],
                "answer": faq["answer"],
                "category": faq["category"],
                "source": "ecommerce_faq",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        df = pd.DataFrame(data)
        
        # Save to file
        output_path = os.path.join(self.raw_dir, "ecommerce_faq_data.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"Saved e-commerce FAQ data to {output_path} ({len(df)} records)")
        
        return df
    
    def collect_product_data(self) -> pd.DataFrame:
        """
        Collect product information data.
        
        Returns:
            DataFrame containing product information data
        """
        logger.info("Collecting product information data")
        
        # In a real implementation, this would scrape product information from e-commerce sites
        # For this example, we'll generate synthetic data for laptop products
        
        # Sample laptop products
        laptops = [
            {
                "name": "TechPro UltraBook X1",
                "specs": "13.3\" FHD Display, Intel Core i5, 8GB RAM, 256GB SSD, Windows 11",
                "price": "799.99",
                "category": "Ultrabook",
                "description": "A slim and lightweight laptop perfect for everyday use and productivity tasks."
            },
            {
                "name": "TechPro UltraBook X3",
                "specs": "14\" QHD Display, Intel Core i7, 16GB RAM, 512GB SSD, Windows 11",
                "price": "1199.99",
                "category": "Ultrabook",
                "description": "Premium ultrabook with powerful performance and long battery life."
            },
            {
                "name": "GameMaster Pro 15",
                "specs": "15.6\" FHD 144Hz Display, AMD Ryzen 7, 16GB RAM, 1TB SSD, NVIDIA RTX 3060, Windows 11",
                "price": "1299.99",
                "category": "Gaming",
                "description": "High-performance gaming laptop with advanced cooling system and RGB keyboard."
            },
            {
                "name": "GameMaster Elite 17",
                "specs": "17.3\" QHD 165Hz Display, Intel Core i9, 32GB RAM, 2TB SSD, NVIDIA RTX 3080, Windows 11",
                "price": "2499.99",
                "category": "Gaming",
                "description": "Ultimate gaming experience with desktop-class performance and immersive display."
            },
            {
                "name": "WorkStation Pro X7",
                "specs": "15.6\" 4K Display, Intel Xeon, 64GB RAM, 2TB SSD, NVIDIA Quadro RTX 5000, Windows 11 Pro",
                "price": "3499.99",
                "category": "Workstation",
                "description": "Professional workstation for 3D modeling, video editing, and other demanding tasks."
            },
            {
                "name": "StudentBook Air",
                "specs": "13.3\" HD Display, Intel Core i3, 8GB RAM, 128GB SSD, Windows 11 S",
                "price": "499.99",
                "category": "Budget",
                "description": "Affordable laptop for students with essential features and good battery life."
            },
            {
                "name": "BusinessBook Elite",
                "specs": "14\" FHD Display, Intel Core i7, 16GB RAM, 512GB SSD, Windows 11 Pro",
                "price": "1399.99",
                "category": "Business",
                "description": "Secure and reliable laptop for business professionals with enhanced security features."
            },
            {
                "name": "CreatorPro 16",
                "specs": "16\" 4K OLED Display, AMD Ryzen 9, 32GB RAM, 1TB SSD, NVIDIA RTX 3070, Windows 11 Pro",
                "price": "2199.99",
                "category": "Creator",
                "description": "Designed for creative professionals with color-accurate display and powerful graphics."
            }
        ]
        
        # Generate Q&A pairs for each laptop
        data = []
        
        for laptop in laptops:
            # Basic information question
            data.append({
                "question": f"What are the specifications of the {laptop['name']}?",
                "answer": f"The {laptop['name']} features {laptop['specs']}. It's priced at ${laptop['price']}.",
                "category": "product_info",
                "source": "product_database",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # Price question
            data.append({
                "question": f"How much does the {laptop['name']} cost?",
                "answer": f"The {laptop['name']} is priced at ${laptop['price']}.",
                "category": "product_info",
                "source": "product_database",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # Use case question
            if "Gaming" in laptop["category"]:
                data.append({
                    "question": f"Is the {laptop['name']} good for gaming?",
                    "answer": f"Yes, the {laptop['name']} is specifically designed for gaming with its {laptop['specs']}. It offers excellent performance for modern games.",
                    "category": "product_recommendation",
                    "source": "product_database",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            elif "Workstation" in laptop["category"]:
                data.append({
                    "question": f"Can I use the {laptop['name']} for video editing?",
                    "answer": f"Absolutely! The {laptop['name']} is a powerful workstation with {laptop['specs']}, making it excellent for video editing, 3D modeling, and other demanding creative tasks.",
                    "category": "product_recommendation",
                    "source": "product_database",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            elif "Budget" in laptop["category"] or "Student" in laptop["name"]:
                data.append({
                    "question": f"Is the {laptop['name']} suitable for students?",
                    "answer": f"Yes, the {laptop['name']} is an excellent choice for students. It's affordable at ${laptop['price']} while offering {laptop['specs']}, which is perfect for schoolwork, research, and everyday tasks.",
                    "category": "product_recommendation",
                    "source": "product_database",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            
            # Add some Vietnamese questions for variety
            if random.random() < 0.3:  # 30% chance
                data.append({
                    "question": f"Laptop {laptop['name']} có phù hợp cho công việc văn phòng không?",
                    "answer": f"Laptop {laptop['name']} với {laptop['specs']} rất phù hợp cho công việc văn phòng, lướt web và các tác vụ hàng ngày.",
                    "category": "product_recommendation",
                    "source": "product_database",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
        
        # Add general laptop recommendation questions
        general_recommendations = [
            {
                "question": "What's the best laptop for programming?",
                "answer": "For programming, we recommend the BusinessBook Elite or WorkStation Pro X7. They offer powerful processors, ample RAM, and comfortable keyboards for long coding sessions.",
                "category": "product_recommendation"
            },
            {
                "question": "I need a laptop with good battery life for travel",
                "answer": "The TechPro UltraBook X3 would be perfect for travel with up to 12 hours of battery life, lightweight design, and powerful enough for most tasks.",
                "category": "product_recommendation"
            },
            {
                "question": "What laptop do you recommend for a college student?",
                "answer": "For college students, we recommend the StudentBook Air or TechPro UltraBook X1. They're affordable, portable, and have enough power for research, writing papers, and online classes.",
                "category": "product_recommendation"
            },
            {
                "question": "Which laptop is best for video editing?",
                "answer": "For video editing, we recommend the CreatorPro 16 or WorkStation Pro X7. They have powerful processors, dedicated graphics cards, and high-resolution displays for precise editing.",
                "category": "product_recommendation"
            },
            {
                "question": "I'm looking for a gaming laptop under $1500",
                "answer": "The GameMaster Pro 15 at $1299.99 is an excellent gaming laptop under $1500. It features an AMD Ryzen 7 processor, NVIDIA RTX 3060 graphics, and a 144Hz display for smooth gaming.",
                "category": "product_recommendation"
            },
            {
                "question": "Laptop nào tốt nhất cho sinh viên?",
                "answer": "Đối với sinh viên, chúng tôi khuyên dùng StudentBook Air hoặc TechPro UltraBook X1. Chúng có giá cả phải chăng, nhẹ và đủ mạnh cho nghiên cứu, viết bài và học trực tuyến.",
                "category": "product_recommendation"
            }
        ]
        
        for rec in general_recommendations:
            data.append({
                "question": rec["question"],
                "answer": rec["answer"],
                "category": rec["category"],
                "source": "product_database",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        df = pd.DataFrame(data)
        
        # Save to file
        output_path = os.path.join(self.raw_dir, "product_data.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"Saved product data to {output_path} ({len(df)} records)")
        
        return df

# Example usage
if __name__ == "__main__":
    collector = DataCollector()
    data = collector.collect_all_data()
    print(f"Collected {len(data)} records in total")
