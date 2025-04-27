import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import API_CONFIG from './config';

// Message types
const MESSAGE_TYPES = {
  USER: 'user',
  BOT: 'bot',
  SYSTEM: 'system'
};

// Language-specific suggestions
const SUGGESTIONS = {
  en: [
    "Track my order",
    "Product information",
    "Return policy",
    "Shipping information",
    "Payment methods",
    "Contact support"
  ],
  vi: [
    "Theo dõi đơn hàng",
    "Thông tin sản phẩm",
    "Chính sách đổi trả",
    "Thông tin vận chuyển",
    "Phương thức thanh toán",
    "Liên hệ hỗ trợ"
  ]
};

// Add translations for common phrases
const TRANSLATIONS = {
  en: {
    next_steps: "Next steps",
    track_order: "Track my order",
    product_info: "Product information",
    return_policy: "Return policy",
    shipping_info: "Shipping information",
    payment_methods: "Payment methods",
    contact_support: "Contact support"
  },
  vi: {
    next_steps: "Các bước tiếp theo",
    track_order: "Theo dõi đơn hàng",
    product_info: "Thông tin sản phẩm",
    return_policy: "Chính sách đổi trả",
    shipping_info: "Thông tin vận chuyển",
    payment_methods: "Phương thức thanh toán",
    contact_support: "Liên hệ hỗ trợ"
  }
};

// Function to translate next steps
const translateNextSteps = (steps, currentLanguage) => {
  if (!steps || steps.length === 0) return [];
  
  return steps.map(step => {
    // Try to find a direct translation
    const translation = TRANSLATIONS[currentLanguage][step.toLowerCase().replace(/\s+/g, '_')];
    if (translation) {
      return translation;
    }
    
    // If no direct translation found, return the original step
    return step;
  });
};

// Chat message component
const ChatMessage = ({ message, onSuggestionClick }) => {
  // Function to format message text with line breaks
  const formatMessage = (text) => {
    return text.split('\n').map((line, index) => (
      <React.Fragment key={index}>
        {line}
        {index < text.split('\n').length - 1 && <br />}
      </React.Fragment>
    ));
  };

  return (
    <div className={`message ${message.type}`}>
      {formatMessage(message.text)}
      {message.type === MESSAGE_TYPES.BOT && message.suggestions && message.suggestions.length > 0 && (
        <div className="suggestions">
          {message.suggestions.map((suggestion, index) => (
            <button 
              key={index} 
              className="suggestion-btn"
              onClick={() => onSuggestionClick(suggestion)}
            >
              {suggestion}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

// Developer login modal component
const DevLoginModal = ({ onLogin, onCancel, language }) => {
  const [password, setPassword] = useState('');
  
  const handleSubmit = (e) => {
    e.preventDefault();
    onLogin(password);
  };
  
  return (
    <div className="dev-login-modal">
      <div className="dev-login-content">
        <h2>{language === 'en' ? 'Developer Login' : 'Đăng nhập nhà phát triển'}</h2>
        <p>{language === 'en' 
          ? 'Enter developer password to access analytics' 
          : 'Nhập mật khẩu nhà phát triển để truy cập phân tích'}
        </p>
        <form onSubmit={handleSubmit}>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder={language === 'en' ? 'Password' : 'Mật khẩu'}
            autoFocus
          />
          <div className="dev-login-buttons">
            <button type="button" onClick={onCancel}>
              {language === 'en' ? 'Cancel' : 'Hủy'}
            </button>
            <button type="submit">
              {language === 'en' ? 'Login' : 'Đăng nhập'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

function App() {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [language, setLanguage] = useState('en');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState('');
  const [userId, setUserId] = useState('');
  const [connectionError, setConnectionError] = useState(false);
  const [showAnalytics, setShowAnalytics] = useState(false);
  const [showDevLogin, setShowDevLogin] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Update useEffect to use only welcome message without hardcoded suggestions
  useEffect(() => {
    setSessionId(`session_${Math.random().toString(36).substring(2, 9)}`);
    setUserId(`user_${Math.random().toString(36).substring(2, 9)}`);
    
    // Add welcome message based on language
    const welcomeMessage = language === 'en' 
      ? "Hello! I'm your customer support assistant. How can I help you today?"
      : "Xin chào! Tôi là trợ lý hỗ trợ khách hàng. Tôi có thể giúp gì cho bạn hôm nay?";
    
    setMessages([
      {
        text: welcomeMessage,
        type: MESSAGE_TYPES.BOT,
        suggestions: SUGGESTIONS[language]
      }
    ]);
    
    // Focus the input field
    if (inputRef.current) {
      inputRef.current.focus();
    }
  }, [language]);

  // Auto-scroll to bottom of messages
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleInputChange = (e) => {
    setInputMessage(e.target.value);
  };

  // Update sendMessageToBackend to translate next steps
  const sendMessageToBackend = async (messageText) => {
    setIsLoading(true);
    
    try {
      setConnectionError(false);
      
      const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.MESSAGE}`, {
        method: 'POST',
        headers: API_CONFIG.HEADERS,
        body: JSON.stringify({
          message: messageText,
          session_id: sessionId
        }),
      });
      
      if (!response.ok) {
        throw new Error(`Server responded with status: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (!data || !data.response) {
        throw new Error('Invalid response from server');
      }
      
      const botMessage = {
        text: data.response,
        type: MESSAGE_TYPES.BOT,
        suggestions: SUGGESTIONS[language],
        language: data.language
      };
      
      setMessages(prevMessages => [...prevMessages, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      setConnectionError(true);
      
      let errorMessage;
      if (error.message.includes('Failed to fetch')) {
        errorMessage = language === 'en'
          ? "Unable to connect to the server. Please check your internet connection and try again."
          : "Không thể kết nối đến máy chủ. Vui lòng kiểm tra kết nối internet và thử lại.";
      } else if (error.message.includes('Invalid response')) {
        errorMessage = language === 'en'
          ? "Received invalid response from server. Please try again later."
          : "Nhận được phản hồi không hợp lệ từ máy chủ. Vui lòng thử lại sau.";
      } else {
        errorMessage = language === 'en'
          ? "An error occurred while processing your request. Please try again later."
          : "Đã xảy ra lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại sau.";
      }
      
      setMessages(prevMessages => [
        ...prevMessages,
        {
          text: errorMessage,
          type: MESSAGE_TYPES.SYSTEM
        }
      ]);
    } finally {
      setIsLoading(false);
      if (inputRef.current) {
        inputRef.current.focus();
      }
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    
    if (inputMessage.trim() === '') return;
    
    // Add user message to chat
    const userMessage = { 
      text: inputMessage, 
      type: MESSAGE_TYPES.USER 
    };
    setMessages(prevMessages => [...prevMessages, userMessage]);
    
    // Store the input value before clearing it
    const messageText = inputMessage;
    setInputMessage('');
    
    // Send the message to the backend
    await sendMessageToBackend(messageText);
  };

  const handleSuggestionClick = (suggestion) => {
    // Add user message to chat immediately for better UX
    const userMessage = { 
      text: suggestion, 
      type: MESSAGE_TYPES.USER 
    };
    setMessages(prevMessages => [...prevMessages, userMessage]);
    
    // Send the message to the backend
    sendMessageToBackend(suggestion);
  };

  const handleFeedback = async (rating) => {
    try {
      const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.FEEDBACK}`, {
        method: 'POST',
        headers: API_CONFIG.HEADERS,
        body: JSON.stringify({
          session_id: sessionId,
          rating: rating,
          comments: ''
        }),
      });
      
      if (!response.ok) {
        throw new Error(`Server responded with status: ${response.status}`);
      }
      
      // Show feedback confirmation based on language
      const feedbackMessage = language === 'en'
        ? `Thank you for your feedback! (${rating}/5)`
        : `Cảm ơn phản hồi của bạn! (${rating}/5)`;
      
      setMessages(prevMessages => [
        ...prevMessages,
        {
          text: feedbackMessage,
          type: MESSAGE_TYPES.SYSTEM
        }
      ]);
    } catch (error) {
      console.error('Error sending feedback:', error);
      const errorMessage = language === 'en'
        ? "Sorry, I couldn't process your feedback. Please try again later."
        : "Xin lỗi, tôi không thể xử lý phản hồi của bạn. Vui lòng thử lại sau.";
      
      setMessages(prevMessages => [
        ...prevMessages,
        {
          text: errorMessage,
          type: MESSAGE_TYPES.SYSTEM
        }
      ]);
    }
  };

  const handleTransferToHuman = async () => {
    try {
      setIsLoading(true);
      
      const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.TRANSFER}`, {
        method: 'POST',
        headers: API_CONFIG.HEADERS,
        body: JSON.stringify({
          session_id: sessionId
        }),
      });
      
      if (!response.ok) {
        throw new Error(`Server responded with status: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Add transfer message based on language
      const transferMessage = language === 'en'
        ? data.message || "A human agent will join this conversation shortly."
        : "Một nhân viên hỗ trợ sẽ tham gia cuộc trò chuyện này trong thời gian ngắn.";
      
      setMessages(prevMessages => [
        ...prevMessages,
        {
          text: transferMessage,
          type: MESSAGE_TYPES.SYSTEM
        }
      ]);
    } catch (error) {
      console.error('Error transferring to human:', error);
      const errorMessage = language === 'en'
        ? "Sorry, I couldn't transfer you to a human agent. Please try again later."
        : "Xin lỗi, tôi không thể chuyển bạn đến nhân viên hỗ trợ. Vui lòng thử lại sau.";
      
      setMessages(prevMessages => [
        ...prevMessages,
        {
          text: errorMessage,
          type: MESSAGE_TYPES.SYSTEM
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const toggleLanguage = () => {
    if (language === 'en') {
      setLanguage('vi');
    } else {
      setLanguage('en');
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSendMessage(e);
    }
  };

  return (
    <div className="app">
      <div className="chat-container">
        <div className="chat-header">
          <h1>{language === 'en' ? 'Customer Support Chat' : 'Hỗ trợ khách hàng'}</h1>
          <button className="language-btn" onClick={toggleLanguage}>
            {language === 'en' ? 'Tiếng Việt' : 'English'}
          </button>
        </div>

        <div className="messages-container">
          {messages.map((message, index) => (
            <ChatMessage 
              key={index} 
              message={message} 
              onSuggestionClick={handleSuggestionClick}
            />
          ))}
          {isLoading && (
            <div className="message bot">
              {language === 'en' ? 'Typing...' : 'Đang nhập...'}
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="input-container">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={language === 'en' ? 'Type your message...' : 'Nhập tin nhắn của bạn...'}
          />
          <button onClick={handleSendMessage}>
            {language === 'en' ? 'Send' : 'Gửi'}
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
