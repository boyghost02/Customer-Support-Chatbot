const API_CONFIG = {
  BASE_URL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  ENDPOINTS: {
    MESSAGE: '/message',
    FEEDBACK: '/feedback',
    TRANSFER: '/transfer',
    ANALYTICS: '/analytics/conversations',
    COLLECT_DATA: '/collect-data',
    ANALYZE_DATA: '/analyze-data',
    RELOAD_MODEL: '/reload-model'
  },
  HEADERS: {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
  }
};

export default API_CONFIG; 