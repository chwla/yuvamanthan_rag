import React, { useState, useEffect, useRef } from 'react';

const CONFIG = {
  BACKEND_API_URL: "http://localhost:8000",
  MAX_RESPONSE_LENGTH: 500,
  TYPING_DELAY: 1000,
  MAX_RETRIES: 3
};

const WELCOME_MESSAGE = "Hello! I'm your School Safety Assistant. I can help you with fire safety protocols, emergency procedures, and safety training programs. What would you like to know?";

const SchoolSafetyChatWidget = () => {
  const [chatOpen, setChatOpen] = useState(false);
  const [messages, setMessages] = useState([
    { type: 'bot', content: WELCOME_MESSAGE, timestamp: new Date() }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [status, setStatus] = useState('Checking connection...');
  const [backendHealth, setBackendHealth] = useState(null);
  
  const chatRef = useRef(null);
  const abortControllerRef = useRef(null);

  // Check backend health on component mount
  useEffect(() => {
    checkBackendHealth();
  }, []);

  const checkBackendHealth = async () => {
    try {
      const response = await fetch(`${CONFIG.BACKEND_API_URL}/api/health`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (response.ok) {
        const healthData = await response.json();
        setBackendHealth(healthData);
        setStatus(healthData.ollama_status ? 'Ready - AI Model Available' : 'Connected - AI Model Loading');
      } else {
        setStatus('Backend Error');
        setBackendHealth(null);
      }
    } catch (error) {
      console.error('Health check failed:', error);
      setStatus('Backend Disconnected');
      setBackendHealth(null);
    }
  };

  const scrollToBottom = () => {
    if (chatRef.current) {
      chatRef.current.scrollTo({ 
        top: chatRef.current.scrollHeight, 
        behavior: 'smooth' 
      });
    }
  };

  const toggleChat = () => {
    setChatOpen(prev => !prev);
    if (!chatOpen) {
      setTimeout(scrollToBottom, 300);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const sendMessage = async () => {
    const message = input.trim();
    if (!message || isTyping) return;

    // Add user message
    const userMessage = { 
      type: 'user', 
      content: message, 
      timestamp: new Date() 
    };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);
    setStatus('AI is thinking...');

    // Cancel any existing request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    // Create new abort controller for this request
    abortControllerRef.current = new AbortController();

    try {
      const response = await fetch(`${CONFIG.BACKEND_API_URL}/api/rag-chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: message,
          max_results: 5
        }),
        signal: abortControllerRef.current.signal
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      // Add bot response
      const botMessage = {
        type: 'bot',
        content: data.answer,
        confidence: data.confidence,
        sources: data.sources,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, botMessage]);
      setStatus(backendHealth?.ollama_status ? 'Ready - AI Model Available' : 'Connected');
      
    } catch (error) {
      if (error.name === 'AbortError') {
        console.log('Request was aborted');
        return;
      }
      
      console.error('Error sending message:', error);
      const errorMessage = {
        type: 'bot',
        content: 'Sorry, I encountered an error while processing your request. Please check if the backend is running and try again.',
        confidence: 0,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
      setStatus('Error - Check Backend Connection');
    } finally {
      setIsTyping(false);
      abortControllerRef.current = null;
      setTimeout(scrollToBottom, 100);
    }
  };

  const getStatusColor = () => {
    if (status.includes('Error') || status.includes('Disconnected')) return '#ff6b6b';
    if (status.includes('Ready')) return '#51cf66';
    if (status.includes('Connected')) return '#ffd43b';
    return '#74c0fc';
  };

  return (
    <div style={{ fontFamily: 'system-ui, -apple-system, sans-serif' }}>
      {/* Chat Toggle Button */}
      <button
        onClick={toggleChat}
        style={{
          position: 'fixed',
          bottom: '24px',
          right: '24px',
          width: '60px',
          height: '60px',
          borderRadius: '50%',
          backgroundColor: chatOpen ? '#dc3545' : '#007bff',
          color: 'white',
          border: 'none',
          fontSize: '24px',
          cursor: 'pointer',
          boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
          zIndex: 1000,
          transition: 'all 0.3s ease',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}
      >
        {chatOpen ? '×' : '🏫'}
      </button>

      {/* Chat Modal */}
      {chatOpen && (
        <div style={{
          position: 'fixed',
          bottom: '100px',
          right: '24px',
          width: '400px',
          height: '600px',
          backgroundColor: 'white',
          borderRadius: '12px',
          boxShadow: '0 8px 32px rgba(0,0,1,0.1)',
          zIndex: 999,
          display: 'flex',
          flexDirection: 'column',
          border: '1px solid #e0e0e0'
        }}>
          {/* Header */}
          <div style={{
            padding: '16px',
            backgroundColor: '#007bff',
            color: 'white',
            borderRadius: '12px 12px 0 0',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            <div>
              <h3 style={{ margin: 0, fontSize: '16px' }}>🏫 School Safety Assistant</h3>
              <div style={{ fontSize: '12px', opacity: 0.9 }}>
                Powered by llama3.2:3b 
              </div>
            </div>
            <button
              onClick={toggleChat}
              style={{
                backgroundColor: 'transparent',
                border: 'none',
                color: 'white',
                fontSize: '20px',
                cursor: 'pointer'
              }}
            >
              ×
            </button>
          </div>

          {/* Messages */}
          <div
            ref={chatRef}
            style={{
              flex: 1,
              padding: '16px',
              overflowY: 'auto',
              backgroundColor: '#f8f9fa'
            }}
          >
            {messages.map((msg, idx) => (
              <div
                key={idx}
                style={{
                  marginBottom: '16px',
                  display: 'flex',
                  flexDirection: msg.type === 'user' ? 'row-reverse' : 'row',
                  alignItems: 'flex-start',
                  gap: '8px'
                }}
              >
                <div style={{
                  width: '32px',
                  height: '32px',
                  borderRadius: '50%',
                  backgroundColor: msg.type === 'user' ? '#007bff' : '#28a745',
                  color: 'white',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '14px',
                  flexShrink: 0
                }}>
                  {msg.type === 'user' ? '👤' : '🏫'}
                </div>
                <div style={{
                  backgroundColor: msg.type === 'user' ? '#007bff' : 'white',
                  color: msg.type === 'user' ? 'white' : '#333',
                  padding: '12px',
                  borderRadius: '12px',
                  maxWidth: '280px',
                  wordWrap: 'break-word',
                  whiteSpace: 'pre-wrap',
                  border: msg.type === 'bot' ? '1px solid #e0e0e0' : 'none'
                }}>
                  {msg.content}
                  {msg.confidence !== undefined && (
                    <div style={{
                      fontSize: '11px',
                      opacity: 0.7,
                      marginTop: '8px',
                      borderTop: '1px solid #eee',
                      paddingTop: '4px'
                    }}>
                      Confidence: {Math.round(msg.confidence * 100)}%
                      {msg.sources && msg.sources.length > 0 && (
                        <div style={{ marginTop: '4px' }}>
                          Sources: {msg.sources.map(s => `${s.filename} (p.${s.page})`).join(', ')}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))}
            
            {/* Typing Indicator */}
            {isTyping && (
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                marginBottom: '16px'
              }}>
                <div style={{
                  width: '32px',
                  height: '32px',
                  borderRadius: '50%',
                  backgroundColor: '#28a745',
                  color: 'white',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '14px'
                }}>
                  🏫
                </div>
                <div style={{
                  backgroundColor: 'white',
                  padding: '12px',
                  borderRadius: '12px',
                  border: '1px solid #e0e0e0'
                }}>
                  <div style={{ display: 'flex', gap: '4px' }}>
                    <div style={{
                      width: '8px',
                      height: '8px',
                      borderRadius: '50%',
                      backgroundColor: '#007bff',
                      animation: 'pulse 1.5s infinite'
                    }}></div>
                    <div style={{
                      width: '8px',
                      height: '8px',
                      borderRadius: '50%',
                      backgroundColor: '#007bff',
                      animation: 'pulse 1.5s infinite 0.5s'
                    }}></div>
                    <div style={{
                      width: '8px',
                      height: '8px',
                      borderRadius: '50%',
                      backgroundColor: '#007bff',
                      animation: 'pulse 1.5s infinite 1s'
                    }}></div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Status Bar */}
          <div style={{
            padding: '8px 16px',
            backgroundColor: getStatusColor(),
            color: 'white',
            fontSize: '12px',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            <span>{status}</span>
          </div>

          {/* Input */}
          <div style={{
            padding: '16px',
            backgroundColor: 'white',
            borderRadius: '0 0 12px 12px',
            borderTop: '1px solid #e0e0e2'
          }}>
            <div style={{ display: 'flex', gap: '8px' }}>
              <input
                type="text"
                placeholder="Ask about safety protocols..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                maxLength={500}
                disabled={isTyping}
                style={{
                  flex: 1,
                  padding: '12px',
                  border: '1px solid #e0e0e0',
                  borderRadius: '6px',
                  fontSize: '14px',
                  outline: 'none'
                }}
              />
              <button
                onClick={sendMessage}
                disabled={!input.trim() || isTyping}
                style={{
                  padding: '12px 16px',
                  backgroundColor: (!input.trim() || isTyping) ? '#ccc' : '#007bff',
                  color: 'white',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: (!input.trim() || isTyping) ? 'not-allowed' : 'pointer',
                  fontSize: '14px'
                }}
              >
                →
              </button>
            </div>
          </div>
        </div>
      )}

      {/* CSS Animations */}
      <style>
        {`
          @keyframes pulse {
            0%, 100% { opacity: 0.4; }
            50% { opacity: 1; }
          }
        `}
      </style>
    </div>
  );
};

export default SchoolSafetyChatWidget;