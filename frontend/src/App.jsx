import { useState, useRef, useEffect } from "react";
import "./App.css";


const API_BASE = "https://rag-chatbot-4a0s.onrender.com";

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  const messagesEndRef = useRef(null);

  // Auto-scroll to the latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const send = async () => {
    const trimmed = input.trim();
    if (!trimmed || isLoading) return;

    // 1. Append user's message immediately
    const userMessage = { role: "user", text: trimmed };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);
    setErrorMessage(null);

    try {
      // 2. POST {"text": ...} to /api/chat
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: trimmed }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(
          errorData.detail || `Server error: ${response.status} ${response.statusText}`
        );
      }

      const data = await response.json();

      // 3. Append assistant's reply and sources
      const assistantMessage = {
        role: "assistant",
        text: data.answer || "No response received.",
        sources: Array.isArray(data.sources) ? data.sources : [],
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      console.error("Chat error:", err);
      setErrorMessage(err.message || "Failed to connect to the chat server.");
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: `Error: ${err.message || "An error occurred while fetching the response."}`,
          isError: true,
          sources: [],
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    send();
  };

  return (
    <div className="chat-app">
      <header className="chat-header">
        <div className="chat-header-title">
          <h1>Times Consultant RAG Knowledge Base</h1>
          <span className="badge">pgvector + LLM</span>
        </div>
      </header>

      {/* Scrollable message list */}
      <main className="messages-list">
        {messages.length === 0 && (
          <div className="empty-state">
            <h2>Welcome to Times Consultant Assistant</h2>
            <p>
              Ask any question regarding study destinations, scholarships, visa
              guidelines, application steps, or upcoming educational events.
            </p>
          </div>
        )}

        {messages.map((msg, index) => (
          <div key={index} className={`message-item ${msg.role}`}>
            <div className={`message-bubble ${msg.isError ? "error" : ""}`}>
              {msg.text}
            </div>

            {/* Render sources array under each assistant message */}
            {msg.role === "assistant" && msg.sources && msg.sources.length > 0 && (
              <div className="message-sources">
                <div className="sources-header">Sources & References:</div>
                <ul className="sources-list">
                  {msg.sources.map((src, srcIndex) => (
                    <li key={srcIndex}>
                      <a
                        href={src.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="source-link"
                      >
                        <span className="source-index">[{srcIndex + 1}]</span>{" "}
                        {src.title || src.url}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ))}

        {/* Loading indicator while waiting */}
        {isLoading && (
          <div className="loading-indicator">
            <div className="dot-pulse">
              <span></span>
              <span></span>
              <span></span>
            </div>
            <span>Searching knowledge base and generating answer...</span>
          </div>
        )}

        {errorMessage && <div className="error-banner">{errorMessage}</div>}

        <div ref={messagesEndRef} />
      </main>

      {/* Input form */}
      <form className="chat-input-form" onSubmit={handleSubmit}>
        <input
          type="text"
          className="chat-input"
          placeholder="Ask a question about study visas, requirements, or programs..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={isLoading}
          autoFocus
        />
        <button
          type="submit"
          className="send-button"
          disabled={isLoading || !input.trim()}
        >
          {isLoading ? "Thinking..." : "Send"}
        </button>
      </form>
    </div>
  );
}
