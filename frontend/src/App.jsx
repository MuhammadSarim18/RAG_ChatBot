import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./App.css";

// const API_BASE = "http://127.0.0.1:8000";
const API_BASE = "https://rag-chatbot-4a0s.onrender.com";

const SUGGESTIONS = [
  "Suggest me study destinations",
  "What are the visa requirements for the UK?",
  "Which countries offer post-study work visas?",
  "What are the cheapest countries to study in?",
];

// Removes citation artifacts like 【1†L1-L22】 that some models add
function cleanAnswer(text) {
  return (text || "")
    .replace(/【[^】]*】/g, "")
    .replace(/[ \t]+\n/g, "\n")
    .trim();
}

function Avatar({ role }) {
  return (
    <div className={`avatar ${role}`}>{role === "user" ? "You" : "TC"}</div>
  );
}

function Message({ msg }) {
  const isUser = msg.role === "user";
  return (
    <div className={`row ${isUser ? "row-user" : "row-bot"}`}>
      {!isUser && <Avatar role="bot" />}
      <div className="msg-col">
        <div
          className={`bubble ${isUser ? "bubble-user" : "bubble-bot"} ${
            msg.isError ? "bubble-error" : ""
          }`}
        >
          {isUser || msg.isError ? (
            msg.text
          ) : (
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                a: ({ node, ...props }) => (
                  <a {...props} target="_blank" rel="noreferrer" />
                ),
                table: ({ node, ...props }) => (
                  <div className="table-wrap">
                    <table {...props} />
                  </div>
                ),
              }}
            >
              {msg.text}
            </ReactMarkdown>
          )}
        </div>

        {!isUser && msg.sources && msg.sources.length > 0 && (
          <details className="sources">
            <summary>Sources ({msg.sources.length})</summary>
            <ol>
              {msg.sources.map((s, i) => (
                <li key={i}>
                  <a href={s.url} target="_blank" rel="noreferrer">
                    {s.title || s.url}
                  </a>
                </li>
              ))}
            </ol>
          </details>
        )}
      </div>
      {isUser && <Avatar role="user" />}
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="row row-bot">
      <Avatar role="bot" />
      <div className="bubble bubble-bot typing">
        <span />
        <span />
        <span />
      </div>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  // Auto-scroll to the latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  // Auto-grow the input box (up to a limit)
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 140) + "px";
  }, [input]);

  const send = async (override) => {
    const trimmed = (override ?? input).trim();
    if (!trimmed || isLoading) return;

    setMessages((prev) => [...prev, { role: "user", text: trimmed }]);
    setInput("");
    setIsLoading(true);

    // Render's free tier can take ~50s to wake up, so allow a long timeout
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 90000);

    try {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: trimmed }),
        signal: controller.signal,
      });

      if (!response.ok) {
        let detail = `Server error (${response.status})`;
        try {
          const err = await response.json();
          if (err.detail) {
            detail =
              typeof err.detail === "string"
                ? err.detail
                : JSON.stringify(err.detail);
          }
        } catch {
          /* response was not JSON */
        }
        throw new Error(detail);
      }

      const data = await response.json();
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: cleanAnswer(data.answer),
          sources: data.sources || [],
        },
      ]);
    } catch (err) {
      let text = err.message;
      if (err.name === "AbortError") {
        text =
          "The server took too long to respond. It may be waking up, so please try again.";
      } else if (err.message === "Failed to fetch") {
        text =
          "Couldn't reach the server. It may be waking up, so try again in a moment.";
      }
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text, isError: true },
      ]);
    } finally {
      clearTimeout(timer);
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    // Enter sends, Shift+Enter adds a new line
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <div className="page">
      <div className="chat">
        <header className="chat-header">
          <div className="header-left">
            <div className="logo">TC</div>
            <div>
              <div className="title">Times Consultant Assistant</div>
              <div className="status">
                <span className="dot" /> Online
              </div>
            </div>
          </div>
          {messages.length > 0 && (
            <button
              className="clear-btn"
              onClick={() => setMessages([])}
              disabled={isLoading}
            >
              New chat
            </button>
          )}
        </header>

        <main className="messages">
          {messages.length === 0 && !isLoading ? (
            <div className="welcome">
              <div className="welcome-logo">TC</div>
              <h2>How can I help you today?</h2>
              <p>Ask about study visas, requirements, or programs abroad.</p>
              <div className="chips">
                {SUGGESTIONS.map((s) => (
                  <button key={s} className="chip" onClick={() => send(s)}>
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((m, i) => <Message key={i} msg={m} />)
          )}
          {isLoading && <TypingIndicator />}
          <div ref={messagesEndRef} />
        </main>

        <footer className="composer">
          <textarea
            ref={textareaRef}
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about study visas, requirements, or programs..."
            disabled={isLoading}
          />
          <button
            className="send-btn"
            onClick={() => send()}
            disabled={isLoading || !input.trim()}
            aria-label="Send"
          >
            Send
          </button>
        </footer>
      </div>
    </div>
  );
}