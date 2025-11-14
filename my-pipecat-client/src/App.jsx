import React, { useState, useEffect, useRef } from 'react';
import { PipecatClient } from "@pipecat-ai/client-js";
import { SmallWebRTCTransport } from "@pipecat-ai/small-webrtc-transport";
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false); // New state to prevent double-clicks
  const pipecatClient = useRef(null);
  const audioRef = useRef(null);

  useEffect(() => {
    // This effect should only run once to initialize the client
    if (!pipecatClient.current) {
        // Corrected: The signaling URL must point to your new server endpoint
      const client = new PipecatClient({
        transport: new SmallWebRTCTransport({
            signalingUrl: "ws://localhost:8000/ws"
        }),
        enableMic: true,
      });

      // --- Event Listeners ---
      client.on("transcription", (text) => {
        if (text) {
          setMessages(prev => [...prev, { sender: 'user', text }]);
        }
      });

      client.on("bot-response", (text) => {
        if (text) {
          setMessages(prev => [...prev, { sender: 'bot', text }]);
        }
      });

      client.on("track-started", (track, participant) => {
        if (participant && !participant.local && track.kind === 'audio') {
          const stream = new MediaStream([track]);
          if (audioRef.current) {
            audioRef.current.srcObject = stream;
            audioRef.current.play().catch(error => console.error("Error playing audio:", error));
          }
        }
      });

      client.on("connected", () => {
        console.log("Successfully connected to Pipecat");
        setIsConnected(true);
        setIsConnecting(false);
      });

      client.on("disconnected", () => {
        console.log("Disconnected from Pipecat");
        setIsConnected(false);
        setIsConnecting(false);
      });

      client.on("error", (error) => {
        console.error("Pipecat client error:", error);
        setIsConnecting(false); // Reset connecting state on error
      });

      pipecatClient.current = client;
    }
  }, []); // Empty dependency array ensures this runs only once

  const handleConnect = async () => {
    if (pipecatClient.current && !isConnected && !isConnecting) {
      setIsConnecting(true);
      try {
        console.log("Attempting to connect...");
        await pipecatClient.current.connect();
      } catch (error) {
        console.error("Connection failed:", error);
        setIsConnecting(false); // Reset connecting state on error
      }
    }
  };

  const handleDisconnect = () => {
    if (pipecatClient.current && isConnected) {
      pipecatClient.current.disconnect();
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Pipecat React Client</h1>
        {!isConnected ? (
          <button onClick={handleConnect} disabled={isConnecting}>
            {isConnecting ? 'Connecting...' : 'Connect'}
          </button>
        ) : (
          <button onClick={handleDisconnect}>Disconnect</button>
        )}
        <p>Status: {isConnecting ? 'Connecting...' : (isConnected ? 'Connected' : 'Disconnected')}</p>
      </header>
      <div className="chat-container">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.sender}`}>
            <p><strong>{msg.sender === 'user' ? 'Ruby' : 'You'}:</strong> {msg.text}</p>
          </div>
        ))}
      </div>
      <audio ref={audioRef} style={{ display: 'none' }} />
    </div>
  );
}

export default App;