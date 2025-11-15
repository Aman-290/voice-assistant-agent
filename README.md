# Voice Agent

A conversational AI voice agent built using Pipecat framework.

## Setup Instructions

To run the voice agent locally, follow these steps:

1. Ensure you have uv installed:
   ```
   pip install uv
   ```

2. Clone the repository:
   ```
   git clone https://github.com/Aman-290/voice-assistant-agent
   ```

3. Navigate to the voice-assistant-agent directory:
   ```
   cd voice-assistant-agent
   ```

4. Copy the environment example file:
   ```
   cp env.example .env
   ```

5. Install dependencies using uv:
   ```
   uv sync
   ```

6. Run the bot:
   ```
   uv run bot.py
   ```

7. After running the bot, the Pipecat client will be available in your browser. The terminal will display something like:
   ```
   🚀 Bot ready!
   → Open http://localhost:7860/client in your browser
   ```
   Open [http://localhost:7860/client](http://localhost:7860/client) to interact with the voice agent. Once opened:
   - Grant microphone permissions when prompted by your browser
   - Click the "Connect" button to connect the client to the server and start the voice agent

The voice agent uses Pipecat as the core orchestrator to manage the conversation flow. The system integrates several services:

- **Speech-to-Text (STT)**: Deepgram for fast and accurate transcription
- **Large Language Model (LLM)**: Gemini for intelligent conversation processing
- **Text-to-Speech (TTS)**: Cartesia for low-latency voice synthesis
- **Memory Management**: Mem0 for conversational memory and context retention
- **Web Search**: Tavily API for real-time information retrieval
- **Weather Data**: Free weather API for location-based weather information

Pipecat handles the orchestration, providing built-in features like interrupt detection and voice activity detection. For every user query, Mem0 fetches the most relevant memories that could be helpful for generating a response, allowing the AI to respond based on historical context. Mem0 can smartly update and manage memory fields to maintain accurate and evolving information.

## Agentic Features

The voice agent includes advanced agentic capabilities that enhance conversational interactions:

- **Seamless Web Search Integration**: The AI can perform real-time web searches using Tavily API and incorporate relevant information directly into the conversation flow.
- **Weather Information**: Access to free weather APIs allows the agent to provide current weather data and forecasts as part of natural dialogue.
- **Intelligent Memory**: The AI maintains persistent memory across conversations, allowing for context-aware responses and personalized interactions.

## Design Decisions

- **Pipecat Framework**: Chosen for its built-in interrupt detection and voice activity detection capabilities, which are essential for natural conversational interactions.
- **Deepgram for STT**: Selected over local Whisper models due to GPU limitations that would slow down processing. Deepgram provides much faster transcription speeds.
- **Cartesia for TTS**: Preferred over ElevenLabs due to significantly lower latency, ensuring more responsive voice output.
- **Mem0 Memory Strategy**: I chose Mem0 API because it offers the best balance of speed, cost, and intelligence for a real-time conversational AI agent. Other options—like Supermemory's API or cloud vector databases—introduce network latency and recurring costs, which break the "instant response" requirement. Simple local storage was also considered but lacks semantic search and cannot scale with long conversations. Mem0 provides millisecond-level retrieval, smart memory filtering, and long-term scalability, all while remaining cost-effective for an MVP. This makes it the most practical and high-performance choice compared to the alternatives.

## Time Spent

Approximately 7-8 hours were spent on this project, including research, design choices, and implementation.

## Known Limitations

- The client expose SDK is available, but integration with custom client environments still needs exploration.
- For testing purposes, the system uses Pipecat's default Uvicorn UI, which includes both audio and video feeds. Currently, only audio is utilized, but video feed can be easily implemented in the future.
- Due to the selection of Pipecat as the framework, changing underlying models (STT, TTS, LLM) is straightforward and won't break existing functionality, reducing vendor lock-in issues.