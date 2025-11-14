"""Voice AI Assistant with Memory and Tools.

This project implements a full-featured voice-based AI assistant using Pipecat,
featuring persistent memory via Mem0, web search via Tavily, and weather lookup
via WeatherAPI. The bot can engage in natural conversations, recall user details,
and provide real-time information.

Required AI services:
- Deepgram (Speech-to-Text)
- Google Gemini (LLM)
- Cartesia (Text-to-Speech)
- Mem0 (Memory)
- Tavily (Web Search)
- WeatherAPI (Weather)

Run the bot using::

    uv run bot.py
"""

import json
import os
from datetime import datetime, timezone
from typing import List, Optional

from dotenv import load_dotenv
from loguru import logger
import requests
from tavily import TavilyClient
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

print("🚀 Starting Pipecat bot...")
print("⏳ Loading models and imports (20 seconds, first run only)\n")

logger.info("Loading Local Smart Turn Analyzer V3...")
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

logger.info("✅ Local Smart Turn Analyzer V3 loaded")
logger.info("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer

logger.info("✅ Silero VAD model loaded")

from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame

logger.info("Loading pipeline components...")
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
# from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.mem0.memory import Mem0MemoryService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.services.llm_service import FunctionCallParams

logger.info("✅ All components loaded successfully!")

load_dotenv(override=True)

DEFAULT_WEATHER_LOCATION = os.getenv("DEFAULT_WEATHER_LOCATION", "San Francisco, CA")
DEFAULT_WEATHER_UNIT = os.getenv("DEFAULT_WEATHER_UNIT", "fahrenheit")
WEATHER_API_URL = "http://api.weatherapi.com/v1/forecast.json"

_tavily_client: Optional[TavilyClient] = None


def _get_tavily_client() -> Optional[TavilyClient]:
    """Create (or reuse) a Tavily client using the configured API key."""

    global _tavily_client
    if _tavily_client:
        return _tavily_client

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.error("TAVILY_API_KEY is not set; web search will return an error message.")
        return None

    try:
        _tavily_client = TavilyClient(api_key)
    except Exception as e:  # pragma: no cover - network/config specific
        logger.error(f"Failed to initialize Tavily client: {e}")
        _tavily_client = None
    return _tavily_client


def web_search(query: str, max_results: int = 3) -> str:
    """Lightweight web search using DuckDuckGo.

    Returns a short, concatenated snippet of top results so Gemini
    can ground its answer in current information.
    """

    client = _get_tavily_client()
    if not client:
        return "Tavily API key missing — please set TAVILY_API_KEY in your environment."

    try:
        response = client.search(
            query=query,
            max_results=max_results,
            include_answer=True,
            search_depth="advanced",
        )
    except Exception as e:  # pragma: no cover - remote call
        logger.error(f"Tavily search failed: {e}")
        return "Tavily search failed. Please retry shortly or adjust the query."

    results = response.get("results", [])
    direct_answer = response.get("answer")

    if not results and not direct_answer:
        return "Tavily did not return any relevant results for that query."

    snippets = []
    if direct_answer:
        snippets.append(f"Tavily summary: {direct_answer}")

    for i, r in enumerate(results, start=1):
        title = r.get("title") or "(no title)"
        body = r.get("content") or ""
        url = r.get("url") or ""
        snippets.append(f"Result {i}: {title}\n{body}\n{url}")

    lowered_query = query.lower()
    timestamp_lines: List[str] = []
    if any(phrase in lowered_query for phrase in ["current time", "time now", "what time"]):
        local_now = datetime.now().astimezone()
        utc_now = datetime.now(timezone.utc)
        timestamp_lines.append(
            f"Assistant local time: {local_now.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        )
        timestamp_lines.append(f"UTC time: {utc_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    if any(phrase in lowered_query for phrase in ["current date", "today's date", "what's today's date"]):
        today = datetime.now().astimezone().strftime("%A, %B %d, %Y")
        timestamp_lines.append(f"Today's date: {today}")

    if timestamp_lines:
        snippets.insert(0, "\n".join(timestamp_lines))

    return "\n\n".join(snippets)


def get_current_weather(location: Optional[str] = None, unit: Optional[str] = None) -> str:
    """Fetch current weather and today's forecast details using WeatherAPI."""

    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        return "Weather API key missing. Please set WEATHER_API_KEY in your environment."

    resolved_location = (location or DEFAULT_WEATHER_LOCATION).strip()
    if not resolved_location:
        resolved_location = DEFAULT_WEATHER_LOCATION

    resolved_unit = (unit or DEFAULT_WEATHER_UNIT or "fahrenheit").lower()
    if resolved_unit not in {"fahrenheit", "celsius"}:
        resolved_unit = "fahrenheit"

    params = {
        "key": api_key,
        "q": resolved_location,
        "days": 1,
        "aqi": "no",
        "alerts": "no",
    }

    try:
        response = requests.get(WEATHER_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:  # pragma: no cover - remote API
        logger.error(f"Weather API request failed: {e}")
        return "Unable to reach the weather service right now. Please try again soon."

    current = data.get("current") or {}
    forecast_days = data.get("forecast", {}).get("forecastday", [])
    if not current or not forecast_days:
        return "Weather data was missing from the provider response."

    today_forecast = forecast_days[0]
    day = today_forecast.get("day", {})
    astro = today_forecast.get("astro", {})

    def pick(f_key: str, c_key: str):
        return f_key if resolved_unit == "fahrenheit" else c_key

    weather_info = {
        "location": resolved_location,
        "unit": resolved_unit,
        "temperature": current.get(pick("temp_f", "temp_c")),
        "feels_like": current.get(pick("feelslike_f", "feelslike_c")),
        "max_temp": day.get(pick("maxtemp_f", "maxtemp_c")),
        "min_temp": day.get(pick("mintemp_f", "mintemp_c")),
        "forecast": current.get("condition", {}).get("text"),
        "wind_speed": current.get(pick("wind_mph", "wind_kph")),
        "wind_direction": current.get("wind_dir"),
        "humidity": current.get("humidity"),
        "pressure": current.get(pick("pressure_in", "pressure_mb")),
        "rain_inches": current.get("precip_in"),
        "sunrise": astro.get("sunrise"),
        "sunset": astro.get("sunset"),
        "moonrise": astro.get("moonrise"),
        "moonset": astro.get("moonset"),
        "moon_phase": astro.get("moon_phase"),
        "visibility": current.get(pick("vis_miles", "vis_km")),
        "will_it_rain": day.get("daily_will_it_rain"),
        "chance_of_rain": day.get("daily_chance_of_rain"),
        "uv": current.get("uv"),
    }

    logger.info(f"[tools] Weather fetched for {resolved_location} ({resolved_unit}).")
    return json.dumps(weather_info)


assistant_tools = ToolsSchema(
    standard_tools=[
        FunctionSchema(
            name="web_search",
            description=(
                "Search the public web for up-to-date information. Call this before answering"
                " questions about current time, date, news, or whenever the user asks you to"
                " 'search the web'."
            ),
            properties={
                "query": {
                    "type": "string",
                    "description": "Natural language description of what to look up.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "How many search snippets to retrieve (1-5).",
                    "minimum": 1,
                    "maximum": 5,
                },
            },
            required=["query"],
        ),
        FunctionSchema(
            name="get_current_weather",
            description=(
                "Retrieve the latest weather conditions and today's forecast for a location."
                " Use this whenever the user asks about weather, temperatures, rain chances,"
                " sunrise/sunset, or similar atmospheric details."
            ),
            properties={
                "location": {
                    "type": "string",
                    "description": "City, address, or lat/long to query. Defaults to your configured city if omitted.",
                },
                "unit": {
                    "type": "string",
                    "enum": ["fahrenheit", "celsius"],
                    "description": "Temperature unit preference. Defaults to fahrenheit.",
                },
            },
            required=[],
        ),
    ]
)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    # stt = WhisperSTTService(model="base", device="cpu")

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    llm = GoogleLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-2.5-flash",
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are Ruby, a friendly AI assistant. You HAVE persistent long-term memory via"
                " Mem0, so never claim you forget. Whenever you see a message that begins with"
                " 'Previous conversations with this user:' or 'Mem0 notes', treat it as ground truth"
                " about the user and blend it naturally into replies. Respond in a warm, concise"
                " tone, cite live data, and avoid saying you are a Google/Gemini model."
                " ALWAYS call the `web_search` tool before answering whenever the user requests"
                " current events, current/real-time facts (time/date/news), or explicitly asks you"
                " to search the web. If the user even hints at the current date, todays date, current time,"
                " what day it is, who currently holds a role, or any other 'right now' fact, you MUST"
                " call `web_search` first and wait for the result before replying. NEVER answer these"
                " from stale knowledge. ALWAYS call `get_current_weather` when the user wants weather,"
                " temperatures, rain chances, or sunrise/sunset information. Once a tool returns,"
                " reason over that data, mention sources when possible, and clearly explain what you found."
            ),
        },
    ]

    context = LLMContext(messages, tools=assistant_tools)
    context_aggregator = LLMContextAggregatorPair(context)

    memory = Mem0MemoryService(
        api_key=os.getenv("MEM0_API_KEY"),
        user_id="demo-user",  # Replace with real user ID for production
        agent_id="voice-assistant-bot",
        params=Mem0MemoryService.InputParams(
            search_limit=3,
            search_threshold=0.35,
            system_prompt=(
                "Mem0 notes for this user (rely on these facts and never say you forgot):\n\n"
            ),
            add_as_system_message=True,
            position=0,
        ),
    )

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            rtvi,  # RTVI processor
            stt,
            context_aggregator.user(),  # User responses
            memory,  # Long-term memory via Mem0
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )
    # Register assistant functions with Pipecat's function-calling API.
    async def web_search_handler(params: FunctionCallParams):
        args = params.arguments or {}
        query = args.get("query", "").strip()
        max_results = args.get("max_results", 3)
        try:
            max_results = int(max_results)
        except (TypeError, ValueError):
            max_results = 3
        max_results = max(1, min(max_results, 5))

        if not query:
            await params.result_callback({"result": "Empty search query provided."})
            return

        logger.info(f"[tools] web_search called with query: {query} (max_results={max_results})")
        result_text = web_search(query, max_results=max_results)
        # Return the result back to the LLM. It will automatically
        # add it into the conversation context and run another completion.
        await params.result_callback({"result": result_text})

    llm.register_function("web_search", web_search_handler, cancel_on_interruption=True)

    async def get_current_weather_handler(params: FunctionCallParams):
        args = params.arguments or {}
        location = args.get("location")
        unit = args.get("unit")
        logger.info(
            f"[tools] get_current_weather called for location='{location or DEFAULT_WEATHER_LOCATION}'"
            f" unit='{unit or DEFAULT_WEATHER_UNIT}'"
        )
        result_text = get_current_weather(location=location, unit=unit)
        await params.result_callback({"result": result_text})

    llm.register_function("get_current_weather", get_current_weather_handler, cancel_on_interruption=True)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        messages.append(
            {
                "role": "system",
                "content": "Greet the user as Ruby, mention you can remember details they share, and offer help.",
            }
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point for the bot starter."""

    transport_params = {
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
    }

    transport = await create_transport(runner_args, transport_params)

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
