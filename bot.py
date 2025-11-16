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


# Hotel Booking Integration
import random
from typing import Dict, List, Any
from google.genai import types
from google.adk.runners import Runner
from google.adk.tools import google_search
from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService

# Hotel booking session management
HOTEL_APP_NAME = "voice_hotel_app"
HOTEL_USER_ID = "voice_user"
HOTEL_SESSION_ID = "voice_session"

hotel_session_service = InMemorySessionService()

# Initialize hotel booking session
async def init_hotel_session():
    await hotel_session_service.create_session(
        app_name=HOTEL_APP_NAME,
        user_id=HOTEL_USER_ID,
        session_id=HOTEL_SESSION_ID
    )

# Run initialization
import asyncio
asyncio.run(init_hotel_session())

# Initialize hotel booking state
hotel_booking_state = {
    "active": False,
    "hotel_name": "",
    "room_type": "",
    "check_in": "",
    "check_out": "",
    "guests": 0,
    "total_price": 0,
    "step": "initial"
}

# Hotel Search Agent Instructions
HOTEL_SEARCH_AGENT_INSTRUCTIONS = """
You are a hotel search assistant. Your goal is to find hotel names and basic information based on the user's query.

**Your Workflow:**
1. Use Google Search to find hotels based on the user's location query.
2. Extract the hotel name, its general location (e.g., the city), a rating, and an estimated cost.
3. STRICTLY follow this JSON output format. The 'hotels' key should contain a list of objects:
    {
        "text": "Optional introductory text about the search results.",
        "hotels": [
            {
                "hotel_name": "Name of Hotel",
                "location": "City or Area, Country",
                "rating": 4.5,
                "cost": 12000
            }
        ]
    }
4. Limit the JSON output to a maximum of 10 hotels.

**Search Guidelines:**
- Use multiple search queries if needed to get comprehensive results.
- Extract hotel names, rating, cost estimates in INR, and the location.
"""

# Hotel Search Agent
hotel_search_agent = LlmAgent(
    name="hotel_search_agent",
    model="gemini-2.0-flash",
    description="Hotel Search Agent",
    instruction=HOTEL_SEARCH_AGENT_INSTRUCTIONS,
    generate_content_config=types.GenerateContentConfig(temperature=0),
    tools=[google_search]
)

hotel_search_runner = Runner(
    agent=hotel_search_agent,
    app_name=HOTEL_APP_NAME,
    session_service=hotel_session_service
)

def mock_api_get_hotel_prices(hotel_name: str, location: str) -> Dict[str, Dict]:
    """Mock API to get hotel prices from multiple booking sites"""
    booking_sites = [
        "Booking.com", "Agoda", "MakeMyTrip", "Trivago",
        "Hotels.com", "Expedia", "Goibibo", "Cleartrip"
    ]
    hotel_prices = {}
    for site in booking_sites:
        if random.random() <= 0.8:  # 80% chance of availability
            price = random.randint(5000, 20000)
            site_domain = site.lower().replace(" ", "").replace(".", "")
            mock_link = f"https://www.{site_domain}.com/hotel/{hotel_name.lower().replace(' ', '-')}"
            hotel_prices[site] = {
                "price": price,
                "available": True,
                "link": mock_link,
                "currency": "INR"
            }
        else:
            hotel_prices[site] = {
                "price": None,
                "available": False,
                "link": None,
                "currency": "INR"
            }
    return hotel_prices

def get_best_hotel_deals(hotels_list: List[Dict]) -> List[Dict]:
    """Get prices for all hotels from multiple sites and return sorted by lowest price."""
    enhanced_hotels = []

    for hotel in hotels_list:
        hotel_name = hotel.get('hotel_name', '')
        location = hotel.get('location', '')
        site_prices = mock_api_get_hotel_prices(hotel_name, location)

        available_prices = [
            {'site': site, 'price': details['price'], 'link': details['link']}
            for site, details in site_prices.items()
            if details['available'] and details.get('price') is not None
        ]

        if available_prices:
            best_deal = min(available_prices, key=lambda x: x['price'])

            rating_from_agent = hotel.get('rating')
            safe_rating = rating_from_agent if rating_from_agent is not None else 0

            enhanced_hotel = {
                'hotel_name': hotel_name,
                'location': location,
                'price': best_deal['price'],
                'rating': safe_rating,
                'link': best_deal['link'],
                'price_source': best_deal['site'],
                'all_prices': site_prices
            }
        else:
            rating_from_agent = hotel.get('rating')
            safe_rating = rating_from_agent if rating_from_agent is not None else 0
            enhanced_hotel = {
                'hotel_name': hotel_name,
                'location': location,
                'price': 0,
                'rating': safe_rating,
                'link': '',
                'price_source': 'Not Available',
                'all_prices': site_prices
            }
        enhanced_hotels.append(enhanced_hotel)

    enhanced_hotels.sort(key=lambda x: (x['price'] == 0, x['price']))
    return enhanced_hotels

def search_hotels(query: str, max_results: int = 8) -> str:
    """Search for hotels using the hotel search agent"""
    try:
        import asyncio
        import json

        async def search_async():
            content = types.Content(role='user', parts=[types.Part(text=query)])
            final_response_text = "Sorry, I couldn't find any hotels."

            async for event in hotel_search_runner.run_async(
                user_id=HOTEL_USER_ID,
                session_id=HOTEL_SESSION_ID,
                new_message=content
            ):
                if event.is_final_response():
                    if event.content and event.content.parts:
                        final_response_text = event.content.parts[0].text
                    break
            return final_response_text

        response_text = asyncio.run(search_async())

        # Parse the JSON response
        cleaned_response = response_text.replace("```json", "").replace("```", "").strip()
        response_json = json.loads(cleaned_response)

        text_output = response_json.get("text", "")
        hotels = response_json.get("hotels", [])

        if hotels:
            enhanced_hotels = get_best_hotel_deals(hotels)
            enhanced_hotels = enhanced_hotels[:max_results]

            result_parts = []
            if text_output:
                result_parts.append(text_output)

            result_parts.append(f"I found {len([h for h in enhanced_hotels if h['price'] > 0])} great hotel options:")

            for i, hotel in enumerate(enhanced_hotels, 1):
                if hotel['price'] > 0:
                    result_parts.append(
                        f"{i}. {hotel['hotel_name']} in {hotel['location']} - ₹{hotel['price']:,} per night "
                        f"(from {hotel['price_source']})"
                    )
                    if hotel['rating'] > 0:
                        result_parts.append(f"   Rating: {hotel['rating']}/5")
                else:
                    result_parts.append(f"{i}. {hotel['hotel_name']} - No prices available")

            return "\n".join(result_parts)
        else:
            return "I couldn't find any hotels matching your search. Please try a different location or be more specific."

    except Exception as e:
        logger.error(f"Hotel search failed: {e}")
        return "I'm having trouble searching for hotels right now. Please try again later."

def get_room_types() -> List[Dict]:
    """Get available room types for booking"""
    return [
        {"type": "Standard Room", "description": "Basic amenities, city view"},
        {"type": "Deluxe Room", "description": "Premium amenities, partial city view"},
        {"type": "Executive Suite", "description": "Spacious suite, city view, executive lounge access"},
        {"type": "Presidential Suite", "description": "Luxury suite, panoramic view, butler service"}
    ]

def process_hotel_booking_step(user_input: str) -> str:
    """Process different steps of hotel booking flow"""
    global hotel_booking_state

    current_step = hotel_booking_state["step"]

    if current_step == "initial":
        # Extract hotel name from user input
        hotel_name = ""
        if "book" in user_input.lower():
            # Try to extract hotel name
            parts = user_input.lower().split("book")
            if len(parts) > 1:
                hotel_part = parts[1].strip()
                hotel_part = hotel_part.replace("the ", "").replace("hotel", "").strip()
                hotel_name = hotel_part.title()

        if hotel_name:
            hotel_booking_state["hotel_name"] = hotel_name
            hotel_booking_state["step"] = "room_selection"
            hotel_booking_state["active"] = True

            room_types = get_room_types()
            response = f"Great! I'll help you book {hotel_name}. Please select a room type:\n"
            for room in room_types:
                response += f"- {room['type']}: {room['description']}\n"
            return response
        else:
            hotel_booking_state["step"] = "room_selection"
            hotel_booking_state["active"] = True

            room_types = get_room_types()
            response = "I'll help you with the booking. Please select a room type:\n"
            for room in room_types:
                response += f"- {room['type']}: {room['description']}\n"
            return response

    elif current_step == "room_selection":
        room_types = get_room_types()
        selected_room = None

        for room in room_types:
            if room["type"].lower() in user_input.lower():
                selected_room = room
                break

        if selected_room:
            hotel_booking_state["room_type"] = selected_room["type"]
            hotel_booking_state["step"] = "date_selection"
            return f"Perfect! You've selected {selected_room['type']}. Now please provide your check-in and check-out dates (e.g., 'Check-in tomorrow, check-out in 3 days')."
        else:
            response = "Please select one of the available room types:\n"
            for room in room_types:
                response += f"- {room['type']}: {room['description']}\n"
            return response

    elif current_step == "date_selection":
        # For demo purposes, set default dates
        hotel_booking_state["check_in"] = "2025-01-15"
        hotel_booking_state["check_out"] = "2025-01-17"
        hotel_booking_state["step"] = "guest_selection"
        return "Thanks! I've set your dates. Now please tell me how many guests will be staying."

    elif current_step == "guest_selection":
        import re
        numbers = re.findall(r'\d+', user_input)
        if numbers:
            guests = int(numbers[0])
            hotel_booking_state["guests"] = guests
            hotel_booking_state["step"] = "confirmation"

            # Generate a random total price for demo purposes
            base_price = random.randint(8000, 15000)
            nights = 2
            total_price = base_price * nights
            hotel_booking_state["total_price"] = total_price

            return f"Perfect! Let me confirm your booking details:\n\n" \
                   f"🏨 Hotel: {hotel_booking_state.get('hotel_name', 'Selected Hotel')}\n" \
                   f"🛏️ Room: {hotel_booking_state['room_type']}\n" \
                   f"📅 Check-in: {hotel_booking_state['check_in']}\n" \
                   f"📅 Check-out: {hotel_booking_state['check_out']}\n" \
                   f"👥 Guests: {guests}\n" \
                   f"💰 Total Price: ₹{total_price:,} (2 nights)\n\n" \
                   f"Please say 'confirm booking' to complete or 'cancel booking' to cancel."
        else:
            return "Please specify the number of guests (e.g., '2 guests' or 'three people')."

    elif current_step == "confirmation":
        if "confirm" in user_input.lower():
            booking_id = f"HTL{random.randint(100000, 999999)}"
            hotel_booking_state["booking_id"] = booking_id
            hotel_booking_state["step"] = "completed"

            response = f"🎉 Booking Confirmed! 🎉\n\n" \
                      f"Your booking has been successfully processed!\n\n" \
                      f"📋 Booking ID: {booking_id}\n" \
                      f"🏨 Hotel: {hotel_booking_state.get('hotel_name', 'Selected Hotel')}\n" \
                      f"🛏️ Room: {hotel_booking_state['room_type']}\n" \
                      f"📅 Check-in: {hotel_booking_state['check_in']}\n" \
                      f"📅 Check-out: {hotel_booking_state['check_out']}\n" \
                      f"👥 Guests: {hotel_booking_state['guests']}\n" \
                      f"💰 Total Paid: ₹{hotel_booking_state['total_price']:,}\n\n" \
                      f"Thank you for choosing our service! Have a wonderful stay!"

            # Reset booking state
            hotel_booking_state = {
                "active": False,
                "hotel_name": "",
                "room_type": "",
                "check_in": "",
                "check_out": "",
                "guests": 0,
                "total_price": 0,
                "step": "initial"
            }
            return response

        elif "cancel" in user_input.lower():
            hotel_booking_state = {
                "active": False,
                "hotel_name": "",
                "room_type": "",
                "check_in": "",
                "check_out": "",
                "guests": 0,
                "total_price": 0,
                "step": "initial"
            }
            return "Booking cancelled. How else can I help you today?"
        else:
            return "Please say 'confirm booking' to complete the booking or 'cancel booking' to cancel."

    return "I'm not sure how to help with that. Can you please clarify?"

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
        FunctionSchema(
            name="search_hotels",
            description=(
                "Search for hotels in a specific location with pricing from multiple booking sites."
                " Use this when users want to find hotels, compare prices, or look for accommodation."
            ),
            properties={
                "query": {
                    "type": "string",
                    "description": "Location and criteria for hotel search (e.g., 'hotels in Mumbai under 10000').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of hotels to return (1-10).",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            required=["query"],
        ),
        FunctionSchema(
            name="book_hotel",
            description=(
                "Handle hotel booking conversations and process booking steps."
                " Use this when users want to book a hotel or continue an existing booking."
            ),
            properties={
                "user_input": {
                    "type": "string",
                    "description": "The user's booking-related input or response.",
                },
            },
            required=["user_input"],
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
                " temperatures, rain chances, or sunrise/sunset information. ALWAYS call `search_hotels`"
                " when users want to find hotels, compare prices, or look for accommodation. ALWAYS call"
                " `book_hotel` when users want to book a hotel or continue an existing booking process."
                " Once a tool returns, reason over that data, mention sources when possible, and clearly"
                " explain what you found."
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

    async def search_hotels_handler(params: FunctionCallParams):
        args = params.arguments or {}
        query = args.get("query", "").strip()
        max_results = args.get("max_results", 8)
        try:
            max_results = int(max_results)
        except (TypeError, ValueError):
            max_results = 8
        max_results = max(1, min(max_results, 10))

        if not query:
            await params.result_callback({"result": "Please provide a location or criteria for hotel search."})
            return

        logger.info(f"[tools] search_hotels called with query: {query} (max_results={max_results})")
        result_text = search_hotels(query, max_results=max_results)
        await params.result_callback({"result": result_text})

    llm.register_function("search_hotels", search_hotels_handler, cancel_on_interruption=True)

    async def book_hotel_handler(params: FunctionCallParams):
        args = params.arguments or {}
        user_input = args.get("user_input", "").strip()

        if not user_input:
            await params.result_callback({"result": "Please provide your booking request or response."})
            return

        logger.info(f"[tools] book_hotel called with input: {user_input}")
        result_text = process_hotel_booking_step(user_input)
        await params.result_callback({"result": result_text})

    llm.register_function("book_hotel", book_hotel_handler, cancel_on_interruption=True)

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
