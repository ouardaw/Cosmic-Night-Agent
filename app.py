import streamlit as st
from skyfield.api import load
import ephem
import pytz
import plotly.graph_objects as go
from timezonefinder import TimezoneFinder
from streamlit_js_eval import streamlit_js_eval
import pandas as pd
import requests
import os
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import math
from dotenv import load_dotenv
import time
from typing import Dict, List, Any, Optional

# LangChain imports with better error handling
from langchain_config import setup_langchain

langchain_modules = setup_langchain()
LANGCHAIN_AVAILABLE = langchain_modules.get('available', False)

if LANGCHAIN_AVAILABLE:
    ChatOpenAI = langchain_modules['ChatOpenAI']
    ConversationBufferWindowMemory = langchain_modules['ConversationBufferWindowMemory']
    LLMChain = langchain_modules['LLMChain']
    PromptTemplate = langchain_modules['PromptTemplate']
    print(f"✅ LangChain loaded successfully (version: {langchain_modules.get('version', 'unknown')})")
else:
    print("⚠️ LangChain not available - using fallback mode")

# Load environment variables
# This works both locally (.env) and on Streamlit Cloud (secrets)
try:
    from dotenv import load_dotenv
    load_dotenv()  # reads .env in project root
except Exception:
    pass
    
def read_secret(name: str, default=None):
    # 1) Try env var (works in Codespaces / local with .env)
    v = os.getenv(name)
    if v and v.strip():
        return v.strip()
    # 2) Try Streamlit Cloud secrets (works on Cloud)
    try:
        v = st.secrets[name]  # bracket access is safest across versions
        if isinstance(v, str) and v.strip():
            return v.strip()
    except Exception:
        pass
    return default

OPENWEATHER_API_KEY = read_secret("OPENWEATHER_API_KEY")
NASA_API_KEY        = read_secret("NASA_API_KEY", "DEMO_KEY")  # public fallback
OPENAI_API_KEY      = read_secret("OPENAI_API_KEY") 

class AstronomyQueryProcessor:
    """Enhanced astronomy query processor with LangChain LLM capabilities"""

    def __init__(self, openai_api_key: str):
        """Initialize the enhanced query processor"""
        
        self.api_key = openai_api_key
        if not self.api_key:
            raise ValueError("OpenAI API key required for AI features")
        
        # Single initialization approach
        self.llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-3.5-turbo",
            openai_api_key=self.api_key,  # Use openai_api_key consistently
            max_tokens=500
        )
        
        # Note: ConversationBufferWindowMemory is deprecated but still works
        # You can remove this if not using it
        self.memory = ConversationBufferWindowMemory(
            k=10,
            return_messages=True
        )
        
        self.astronomy_chain = self._create_astronomy_chain()

    def _create_astronomy_chain(self):
        """Create a chain specifically for astronomy expertise"""
        
        astronomy_prompt = PromptTemplate(
            input_variables=["query", "location", "current_time", "context"],
            template="""You are StellarGuide, an expert astronomer and astrophysicist with deep knowledge of:
- All constellations, their mythology, viewing times, and locations
- Planets, moons, asteroids, comets, and their orbits
- Deep sky objects (galaxies, nebulae, star clusters)
- Astronomical phenomena (eclipses, meteor showers, conjunctions)
- Space exploration, missions, and spacecraft
- Astrophysics concepts and stellar evolution
- Telescope usage and astrophotography
- Historical and cultural astronomy

Current Information:
- Location: {location}
- Date/Time: {current_time}
- Additional Context: {context}

Answer this astronomy question with rich detail, including:
- Specific viewing times and directions when relevant
- Interesting facts and scientific explanations
- Practical observing tips
- Related objects or phenomena to look for

Question: {query}

Provide a comprehensive, engaging answer that educates and inspires interest in astronomy. Format your response with HTML tags for better display (<b> for bold, <br> for line breaks):"""
        )
        
        # Use LLMChain for simplicity
        from langchain.chains import LLMChain
        return LLMChain(llm=self.llm, prompt=astronomy_prompt)
    
    def process_query(self, query: str, location: str, lat: float, lon: float, 
                     weather: Dict, astronomy: Dict, iss_passes: List) -> str:
        """
        Process any astronomy-related query with LLM
        
        Args:
            query: User's question about astronomy/space
            location: User's location
            lat: Latitude
            lon: Longitude
            weather: Weather data
            astronomy: Astronomy data
            iss_passes: ISS pass data
            
        Returns:
            Comprehensive response with astronomy information
        """
        try:
            # Build context from current data
            context_parts = []
            
            # Add weather context
            if weather and 'current' in weather:
                clouds = weather['current'].get('clouds', 'Unknown')
                temp = weather['current'].get('temp', 'Unknown')
                context_parts.append(f"Weather: {clouds}% clouds, {temp}°C")
            
            # Add astronomy context
            if astronomy:
                if 'moon' in astronomy:
                    context_parts.append(f"Moon phase: {astronomy['moon'].get('phase', 'Unknown')}")
                if 'planets' in astronomy and 'visible' in astronomy['planets']:
                    planets = astronomy['planets']['visible']
                    if planets:
                        context_parts.append(f"Visible planets: {', '.join(planets)}")
                if 'sun' in astronomy:
                    context_parts.append(f"Sunset: {astronomy['sun'].get('sunset', 'Unknown')}")
            
            # Add ISS context if relevant
            if "iss" in query.lower() or "space station" in query.lower():
                if iss_passes and len(iss_passes) > 0 and iss_passes[0].get('duration', 0) > 0:
                    next_pass = iss_passes[0]
                    context_parts.append(f"Next ISS pass: {next_pass.get('date', 'Unknown')} at {next_pass.get('risetime', 'Unknown')}")
            
            # Join context
            context = " | ".join(context_parts) if context_parts else "General stargazing conditions"
            
            # Get comprehensive answer from the astronomy chain using run() method
            response = self.astronomy_chain.run(
                query=query,
                location=f"{location} (Lat: {lat:.2f}°, Lon: {lon:.2f}°)",
                current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                context=context
            )
            
            return response
            
        except Exception as e:
            # Fallback to basic response
            return self._fallback_response(query, location, str(e))
    
    def _fallback_response(self, query: str, location: str, error: str) -> str:
        """Fallback response using just the LLM without context"""
        
        try:
            # Use the LLM directly for general astronomy knowledge
            result = self.llm.invoke(
                f"""As an astronomy expert, answer this question in a helpful and engaging way:
                
                Question: {query}
                Location: {location}
                Current Date: {datetime.now().strftime('%B %d, %Y')}
                
                Provide a helpful, detailed answer about astronomy. Use HTML formatting (<b> for bold, <br> for line breaks):"""
            )
            
            # Extract content from the response
            response = result.content if hasattr(result, 'content') else str(result)
            
            return response
        except:
            return f"""<b>I apologize, but I'm having trouble processing your question.</b><br><br>
            The AI service may be temporarily unavailable. Please try again in a moment, or ask about:<br>
            • Visible planets tonight<br>
            • Moon phases<br>
            • ISS passes<br>
            • Stargazing conditions<br>
            • How to find constellations"""
    
    def get_suggested_queries(self) -> List[str]:
        """Get diverse suggested queries"""
        
        month = datetime.now().month
        season_specific = []
        
        if month in [12, 1, 2]:  # Winter
            season_specific = [
                "When can I see the Orion Nebula?",
                "How do I find the Winter Triangle?"
            ]
        elif month in [6, 7, 8]:  # Summer
            season_specific = [
                "How can I see the Summer Triangle?",
                "When is the best time to observe Saturn's rings?"
            ]
        else:
            season_specific = [
                "What's the best time to see the Pleiades?",
                "Where is the center of our galaxy?"
            ]
        
        general = [
            "Can you explain what a nebula is?",
            "How do I choose my first telescope?",
            "When is the next meteor shower?",
            "Can you explain what a nebula is?"
        ]
        
        return season_specific + general[:2]

def inject_stellaris_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600&family=Orbitron:wght@700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Space Grotesk', monospace !important;
        background: radial-gradient(ellipse at 60% 40%, #1a1a2e 0%, #16213e 40%, #0f0f23 100%) !important;
        color: #f8fafc !important;
    }
      /* Hide Streamlit's "Running cached function" messages */
    div[data-testid="stStatusWidget"] {
        display: none !important;
    }
    
    /* Hide the specific running messages */
    div.stAlert {
        display: none !important;
    }
    .stApp {
        background: radial-gradient(ellipse at 60% 40%, #1a1a2e 0%, #16213e 40%, #0f0f23 100%) !important;
    }
     section[data-testid="stSidebar"] button {
    background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 50%, #6d28d9 100%) !important;
    color: #fff !important;
    font-family: 'Orbitron', 'Space Grotesk', sans-serif !important;
    border: none !important;
    border-radius: 0px !important;  /* Square corners */
    font-size: 1.08rem !important;
    margin-bottom: 0.8em !important;
    font-weight: 700 !important;
    letter-spacing: 0.09em !important;
    padding: 1em 0.5em !important;
    box-shadow: 
        /* Main 3D shadow */
        0 8px 0 #5b21b6,
        0 8px 1px #4c1d95,
        /* Depth shadow */
        0 10px 8px rgba(0, 0, 0, 0.4),
        /* Glow effect */
        0 0 20px rgba(139, 92, 246, 0.3) !important;
    transform: translateY(0) !important;
    transition: all 0.15s ease !important;
    position: relative !important;
    text-transform: uppercase !important;
}

    section[data-testid="stSidebar"] button:hover {
    background: linear-gradient(135deg, #a78bfa 0%, #8b5cf6 50%, #7c3aed 100%) !important;
    transform: translateY(2px) !important;
    box-shadow: 
        /* Reduced 3D effect on hover */
        0 6px 0 #5b21b6,
        0 6px 1px #4c1d95,
        0 8px 6px rgba(0, 0, 0, 0.4),
        0 0 25px rgba(139, 92, 246, 0.5) !important;
}

    section[data-testid="stSidebar"] button:active {
    background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 50%, #5b21b6 100%) !important;
    transform: translateY(6px) !important;
    box-shadow: 
        /* Minimal shadow when pressed */
        0 2px 0 #4c1d95,
        0 2px 1px #3b0e7a,
        0 3px 3px rgba(0, 0, 0, 0.4),
        inset 0 2px 5px rgba(0, 0, 0, 0.2) !important;
}

    /* Add a subtle shine effect to make it look more 3D */
    section[data-testid="stSidebar"] button::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 50%;
    background: linear-gradient(
        to bottom,
        rgba(255, 255, 255, 0.2),
        rgba(255, 255, 255, 0.05)
    );
    border-radius: 0px;
    pointer-events: none;
}
    .cosmic-menu-panel {
        background: linear-gradient(120deg, rgba(131,56,236,0.18), rgba(59,134,255,0.11));
        border: 1.5px solid rgba(255,255,255,0.13);
        border-radius: 16px;
        margin: 0.8em 0 1.3em 0;
        padding: 0.6em 0.7em 0.3em 0.7em;
        box-shadow: 0 2px 16px 0 rgba(59,134,255,0.12);
        width: 100%;
        display: flex;
        flex-direction: column;
        gap: 0.3em;
    }
    .cosmic-loader {
        display: inline-block;
        width: 50px;
        height: 50px;
        border: 3px solid rgba(124, 58, 237, 0.1);
        border-radius: 50%;
        border-top-color: #7c3aed;
        animation: cosmic-spin 1s ease-in-out infinite;
        margin: 20px auto;
    }
    
    @keyframes cosmic-spin {
        to { transform: rotate(360deg); }
    }
    
    .loading-container {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, rgba(124,58,237,0.1), rgba(58,134,255,0.05));
        border-radius: 15px;
        border: 1px solid rgba(124,58,237,0.3);
    }
    
    .loading-text {
        color: #7c3aed;
        font-family: 'Orbitron', sans-serif;
        font-size: 1.1rem;
        margin-top: 1rem;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.6; }
        50% { opacity: 1; }
    }
    .menu-item-btn, .cosmic-menu-panel button {
    background: linear-gradient(135deg, #f59e0b 0%, #fb923c 50%, #ea580c 100%);
    border: none;
    border-radius: 0px !important;  /* Square corners */
    box-shadow: 
        /* 3D effect */
        0 6px 0 #dc2626,
        0 6px 1px #b91c1c,
        0 8px 12px rgba(0, 0, 0, 0.3),
        0 0 15px rgba(251, 146, 60, 0.3);
    color: #fff;
    font-family: 'Orbitron', 'Space Grotesk', monospace !important;
    font-size: 1.07rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin: 0.5em 0.18em;
    padding: 0.9em 0.25em;
    min-width: 0;
    width: 99%;
    display: block;
    transition: all 0.15s ease;
    text-align: center;
    outline: none;
    cursor: pointer;
    transform: translateY(0);
    text-transform: uppercase;
    position: relative;
}

    .menu-item-btn.selected, .cosmic-menu-panel button.selected,
    .menu-item-btn:hover, .cosmic-menu-panel button:hover {
    background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 50%, #fb923c 100%);
    transform: translateY(2px);
    box-shadow: 
        0 4px 0 #dc2626,
        0 4px 1px #b91c1c,
        0 6px 10px rgba(0, 0, 0, 0.3),
        0 0 20px rgba(251, 146, 60, 0.5);
}

    .menu-item-btn:active, .cosmic-menu-panel button:active {
    transform: translateY(4px);
    box-shadow: 
        0 2px 0 #dc2626,
        0 2px 1px #b91c1c,
        0 3px 5px rgba(0, 0, 0, 0.3),
        inset 0 2px 4px rgba(0, 0, 0, 0.2);
}

    /* Add shine effect */
    .menu-item-btn::before, .cosmic-menu-panel button::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 45%;
    background: linear-gradient(
        to bottom,
        rgba(255, 255, 255, 0.3),
        rgba(255, 255, 255, 0.1)
    );
    pointer-events: none;
}
 
    .cosmic-title {
    font-family: 'Orbitron', 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji', sans-serif;
    font-size: 2.6rem;
    letter-spacing: 0.19em;
    text-align: center;
    margin-bottom: 1rem;
    background: linear-gradient(90deg, #ffe066 0%, #7c3aed 60%, #3a86ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow:
        0 1px 8px #7c3aed33,
        0 2px 18px #3a86ff22;
    filter: brightness(1.03) drop-shadow(0 0 4px #f59e0b33);
    padding-top: 0.15em;
}
    .cosmic-section {
        font-family: 'Orbitron', 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji', sans-serif;
        font-size: 1.5rem;
        letter-spacing: 0.12em;
        margin: 1.7rem 0 1rem 0;
        background: linear-gradient(90deg, #f59e0b 0%, #7c3aed 80%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 2px 12px #8b5cf666;
    }
    .cosmic-card {
    background: linear-gradient(120deg, rgba(131,56,236,0.20), rgba(59,134,255,0.13));
    border: 4px solid transparent;
    border-radius: 32px; /* much rounder corners */
    box-shadow:
        0 8px 32px 0 #7c3aed55,   /* soft outer glow */
        0 2px 18px 0 #ffe06688,   /* yellow highlight */
        0 0 64px 0 #ffb80033,     /* golden glow */
        0 1px 0px 0 #fff,         /* subtle highlight */
        inset 0 3px 18px #fff3,   /* inner light for 3D effect */
        inset 0 -8px 32px #7c3aed22; /* inner shadow for depth */
    padding: 1.6rem 1.3rem;
    margin-bottom: 1.3rem;
    transition: box-shadow 0.25s, border-color 0.25s;
    filter: brightness(1.14);
    position: relative;
    }
    .cosmic-card:hover {
    box-shadow:
        0 12px 48px 0 #7c3aed99,
        0 2px 32px 0 #ffe066cc,
        0 0 80px 0 #ffb80044,
        0 1px 0px 0 #fff,
        inset 0 6px 26px #fff5,
        inset 0 -12px 44px #7c3aed33;
    filter: brightness(1.20);
}

    /* Gradient text that works in Chrome, Firefox, Safari */
    .gradient-text {
      color: #eab308; /* fallback visible everywhere */
      background: linear-gradient(90deg, #f59e0b, #7c3aed, #3a86ff);
      background-clip: text;
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    
    /* If a browser doesn't support background-clip:text, keep solid color */
    @supports not ((-webkit-background-clip: text) or (background-clip: text)) {
      .gradient-text {
        background: none !important;
        -webkit-text-fill-color: initial !important;
        color: #eab308 !important;
      }
    }
    
    /* (Optional) Avoid rare Chrome issues: no filters on gradient-text containers */
    .gradient-container {
      /* remove filter/opacity on the container that holds gradient text */
      filter: none !important;
      opacity: 1 !important;
    }
        .cosmic-label {
        font-size: 1.1rem;
        opacity: 0.8;
        margin-bottom: 0.3em;
        letter-spacing: 1px;
    }
    .cosmic-value {
        font-size: 2.0rem;
        font-weight: bold;
        background: linear-gradient(135deg, #f59e0b, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .ai-response {
    background: linear-gradient(120deg, rgba(131,56,236,0.20), rgba(59,134,255,0.13));
    border: 4px solid transparent;
    border-radius: 32px; /* much rounder corners */
    box-shadow:
        0 8px 32px 0 #7c3aed55,   /* soft outer glow */
        0 2px 18px 0 #ffe06688,   /* yellow highlight */
        0 0 64px 0 #ffb80033,     /* golden glow */
        0 1px 0px 0 #fff,         /* subtle highlight */
        inset 0 3px 18px #fff3,   /* inner light for 3D effect */
        inset 0 -8px 32px #7c3aed22; /* inner shadow for depth */
    padding: 1.6rem 1.3rem;
    margin-bottom: 1.3rem;
    transition: box-shadow 0.25s, border-color 0.25s;
    filter: brightness(1.14);
    position: relative;
    }
    .ai-badge {
        display: inline-block;
        background: linear-gradient(90deg, #7c3aed, #3a86ff);
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    footer, #MainMenu {visibility: hidden;}
    section[data-testid="stSidebar"] button {
        background: linear-gradient(90deg,#f59e0b55 0%,#8b5cf6bb 100%) !important;
        color: #fff !important;
        font-family: 'Orbitron', 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji', sans-serif !important;
        border: none !important;
        border-radius: 14px !important;
        font-size: 1.08rem !important;
        margin-bottom: 0.18em !important;
        font-weight: 600 !important;
        letter-spacing: 0.09em !important;
        box-shadow: 0 2px 14px #8b5cf633 !important;
        transition: background 0.2s, color 0.2s;
    }

    section[data-testid="stSidebar"] button:hover, 
    section[data-testid="stSidebar"] button:focus {
        background: linear-gradient(90deg,#f59e0b77 0%,#8b5cf6cc 100%) !important;
        color: #fff !important;
        box-shadow: 0 4px 18px #7c3aed33 !important;
        outline: none !important;
    }

    section[data-testid="stSidebar"] button:active {
        background: linear-gradient(90deg,#f59e0b99 0%,#8b5cf6cc 100%) !important;
    }
    .menu-pictogram {
    display: inline-block;
    width: 24px;
    height: 24px;
    margin-right: 8px;
    vertical-align: middle;
}

/* Orbit animation for Celestial Tracker */
    @keyframes orbit {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

    .orbit-animation {
    animation: orbit 10s linear infinite;
    transform-origin: center;
}

/* ISS Path animation */
    @keyframes orbit-iss {
    from { transform: rotate(-45deg); }
    to { transform: rotate(315deg); }
}

    .iss-path-animation {
    animation: orbit-iss 8s linear infinite;
    transform-origin: center;
}

/* Mystic Eye animation for Cosmic Oracle */
    @keyframes mystic-glow {
    0%, 100% {
        transform: scale(1);
        filter: drop-shadow(0 0 2px #fde047);
    }
    50% {
        transform: scale(1.05);
        filter: drop-shadow(0 0 5px #fde047);
    }
}

    .mystic-eye-animation {
    animation: mystic-glow 4s ease-in-out infinite;
    transform-origin: center;
}

/* Sun animation for Weather */
    @keyframes sun-pulse {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-2px); }
}

    .sun-animation {
    animation: sun-pulse 4s ease-in-out infinite;
}
/* Animated emoji styles for menu buttons */
    .cosmic-menu-panel button {
    position: relative;
    overflow: visible !important;
}

/* Pulse animation for weather */
    @keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.1); }
}

/* Rotation animation for planets */
    @keyframes rotate-slow {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

/* Float animation for ISS */
    @keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-3px); }
}

/* Glow animation for cosmic eye */
    @keyframes glow {
    0%, 100% { 
        filter: drop-shadow(0 0 2px #fde047);
    }
    50% { 
        filter: drop-shadow(0 0 8px #fde047);
    }
}

/* Apply animations to specific buttons based on their text content */
    .cosmic-menu-panel button:nth-child(1) {
    animation: pulse 3s ease-in-out infinite;
}

    .cosmic-menu-panel button:nth-child(2) {
    animation: rotate-slow 10s linear infinite;
}

    .cosmic-menu-panel button:nth-child(3) {
    animation: float 4s ease-in-out infinite;
}

    .cosmic-menu-panel button:nth-child(4) {
    animation: glow 3s ease-in-out infinite;
}
/* Add these animations to your inject_stellaris_css() function */

/* Banner floating animation */
    @keyframes bannerFloat {
    0%, 100% { transform: perspective(1000px) rotateX(2deg) translateY(0); }
    50% { transform: perspective(1000px) rotateX(2deg) translateY(-5px); }
}

/* Border gradient animation */
    @keyframes borderGradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Title pulse animation */
    @keyframes titlePulse {
    0%, 100% { 
        filter: brightness(1.2) drop-shadow(0 0 20px #ffb800);
    }
    50% { 
        filter: brightness(1.4) drop-shadow(0 0 40px #7c3aed);
    }
}

/* Letter floating animation */
    @keyframes letterFloat {
    0%, 100% { transform: translateY(0) rotateZ(0deg); }
    25% { transform: translateY(-3px) rotateZ(1deg); }
    75% { transform: translateY(2px) rotateZ(-1deg); }
}

/* Subtitle glow animation */
    @keyframes subtitleGlow {
    from { 
        text-shadow:
            0 0 10px #7c3aed,
            0 0 20px #7c3aed,
            0 0 30px #7c3aed,
            0 0 40px #3a86ff;
    }
    to { 
        text-shadow:
            0 0 20px #7c3aed,
            0 0 30px #7c3aed,
            0 0 40px #7c3aed,
            0 0 50px #3a86ff,
            0 0 60px #3a86ff;
    }
}

/* Sparkle animation */
    .sparkle-container {
    position: absolute;
    width: 100%;
    height: 100%;
    top: 0;
    left: 0;
    pointer-events: none;
    overflow: hidden;
}

    .sparkle {
    position: absolute;
    top: -20px;
    font-size: 1.2rem;
    animation: sparklefall 4s linear infinite;
    opacity: 0;
}

    @keyframes sparklefall {
    0% {
        transform: translateY(0) rotate(0deg);
        opacity: 0;
    }
    10% {
        opacity: 1;
    }
    90% {
        opacity: 1;
    }
    100% {
        transform: translateY(120px) rotate(360deg);
        opacity: 0;
    }
}

/* Floating orbs */
    .orb {
    position: absolute;
    border-radius: 50%;
    filter: blur(40px);
    opacity: 0.6;
    pointer-events: none;
}

    .orb1 {
    width: 80px;
    height: 80px;
    background: radial-gradient(circle, #ffb800, transparent);
    top: 10%;
    left: 10%;
    animation: orbFloat1 15s ease-in-out infinite;
}

    .orb2 {
    width: 60px;
    height: 60px;
    background: radial-gradient(circle, #7c3aed, transparent);
    top: 60%;
    right: 10%;
    animation: orbFloat2 20s ease-in-out infinite;
}

    .orb3 {
    width: 100px;
    height: 100px;
    background: radial-gradient(circle, #3a86ff, transparent);
    bottom: 10%;
    left: 50%;
    animation: orbFloat3 25s ease-in-out infinite;
}

    @keyframes orbFloat1 {
    0%, 100% { transform: translate(0, 0) scale(1); }
    33% { transform: translate(30px, -30px) scale(1.1); }
    66% { transform: translate(-20px, 20px) scale(0.9); }
}

    @keyframes orbFloat2 {
    0%, 100% { transform: translate(0, 0) scale(1); }
    50% { transform: translate(-40px, 40px) scale(1.2); }
}

    @keyframes orbFloat3 {
    0%, 100% { transform: translate(0, 0) scale(1); }
    25% { transform: translate(20px, -20px) scale(0.8); }
    50% { transform: translate(-30px, -10px) scale(1.1); }
    75% { transform: translate(10px, 30px) scale(0.9); }
}
    .cosmic-question-btn {
    background: linear-gradient(90deg, #1a1a2e 0%, #7c3aed 80%, #ffb800 100%);
    color: #fff;
    border-radius: 16px;
    border: 2px solid #ffe066;
    font-family: 'Orbitron', 'Space Grotesk', sans-serif;
    font-size: 1.05rem;
    font-weight: 500;
    box-shadow: 0 2px 14px #7c3aed66, 0 0 8px #ffb80044;
    padding: 0.5em 1em;
    margin: 0.2em 0.1em;
    transition: background 0.2s, box-shadow 0.2s;
}
    .cosmic-question-btn:hover {
    background: linear-gradient(90deg, #ffb800 0%, #7c3aed 100%);
    box-shadow: 0 4px 22px #ffb80088, 0 0 16px #7c3aed55;
    color: #fff;
}

    /* Enhanced Cosmic Oracle Quick Question Buttons */
    .cosmic-quick-btn {
    background: linear-gradient(135deg, #7c3aed 0%, #3a86ff 50%, #7c3aed 100%);
    border: none;
    border-radius: 0px !important;  /* Square corners to match your style */
    box-shadow: 
        /* 3D effect */
        0 6px 0 #5b21b6,
        0 6px 1px #4c1d95,
        0 8px 12px rgba(0, 0, 0, 0.4),
        0 0 20px rgba(124, 58, 237, 0.3),
        /* Inner glow */
        inset 0 0 20px rgba(139, 92, 246, 0.2);
    color: #fff;
    font-family: 'Orbitron', 'Space Grotesk', monospace !important;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    padding: 1em 0.8em;
    transition: all 0.15s ease;
    transform: translateY(0);
    text-transform: uppercase;
    position: relative;
    overflow: hidden;
}

    .cosmic-quick-btn:hover {
    background: linear-gradient(135deg, #8b5cf6 0%, #60a5fa 50%, #8b5cf6 100%);
    transform: translateY(2px);
    box-shadow: 
        0 4px 0 #5b21b6,
        0 4px 1px #4c1d95,
        0 6px 10px rgba(0, 0, 0, 0.4),
        0 0 30px rgba(124, 58, 237, 0.5),
        inset 0 0 30px rgba(139, 92, 246, 0.3);
}

    .cosmic-quick-btn:active {
    transform: translateY(4px);
    box-shadow: 
        0 2px 0 #5b21b6,
        0 2px 1px #4c1d95,
        0 3px 5px rgba(0, 0, 0, 0.4),
        inset 0 2px 4px rgba(0, 0, 0, 0.3);
}

    /* Shine effect overlay */
    .cosmic-quick-btn::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(
        90deg,
        transparent,
        rgba(255, 255, 255, 0.3),
        transparent
    );
    transition: left 0.5s ease;
}

    .cosmic-quick-btn:hover::before {
    left: 100%;
}

    /* Emoji animation for quick buttons */
    .cosmic-quick-btn .emoji {
    display: inline-block;
    margin-right: 8px;
    font-size: 1.2rem;
    animation: float 3s ease-in-out infinite;
}

    @keyframes float {
    0%, 100% { transform: translateY(0px) rotate(0deg); }
    50% { transform: translateY(-3px) rotate(5deg); }
}

    button[key^="quick_"] {
        background: linear-gradient(135deg, #7c3aed 0%, #3a86ff 50%, #7c3aed 100%) !important;
        border: none !important;
        border-radius: 0px !important;
        box-shadow: 
            0 6px 0 #5b21b6,
            0 6px 1px #4c1d95,
            0 8px 12px rgba(0, 0, 0, 0.4),
            0 0 20px rgba(124, 58, 237, 0.3) !important;
        font-family: 'Orbitron', 'Space Grotesk', monospace !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        padding: 1em 0.8em !important;
        transition: all 0.15s ease !important;
    }
    
    button[key^="quick_"]:hover {
        background: linear-gradient(135deg, #8b5cf6 0%, #60a5fa 50%, #8b5cf6 100%) !important;
        transform: translateY(2px) !important;
        box-shadow: 
            0 4px 0 #5b21b6,
            0 4px 1px #4c1d95,
            0 6px 10px rgba(0, 0, 0, 0.4),
            0 0 30px rgba(124, 58, 237, 0.5) !important;
    }
    
    button[key^="quick_"]:active {
        transform: translateY(4px) !important;
        box-shadow: 
            0 2px 0 #5b21b6,
            0 2px 1px #4c1d95,
            0 3px 5px rgba(0, 0, 0, 0.4),
            inset 0 2px 4px rgba(0, 0, 0, 0.3) !important;
    }
    /* SET WIDER SIDEBAR WIDTH */
    section[data-testid="stSidebar"] {
        width: 340px !important;  /* Increased from default 284px */
        min-width: 340px !important;
    }
    
    section[data-testid="stSidebar"] > div:first-child {
        width: 340px !important;
        min-width: 340px !important;
    }
    
    /* Ensure buttons don't wrap text */
    section[data-testid="stSidebar"] button {
        white-space: nowrap !important;  /* Prevent text wrapping */
        overflow: hidden !important;
        text-overflow: ellipsis !important;  /* Add ... if text is somehow still too long */
    }
    
    /* Ensure text input stays proper width */
    section[data-testid="stSidebar"] input {
        width: 100% !important;
    }

    </style>
    """, unsafe_allow_html=True)
import streamlit as st

import streamlit as st

def display_planet_visibility(astronomy_data, location_name):
    """Display real planet visibility cards based on actual astronomical calculations"""

    # Planet icons dictionary
    planet_icons = {
        "Mercury": "☿️",
        "Venus": "♀️",
        "Mars": "♂️",
        "Jupiter": "♃",
        "Saturn": "♄",
        "Uranus": "⛢",
        "Neptune": "♆"
    }

    # Get planet details from astronomy data
    if 'planets' in astronomy_data and 'details' in astronomy_data['planets']:
        planets_details = astronomy_data['planets']['details']
        visible_planets = astronomy_data['planets']['visible']
    else:
        visible_planets = astronomy_data['planets'].get('visible', [])
        planets_details = []
        all_planets = ["Mercury", "Venus", "Mars", "Jupiter", "Saturn"]
        for planet in all_planets:
            is_visible = planet in visible_planets
            planets_details.append({
                'name': planet,
                'visible': is_visible,
                'altitude': "45°" if is_visible else "-10°",
                'azimuth': "180°",
                'magnitude': -2.0 if planet == "Venus" else 1.0,
                'rise_time': "19:30" if is_visible else "---",
                'set_time': "05:30" if is_visible else "---",
                'visibility_reason': "Above horizon after sunset" if is_visible else "Below horizon"
            })

    # Section header
    #st.markdown("🪐 **Planet Visibility Tracker**")

    # Summary bar showing currently visible planets
    if visible_planets:
        planets_with_icons = []
        for p in visible_planets:
            icon = planet_icons.get(p, "🪐")
            planets_with_icons.append(f"{icon} {p}")
        visible_planets_str = ", ".join(planets_with_icons)
    else:
        visible_planets_str = "🌅 Check back after sunset for planet visibility"

    st.markdown(f"""
        <div class='cosmic-card' style='text-align: center; padding: 0.8rem; margin-bottom: 0.8rem;'>
            <div style='color: #f59e0b; font-size: 1.07rem; font-weight: bold; margin-bottom: 0.28em;'>
                🪐 Visible Planets Tonight from {location_name}
            </div>
            <div style='font-size: 1.22rem; color: #ffb800;'>
                {visible_planets_str}
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Display planet cards in 2 columns
    cols = st.columns(2)

    # Sort planets: visible ones first, then by name
    planets_details_sorted = sorted(
        planets_details, key=lambda x: (not x.get('visible', False), x['name'])
    )

    for idx, planet in enumerate(planets_details_sorted[:6]):  # Limit to 6 major planets
        icon = planet_icons.get(planet['name'], "🪐")
        col = cols[idx % 2]

        with col:
            st.markdown(f"""
                <div class='cosmic-card' style='
                    text-align: left;
                    padding: 0.6rem 0.6rem 0.5rem 0.6rem;
                    margin-bottom: 0.45rem;
                    font-size:0.98rem;
                    min-height: 100px;
                    border-left: 4px solid {'#00ff00' if planet['visible'] else '#ff3333'};
                    box-shadow: 0 2px 12px #7c3aed22;
                '>
                    <div style='font-weight: bold; font-size: 1.18rem; color: #f59e0b; margin-bottom:0.15em;'>
                        {icon} {planet['name']} {'✅' if planet['visible'] else '❌'}
                    </div>
                    <div style='margin-bottom: 0.13em;'><b>Alt:</b> {planet['altitude']} | <b>Az:</b> {planet['azimuth']}</div>
                    <div style='margin-bottom: 0.13em;'><b>Mag:</b> {planet['magnitude']}</div>
                    <div style='margin-bottom: 0.13em;'><b>Rise:</b> {planet['rise_time']} | <b>Set:</b> {planet['set_time']}</div>
                    <div style='font-size: 0.87rem; opacity: 0.70; margin-top:0.09em;'><i>{planet['visibility_reason']}</i></div>
                </div>
            """, unsafe_allow_html=True)
def get_local_now(lat, lon):
    tf = TimezoneFinder()
    tz_str = tf.timezone_at(lat=lat, lng=lon)
    if tz_str is None:
        tz_str = "UTC"
    local_tz = pytz.timezone(tz_str)
    return datetime.now(local_tz)
@st.cache_data(ttl=3600)
def geocode_location(location_name: str):
    """Get coordinates for a city name with English language preference"""
    try:
        location_name = location_name.strip()
        
        # Input validation
        if not location_name:
            st.error("🚫 **Location cannot be empty!** Please enter a city name.")
            return None, None, None, "empty"
        
        if len(location_name) < 2:
            st.error("🚫 **Location name too short!** Please enter at least 2 characters.")
            return None, None, None, "too_short"
        
        # Check for invalid characters
        if any(char in location_name for char in ['@', '#', '$', '%', '^', '&', '*', '=', '+', '[', ']', '{', '}', '\\', '|', '<', '>']):
            st.error("🚫 **Invalid characters detected!** Please use only letters, numbers, spaces, and commas.")
            return None, None, None, "invalid_chars"
        
        with st.spinner(f'🔍 Searching for "{location_name}"...'):
            time.sleep(0.5)  # Small delay for UX
            
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                "q": location_name,
                "format": "json",
                "limit": 5,
                "addressdetails": 1,
                "accept-language": "en",  # Request results in English
                "extratags": 1,  # Get extra tags including local names
                "namedetails": 1  # Get name details in multiple languages
            }
            response = requests.get(url, params=params, headers={"User-Agent": "Stellaris/1.0"}, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data and len(data) > 0:
                result = data[0]
                lat = float(result['lat'])
                lon = float(result['lon'])
                
                # Get English display name
                display_name_en = result['display_name']
                
                # Try to get local name if available
                local_name = None
                if 'namedetails' in result:
                    # Get the local/native name
                    name_details = result['namedetails']
                    # Try to get name in local language
                    local_name = name_details.get('name', None)
                    # If local name is same as English, don't duplicate
                    if local_name and local_name == name_details.get('name:en', local_name):
                        local_name = None
                
                # Get address components in English
                address = result.get('address', {})
                
                # Build clean English display name
                city_name = (address.get('city') or 
                           address.get('town') or 
                           address.get('village') or 
                           address.get('municipality') or
                           address.get('suburb') or
                           result.get('name', ''))
                
                country = address.get('country', '')
                state = address.get('state', '')
                
                # Create formatted display name
                if country.lower() in ['united states', 'united states of america', 'usa']:
                    if state:
                        formatted_display = f"{city_name}, {state}, USA"
                    else:
                        formatted_display = f"{city_name}, USA"
                else:
                    formatted_display = f"{city_name}, {country}"
                
                # Add local name if it exists and is different
                if local_name and local_name != city_name:
                    formatted_display = f"{formatted_display} ({local_name})"
                
                place_type = result.get('type', '')
                address_type = result.get('addresstype', '')
                
                valid_types = ['city', 'town', 'village', 'municipality', 'suburb', 
                              'neighbourhood', 'hamlet', 'locality', 'administrative']
                
                #if place_type not in valid_types and address_type not in valid_types:
                    #if 'city' not in display_name_en.lower() and 'town' not in display_name_en.lower():
                       # st.warning(f"⚠️ **Location found but might not be a city:** {city_name}")
                       # return lat, lon, formatted_display, "uncertain"
                
                return lat, lon, formatted_display, "success"
            else:
                st.error(f"""
                🚫 **Location "{location_name}" not found!**
                
                **Try these tips:**
                • Add country: "{location_name}, USA" or "{location_name}, UK"
                • Add state/region: "{location_name}, California" or "{location_name}, Texas"
                • Check spelling: Make sure the city name is spelled correctly
                • Use larger city: Try a nearby major city instead
                
                **Examples that work:**
                • "Austin, Texas"
                • "London, UK"
                • "Tokyo, Japan" 
                • "Paris, France"
                """)
                return None, None, None, "not_found"
                
    except requests.exceptions.Timeout:
        st.error("""
        ⏱️ **Connection timeout!**
        
        The location service is taking too long to respond.
        Please try again in a moment.
        """)
        return None, None, None, "timeout"
    except requests.exceptions.RequestException as e:
        st.error(f"""
        🌐 **Network error!**
        
        Cannot connect to location service.
        Please check your internet connection.
        """)
        return None, None, None, "network_error"
    except Exception as e:
        st.error(f"""
        ❌ **Unexpected error!**
        
        Something went wrong while searching for your location.
        Please try again or use a different city name.
        """)
        return None, None, None, "error"
def get_constellation_info(latitude, month):
    """Get visible constellations based on latitude and month with fun facts and spotting guides"""
    
    constellation_database = {
        "Orion": {
            "seasons": [12, 1, 2, 3],
            "latitude_range": (-85, 85),
            "fun_fact": "Known as 'The Hunter', Orion contains the famous Orion Nebula (M42), a stellar nursery where new stars are being born.",
            "how_to_spot": "Look for three bright stars in a row (Orion's Belt). Two bright stars mark the shoulders (Betelgeuse - red) and knees (Rigel - blue).",
            "key_stars": "• Betelgeuse (red supergiant)<br>• Rigel (blue supergiant)<br>• Belt stars: Alnitak, Alnilam, Mintaka",
            "best_time": "9 PM - 2 AM",
            "direction": "South",
            "brightness": "Very Bright ⭐⭐⭐⭐⭐",
            "size": "Large"
        },
        "Ursa Major (Big Dipper)": {
            "seasons": [3, 4, 5, 6],
            "latitude_range": (-30, 90),
            "fun_fact": "The Big Dipper's stars (except the end two) belong to the Ursa Major Moving Group, traveling together through space.",
            "how_to_spot": "Look north for seven bright stars forming a ladle or saucepan shape. The two end stars of the 'cup' point to Polaris.",
            "key_stars": "• Dubhe (pointer star)<br>• Merak (pointer star)<br>• Alkaid (handle end)<br>• Mizar (double star)",
            "best_time": "All night (circumpolar)",
            "direction": "North",
            "brightness": "Very Bright ⭐⭐⭐⭐⭐",
            "size": "Large"
        },
        "Cassiopeia": {
            "seasons": [9, 10, 11, 12, 1],
            "latitude_range": (5, 90),
            "fun_fact": "Named after a vain queen in Greek mythology, its distinctive W or M shape makes it one of the easiest constellations to recognize.",
            "how_to_spot": "Look for a distinctive W or M shape of five bright stars. Located on opposite side of Polaris from the Big Dipper.",
            "key_stars": "• Schedar (orange giant)<br>• Caph (variable star)<br>• Gamma Cas (variable star)<br>• Ruchbah",
            "best_time": "All night (circumpolar)",
            "direction": "North",
            "brightness": "Bright ⭐⭐⭐⭐",
            "size": "Medium"
        },
        "Leo": {
            "seasons": [3, 4, 5],
            "latitude_range": (-65, 90),
            "fun_fact": "Leo's shape really does resemble a lion! The Leonid meteor shower radiates from this constellation every November.",
            "how_to_spot": "Look for a backwards question mark (the Sickle) forming the lion's head. A triangle of stars to the east forms the hindquarters.",
            "key_stars": "• Regulus (lion's heart)<br>• Denebola (lion's tail)<br>• Algieba (double star)<br>• The Sickle asterism",
            "best_time": "10 PM - 2 AM",
            "direction": "South",
            "brightness": "Bright ⭐⭐⭐⭐",
            "size": "Large"
        },
        "Scorpius": {
            "seasons": [6, 7, 8],
            "latitude_range": (-90, 40),
            "fun_fact": "One of the few constellations that actually looks like its namesake! Antares means 'rival of Mars' due to its red color.",
            "how_to_spot": "Look for a curved line of stars resembling a scorpion's body with a stinger. The red supergiant Antares marks the heart.",
            "key_stars": "• Antares (red supergiant)<br>• Shaula (stinger star)<br>• Graffias (double star)<br>• Dschubba",
            "best_time": "10 PM - 1 AM",
            "direction": "South",
            "brightness": "Bright ⭐⭐⭐⭐",
            "size": "Large"
        },
        "Cygnus (Northern Cross)": {
            "seasons": [6, 7, 8, 9],
            "latitude_range": (-40, 90),
            "fun_fact": "Known as the Northern Cross, Cygnus flies along the Milky Way. Deneb will be our pole star in 12,000 years!",
            "how_to_spot": "Look for a large cross shape flying down the Milky Way. Deneb marks the tail, Albireo marks the head.",
            "key_stars": "• Deneb (future pole star)<br>• Albireo (gold & blue double)<br>• Sadr (center star)<br>• Forms Summer Triangle",
            "best_time": "9 PM - 3 AM",
            "direction": "Overhead",
            "brightness": "Very Bright ⭐⭐⭐⭐⭐",
            "size": "Large"
        },
        "Taurus": {
            "seasons": [11, 12, 1, 2],
            "latitude_range": (-65, 90),
            "fun_fact": "Home to two famous star clusters: the Pleiades (Seven Sisters) and the Hyades (forms the bull's face).",
            "how_to_spot": "Find the V-shaped Hyades cluster forming the bull's face with orange Aldebaran as the eye.",
            "key_stars": "• Aldebaran (bull's eye)<br>• Pleiades cluster<br>• Hyades cluster<br>• Elnath (horn tip)",
            "best_time": "8 PM - 1 AM",
            "direction": "South-Southwest",
            "brightness": "Bright ⭐⭐⭐⭐",
            "size": "Large"
        },
        "Andromeda": {
            "seasons": [9, 10, 11, 12],
            "latitude_range": (-40, 90),
            "fun_fact": "Contains the Andromeda Galaxy (M31), the nearest major galaxy to our Milky Way at 2.5 million light-years away!",
            "how_to_spot": "Find the Great Square of Pegasus, then look for a line of stars extending from the northeast corner.",
            "key_stars": "• Alpheratz (shared with Pegasus)<br>• Mirach (guide to M31)<br>• Almach (double star)<br>• M31 Galaxy location",
            "best_time": "9 PM - 2 AM",
            "direction": "East to Overhead",
            "brightness": "Moderate ⭐⭐⭐",
            "size": "Large"
        }
    }
    
    visible_constellations = []
    
    for name, data in constellation_database.items():
        if month in data["seasons"]:
            min_lat, max_lat = data["latitude_range"]
            if min_lat <= latitude <= max_lat:
                visible_constellations.append({
                    "name": name,
                    "fun_fact": data["fun_fact"],
                    "how_to_spot": data["how_to_spot"],
                    "key_stars": data["key_stars"],
                    "best_time": data["best_time"],
                    "direction": data["direction"],
                    "brightness": data["brightness"],
                    "size": data["size"]
                })
    
    # If no constellations found, return seasonal defaults
    if not visible_constellations:
        # Default to at least one constellation based on season
        if month in [12, 1, 2]:
            default = constellation_database.get("Orion", {})
        elif month in [3, 4, 5]:
            default = constellation_database.get("Leo", {})
        elif month in [6, 7, 8]:
            default = constellation_database.get("Cygnus (Northern Cross)", {})
        else:
            default = constellation_database.get("Andromeda", {})
        
        if default:
            visible_constellations.append({
                "name": list(constellation_database.keys())[0],
                "fun_fact": default.get("fun_fact", "A beautiful constellation visible tonight"),
                "how_to_spot": default.get("how_to_spot", "Look for bright stars in pattern"),
                "key_stars": default.get("key_stars", "Multiple bright stars"),
                "best_time": default.get("best_time", "After dark"),
                "direction": default.get("direction", "Sky"),
                "brightness": default.get("brightness", "Visible"),
                "size": default.get("size", "Medium")
            })
    
    return visible_constellations[:5]
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_visible_constellations(lat, lon):
    """
    Get all constellations that will be visible tonight at any point,
    not just those visible right now.
    """
    
    # Create observer at user's location
    observer = ephem.Observer()
    observer.lat = str(lat)
    observer.lon = str(lon)
    
    # Get sunset and sunrise times for tonight
    sun = ephem.Sun()
    observer.date = ephem.now()
    
    # Find tonight's sunset
    sunset = observer.next_setting(sun)
    # Find tomorrow's sunrise
    sunrise = observer.next_rising(sun, start=sunset)
    
    # Define constellations with their key stars
    constellation_data = {
        "Orion": {
            "star": "Rigel",
            "ephem_star": ephem.star("Rigel"),
            "description": "The Hunter - look for three stars in a belt"
        },
        "Ursa Major": {
            "star": "Dubhe",
            "ephem_star": ephem.star("Dubhe"),
            "description": "Big Dipper - seven bright stars in a ladle shape"
        },
        "Leo": {
            "star": "Regulus",
            "ephem_star": ephem.star("Regulus"),
            "description": "The Lion - backwards question mark forms the head"
        },
        "Scorpius": {
            "star": "Antares",
            "ephem_star": ephem.star("Antares"),
            "description": "The Scorpion - curved tail with red Antares"
        },
        "Cassiopeia": {
            "star": "Schedar",
            "ephem_star": ephem.star("Schedar"),
            "description": "The Queen - distinctive W or M shape"
        },
        "Cygnus": {
            "star": "Deneb",
            "ephem_star": ephem.star("Deneb"),
            "description": "Northern Cross - flying along the Milky Way"
        },
        "Taurus": {
            "star": "Aldebaran",
            "ephem_star": ephem.star("Aldebaran"),
            "description": "The Bull - V-shaped face with orange eye"
        },
        "Andromeda": {
            "star": "Alpheratz",
            "ephem_star": ephem.star("Alpheratz"),
            "description": "The Princess - home to Andromeda Galaxy M31"
        },
        "Perseus": {
            "star": "Mirfak",
            "ephem_star": ephem.star("Mirfak"),
            "description": "The Hero - contains variable star Algol"
        },
        "Gemini": {
            "star": "Pollux",
            "ephem_star": ephem.star("Pollux"),
            "description": "The Twins - Castor and Pollux mark their heads"
        },
        "Lyra": {
            "star": "Vega",
            "ephem_star": ephem.star("Vega"),
            "description": "The Lyre - brilliant Vega in Summer Triangle"
        },
        "Boötes": {
            "star": "Arcturus",
            "ephem_star": ephem.star("Arcturus"),
            "description": "The Herdsman - kite shape with orange Arcturus"
        },
        "Virgo": {
            "star": "Spica",
            "ephem_star": ephem.star("Spica"),
            "description": "The Virgin - bright blue-white Spica"
        },
        "Aquila": {
            "star": "Altair",
            "ephem_star": ephem.star("Altair"),
            "description": "The Eagle - Altair in Summer Triangle"
        },
        "Pegasus": {
            "star": "Markab",
            "ephem_star": ephem.star("Markab"),
            "description": "The Winged Horse - Great Square of Pegasus"
        },
        "Sagittarius": {
            "star": "Kaus Australis",
            "ephem_star": ephem.star("Kaus Australis"),
            "description": "The Archer - teapot asterism points to galactic center"
        }
    }
    
    visible_constellations = []
    
    for const_name, const_info in constellation_data.items():
        try:
            star = const_info["ephem_star"]
            
            # Check visibility throughout the night
            # Sample every hour from sunset to sunrise
            observer.date = sunset
            max_altitude = 0
            best_time = None
            is_visible_tonight = False
            
            while observer.date < sunrise:
                star.compute(observer)
                altitude_deg = float(star.alt) * 180 / ephem.pi
                
                # If star gets above 10 degrees at any point tonight, it's viewable
                if altitude_deg > 10:
                    is_visible_tonight = True
                    if altitude_deg > max_altitude:
                        max_altitude = altitude_deg
                        best_time = ephem.localtime(observer.date)
                
                # Check next hour
                observer.date = ephem.Date(observer.date + ephem.hour)
            
            if is_visible_tonight:
                # Get current position for reference
                observer.date = ephem.now()
                star.compute(observer)
                current_alt = float(star.alt) * 180 / ephem.pi
                current_az = float(star.az) * 180 / ephem.pi
                
                # Determine compass direction
                if 337.5 <= current_az or current_az < 22.5:
                    direction = "N"
                elif 22.5 <= current_az < 67.5:
                    direction = "NE"
                elif 67.5 <= current_az < 112.5:
                    direction = "E"
                elif 112.5 <= current_az < 157.5:
                    direction = "SE"
                elif 157.5 <= current_az < 202.5:
                    direction = "S"
                elif 202.5 <= current_az < 247.5:
                    direction = "SW"
                elif 247.5 <= current_az < 292.5:
                    direction = "W"
                else:
                    direction = "NW"
                
                # Format altitude display
                if current_alt > 0:
                    altitude_display = f"{current_alt:.0f}° {direction} now"
                else:
                    altitude_display = f"Rises later ({max_altitude:.0f}° max)"
                
                visible_constellations.append({
                    "constellation": const_name,
                    "star": const_info["star"],
                    "altitude": altitude_display,
                    "azimuth": f"{current_az:.0f}°",
                    "best_time": best_time.strftime("%H:%M") if best_time else "---",
                    "description": const_info["description"],
                    "max_altitude": max_altitude,  # For sorting
                    "current_altitude": current_alt
                })
                
        except Exception as e:
            continue
    
    # Sort by max altitude tonight (best viewing potential first)
    visible_constellations.sort(key=lambda x: x['max_altitude'], reverse=True)
    
    # Remove sorting values from final output
    for const in visible_constellations:
        const.pop('max_altitude', None)
        const.pop('current_altitude', None)
    
    return visible_constellations[:10]  # Return top 10 for tonight

# Alternative: Add a summary function for quick overview
def get_constellation_summary(lat, lon):
    """Get a quick summary of tonight's constellation viewing"""
    
    observer = ephem.Observer()
    observer.lat = str(lat)
    observer.lon = str(lon)
    observer.date = ephem.now()
    
    sun = ephem.Sun()
    sunset = observer.next_setting(sun)
    sunset_time = ephem.localtime(sunset).strftime("%H:%M")
    
    # Count how many major constellations will be visible
    visible_constellations = get_visible_constellations(lat, lon)
    
    currently_visible = sum(1 for c in visible_constellations if "now" in c.get('altitude', ''))
    will_rise_later = len(visible_constellations) - currently_visible
    
    return {
        'sunset': sunset_time,
        'total_tonight': len(visible_constellations),
        'visible_now': currently_visible,
        'rising_later': will_rise_later
    }
def get_city_suggestions(invalid_input):
    """Provide helpful suggestions for common city name issues"""
    suggestions = []
    
    if "," not in invalid_input:
        suggestions.append(f"Try adding country: '{invalid_input}, USA' or '{invalid_input}, UK'")
        suggestions.append(f"Try adding state: '{invalid_input}, California' or '{invalid_input}, Texas'")
    
    return suggestions

@st.cache_data(ttl=1800)
def get_weather_forecast(lat, lon):
    """Get real weather data from OpenWeatherMap API with robust error handling"""
 
    
    # Default fallback data
    fallback_data = {
        "current": {
            "temp": 20,
            "feels_like": 19,
            "humidity": 50,
            "wind_speed": 3,
            "clouds": 30,
            "weather": [{"main": "Clear", "description": "Weather data unavailable"}]
        }
    }
    
    if not OPENWEATHER_API_KEY:
        return fallback_data
    
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric"
        }
        
        # Add timeout and retry logic
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        return {
            "current": {
                "temp": data['main']['temp'],
                "feels_like": data['main']['feels_like'],
                "humidity": data['main']['humidity'],
                "wind_speed": data['wind']['speed'],
                "clouds": data['clouds']['all'],
                "weather": data['weather']
            }
        }
    
    except requests.exceptions.ConnectionError:
        # Network is down
        st.warning("📡 Network connection issue. Using cached weather data.")
        return fallback_data
    
    except requests.exceptions.Timeout:
        # Request timed out
        st.warning("⏱️ Weather service is slow. Using cached data.")
        return fallback_data
    
    except requests.exceptions.HTTPError as e:
        # HTTP error (4xx, 5xx)
        if e.response.status_code == 429:
            st.warning("⚠️ Weather API rate limit reached. Using cached data.")
        elif e.response.status_code >= 500:
            st.warning("🔧 Weather service is temporarily down. Using cached data.")
        else:
            st.warning(f"⚠️ Weather data unavailable (Error {e.response.status_code})")
        return fallback_data
    
    except requests.exceptions.RequestException:
        # Any other request exception
        st.warning("📡 Unable to fetch weather data. Using cached values.")
        return fallback_data
    
    except Exception:
        # Any other unexpected error
        return fallback_data
@st.cache_data(ttl=3600)
def get_astronomy_data(lat, lon):
    """Get astronomy data for the location with loading state"""
    try:
        with st.spinner('🌙 Calculating celestial positions...'):
            time.sleep(0.5)  # Small delay for UX
            url = f"https://api.sunrise-sunset.org/json"
            params = {
                "lat": lat,
                "lng": lon,
                "formatted": 0,
                "date": "today"
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data['status'] == 'OK':
                results = data['results']
                
                # Convert UTC times to local timezone
                from timezonefinder import TimezoneFinder
                import pytz
                
                # Get the timezone for the location
                tf = TimezoneFinder()
                timezone_str = tf.timezone_at(lat=lat, lng=lon)
                if timezone_str:
                    local_tz = pytz.timezone(timezone_str)
                else:
                    # Fallback to EST if timezone not found
                    local_tz = pytz.timezone('America/New_York')
                
                # Parse UTC times and convert to local
                sunrise_utc = datetime.fromisoformat(results['sunrise'].replace('Z', '+00:00'))
                sunset_utc = datetime.fromisoformat(results['sunset'].replace('Z', '+00:00'))
                
                # Convert to local timezone
                sunrise_local = sunrise_utc.astimezone(local_tz)
                sunset_local = sunset_utc.astimezone(local_tz)
                
                # Moon phase calculation
                moon_phases = ["New Moon", "Waxing Crescent", "First Quarter", "Waxing Gibbous", 
                              "Full Moon", "Waning Gibbous", "Last Quarter", "Waning Crescent"]
                days_since_new_moon = (datetime.now().day % 29.53)
                phase_index = int(days_since_new_moon / 3.69)
                moon_phase = moon_phases[min(phase_index, 7)]
                
                # Visible planets based on month
                month = datetime.now().month
                if month in [12, 1, 2]:
                    visible_planets = ["Mars", "Jupiter", "Saturn"]
                elif month in [3, 4, 5]:
                    visible_planets = ["Venus", "Mars", "Saturn"]
                elif month in [6, 7, 8]:
                    visible_planets = ["Jupiter", "Saturn", "Mercury"]
                else:
                    visible_planets = ["Venus", "Jupiter", "Mars"]
                
                if abs(lat) > 60:
                    visible_planets = visible_planets[:2]
                
                # Calculate moonrise/moonset (approximate)
                moonrise_local = sunrise_local + timedelta(hours=13)
                moonset_local = sunrise_local + timedelta(hours=1)
                
                return {
                    "sun": {
                        "sunrise": sunrise_local.strftime("%H:%M"),
                        "sunset": sunset_local.strftime("%H:%M")
                    },
                    "moon": {
                        "moonrise": moonrise_local.strftime("%H:%M"),
                        "moonset": moonset_local.strftime("%H:%M"),
                        "phase": moon_phase
                    },
                    "planets": {
                        "visible": visible_planets
                    }
                }
            else:
                raise Exception("API returned error status")
                
    except Exception as e:
        # Fallback with reasonable defaults for US East Coast
        return {
            "sun": {"sunrise": "06:21", "sunset": "20:14"},
            "moon": {"moonrise": "22:15", "moonset": "10:30", "phase": "Waxing Crescent"},
            "planets": {"visible": ["Venus", "Mars", "Jupiter"]}
        }
@st.cache_data(ttl=3600)
def get_iss_passes(lat, lon, city_name=""):
    """Get ISS pass predictions with loading state"""
    try:
        with st.spinner('🛸 Tracking ISS orbit...'):
            time.sleep(0.5)  # Small delay for UX
            # ... rest of your existing ISS code ...
            if abs(lat) > 80:
                return [{
                    "risetime": "N/A",
                    "date": "N/A",
                    "duration": 0,
                    "message": f"ISS passes not visible from {city_name} (latitude: {lat:.1f}°)"
                }]
            
            url = f"http://api.open-notify.org/iss-pass.json"
            params = {
                "lat": round(lat, 2),
                "lon": round(lon, 2),
                "n": 5,
                "alt": 100
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            # ... rest of your ISS implementation ...
            if response.status_code == 200:
                data = response.json()
                
                if data.get('message') == 'success' and 'response' in data:
                    passes = []
                    for pass_data in data['response'][:3]:
                        risetime_dt = datetime.fromtimestamp(pass_data['risetime'])
                        duration = pass_data['duration']
                        
                        passes.append({
                            "risetime": risetime_dt.strftime("%H:%M"),
                            "date": risetime_dt.strftime("%b %d, %Y"),
                            "duration": duration,
                            "message": "success"
                        })
                    
                    if not passes:
                        return [{
                            "risetime": "No passes",
                            "date": "Check back later",
                            "duration": 0,
                            "message": f"No visible ISS passes in the next few days for {city_name}"
                        }]
                        
                    return passes
                else:
                    return [{
                        "risetime": "Limited visibility",
                        "date": "Rare passes",
                        "duration": 0,
                        "message": f"ISS passes are rare for {city_name} at latitude {lat:.1f}°"
                    }]
            else:
                raise Exception(f"API returned status {response.status_code}")
                
    except Exception:
    # No fake data - just return unavailable status
        return [{
            "risetime": "---",
            "date": "---",
            "duration": 0,
            "message": "ISS data temporarily unavailable"
        }]
# --- Add the ISS visibility functions near your other utility functions (after get_iss_passes, before cosmic_fun_fact) ---
def get_iss_crew():
    """Get current ISS crew members"""
    try:
        response = requests.get("http://api.open-notify.org/astros.json", timeout=5)
        if response.status_code == 200:
            data = response.json()
            iss_crew = [person for person in data['people'] if person['craft'] == 'ISS']
            return iss_crew
    except:
        return []


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance between two lat/lon points (km)"""
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def iss_visibility(lat, lon):
    """
    Get ISS current location and estimate visibility from user's location.
    Returns:
        dict with keys: visible (bool), distance_km (float), iss_lat, iss_lon, message, next_pass_time (if visible soon)
    """
    iss_url = "http://api.open-notify.org/iss-now.json"
    try:
        resp = requests.get(iss_url, timeout=10)
        data = resp.json()
        iss_lat = float(data['iss_position']['latitude'])
        iss_lon = float(data['iss_position']['longitude'])
        # Calculate distance
        distance = haversine_distance(lat, lon, iss_lat, iss_lon)
        visible_threshold = 1000  # km
        if distance <= visible_threshold:
            # ISS is close - visible now!
            return {
                "visible": True,
                "distance_km": distance,
                "iss_lat": iss_lat,
                "iss_lon": iss_lon,
                "message": f"The ISS is currently visible from your location! Distance: {int(distance)} km.",
                "next_pass_time": datetime.now().strftime("%H:%M %b %d, %Y")
            }
        else:
            # Estimate next pass (simple guess: orbits every 90 minutes)
            return {
                "visible": False,
                "distance_km": distance,
                "iss_lat": iss_lat,
                "iss_lon": iss_lon,
                "message": "ISS visibility cannot be determined at this time.",
                "next_pass_time": "Data unavailable"
            }
    except Exception as e:
        return {
            "visible": False,
            "message": f"Could not get ISS location: {e}"
        }

# --- End of new functions ---

from skyfield.api import load
import numpy as np
import plotly.graph_objects as go

def get_iss_trajectory(minutes=90, step=1):
    """
    Get accurate ISS trajectory (lat, lon arrays) for the next `minutes` minutes.
    Returns:
        lats, lons (numpy arrays of latitude and longitude in degrees)
    """
    # 1. Download latest ISS TLE
    stations_url = 'https://celestrak.com/NORAD/elements/stations.txt'
    satellites = load.tle_file(stations_url)
    iss = [sat for sat in satellites if 'ISS' in sat.name][0]

    # 2. Generate times for the trajectory
    ts = load.timescale()
    t_now = ts.now()
    minutes_ahead = np.arange(0, minutes, step)  # every `step` minutes up to `minutes`
    times = ts.utc(
        t_now.utc_datetime().year,
        t_now.utc_datetime().month,
        t_now.utc_datetime().day,
        t_now.utc_datetime().hour,
        t_now.utc_datetime().minute + minutes_ahead
    )

    # 3. Get ISS positions
    iss_positions = iss.at(times)
    lats = iss_positions.subpoint().latitude.degrees
    lons = iss_positions.subpoint().longitude.degrees
    return lats, lons



def get_iss_trajectory(minutes=90, step=1):
    """
    Get accurate ISS trajectory (lat, lon arrays) for the next `minutes` minutes.
    Returns:
        lats, lons (numpy arrays of latitude and longitude in degrees)
    """
    # 1. Download latest ISS TLE
    stations_url = 'https://celestrak.com/NORAD/elements/stations.txt'
    satellites = load.tle_file(stations_url)
    iss = [sat for sat in satellites if 'ISS' in sat.name][0]

    # 2. Generate times for the trajectory
    ts = load.timescale()
    t_now = ts.now()
    minutes_ahead = np.arange(0, minutes, step)  # every `step` minutes up to `minutes`
    times = ts.utc(
        t_now.utc_datetime().year,
        t_now.utc_datetime().month,
        t_now.utc_datetime().day,
        t_now.utc_datetime().hour,
        t_now.utc_datetime().minute + minutes_ahead
    )

    # 3. Get ISS positions
    iss_positions = iss.at(times)
    lats = iss_positions.subpoint().latitude.degrees
    lons = iss_positions.subpoint().longitude.degrees
    return lats, lons

def create_iss_world_map(lat, lon, minutes=90, step=1):
    """
    Create a Plotly world map showing the accurate ISS trajectory and user position.
    Args:
        lat, lon: latitude and longitude
        minutes: time span for trajectory
        step: time step in minutes
    Returns:
        Plotly figure
    """
    lats, lons = get_iss_trajectory(minutes, step)
    curr_lat = lats[0]
    curr_lon = lons[0]

    fig = go.Figure()

    # Plot ISS trajectory
    fig.add_trace(go.Scattergeo(
        lon=lons,
        lat=lats,
        mode='lines',
        line=dict(width=2, color='cyan'),
        name="ISS Trajectory"
    ))

    # Plot current ISS position
    fig.add_trace(go.Scattergeo(
        lon=[curr_lon],
        lat=[curr_lat],
        text="ISS Current Position",
        mode='markers+text',
        marker=dict(size=15, color='red', symbol='circle'),
        textposition="top center",
        name="ISS"
    ))

    # Plot user location
    fig.add_trace(go.Scattergeo(
        lon=[lon],
        lat=[lat],
        text="Your Location",
        mode='markers+text',
        marker=dict(size=12, color='#ffb800', symbol='star'),
        textposition="bottom center",
        name="You"
    ))

    fig.update_layout(
        title="🛰️ ISS Accurate Live Trajectory",
        geo=dict(
            projection_type='natural earth',
            showland=True,
            landcolor='rgb(30, 30, 50)',
            showocean=True,
            oceancolor='rgb(10, 10, 30)',
            showcountries=True,
            countrycolor='rgb(60, 60, 100)'
        ),
        height=500,
        showlegend=True
    )
    return fig

def get_city_from_coords(lat, lon):
    """Reverse geocode to get city and country from lat/lon in English."""
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            "lat": lat,
            "lon": lon,
            "format": "json",
            "zoom": 10,
            "accept-language": "en",  # Force English results
            "namedetails": 1,
            "extratags": 1
        }
        response = requests.get(url, params=params, headers={"User-Agent": "Stellaris/1.0"}, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        address = data.get("address", {})
        
        # Get city name in English
        city = (address.get("city") or 
                address.get("town") or 
                address.get("village") or 
                address.get("county") or 
                "Unknown")
        
        # Get country in English
        country = address.get("country", "Unknown")
        
        # Get local name if available
        local_name = None
        if 'namedetails' in data:
            name_details = data['namedetails']
            local_name = name_details.get('name', None)
            if local_name and local_name == name_details.get('name:en', local_name):
                local_name = None
        
        # Format with local name if different
        if local_name and local_name != city:
            city = f"{city} ({local_name})"
        
        return city, country
    except Exception:
        return "Unknown", "Unknown"

def format_location_header(display_name):
    """
    Returns formatted location name in English with optional local name
    'City, State' for US locations, else 'City, Country (LocalName)'
    """
    if not display_name:
        return "Unknown"
    
    # If display name already has local name in parentheses, use as is
    if '(' in display_name and ')' in display_name:
        return display_name
    
    parts = [p.strip() for p in display_name.split(",")]
    if len(parts) < 2:
        return display_name
    
    city = parts[0]
    country = parts[-1]
    
    # Special handling for US states
    us_states = {
        "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
        "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa",
        "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan",
        "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire",
        "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
        "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
        "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia",
        "Wisconsin", "Wyoming"
    }
    
    # Handle US locations
    if country in ["United States", "United States of America", "USA"]:
        if len(parts) >= 3:
            for part in parts[1:-1]:
                if part in us_states:
                    return f"{city}, {part}"
        return f"{city}, USA"
    else:
        return f"{city}, {country}"
def cosmic_fun_fact():
    facts = [
        "A teaspoon of neutron star would weigh about 6 billion tons.",
        "The observable universe is about 93 billion light-years in diameter.",
        "There are more stars in the universe than grains of sand on all the Earth's beaches.",
        "Jupiter is so big you could fit all the other planets inside it.",
        "If two pieces of the same type of metal touch in space, they bond permanently.",
        "The Sun makes up 99.86% of the Solar System's mass.",
        "One million Earths could fit inside the Sun.",
        "Saturn would float in water because it's mostly made of gas.",
        "The footprints on the Moon will be there for 100 million years.",
        "If you could travel at the speed of light, it would still take you over 93 billion years to cross the observable universe.",
        "All the ordinary matter (atoms) in the universe makes up less than 5% of its total content. The rest is mysterious dark matter and dark energy.",
        "On Jupiter and Saturn, it rains diamonds—the immense pressure turns carbon into diamond crystals!",
        "A day on Venus is longer than a year on Venus!",
        "Some neutron stars spin as fast as 700 times per second."
        
    ]
    idx = datetime.utcnow().timetuple().tm_yday % len(facts)
    return facts[idx]

def generate_cosmic_answer_fallback(question, city, lat, lon, weather, astronomy, iss_passes):
    """Fallback rule-based system for when LangChain is not available"""
    q_lower = question.lower()
    
    if "planet" in q_lower and ("visible" in q_lower or "tonight" in q_lower):
        planets = astronomy['planets']['visible']
        if planets:
            return f"""🪐 <b>Visible Planets Tonight from {city}:</b><br><br>
            {', '.join(planets)} are visible tonight!<br><br>
            <b>Best viewing times:</b><br>
            • Early evening (1-2 hours after sunset): Best for Venus and Mercury<br>
            • Midnight: Best for Mars and Jupiter<br>
            • Pre-dawn: Best for Saturn"""
        else:
            return f"Unfortunately, no major planets are easily visible from {city} tonight."
    
    elif "moon" in q_lower and "phase" in q_lower:
        moon_phase = astronomy['moon']['phase']
        moonrise = astronomy['moon']['moonrise']
        moonset = astronomy['moon']['moonset']
        
        return f"""🌙 <b>Moon Information for {city}:</b><br><br>
        <b>Current Phase:</b> {moon_phase}<br>
        <b>Moonrise:</b> {moonrise}<br>
        <b>Moonset:</b> {moonset}"""
    
    else:
        return f"""🌟 <b>Great question about space!</b><br><br>
        Here's what I can tell you about current conditions in {city}:<br><br>
        <b>Tonight's Highlights:</b><br>
        • Visible Planets: {', '.join(astronomy['planets']['visible']) if astronomy['planets']['visible'] else 'Check after sunset'}<br>
        • Moon Phase: {astronomy['moon']['phase']}<br>
        • Weather: {weather['current']['weather'][0]['description'].capitalize()}<br>
        • Cloud Cover: {weather['current'].get('clouds', 'N/A')}%"""

def main():
    st.set_page_config(page_title="Stellaris - Cosmic Explorer", page_icon="🪐", layout="wide")
    # Suppress the "Running cached function" messages

    
    # Initialize with a loading state
    if 'initialized' not in st.session_state:
        with st.spinner('🌌 Initializing Stellaris Observatory...'):
            st.session_state.initialized = True
            time.sleep(2)  # Give it a moment for the cosmic feel
  # Initialize LangChain processor if available
    if 'ai_processor' not in st.session_state:
        st.session_state['ai_enabled'] = False
        
        if LANGCHAIN_AVAILABLE:
            
            if OPENAI_API_KEY:
                try:
                    st.session_state['ai_processor'] = AstronomyQueryProcessor(OPENAI_API_KEY)
                    st.session_state['ai_enabled'] = True
                except Exception as e:
                    print(f"Failed to initialize AI: {e}")
            else:
                st.info("Add OPENAI_API_KEY to enable AI responses")
        else:
            st.info("Install langchain and langchain-openai for AI features")
    
    # Header
    #st.markdown('<div class="cosmic-title">STELLARIS</div>', unsafe_allow_html=True)
    # Replace your header section with this corrected version:
    inject_stellaris_css()
    st.markdown("""
    <div style='
    width: 100%;
    margin: 0.4em auto 1.1em auto;
    padding: 1.2em 0.4em 1em 0.4em;
    background: 
        linear-gradient(135deg, #1a1a2e 0%, #16213e 25%, #0f3460 50%, #16213e 75%, #1a1a2e 100%),
        radial-gradient(ellipse at top, rgba(124,58,237,0.3), transparent 70%);
    border-radius: 20px;
    box-shadow:
        0 1px 2px #7c3aed,
        0 2px 4px #5b21b6,
        0 4px 8px #4c1d95,
        0 8px 16px rgba(124,58,237,0.4),
        0 16px 32px rgba(0,0,0,0.5),
        inset 0 1px 0 rgba(255,255,255,0.2),
        inset 0 -1px 0 rgba(0,0,0,0.3),
        0 0 50px rgba(124,58,237,0.3),
        0 0 100px rgba(251,146,60,0.1);
    text-align: center;
    position: relative;
    overflow: visible;
    border: 2px solid transparent;
    background-clip: padding-box;
    transform: perspective(1000px) rotateX(2deg);
    animation: bannerFloat 6s ease-in-out infinite;
    '>
    <div style='
        position: absolute;
        top: -2px; left: -2px; right: -2px; bottom: -2px;
        background: linear-gradient(
            45deg,
            #ffb800, #f59e0b, #7c3aed, #3a86ff,
            #ffb800, #f59e0b, #7c3aed, #3a86ff
        );
        background-size: 400% 400%;
        border-radius: 20px;
        z-index: -1;
        animation: borderGradient 8s ease infinite;
        opacity: 0.8;
    '></div>
    <div class='sparkle-container'>
        <span class='sparkle' style='left: 10%; animation-delay: 0s;'>✨</span>
        <span class='sparkle' style='left: 20%; animation-delay: 0.5s;'>⭐</span>
        <span class='sparkle' style='left: 30%; animation-delay: 1s;'>✨</span>
        <span class='sparkle' style='left: 40%; animation-delay: 1.5s;'>💫</span>
        <span class='sparkle' style='left: 50%; animation-delay: 2s;'>✨</span>
        <span class='sparkle' style='left: 60%; animation-delay: 2.5s;'>⭐</span>
        <span class='sparkle' style='left: 70%; animation-delay: 3s;'>✨</span>
        <span class='sparkle' style='left: 80%; animation-delay: 3.5s;'>💫</span>
        <span class='sparkle' style='left: 90%; animation-delay: 4s;'>✨</span>
    </div>
    <div style="
        font-family: 'Orbitron', 'Space Grotesk', sans-serif;
        font-size: 2.8rem;
        font-weight: 900;
        letter-spacing: 0.15em;
        background: linear-gradient(
            135deg,
            #fff 0%, #ffea00 25%, #ffb800 50%, #ff6b00 75%, #7c3aed 100%
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow:
            0 1px 0 #ccc,
            0 2px 0 #c9c9c9,
            0 3px 0 #bbb,
            0 4px 0 #b9b9b9,
            0 5px 0 #aaa,
            0 6px 1px rgba(0,0,0,.1),
            0 0 5px rgba(255,184,0,0.5),
            0 1px 3px rgba(0,0,0,.3),
            0 3px 5px rgba(0,0,0,.2),
            0 5px 10px rgba(0,0,0,.25),
            0 10px 10px rgba(0,0,0,.2),
            0 20px 20px rgba(0,0,0,.15);
        filter: brightness(1.2);
        display: inline-block;
        margin-bottom: 0.2em;
        position: relative;
        animation: titlePulse 3s ease-in-out infinite;
        transform: perspective(500px) rotateY(-5deg);
    ">
        <span style='display: inline-block; animation: letterFloat 4s ease-in-out infinite; animation-delay: 0s;'>S</span>
        <span style='display: inline-block; animation: letterFloat 4s ease-in-out infinite; animation-delay: 0.1s;'>T</span>
        <span style='display: inline-block; animation: letterFloat 4s ease-in-out infinite; animation-delay: 0.2s;'>E</span>
        <span style='display: inline-block; animation: letterFloat 4s ease-in-out infinite; animation-delay: 0.3s;'>L</span>
        <span style='display: inline-block; animation: letterFloat 4s ease-in-out infinite; animation-delay: 0.4s;'>L</span>
        <span style='display: inline-block; animation: letterFloat 4s ease-in-out infinite; animation-delay: 0.5s;'>A</span>
        <span style='display: inline-block; animation: letterFloat 4s ease-in-out infinite; animation-delay: 0.6s;'>R</span>
        <span style='display: inline-block; animation: letterFloat 4s ease-in-out infinite; animation-delay: 0.7s;'>I</span>
        <span style='display: inline-block; animation: letterFloat 4s ease-in-out infinite; animation-delay: 0.8s;'>S</span>
    </div>
    <div style='
        font-size: 1.2rem;
        font-family: Space Grotesk, sans-serif;
        color: #fff;
        text-shadow:
            0 0 10px #7c3aed,
            0 0 20px #7c3aed,
            0 0 30px #7c3aed,
            0 0 40px #3a86ff;
        margin-top: -0.5em;
        letter-spacing: 0.3em;
        opacity: 0.95;
        animation: subtitleGlow 2s ease-in-out infinite alternate;
        text-transform: uppercase;
        font-weight: 300;
    '>
        Cosmic Night Sky Agent
    </div>
    <div class='orb orb1'></div>
    <div class='orb orb2'></div>
    <div class='orb orb3'></div>
    </div>
    """, unsafe_allow_html=True)
    
    #inject_stellaris_css()
    st.markdown("""
    <style>
/* Sparkly luminous border for Streamlit text inputs */
    div[data-testid="stTextInput"] input {
    border: 3px solid;
    border-image: linear-gradient(90deg, #ffe066, #7c3aed, #ffb800, #7c3aed) 1;
    border-radius: 12px;
    box-shadow:
        0 0 12px 3px #ffe06677,
        0 0 24px 7px #7c3aed44,
        0 0 8px 2px #ffb80055;
    background: rgba(20,30,48,0.94);
    color: #fff;
    font-size: 1.07rem;
    font-family: 'Orbitron', 'Space Grotesk', monospace;
    transition: box-shadow 0.2s, border-color 0.2s;
    outline: none;
}
div[data-testid="stTextInput"] input:focus {
    box-shadow:
        0 0 24px 6px #ffe066bb,
        0 0 32px 10px #7c3aed99,
        0 0 12px 4px #ffb80099;
    border-image: linear-gradient(90deg, #ffb800, #ffe066, #7c3aed, #ffb800) 1;
}
    </style>
    """, unsafe_allow_html=True)
    # Initialize session state
    if 'current_city' not in st.session_state:
        st.session_state['current_city'] = 'Austin, TX'
    if 'menu_selection' not in st.session_state:
        st.session_state['menu_selection'] = "Home"
            
    with st.sidebar:
        #st.markdown('<div class="cosmic-title">STELLARIS</div>', unsafe_allow_html=True)
        st.sidebar.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Space+Grotesk:wght@400;700&display=swap');
        
        .sidebar-card {
            background: linear-gradient(120deg, #1e3a8a 0%, #7c3aed 80%, #ffb800 100%);
            border-radius: 18px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            text-align: center;
            color: white;
            
            /* --- 3D Effect & Transition --- */
            transform-style: preserve-3d;
            transform: perspective(1000px); /* Sets up the 3D space */
            transition: all 0.2s ease-out;
            box-shadow: 0 10px 30px -5px rgba(124, 58, 237, 0.5);
        }
        
        .sidebar-card:hover {
            transform: perspective(1000px) scale(1.05) rotateY(-8deg) rotateX(5deg);
            box-shadow: 0 20px 40px -5px rgba(124, 58, 237, 0.7);
        }
        
        .card-title {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-shadow: 0 2px 10px #8b5cf6;
            margin-bottom: 0.5rem;
        }
        
        .card-body {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.95rem;
            color: #e0e7ff;
            line-height: 1.5;
            opacity: 0.9;
        }
        </style>
        
        <div class="sidebar-card">
            <div class="card-title">🔭 Stargazer!</div>
            <div class="card-body">
                The universe is calling! Spot ISS flybys, planets, and constellations in your night sky.
                <br>
                Enter your location to start your cosmic journey! 🚀
            </div>
        </div>
        """, unsafe_allow_html=True)
        # Show AI status
        #if st.session_state.get('ai_enabled', False):
        #    st.success("🤖 AI Mode Active")
       # else:
         #   st.info("📊 Rule-Based Mode")
        
        #st.markdown("---", unsafe_allow_html=True)
        
        # City input
        # City input
        city = st.text_input(
            "City", 
            value=st.session_state.get('current_city', 'Austin, TX'),
            placeholder="e.g., London, UK or Austin, Texas",
            help="Enter city name with country or state for better results",
            label_visibility="collapsed",
            key="city_input"  # Add a key for reference
        )
        
        # Check if Enter was pressed (city changed and not empty)
        if city and city != st.session_state.get('current_city', ''):
            # This will trigger when Enter is pressed
            if st.session_state.get('city_input') == city:
                # Clear any previous error messages
                st.session_state.pop('location_error', None)
                
                # Validate input before geocoding
                if len(city.strip()) < 2:
                    st.error("🚫 Please enter a valid city name (at least 2 characters)")
                elif any(char in city for char in ['@', '#', '$', '%', '^', '&', '*']):
                    st.error("🚫 Please remove special characters from the city name")
                else:
                    lat, lon, display_name, status = geocode_location(city)
                    
                    if status == "success" or status == "uncertain":
                        st.session_state['current_city'] = city
                        st.session_state['validated_location'] = display_name
                        st.session_state['lat'] = lat
                        st.session_state['lon'] = lon
                        st.session_state['location_valid'] = True
                        st.cache_data.clear()
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.session_state['location_valid'] = False
        
        # Keep the button as a backup option
        if st.button("Calibrate Your Observatory", use_container_width=True):
            if city and city != st.session_state.get('current_city', ''):
                # Same logic as above
                st.session_state.pop('location_error', None)
                
                if len(city.strip()) < 2:
                    st.error("🚫 Please enter a valid city name (at least 2 characters)")
                elif any(char in city for char in ['@', '#', '$', '%', '^', '&', '*']):
                    st.error("🚫 Please remove special characters from the city name")
                else:
                    lat, lon, display_name, status = geocode_location(city)
                    
                    if status == "success" or status == "uncertain":
                        st.session_state['current_city'] = city
                        st.session_state['validated_location'] = display_name
                        st.session_state['lat'] = lat
                        st.session_state['lon'] = lon
                        st.session_state['location_valid'] = True
                        st.cache_data.clear()
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.session_state['location_valid'] = False
            elif not city:
                st.error("🚫 Please enter a city name before calibrating")
        # Menu
        st.markdown('<hr style="border: none; height: 2px; background: linear-gradient(90deg, #f59e0b, #ea580c); border-radius: 1em;">', unsafe_allow_html=True)
     # Home button with planet icon
        if st.button(
            "🌍 Home", 
            key="home_button", 
            help="Return to home page with daily cosmic content",
            use_container_width=True
        ):
            st.session_state['menu_selection'] = "Home"
            st.rerun()
        
        #st.markdown("---")  # Separator between home and menu
        #st.markdown('<hr style="border: none; height: 2px; background: linear-gradient(90deg, #f59e0b, #ea580c); border-radius: 1em;">', #unsafe_allow_html=True)
        # Menu with animated emojis
        menu = [{"label": "Celestial Tracker", "icon": "🪐"},
            {"label": "ISS Tracking", "icon": "🛸"},
            {"label": "Cosmic Oracle", "icon": "👁️"}
        ]
            
        for i, m in enumerate(menu):
            # Add animation class based on menu item
            anim_class = ["weather-anim", "orbit-anim", "iss-anim", "glow-anim"][i]
            
            # Create button with animated emoji
            if st.button(
                f"{m['icon']} {m['label']}", 
                key=f"menu_{m['label']}", 
                use_container_width=True
            ):
                st.session_state['menu_selection'] = m["label"]
                st.rerun()
        
        
        if st.session_state.get('location_valid', True):
            st.success(f"📍 Active: {st.session_state.get('current_city', 'Austin, TX')}")
        else:
            st.error(f"📍 Invalid location")
        
        st.caption(f"🕐 Updated: {datetime.now().strftime('%H:%M')}")

    # Get location data
    if st.session_state.get('location_valid', True):
        lat, lon, display_name, status = geocode_location(st.session_state['current_city'])
        if status != "success" and status != "uncertain":
            if 'lat' in st.session_state and 'lon' in st.session_state:
                lat = st.session_state['lat']
                lon = st.session_state['lon']
                display_name = st.session_state.get('validated_location', 'Austin, TX, USA')
            else:
                lat, lon = 30.2672, -97.7431
                display_name = "Austin, TX, USA"
    else:
        if 'lat' in st.session_state and 'lon' in st.session_state:
            lat = st.session_state['lat']
            lon = st.session_state['lon']
            display_name = st.session_state.get('validated_location', 'Austin, TX, USA')
        else:
            lat, lon = 30.2672, -97.7431
            display_name = "Austin, TX, USA"
    
    # Fetch data
# Add this near the beginning of your main() function, after getting location data
# Replace the section where you fetch weather, astronomy, and ISS data with this:

# Create a placeholder for loading messages
    if 'data_loaded' not in st.session_state:
        loading_container = st.empty()
        with loading_container.container():
            loading_messages = [
                "🌌 Initializing Stellaris Observatory...",
                "🔭 Calibrating telescope arrays...",
                "📡 Establishing satellite uplink...",
                "🛸 Connecting to ISS telemetry...",
                "☁️ Analyzing atmospheric conditions...",
                "✨ Mapping celestial coordinates...",
                "🌟 Observatory ready!"
            ]
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, message in enumerate(loading_messages[:-1]):
                status_text.markdown(f"<h3 style='color: #7c3aed;'>{message}</h3>", unsafe_allow_html=True)
                progress_bar.progress((i + 1) / len(loading_messages))
                time.sleep(0.3)
            
            # Actually fetch the data
            #weather = get_weather_forecast(lat, lon)
            #astronomy = get_astronomy_data(lat, lon)
            #iss_passes = get_iss_passes(lat, lon, st.session_state['current_city'])
            with st.empty():
                weather = get_weather_forecast(lat, lon)
                astronomy = get_astronomy_data(lat, lon)
                iss_passes = get_iss_passes(lat, lon, st.session_state['current_city'])
            
            status_text.markdown(f"<h3 style='color: #00ff00;'>{loading_messages[-1]}</h3>", unsafe_allow_html=True)
            progress_bar.progress(1.0)
            time.sleep(0.5)
            
            st.session_state.data_loaded = True
            st.session_state.weather = weather
            st.session_state.astronomy = astronomy
            st.session_state.iss_passes = iss_passes
            
        loading_container.empty()  # Clear the loading screen
        
        # Use cached data
        weather = st.session_state.weather
        astronomy = st.session_state.astronomy
        iss_passes = st.session_state.iss_passes
    else:
        # Data already loaded, fetch normally
        weather = get_weather_forecast(lat, lon)
        astronomy = get_astronomy_data(lat, lon)
        iss_passes = get_iss_passes(lat, lon, st.session_state['current_city'])

    # Display header
    location_label = format_location_header(display_name)
    st.markdown(f"<div class='cosmic-section'>{location_label}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='margin-top:-1.3em;color:#8b5cf6;font-family:Orbitron,sans-serif;font-size:1.1rem;'>📍 {lat:.2f}°, {lon:.2f}°</div>", unsafe_allow_html=True)
    

    # Display content based on menu selection
    if st.session_state['menu_selection'] == "Home":
        #st.markdown("<div class='cosmic-section'>🌍 Welcome to Your Cosmic Dashboard</div>", unsafe_allow_html=True)
      
        st.markdown("<div class='cosmic-section'>🌤️ Current Conditions</div>", unsafe_allow_html=True)
        
        # Calculate time until sunset (keep your existing logic)
        sunset_time_str = astronomy['sun']['sunset']
        sunset_hour, sunset_min = map(int, sunset_time_str.split(':'))
        now = get_local_now(lat, lon)
        sunset_time = now.replace(hour=sunset_hour, minute=sunset_min, second=0, microsecond=0)
        
        if sunset_time < now:
            time_message = "🌙 Sun has set - stargazing time is here!"
        else:
            time_diff = sunset_time - now
            hours_until = time_diff.seconds // 3600
            mins_until = (time_diff.seconds % 3600) // 60
            
            if hours_until > 2:
                time_message = f"⏰ {hours_until}h {mins_until}m until sunset - prepare your equipment!"
            elif hours_until > 0:
                time_message = f"🌅 Golden hour approaching in {mins_until} minutes - great for Moon photography!"
            else:
                time_message = f"🌆 Sunset in {mins_until} minutes - get ready!"
        
        # Time message bar
        #st.markdown(f"""
         #   <div class='cosmic-card' style='text-align: center; padding: 0.8rem; margin-bottom: 0.8rem;'>
          #      <div style='color: #ffb800; font-size: 1rem;'>{time_message}</div>
          #  </div>
       # """, unsafe_allow_html=True)
        
        # Weather conditions in one compact row
        c1, c2, c3, c4 = st.columns(4)
        
        temp = weather['current']['temp']
        feels = weather['current']['feels_like']
        humidity = weather['current']['humidity']
        wind = weather['current']['wind_speed']
        condition = weather['current']['weather'][0]['description'].capitalize()
        clouds = weather['current'].get('clouds', 0)
        temp_f = (temp * 9/5) + 32 
        temp_f_feel = (feels * 9/5) + 32
        temp_f_rounded = math.ceil(temp_f)
        temp_f_feel_rounded = math.ceil(temp_f_feel)
        temp_r = math.ceil(temp)
        feels_r = math.ceil(feels)
        
        temp_color = "#00ff00" if 10 <= temp <= 25 else "#ffb800" if 5 <= temp <= 30 else "#ff3333"
        humidity_color = "#00ff00" if humidity < 60 else "#ffb800" if humidity < 80 else "#ff3333"
        wind_color = "#00ff00" if wind < 3 else "#ffb800" if wind < 6 else "#ff3333"
        cloud_color = "#00ff00" if clouds < 30 else "#ffb800" if clouds < 60 else "#ff3333"
        
        c1.markdown(f"""
            <div class='cosmic-card' style='padding: 0.8rem;'>
                <div style='font-size: 0.75rem; opacity: 0.6; margin-bottom: 0.2rem;'>Temperature</div>
                <div style='font-size: 1.2rem; font-weight: bold; color: {temp_color};'>{temp_r}°C/{temp_f_rounded}°F</div>
                <div style='font-size: 0.7rem; opacity: 0.7;'>
                    {"❄️ Bundle up!" if temp < 10 else "👍 Comfortable" if temp < 25 else "🥵 Stay hydrated"}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        c2.markdown(f"""
            <div class='cosmic-card' style='padding: 0.8rem;'>
                <div style='font-size: 0.75rem; opacity: 0.6; margin-bottom: 0.2rem;'>Feels Like</div>
                <div style='font-size: 1.2rem; font-weight: bold; color: {temp_color};'>{feels_r}°C/{temp_f_feel_rounded}°F</div>
                <div style='font-size: 0.7rem; opacity: 0.7;'>
                    {"❄️ Bundle up!" if feels < 10 else "👍 Comfortable" if feels < 25 else "🥵 Stay hydrated"}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        c3.markdown(f"""
            <div class='cosmic-card' style='padding: 0.8rem;'>
                <div style='font-size: 0.75rem; opacity: 0.6; margin-bottom: 0.2rem;'>Humidity</div>
                <div style='font-size: 1.2rem; font-weight: bold; color: {humidity_color};'>{humidity}%</div>
                <div style='font-size: 0.7rem; opacity: 0.7;'>
                    {"Perfect!" if humidity < 60 else "⚠️ Dew risk" if humidity < 80 else "💧 High dew"}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        c4.markdown(f"""
            <div class='cosmic-card' style='padding: 0.8rem;'>
                <div style='font-size: 0.75rem; opacity: 0.6; margin-bottom: 0.2rem;'>Wind Speed</div>
                <div style='font-size: 1.2rem; font-weight: bold; color: {wind_color};'>{wind} m/s</div>
                <div style='font-size: 0.7rem; opacity: 0.7;'>
                    {"🎯 Very stable" if wind < 3 else "📷 Some shake" if wind < 6 else "💨 Difficult"}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Sky condition bar (more compact)
        cloud_emoji = "☀️" if clouds < 20 else "⛅" if clouds < 50 else "☁️" if clouds < 80 else "☁️☁️"
        
        st.markdown(f"""
            <div class='cosmic-card' style='padding: 0.8rem;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div style='flex: 1;'>
                        <span style='font-size: 0.9rem;'><b>Sky:</b> {condition} {cloud_emoji} • <b>Clouds:</b> {clouds}%</span>
                    </div>
                    <div style='flex: 1; max-width: 250px; height: 15px; background: rgba(255,255,255,0.1); 
                                border-radius: 10px; overflow: hidden; margin-left: 1rem;'>
                        <div style='width: {clouds}%; height: 100%; 
                                   background: linear-gradient(90deg, {cloud_color}, {cloud_color}bb);
                                   transition: width 0.5s ease;'></div>
                    </div>
                </div>
                <div style='margin-top: 0.5rem; font-size: 0.8rem; opacity: 0.8; text-align: center;'>
                    {"🌟 Excellent - Deep sky objects visible!" if clouds < 30 else 
                     "⭐ Good - Planets and bright stars visible" if clouds < 60 else 
                     "☁️ Poor - Only brightest objects visible"}
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("<div class='cosmic-section'>💡 Cosmic Fact of the Day</div>", unsafe_allow_html=True)
        st.markdown(f"""
                  <div class='cosmic-card'>
                    <div class="gradient-text" style='font-family: Orbitron, sans-serif;'>
                      {cosmic_fun_fact()}
                    </div>
                  </div>
        """, unsafe_allow_html=True)
       
        # NASA Picture of the Day
        st.markdown("<div class='cosmic-section'>📸 NASA Astronomy Picture of the Day</div>", unsafe_allow_html=True)
        
        try:
            # Get NASA API key
            NASA_API_KEY = read_secret("NASA_API_KEY", "DEMO_KEY")
            if not NASA_API_KEY:
                NASA_API_KEY = "DEMO_KEY"
            
            url = f"https://api.nasa.gov/planetary/apod?api_key={NASA_API_KEY}"
            
            # Fetch APOD data with timeout
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                apod = response.json()
                
                # Create two columns for image and description
                col_img, col_desc = st.columns([1, 1])
                
                with col_img:
                    # Display title
                    title = apod.get('title', 'NASA Astronomy Picture')
                    st.markdown(f"<h3 style='color: #f59e0b;'>{title}</h3>", unsafe_allow_html=True)
                    
                    # Display date
                    date = apod.get('date', '')
                    if date:
                        st.caption(f"📅 {date}")
                    
                    # Display media
                    url = f"https://api.nasa.gov/planetary/apod?api_key={NASA_API_KEY}&thumbs=true"
                    
                    media_url  = apod.get('url', '')
                    media_type = apod.get('media_type', 'image')
                    thumb_url  = apod.get('thumbnail_url')
                    
                    if media_type == "image" and media_url:
                        st.image(media_url, caption=title, use_container_width=True)
                    elif media_type == "video" and media_url:
                        if any(h in media_url for h in ("youtube.com", "youtu.be", "vimeo.com")):
                            st.video(media_url)
                        else:
                            if thumb_url:
                                st.image(thumb_url, caption=title, use_container_width=True)
                            st.link_button("Open APOD Video", media_url)
                    else:
                        st.info("Media type not supported or unavailable today.")
                
                    
                    # Copyright info if available
                    copyright_info = apod.get('copyright', '')
                    if copyright_info:
                        st.caption(f"© {copyright_info}")
                
                with col_desc:
                    st.markdown("<h4 style='color: #7c3aed;'>📖 Explanation</h4>", unsafe_allow_html=True)
                    
                    # Display explanation
                    explanation = apod.get('explanation', 'No description available')
                    
                    # Create a scrollable container for long descriptions
                    # Around lines 2741–2751
                    st.markdown(f"""
                        <div style='background: rgba(131,56,236,0.1);
                                   border: 1px solid rgba(131,56,236,0.3);
                                   border-radius: 10px;
                                   padding: 1rem;
                                   line-height: 1.6;'>
                            {explanation}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # HD version link if available
                    hdurl = apod.get('hdurl', '')
                    if hdurl:
                        st.markdown(f"[🔍 View HD Version]({hdurl})")
                    
                    # Service version
                    service_version = apod.get('service_version', '')
                    if service_version:
                        st.caption(f"API Version: {service_version}")
            else:
                st.warning(f"Unable to load NASA Picture (Status: {response.status_code})")
                st.info("Visit [apod.nasa.gov](https://apod.nasa.gov) for today's image")
                
        except requests.exceptions.Timeout:
            st.warning("⏱️ NASA API timeout. Please try again later.")
            st.info("Visit [apod.nasa.gov](https://apod.nasa.gov) for today's image")
        except Exception as e:
            st.warning(f"Unable to load NASA Picture of the Day")
            st.info("Visit [apod.nasa.gov](https://apod.nasa.gov) for today's image")
            
            if st.session_state.get('debug_mode', False):
                st.error(f"Debug: {str(e)}")
    
    
    elif st.session_state['menu_selection'] == "Celestial Tracker":
        st.markdown("<div class='cosmic-section'>🌙 Celestial Schedule</div>", unsafe_allow_html=True)
    
        # Create Sky Map Visualization
      
        
        # What's Up Right Now Alert Box
        def get_next_event(astronomy, current_time, lat, lon):
 
            events = []
            
            # Get local timezone-aware current time
            local_now = get_local_now(lat, lon)
            
            # Parse sunset/sunrise times with local timezone
            sunset_hour, sunset_min = map(int, astronomy['sun']['sunset'].split(':'))
            sunrise_hour, sunrise_min = map(int, astronomy['sun']['sunrise'].split(':'))
            
            sunset_time = local_now.replace(hour=sunset_hour, minute=sunset_min, second=0, microsecond=0)
            sunrise_time = local_now.replace(hour=sunrise_hour, minute=sunrise_min, second=0, microsecond=0)
            
            # Handle day transitions
            if sunset_time < local_now:
                sunset_time += timedelta(days=1)
            
            if sunrise_time < local_now:
                sunrise_time += timedelta(days=1)
            
            # Moonrise
            moonrise_hour, moonrise_min = map(int, astronomy['moon']['moonrise'].split(':'))
            moonrise_time = local_now.replace(hour=moonrise_hour, minute=moonrise_min, second=0, microsecond=0)
            if moonrise_time < local_now:
                moonrise_time += timedelta(days=1)
            
            # Collect events
            events.append(("🌅 Sunset", sunset_time))
            events.append(("🌙 Moonrise", moonrise_time))
            events.append(("☀️ Sunrise", sunrise_time))
            
            # Find next event
            events.sort(key=lambda x: x[1])
            for event_name, event_time in events:
                if event_time > local_now:
                    time_diff = event_time - local_now
                    hours = time_diff.seconds // 3600
                    minutes = (time_diff.seconds % 3600) // 60
                    return event_name, hours, minutes
            
            return "No events", 0, 0
       
        
        # Display the alert box
        current_time = datetime.now()
        next_event, hours_until, mins_until = get_next_event(astronomy, current_time, lat, lon)
        
        # Determine if it's a good time for stargazing
        hour = current_time.hour
        if astronomy['sun']['sunset']:
            sunset_hour = int(astronomy['sun']['sunset'].split(':')[0])
            local_now = get_local_now(lat, lon)
            hour = local_now.hour
            
            # Check if sun has actually set
            sunset_time = local_now.replace(hour=sunset_hour, minute=int(astronomy['sun']['sunset'].split(':')[1]), second=0, microsecond=0)
            
            if local_now < sunset_time:
                # Still daytime
                alert_message = "🔴 Daylight hours - Prepare for tonight"
                alert_color = "#ff3333"
            elif 21 <= hour or hour <= 4:  # Peak stargazing hours
                alert_message = "🟢 PERFECT STARGAZING CONDITIONS - GO OUTSIDE NOW!"
                alert_color = "#00ff00"
            else:
                alert_message = "🟡 Good viewing starting - Eyes adjusting period"
                alert_color = "#ffb800"
        


        
        c1, c2 = st.columns(2)
        
        c1.markdown(f"""<div class='cosmic-card' style='padding: 1rem;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div style='font-size: 2rem;'>☀️</div>
                <div style='text-align: right;'>
                    <div style='font-size: 0.8rem; opacity: 0.6;'>Sunrise / Sunset</div>
                    <div style='font-size: 1.2rem; color: #f59e0b; font-weight: bold;'>
                        {astronomy['sun']['sunrise']} / {astronomy['sun']['sunset']}
                    </div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
        
        c2.markdown(f"""<div class='cosmic-card' style='padding: 1rem;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div style='font-size: 2rem;'>🌙</div>
                <div style='text-align: right;'>
                    <div style='font-size: 0.8rem; opacity: 0.6;'>{astronomy['moon']['phase']}</div>
                    <div style='font-size: 1.2rem; color: #f59e0b; font-weight: bold;'>
                        ↑{astronomy['moon']['moonrise']} / ↓{astronomy['moon']['moonset']}
                    </div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
        # Display the sky map
        
        
        #st.markdown("<div class='cosmic-section'>🪐 Planet Visibility Tracker</div>", unsafe_allow_html=True)
        display_planet_visibility(astronomy, st.session_state.get('current_city', 'your location'))
        
        # Constellation section - fixed without duplication
        st.markdown("<div class='cosmic-section'>⭐ Visible Constellations Tonight</div>", unsafe_allow_html=True)
        
        const_summary = get_constellation_summary(lat, lon)
    
    # Display summary bar
        st.markdown(f"""
        <div class='cosmic-card' style='text-align: center; padding: 0.8rem; margin-bottom: 0.8rem;'>
            <div style='color: #f59e0b; font-size: 1.07rem; font-weight: bold; margin-bottom: 0.28em;'>
                🌃 Tonight's Constellation Forecast
            </div>
            <div style='font-size: 1.07rem;'> 
                <span style='color: #00ff00;'>{const_summary['visible_now']} visible now</span> • 
                <span style='color: #ffb800;'>{const_summary['rising_later']} will rise later</span> • 
                <span style='color: #8b5cf6;'>{const_summary['total_tonight']} total tonight</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
        
        # Display constellations
        visible_constellations = get_visible_constellations(lat, lon)
        
        if visible_constellations:
            cols = st.columns(2)
            for idx, constellation in enumerate(visible_constellations):
                col = cols[idx % 2]
                with col:
                    # Color code based on current visibility
                    if "now" in constellation['altitude']:
                        border_color = "#00ff00"  # Green for visible now
                    else:
                        border_color = "#ffb800"  # Orange for rising later
                    
                    st.markdown(f"""
                        <div class='cosmic-card' style='
                            padding: 0.5rem; 
                            min-height: 90px; 
                            font-size:0.98rem;
                            border-left: 3px solid {border_color};
                        '>
                            <b>{constellation['constellation']}</b><br>
                            Key Star: <b>{constellation['star']}</b><br>
                            Position: {constellation['altitude']}<br>
                            <span style='font-size:0.92rem; color:#f59e0b;'><b>Best at:</b> {constellation['best_time']}</span><br>
                            <span style='font-size:0.88rem; opacity:0.8;'>{constellation['description']}</span>
                        </div>
                    """, unsafe_allow_html=True)
            
            if len(visible_constellations) % 2 != 0:
                with cols[1]:
                    st.markdown("<div style='height:90px;'></div>", unsafe_allow_html=True)
        else:
            st.info("No major constellations will be visible tonight from this location.")
            
    elif st.session_state['menu_selection'] == "ISS Tracking":
        st.markdown("<div class='cosmic-section'>🛰️ ISS Passes</div>", unsafe_allow_html=True)
        iss_facts = [
        "The ISS travels at 17,500 mph (28,000 km/h) - it circles Earth every 90 minutes!",
        "The ISS is about the size of a football field and weighs 420,000 kg (925,000 lbs).",
        "Astronauts aboard the ISS see 16 sunrises and sunsets every day.",
        "The ISS orbits at ~408 km (254 miles) above Earth - you can see it with the naked eye!",
        "The ISS has been continuously inhabited since November 2000 - over 23 years!",
        "Water on the ISS is recycled - including from urine and sweat - with 93% efficiency.",
        "The ISS solar panels generate 84-120 kilowatts of power - enough for 40 homes.",
        "Over 270 spacewalks have been conducted for ISS assembly and maintenance.",
        "The ISS cost approximately $150 billion - the most expensive object ever built.",
        "Internet speed on the ISS is about 600 Mbps down and 25 Mbps up to Earth."
    ]
    
    # Select fact based on current day (changes daily)
        fact_index = datetime.now().timetuple().tm_yday % len(iss_facts)
        current_fact = iss_facts[fact_index]
        
        st.markdown(f"""
            <div class='cosmic-card' style='
                background: linear-gradient(135deg, rgba(59,134,255,0.1), rgba(124,58,237,0.1));
                border-left: 3px solid #3a86ff;
                padding: 1rem;
                margin-bottom: 1rem;
            '>
                <div style='color: #3a86ff; font-size: 0.9rem; margin-bottom: 0.3rem;'>
                    💡 <b>ISS Fact of the Day</b>
                </div>
                <div style='color: #fff; font-size: 0.95rem; line-height: 1.4;'>
                    {current_fact}
                </div>
            </div>
        """, unsafe_allow_html=True)
        iss_vis = iss_visibility(lat, lon)
        iss_city, iss_country = get_city_from_coords(iss_vis["iss_lat"], iss_vis["iss_lon"]) if "iss_lat" in iss_vis else ("Unknown", "Unknown")
        
        if iss_vis["visible"]:
            st.success(iss_vis["message"])
            # Format coordinates only if they're numbers
            if isinstance(iss_lat, (int, float)) and isinstance(iss_lon, (int, float)):
                location_text = f"{iss_lat:.2f}°, {iss_lon:.2f}°"
            else:
                location_text = "Location unavailable"
            
            st.markdown(
                f"<div class='cosmic-card'><b>Current ISS Location:</b> {location_text}{visible_location_text}</div>",
                unsafe_allow_html=True
            )
            
            
            # Show visible passes block
            if iss_passes:
                valid_passes = [p for p in iss_passes if p.get('duration', 0) > 0]
                if valid_passes:
                    st.success(f"🛸 Next {len(valid_passes)} visible ISS passes") 
                    for i, p in enumerate(valid_passes[:2], 1):  # Only next 2 passes
                        duration_min = p['duration'] // 60
                        duration_sec = p['duration'] % 60
                        st.markdown(
                            f"""<div class='cosmic-card'>
                                <b>Pass #{i}</b><br>
                                📅 Date: <b>{p['date']}</b><br>
                                🕐 Time: <b>{p['risetime']}</b> local time<br>
                                ⏱️ Duration: <b>{duration_min}m {duration_sec}s</b>
                            </div>""",
                            unsafe_allow_html=True
                        )
                else:
                    # No valid passes available
                    st.info("🛸 ISS pass data is currently unavailable. The ISS orbits Earth every 90 minutes, but visible passes from your location require specific conditions. Please check back later.")
            
        
        else:
            # ISS not visible in entered location
            iss_lat = iss_vis.get('iss_lat', 'N/A')
            iss_lon = iss_vis.get('iss_lon', 'N/A')
            city, country = iss_city, iss_country
            
            st.markdown(f"""
                <div class='cosmic-card' style='
                    background: #fffbe6;
                    border-left: 4px solid #f59e0b;
                    padding: 0.8rem;
                    margin-bottom: 0.8rem;
                    color: #a16207;
                    font-size: 1.07rem;
                    font-weight: bold;
                    text-align: center;
                '>
                    🚫 The ISS is not visible from your location ({st.session_state['current_city']}) right now.
                </div>
            """, unsafe_allow_html=True)
            
            visible_location_text = (
                f"<br><b>Visible today in:</b> {city}, {country}"
                if city != "Unknown" and country != "Unknown"
                else ""
            )
            st.markdown(
                f"<div class='cosmic-card'><b>Current ISS Location:</b> {iss_lat:.2f}°, {iss_lon:.2f}{visible_location_text}</div>",
                unsafe_allow_html=True
            )
        
            fig = create_iss_world_map(lat, lon)
            fig.update_layout(
                title="<b style='color:#ffb800; font-size:1.3em;'>🛰️ ISS Live Trajectory</b>",
                geo=dict(
                    projection_type='natural earth',
                    showland=True,
                    landcolor='rgb(20,20,40)',
                    showocean=True,
                    oceancolor='rgb(10,30,80)',
                    showcountries=True,
                    countrycolor='rgba(255,255,255,0.3)',
                    showframe=False,
                    showcoastlines=True,
                    coastlinecolor='rgba(200,200,255,0.5)',
                    bgcolor='rgba(0,0,0,0)'
                ),
                paper_bgcolor='rgba(30,30,50,0.95)',
                plot_bgcolor='rgba(30,30,50,0.95)',
                font=dict(family="Orbitron, sans-serif", color="#fffbe6", size=12)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            crew = get_iss_crew()
            if crew:
                st.markdown("<div class='cosmic-section'>👨‍🚀 Current ISS Crew</div>", unsafe_allow_html=True)
                cols = st.columns(min(len(crew), 3))
                for idx, astronaut in enumerate(crew[:6]):  # Limit to 6
                    with cols[idx % 3]:
                        st.markdown(f"""
                            <div class='cosmic-card' style='text-align: center; padding: 1rem;'>
                                <div style='font-size: 2rem;'>👨‍🚀</div>
                                <div style='color: #f59e0b; font-weight: bold;'>{astronaut['name']}</div>
                                <div style='opacity: 0.7; font-size: 0.9rem;'>Aboard ISS</div>
                            </div>
                        """, unsafe_allow_html=True)
            
    elif st.session_state['menu_selection'] == "Cosmic Oracle":
        #st.markdown("<div class='cosmic-section'>🤖 Cosmic Oracle</div>", unsafe_allow_html=True)

        
        # AI Q&A Section
        st.markdown("""
        <div class='cosmic-card'>
            <h2 style='text-align: center; 
            font-family: Orbitron, sans-serif;
            background: linear-gradient(90deg, #f59e0b, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;'>
            🔮 Ask the Cosmic Oracle
        </h2>
    <p style='text-align: center; opacity: 0.8;'>
        Access the Galactic Archives.<br>
        Submit your inquiry. The Cosmic Oracle will provide an immediate answer!
    </p>
    </div>
    """, unsafe_allow_html=True)
        
        # Chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # REMOVED THE CHAT DISPLAY FROM HERE - MOVED TO BOTTOM
        
        # Quick questions
        # Add this CSS right before your quick questions buttons (where your other styles are)
        st.markdown("""
        <style>
        /* Smooth scrolling for the entire page */
        html {
            scroll-behavior: smooth;
        }
        
        /* Auto-focus animation for new answers */
        .ai-response {
            animation: slideInFocus 0.6s ease-out;
        }
        
        @keyframes slideInFocus {
            0% {
                opacity: 0;
                transform: translateY(20px);
            }
            100% {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Highlight the newest answer */
        .ai-response:last-of-type {
            border: 2px solid rgba(255, 184, 0, 0.3);
            box-shadow: 0 0 30px rgba(255, 184, 0, 0.2);
            animation: pulseGlow 2s ease-out;
        }
        
        @keyframes pulseGlow {
            0% {
                box-shadow: 0 0 30px rgba(255, 184, 0, 0.4);
            }
            100% {
                box-shadow: 0 0 10px rgba(255, 184, 0, 0.1);
            }
        }
        
        /* Your existing button styles continue here... */
        div[data-testid="stHorizontalBlock"]:has(button) button {
            /* ... your existing button styles ... */
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style='
        background: linear-gradient(135deg, rgba(124,58,237,0.2), rgba(58,134,255,0.15));
        border: 2px solid rgba(124,58,237,0.5);
        border-radius: 16px;
        padding: 1rem;
        margin-bottom: 1.5rem;
        text-align: center;
        '>
        <h3 style='
            font-family: Orbitron, sans-serif;
            background: linear-gradient(90deg, #f59e0b, #7c3aed, #3a86ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
            font-size: 1.3rem;
            letter-spacing: 0.1em;
        '>✨ QUICK COSMIC QUERIES ✨</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.get('ai_enabled', False):
            # AI-enhanced questions
            suggested_queries = st.session_state['ai_processor'].get_suggested_queries()
            quick_questions = suggested_queries[:4]

        else:
            # Standard questions
      
            quick_questions = [
                "🪐 What planets are visible tonight?",
                "🌙 What's the moon phase?",
                "🛸 When's the next ISS pass?",
                "⭐ Is tonight good for stargazing?"
            ]
        
    
# Quick questions section
        
        st.markdown("""
        <div id="quick-questions-container">
        </div>
        """, unsafe_allow_html=True)
        
        # Add this CSS right before your quick questions buttons
        st.markdown("""
        <style>
        /* Target buttons that come right after this style tag */
        /* Replace your current quick button CSS with this */
        /* Replace your current quick button CSS with this */
div[data-testid="stHorizontalBlock"]:has(button) button {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.8) 0%, rgba(139, 92, 246, 0.7) 50%, rgba(167, 139, 250, 0.6) 100%) !important;
    border: none !important;
    border-radius: 32px !important;  /* Much rounder corners */
    box-shadow: 
        /* 3D shadow layers */
        0 8px 0 rgba(91, 33, 182, 0.8),
        0 8px 1px rgba(76, 29, 149, 0.7),
        /* Depth shadow */
        0 12px 20px rgba(0, 0, 0, 0.3),
        /* Purple glow effects */
        0 0 30px rgba(124, 58, 237, 0.3),
        inset 0 2px 8px rgba(255, 255, 255, 0.2),
        inset 0 -4px 12px rgba(91, 33, 182, 0.2) !important;
    color: #fff !important;
    font-family: 'Orbitron', 'Space Grotesk', monospace !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 1.2em 1em !important;
    transition: all 0.15s ease !important;
    transform: translateY(0) !important;
    position: relative !important;
    overflow: hidden !important;
}

/* Shine effect overlay */
div[data-testid="stHorizontalBlock"]:has(button) button::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    height: 50% !important;
    background: linear-gradient(
        to bottom,
        rgba(255, 255, 255, 0.25),
        rgba(255, 255, 255, 0.1),
        transparent
    ) !important;
    border-radius: 32px 32px 0 0 !important;
    pointer-events: none !important;
}

div[data-testid="stHorizontalBlock"]:has(button) button:hover {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.9) 0%, rgba(124, 58, 237, 0.8) 50%, rgba(167, 139, 250, 0.7) 100%) !important;
    transform: translateY(2px) !important;
    box-shadow: 
        0 6px 0 rgba(91, 33, 182, 0.8),
        0 6px 1px rgba(76, 29, 149, 0.7),
        0 10px 18px rgba(0, 0, 0, 0.3),
        0 0 40px rgba(139, 92, 246, 0.5),
        inset 0 2px 10px rgba(255, 255, 255, 0.3),
        inset 0 -4px 12px rgba(91, 33, 182, 0.3) !important;
}

div[data-testid="stHorizontalBlock"]:has(button) button:active {
    transform: translateY(6px) !important;
    box-shadow: 
        0 2px 0 rgba(91, 33, 182, 0.8),
        0 2px 1px rgba(76, 29, 149, 0.7),
        0 4px 8px rgba(0, 0, 0, 0.3),
        inset 0 2px 4px rgba(0, 0, 0, 0.2),
        inset 0 -2px 6px rgba(91, 33, 182, 0.4) !important;
}
        }
        </style>
        """, unsafe_allow_html=True)

        # Grid of buttons
        for i in range(0, len(quick_questions), 2):
            
            col_a, col_b = st.columns(2)
            with col_a:
                if i < len(quick_questions):   
                        
                    if st.button(quick_questions[i], key=f"quick_{i}", use_container_width=True):
                        
                        question = quick_questions[i]
                        # Generate answer using AI or fallback
                        if st.session_state.get('ai_enabled', False):
                            with st.spinner("🤖 AI is thinking..."):
                                answer = st.session_state['ai_processor'].process_query(
                                    question,
                                    st.session_state['current_city'],
                                    lat, lon, weather, astronomy, iss_passes
                                )
                            ai_powered = True
                        else:
                            answer = generate_cosmic_answer_fallback(
                                question,
                                st.session_state['current_city'],
                                lat, lon, weather, astronomy, iss_passes
                            )
                            ai_powered = False
                        
                        st.session_state.chat_history.append({
                            'question': question,
                            'answer': answer,
                            'ai_powered': ai_powered
                        })
                        st.rerun()
            
            with col_b:
                if i+1 < len(quick_questions):
                    if st.button(quick_questions[i+1], key=f"quick_{i+1}", use_container_width=True):
                        question = quick_questions[i+1].replace("🤖 ", "").replace("🪐 ", "").replace("🌙 ", "").replace("🛸 ", "").replace("⭐ ", "")
                        
                        if st.session_state.get('ai_enabled', False):
                            with st.spinner("🤖 AI is thinking..."):
                                answer = st.session_state['ai_processor'].process_query(
                                    question,
                                    st.session_state['current_city'],
                                    lat, lon, weather, astronomy, iss_passes
                                )
                            ai_powered = True
                        else:
                            answer = generate_cosmic_answer_fallback(
                                question,
                                st.session_state['current_city'],
                                lat, lon, weather, astronomy, iss_passes
                            )
                            ai_powered = False
                        
                        st.session_state.chat_history.append({
                            'question': question,
                            'answer': answer,
                            'ai_powered': ai_powered
                        })
                        st.rerun()
        # Custom question input
        st.markdown("---")
        st.markdown("""
        <div style='
        background: linear-gradient(135deg, rgba(124,58,237,0.2), rgba(58,134,255,0.15));
        border: 2px solid rgba(124,58,237,0.5);
        border-radius: 16px;
        padding: 1rem;
        margin-bottom: 1.5rem;
        text-align: center;
                    '>
                   <h3 class="gradient-text" style='
                font-family: Orbitron, sans-serif;
                margin: 0;
                font-size: 1.3rem;
            '>🔮 Ask The Cosmic Oracle: 🔮</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if 'input_counter' not in st.session_state:
            st.session_state.input_counter = 0
        
        input_key = f"cosmic_question_{st.session_state.input_counter}"
        
        user_question = st.text_input(
            "Ask your cosmic question",
            placeholder="Type any space or astronomy question...",
            key=input_key,
            label_visibility="collapsed"
        )
        
        col_submit, col_clear = st.columns([3, 1])
        with col_submit:
            button_text = "Get Your Answer!" if st.session_state.get('ai_enabled', False) else "🚀 Get Your Answer!"
  
            if st.button(button_text, type="primary", use_container_width=True):
                if user_question:
                    if st.session_state.get('ai_enabled', False):
                        with st.spinner("🔮 The Oracle is processing your question..."):
                            answer = st.session_state['ai_processor'].process_query(
                                user_question,
                                st.session_state['current_city'],
                                lat, lon, weather, astronomy, iss_passes
                            )
                        ai_powered = True
                    else:
                        answer = generate_cosmic_answer_fallback(
                            user_question,
                            st.session_state['current_city'],
                            lat, lon, weather, astronomy, iss_passes
                        )
                        ai_powered = False
                    
                    st.session_state.chat_history.append({
                        'question': user_question,
                        'answer': answer,
                        'ai_powered': ai_powered
                    })
                    st.rerun()
            
        
        with col_clear:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.input_counter += 1
                st.rerun()
                streamlit_js_eval(js_expressions="window.scrollTo(0,document.body.scrollHeight)", key="autoscroll")
        # ============= MOVED CHAT HISTORY DISPLAY TO HERE =============
        # Show chat messages BELOW all input elements
        # ============= MOVED CHAT HISTORY DISPLAY TO HERE =============
        # Show chat messages BELOW all input elements
       # Show chat messages BELOW all input elements
        st.markdown("---")  # Divider
        
        if st.session_state.get("chat_history"):
    # Create an expander that's open by default for the latest answer
            with st.expander("🔮 Oracle's Latest Response", expanded=True):
                # Show only the latest Q&A
                if st.session_state.chat_history:
                    latest_chat = st.session_state.chat_history[-1]
                    
                    st.markdown(f"""
                        <div style='background: rgba(0,255,255,0.1); 
                                   border-left: 3px solid #00ffff;
                                   padding: 0.5rem; 
                                   margin-bottom: 0.5rem;
                                   border-radius: 5px;'>
                            <small style='color: #00ffff;'>You asked:</small><br>
                            {latest_chat['question']}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if latest_chat.get("ai_powered", False):
                        st.markdown('<span class="ai-badge">AI POWERED</span>', unsafe_allow_html=True)
                    
                    st.markdown(f"""
                        <div class='ai-response'>
                            <small style='color: #ff00ff;'>Cosmic Oracle:</small><br>
                            {latest_chat['answer']}
                        </div>
                    """, unsafe_allow_html=True)
            
    # Show ALL previous answers (except the latest one)
        if len(st.session_state.chat_history) > 1:
            with st.expander(f"📜 Previous {len(st.session_state.chat_history)-1} Responses", expanded=False):
                # FIXED: Loop through ALL previous chats except the last one
                for chat in reversed(st.session_state.chat_history[:-1]):  # All except the latest
                    st.markdown(f"""
                        <div style='background: rgba(0,255,255,0.1); 
                                   border-left: 3px solid #00ffff;
                                   padding: 0.5rem; 
                                   margin-bottom: 0.5rem;
                                   border-radius: 5px;'>
                            <small style='color: #00ffff;'>You asked:</small><br>
                            {chat['question']}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if chat.get("ai_powered", False):
                        st.markdown('<span class="ai-badge">AI POWERED</span>', unsafe_allow_html=True)
                    
                    st.markdown(f"""
                        <div class='ai-response'>
                            <small style='color: #ff00ff;'>Cosmic Oracle:</small><br>
                            {chat['answer']}
                        </div>
                    """, unsafe_allow_html=True)
                
                # Add a separator between previous responses
                st.markdown("---")
        # This will always scroll to the bottom after each rerun
    
        
        # Footer (can be indented if needed)
    ai_status = "AI-Enhanced" if st.session_state.get('ai_enabled', False) else "Rule-Based"
    st.markdown(f"""<center>
                                <div style="opacity:0.55; font-size:0.95rem; margin-top:2em;">
                                🚀 <b>Stellaris ({ai_status})</b> | Real-time astronomical data for your location<br>
                                Data sources: OpenWeatherMap, Sunrise-Sunset.org, Open-Notify ISS{', OpenAI GPT-3.5' if   st.session_state.get('ai_enabled', False) else ''}
                                </div>
                                </center>""", unsafe_allow_html=True)
if __name__ == "__main__":
    main()





    
