"""
Weather module with enhanced debugging
Save this as src/weather_debug.py for testing
"""

import os
import streamlit as st
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# Load environment variables with explicit path
import os

# Try to load from .env for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def get_weather_forecast(lat: float, lon: float) -> Dict[str, Any]:
    """
    Get weather forecast with detailed debugging
    """
    
    # Get API key from environment
    api_key = os.getenv("OPENWEATHER_API_KEY") or st.secrets.get("OPENWEATHER_API_KEY", None)
    
    print(f"DEBUG: API Key loaded: {api_key is not None}")
    print(f"DEBUG: API Key length: {len(api_key) if api_key else 0}")
    
    if not api_key:
        print("DEBUG: No API key found in environment")
        return {
            "error": "OpenWeatherMap API key not configured",
            "debug_info": "API key not found in environment variables",
            "current": _get_demo_weather()
        }
    
    try:
        # Try the current weather endpoint first (simpler)
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': api_key,
            'units': 'metric'
        }
        
        print(f"DEBUG: Making API call to {url}")
        print(f"DEBUG: Parameters: lat={lat}, lon={lon}")
        
        response = requests.get(url, params=params, timeout=10)
        
        print(f"DEBUG: Response status code: {response.status_code}")
        print(f"DEBUG: Response text (first 200 chars): {response.text[:200]}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Convert current weather format to our format
            return {
                "current": {
                    "temp": data['main']['temp'],
                    "feels_like": data['main']['feels_like'],
                    "clouds": data['clouds']['all'],
                    "humidity": data['main']['humidity'],
                    "wind_speed": data['wind']['speed'],
                    "wind_deg": data['wind'].get('deg', 0),
                    "description": data['weather'][0]['description'] if data['weather'] else "Clear"
                },
                "daily": []  # Current weather API doesn't provide forecast
            }
        elif response.status_code == 401:
            return {
                "error": "Invalid API key",
                "debug_info": f"API returned 401 Unauthorized. Check if your API key is activated.",
                "current": _get_demo_weather()
            }
        elif response.status_code == 404:
            return {
                "error": "Location not found",
                "debug_info": f"API returned 404 for coordinates {lat}, {lon}",
                "current": _get_demo_weather()
            }
        else:
            return {
                "error": f"API error: {response.status_code}",
                "debug_info": f"Response: {response.text[:200]}",
                "current": _get_demo_weather()
            }
            
    except requests.exceptions.ConnectionError as e:
        print(f"DEBUG: Connection error: {e}")
        return {
            "error": "Connection error",
            "debug_info": f"Could not connect to OpenWeatherMap API: {str(e)}",
            "current": _get_demo_weather()
        }
    except requests.exceptions.Timeout as e:
        print(f"DEBUG: Timeout error: {e}")
        return {
            "error": "Request timeout",
            "debug_info": "API request timed out after 10 seconds",
            "current": _get_demo_weather()
        }
    except requests.exceptions.RequestException as e:
        print(f"DEBUG: Request error: {e}")
        return {
            "error": f"Network error: {str(e)}",
            "debug_info": f"Request failed: {str(e)}",
            "current": _get_demo_weather()
        }
    except Exception as e:
        print(f"DEBUG: Unexpected error: {e}")
        return {
            "error": f"Unexpected error: {str(e)}",
            "debug_info": f"Error type: {type(e).__name__}",
            "current": _get_demo_weather()
        }

def _get_demo_weather() -> Dict[str, Any]:
    """Return demo weather data when API is unavailable"""
    return {
        "temp": 22,
        "feels_like": 21,
        "clouds": 15,
        "humidity": 55,
        "wind_speed": 2.5,
        "wind_deg": 225,
        "description": "Clear sky (demo data)"
    }

def test_api_key():
    """Test function to verify API key is working"""
    api_key = os.getenv('OPENWEATHER_API_KEY')
    
    if not api_key:
        print("❌ No API key found in environment")
        print("Make sure OPENWEATHER_API_KEY is in your .env file")
        return False
    
    print(f"✓ API key found (length: {len(api_key)})")
    
    # Test with a simple API call
    url = f"https://api.openweathermap.org/data/2.5/weather?q=London&appid={api_key}"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print("✅ API key is valid and working!")
            return True
        elif response.status_code == 401:
            print("❌ API key is invalid or not activated")
            print("Please check your OpenWeatherMap account")
            return False
        else:
            print(f"⚠️ Unexpected response: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False

if __name__ == "__main__":
    print("Testing OpenWeatherMap API...")
    print("-" * 40)
    test_api_key()
    print("-" * 40)
    print("\nTesting weather forecast for Austin, TX...")
    result = get_weather_forecast(30.2672, -97.7431)
    if "error" in result:
        print(f"Error: {result['error']}")
        if "debug_info" in result:
            print(f"Debug: {result['debug_info']}")
    else:
        print(f"Success! Temperature: {result['current']['temp']}°C")
