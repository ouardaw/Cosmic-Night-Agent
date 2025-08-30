"""
Weather module for StellarGuide - Working Version
Handles weather data fetching and stargazing condition assessment
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
    Get weather forecast for given coordinates
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        Dictionary with weather data including current and forecast
    """
    
    # Get API key from environment
    api_key = os.getenv("OPENWEATHER_API_KEY") or st.secrets.get("OPENWEATHER_API_KEY", None)
    
    if not api_key:
        return {
            "error": "OpenWeatherMap API key not configured",
            "current": _get_demo_weather(),
            "daily": _get_demo_forecast()
        }
    
    try:
        # Use the current weather endpoint (free tier)
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': api_key,
            'units': 'metric'
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Process and return weather data
            return {
                "current": {
                    "temp": data['main']['temp'],
                    "feels_like": data['main']['feels_like'],
                    "clouds": data['clouds']['all'],
                    "humidity": data['main']['humidity'],
                    "wind_speed": data['wind']['speed'],
                    "wind_deg": data['wind'].get('deg', 0),
                    "uvi": 5,  # UV index not available in free tier
                    "visibility": data.get('visibility', 10000),
                    "description": data['weather'][0]['description'] if data['weather'] else "Clear"
                },
                "daily": _get_demo_forecast()  # Forecast requires paid tier
            }
        else:
            return {
                "error": f"API error: {response.status_code}",
                "current": _get_demo_weather(),
                "daily": _get_demo_forecast()
            }
            
    except requests.exceptions.RequestException as e:
        return {
            "error": f"Network error: {str(e)}",
            "current": _get_demo_weather(),
            "daily": _get_demo_forecast()
        }
    except Exception as e:
        return {
            "error": f"Unexpected error: {str(e)}",
            "current": _get_demo_weather(),
            "daily": _get_demo_forecast()
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
        "uvi": 3,
        "visibility": 10000,
        "description": "Clear sky (demo data)"
    }

def _get_demo_forecast() -> List[Dict[str, Any]]:
    """Return demo forecast data"""
    return [
        {"temp": {"max": 25, "min": 15}, "clouds": 20},
        {"temp": {"max": 24, "min": 14}, "clouds": 30},
        {"temp": {"max": 23, "min": 13}, "clouds": 40},
        {"temp": {"max": 22, "min": 12}, "clouds": 35},
        {"temp": {"max": 24, "min": 14}, "clouds": 25}
    ]

def assess_stargazing_conditions(weather_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assess stargazing conditions based on weather data
    
    Args:
        weather_data: Weather data dictionary
        
    Returns:
        Assessment with rating and recommendations
    """
    
    if "error" in weather_data:
        return {
            "rating": "unknown",
            "score": 0,
            "factors": {},
            "recommendation": "Unable to assess conditions"
        }
    
    current = weather_data.get("current", {})
    
    # Scoring factors (0-100, higher is better)
    factors = {
        "cloud_cover": 100 - current.get("clouds", 100),
        "humidity": 100 - min(current.get("humidity", 100), 100),
        "wind": max(0, 100 - (current.get("wind_speed", 0) * 10)),
        "visibility": min(100, current.get("visibility", 0) / 100)
    }
    
    # Calculate overall score
    score = sum(factors.values()) / len(factors)
    
    # Determine rating
    if score >= 80:
        rating = "excellent"
        recommendation = "Perfect conditions for stargazing! Don't miss tonight!"
    elif score >= 60:
        rating = "good"
        recommendation = "Good conditions for observing bright objects"
    elif score >= 40:
        rating = "fair"
        recommendation = "Limited visibility - focus on bright planets and moon"
    else:
        rating = "poor"
        recommendation = "Poor conditions - consider postponing observation"
    
    return {
        "rating": rating,
        "score": round(score),
        "factors": factors,
        "recommendation": recommendation,
        "details": {
            "cloud_interpretation": _interpret_clouds(current.get("clouds", 0)),
            "humidity_interpretation": _interpret_humidity(current.get("humidity", 0)),
            "wind_interpretation": _interpret_wind(current.get("wind_speed", 0))
        }
    }

def _interpret_clouds(cloud_percentage: float) -> str:
    """Interpret cloud cover for stargazing"""
    if cloud_percentage < 10:
        return "Crystal clear skies"
    elif cloud_percentage < 30:
        return "Mostly clear with minimal clouds"
    elif cloud_percentage < 50:
        return "Partly cloudy - some objects visible"
    elif cloud_percentage < 70:
        return "Mostly cloudy - limited visibility"
    else:
        return "Heavily clouded - poor visibility"

def _interpret_humidity(humidity: float) -> str:
    """Interpret humidity for stargazing"""
    if humidity < 40:
        return "Low humidity - excellent transparency"
    elif humidity < 60:
        return "Moderate humidity - good viewing"
    elif humidity < 80:
        return "High humidity - some atmospheric distortion"
    else:
        return "Very high humidity - poor seeing conditions"

def _interpret_wind(wind_speed: float) -> str:
    """Interpret wind speed for telescope stability"""
    if wind_speed < 2:
        return "Calm - perfect for high magnification"
    elif wind_speed < 5:
        return "Light breeze - stable viewing"
    elif wind_speed < 10:
        return "Moderate wind - some vibration possible"
    else:
        return "Strong wind - difficult for telescope use"

def get_hourly_forecast(lat: float, lon: float, hours: int = 12) -> List[Dict[str, Any]]:
    """
    Get hourly forecast for stargazing planning
    Note: Requires paid API tier - returns demo data for free tier
    """
    
    # For free tier, return demo data
    return [
        {
            "time": (datetime.now() + timedelta(hours=i)).strftime("%H:00"),
            "clouds": 20 + (i * 5) % 40,
            "temp": 20 - i,
            "conditions": "Clear" if i % 3 == 0 else "Partly cloudy"
        }
        for i in range(hours)
    ]

# For backward compatibility
def check_weather_conditions(location: str) -> Dict[str, Any]:
    """
    Legacy function for checking weather by location name
    """
    # Default to Austin, TX coordinates for demo
    return get_weather_forecast(30.2672, -97.7431)

# Test function
if __name__ == "__main__":
    print("Testing weather module...")
    result = get_weather_forecast(30.2672, -97.7431)
    if "error" not in result:
        print(f"✅ Success! Temperature in Austin: {result['current']['temp']}°C")
        print(f"   Conditions: {result['current']['description']}")
        print(f"   Cloud cover: {result['current']['clouds']}%")
    else:
        print(f"❌ Error: {result['error']}")
