"""
Astronomy module for StellarGuide
Core astronomical calculations and celestial object tracking
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json

try:
    from skyfield.api import load, Star, wgs84
    from skyfield.almanac import find_discrete, risings_and_settings
    SKYFIELD_AVAILABLE = True
except ImportError:
    SKYFIELD_AVAILABLE = False
    print("Warning: Skyfield not installed. Using simplified calculations.")

try:
    import ephem
    EPHEM_AVAILABLE = True
except ImportError:
    EPHEM_AVAILABLE = False

def get_visible_objects(location: str) -> Dict[str, Any]:
    """
    Get all visible celestial objects for a given location
    
    Args:
        location: City name or coordinates
        
    Returns:
        Dictionary with visible planets, stars, and deep sky objects
    """
    
    # For demo purposes, return sample data
    # In production, this would use actual astronomical calculations
    
    current_hour = datetime.now().hour
    is_night = current_hour < 6 or current_hour > 18
    
    if not is_night:
        return {
            "visible_objects": {
                "planets": [],
                "bright_stars": ["Sun"],
                "deep_sky": [],
                "message": "Daytime - limited celestial objects visible"
            },
            "observation_quality": "poor",
            "best_time": "Wait for sunset"
        }
    
    # Sample visible objects
    visible_data = {
        "visible_objects": {
            "planets": _get_visible_planets(),
            "bright_stars": _get_bright_stars(),
            "deep_sky": _get_deep_sky_objects(),
            "satellites": _get_visible_satellites()
        },
        "observation_quality": "good",
        "best_time": "Now through 2 AM",
        "limiting_magnitude": 5.5
    }
    
    return visible_data

def _get_visible_planets() -> List[str]:
    """Get list of currently visible planets"""
    
    # Simplified visibility - in reality would calculate actual positions
    current_month = datetime.now().month
    
    # Rough visibility by month (simplified)
    planet_visibility = {
        1: ["Mars", "Jupiter"],
        2: ["Venus", "Mars", "Jupiter"],
        3: ["Venus", "Mars", "Saturn"],
        4: ["Venus", "Mars", "Saturn"],
        5: ["Venus", "Jupiter", "Saturn"],
        6: ["Jupiter", "Saturn"],
        7: ["Jupiter", "Saturn", "Mars"],
        8: ["Jupiter", "Saturn", "Mars"],
        9: ["Jupiter", "Saturn", "Venus"],
        10: ["Venus", "Jupiter", "Saturn"],
        11: ["Venus", "Mars", "Jupiter"],
        12: ["Venus", "Mars", "Jupiter"]
    }
    
    return planet_visibility.get(current_month, ["Jupiter", "Saturn"])

def _get_bright_stars() -> List[str]:
    """Get list of bright stars visible tonight"""
    
    # Seasonal bright stars
    current_month = datetime.now().month
    
    if current_month in [12, 1, 2]:  # Winter
        return ["Sirius", "Betelgeuse", "Rigel", "Aldebaran", "Capella", 
                "Procyon", "Pollux", "Castor"]
    elif current_month in [3, 4, 5]:  # Spring
        return ["Arcturus", "Spica", "Regulus", "Denebola", "Alphard",
                "Cor Caroli", "Vindemiatrix"]
    elif current_month in [6, 7, 8]:  # Summer
        return ["Vega", "Altair", "Deneb", "Antares", "Arcturus",
                "Spica", "Shaula", "Rasalhague"]
    else:  # Autumn
        return ["Fomalhaut", "Vega", "Altair", "Deneb", "Capella",
                "Aldebaran", "Mirfak", "Hamal"]

def _get_deep_sky_objects() -> List[str]:
    """Get list of visible deep sky objects"""
    
    current_month = datetime.now().month
    
    if current_month in [12, 1, 2]:  # Winter
        return ["M42 (Orion Nebula)", "M45 (Pleiades)", "M31 (Andromeda)",
                "M35", "M37", "M1 (Crab Nebula)"]
    elif current_month in [3, 4, 5]:  # Spring
        return ["M51 (Whirlpool Galaxy)", "M81/M82", "M104 (Sombrero)",
                "M87 (Virgo A)", "M13 (Hercules Cluster)"]
    elif current_month in [6, 7, 8]:  # Summer
        return ["M13 (Hercules Cluster)", "M57 (Ring Nebula)", "M27 (Dumbbell)",
                "M8 (Lagoon)", "M20 (Trifid)", "M31 (Andromeda)"]
    else:  # Autumn
        return ["M31 (Andromeda)", "M33 (Triangulum)", "M15", "M2",
                "NGC 869/884 (Double Cluster)", "M45 (Pleiades)"]

def _get_visible_satellites() -> List[str]:
    """Get visible satellites"""
    return ["ISS (if passing)", "Starlink train (possible)", "Iridium flares (rare)"]

def calculate_moon_phase(date: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Calculate the current moon phase
    
    Args:
        date: Optional date (defaults to today)
        
    Returns:
        Dictionary with moon phase information
    """
    
    if date is None:
        date = datetime.now()
    
    # Simple moon phase calculation
    # Using the synodic month period (29.530588 days)
    
    # Known new moon date (January 11, 2024)
    known_new_moon = datetime(2024, 1, 11, 11, 57)
    
    # Calculate days since known new moon
    days_since = (date - known_new_moon).total_seconds() / 86400
    
    # Calculate phase (0-29.53 days)
    synodic_month = 29.530588
    phase_days = days_since % synodic_month
    
    # Calculate illumination percentage
    phase_angle = (phase_days / synodic_month) * 360
    illumination = (1 - math.cos(math.radians(phase_angle))) / 2 * 100
    
    # Determine phase name
    if phase_days < 1.85:
        phase_name = "New Moon"
    elif phase_days < 5.54:
        phase_name = "Waxing Crescent"
    elif phase_days < 9.23:
        phase_name = "First Quarter"
    elif phase_days < 12.91:
        phase_name = "Waxing Gibbous"
    elif phase_days < 16.60:
        phase_name = "Full Moon"
    elif phase_days < 20.29:
        phase_name = "Waning Gibbous"
    elif phase_days < 23.98:
        phase_name = "Last Quarter"
    elif phase_days < 27.66:
        phase_name = "Waning Crescent"
    else:
        phase_name = "New Moon"
    
    return {
        "phase_name": phase_name,
        "illumination": round(illumination, 1),
        "age": round(phase_days, 1),
        "phase_angle": round(phase_angle, 1),
        "next_new_moon": known_new_moon + timedelta(days=((days_since // synodic_month + 1) * synodic_month)),
        "next_full_moon": known_new_moon + timedelta(days=((days_since // synodic_month) * synodic_month + 14.765))
    }

def calculate_planet_positions(lat: float, lon: float, 
                              date: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Calculate positions of visible planets
    
    Args:
        lat: Observer latitude
        lon: Observer longitude
        date: Optional observation date
        
    Returns:
        Dictionary with planet positions
    """
    
    if date is None:
        date = datetime.now()
    
    # Simplified planet data
    # In production, would use ephemeris calculations
    planets = {}
    
    visible_planets = _get_visible_planets()
    
    for planet in visible_planets:
        # Mock data - in reality would calculate actual positions
        planets[planet] = {
            "altitude": 30 + (hash(planet) % 40),  # Random altitude 30-70°
            "azimuth": (hash(planet) % 360),
            "magnitude": _get_planet_magnitude(planet),
            "constellation": _get_planet_constellation(planet),
            "rise_time": "19:30",  # Would calculate actual times
            "set_time": "04:30",
            "transit_time": "00:00"
        }
    
    return {
        "observer_location": {"latitude": lat, "longitude": lon},
        "observation_time": date.isoformat(),
        "planets": planets
    }

def _get_planet_magnitude(planet: str) -> float:
    """Get typical apparent magnitude of planet"""
    magnitudes = {
        "Mercury": -0.5,
        "Venus": -4.0,
        "Mars": 0.5,
        "Jupiter": -2.5,
        "Saturn": 0.5,
        "Uranus": 5.7,
        "Neptune": 7.8
    }
    return magnitudes.get(planet, 0.0)

def _get_planet_constellation(planet: str) -> str:
    """Get current constellation for planet (simplified)"""
    # In reality, would calculate actual position
    constellations = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
                     "Libra", "Scorpius", "Sagittarius", "Capricornus", 
                     "Aquarius", "Pisces"]
    return constellations[hash(planet) % 12]

def get_constellation_info(constellation_name: str) -> Dict[str, Any]:
    """
    Get information about a specific constellation
    
    Args:
        constellation_name: Name of constellation
        
    Returns:
        Dictionary with constellation information
    """
    
    constellation_data = {
        "Orion": {
            "bright_stars": ["Betelgeuse", "Rigel", "Bellatrix", "Mintaka", "Alnilam", "Alnitak"],
            "best_months": ["December", "January", "February"],
            "mythology": "The Hunter in Greek mythology",
            "notable_features": ["Orion Nebula (M42)", "Horsehead Nebula", "Orion's Belt"],
            "area_sq_degrees": 594
        },
        "Ursa Major": {
            "bright_stars": ["Dubhe", "Merak", "Phecda", "Megrez", "Alioth", "Mizar", "Alkaid"],
            "best_months": ["April", "May", "June"],
            "mythology": "The Great Bear in Greek mythology",
            "notable_features": ["Big Dipper asterism", "M81/M82 galaxies", "M101 Pinwheel Galaxy"],
            "area_sq_degrees": 1280
        },
        "Cassiopeia": {
            "bright_stars": ["Schedar", "Caph", "Gamma Cas", "Ruchbah", "Segin"],
            "best_months": ["October", "November", "December"],
            "mythology": "The vain Queen in Greek mythology",
            "notable_features": ["W-shaped asterism", "Heart and Soul Nebulae"],
            "area_sq_degrees": 598
        }
    }
    
    return constellation_data.get(constellation_name, {
        "bright_stars": [],
        "best_months": [],
        "mythology": "Information not available",
        "notable_features": [],
        "area_sq_degrees": 0
    })

def calculate_best_viewing_times(object_name: str, lat: float, lon: float,
                                date: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Calculate best viewing times for a celestial object
    
    Args:
        object_name: Name of celestial object
        lat: Observer latitude
        lon: Observer longitude
        date: Optional date
        
    Returns:
        Dictionary with viewing recommendations
    """
    
    if date is None:
        date = datetime.now()
    
    # Simplified calculation
    # In production would use actual ephemeris data
    
    return {
        "object": object_name,
        "date": date.strftime("%Y-%m-%d"),
        "location": {"latitude": lat, "longitude": lon},
        "rise_time": "20:15",
        "transit_time": "01:30",  # Highest point in sky
        "set_time": "06:45",
        "best_viewing_window": "22:00 - 04:00",
        "altitude_at_transit": 65,  # degrees
        "notes": "Best viewed when at least 30° above horizon"
    }

def get_meteor_showers(month: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Get active meteor showers for given month
    
    Args:
        month: Optional month number (defaults to current)
        
    Returns:
        List of active meteor showers
    """
    
    if month is None:
        month = datetime.now().month
    
    all_showers = [
        {"name": "Quadrantids", "peak": "Jan 3-4", "rate": 120, "months": [1]},
        {"name": "Lyrids", "peak": "Apr 22-23", "rate": 18, "months": [4]},
        {"name": "Eta Aquariids", "peak": "May 6-7", "rate": 60, "months": [5]},
        {"name": "Delta Aquariids", "peak": "Jul 29-30", "rate": 20, "months": [7]},
        {"name": "Perseids", "peak": "Aug 12-13", "rate": 100, "months": [8]},
        {"name": "Orionids", "peak": "Oct 21-22", "rate": 20, "months": [10]},
        {"name": "Leonids", "peak": "Nov 17-18", "rate": 15, "months": [11]},
        {"name": "Geminids", "peak": "Dec 13-14", "rate": 120, "months": [12]}
    ]
    
    active_showers = [shower for shower in all_showers if month in shower["months"]]
    
    return active_showers

def calculate_sunrise_sunset(lat: float, lon: float, 
                            date: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Calculate sunrise and sunset times
    
    Args:
        lat: Latitude
        lon: Longitude
        date: Optional date
        
    Returns:
        Dictionary with sun times
    """
    
    if date is None:
        date = datetime.now()
    
    # Simplified calculation
    # In production would use proper algorithms
    
    # Very rough approximation
    day_of_year = date.timetuple().tm_yday
    
    # Equation of time (minutes)
    B = 2 * math.pi * (day_of_year - 81) / 365
    E = 9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)
    
    # Solar declination
    decl = 23.45 * math.sin(math.radians((360 * (284 + day_of_year)) / 365))
    
    # Hour angle
    lat_rad = math.radians(lat)
    decl_rad = math.radians(decl)
    
    try:
        hour_angle = math.degrees(math.acos(-math.tan(lat_rad) * math.tan(decl_rad)))
    except:
        hour_angle = 90  # Default for edge cases
    
    # Sunrise/sunset times (very approximate)
    sunrise_hour = 12 - hour_angle / 15 - E / 60 - lon / 15
    sunset_hour = 12 + hour_angle / 15 - E / 60 - lon / 15
    
    sunrise = date.replace(hour=int(sunrise_hour), minute=int((sunrise_hour % 1) * 60))
    sunset = date.replace(hour=int(sunset_hour), minute=int((sunset_hour % 1) * 60))
    
    # Calculate twilight times
    civil_twilight_begin = sunrise - timedelta(minutes=30)
    civil_twilight_end = sunset + timedelta(minutes=30)
    nautical_twilight_begin = sunrise - timedelta(minutes=60)
    nautical_twilight_end = sunset + timedelta(minutes=60)
    astronomical_twilight_begin = sunrise - timedelta(minutes=90)
    astronomical_twilight_end = sunset + timedelta(minutes=90)
    
    return {
        "sunrise": sunrise.strftime("%H:%M"),
        "sunset": sunset.strftime("%H:%M"),
        "civil_twilight_begin": civil_twilight_begin.strftime("%H:%M"),
        "civil_twilight_end": civil_twilight_end.strftime("%H:%M"),
        "nautical_twilight_begin": nautical_twilight_begin.strftime("%H:%M"),
        "nautical_twilight_end": nautical_twilight_end.strftime("%H:%M"),
        "astronomical_twilight_begin": astronomical_twilight_begin.strftime("%H:%M"),
        "astronomical_twilight_end": astronomical_twilight_end.strftime("%H:%M"),
        "day_length": f"{hour_angle * 2 / 15:.1f} hours"
    }
