"""
ISS Tracker module for StellarGuide
Tracks International Space Station passes and visibility
"""

import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import math

def get_iss_passes(lat: float, lon: float, alt: float = 0, n: int = 5) -> Dict[str, Any]:
    """
    Get ISS pass predictions for a location
    
    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees
        alt: Altitude in meters (default 0)
        n: Number of passes to return (default 5)
        
    Returns:
        Dictionary with ISS pass information
    """
    
    try:
        # Open Notify API for ISS passes
        url = "http://api.open-notify.org/iss-pass.json"
        params = {
            'lat': lat,
            'lon': lon,
            'alt': alt,
            'n': n
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Process the pass data
            passes = []
            for pass_data in data.get('response', []):
                rise_time = datetime.fromtimestamp(pass_data['risetime'])
                duration = pass_data['duration']
                
                passes.append({
                    'risetime': rise_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'duration': duration,
                    'duration_minutes': duration / 60,
                    'pass_type': _determine_pass_type(rise_time.hour),
                    'magnitude': _estimate_magnitude(duration),
                    'max_elevation': _estimate_max_elevation(duration)
                })
            
            return {
                'location': {'latitude': lat, 'longitude': lon, 'altitude': alt},
                'request_time': datetime.now().isoformat(),
                'passes': passes
            }
        else:
            return _get_demo_passes(lat, lon, n)
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching ISS data: {e}")
        return _get_demo_passes(lat, lon, n)
    except Exception as e:
        print(f"Unexpected error: {e}")
        return _get_demo_passes(lat, lon, n)

def _get_demo_passes(lat: float, lon: float, n: int = 5) -> Dict[str, Any]:
    """Return demo ISS pass data when API is unavailable"""
    
    passes = []
    base_time = datetime.now().replace(hour=20, minute=0, second=0, microsecond=0)
    
    for i in range(n):
        pass_time = base_time + timedelta(days=i, hours=i % 3, minutes=(i * 17) % 60)
        duration = 180 + (i * 47) % 300  # 3-8 minutes
        
        passes.append({
            'risetime': pass_time.strftime('%Y-%m-%d %H:%M:%S'),
            'duration': duration,
            'duration_minutes': duration / 60,
            'pass_type': _determine_pass_type(pass_time.hour),
            'magnitude': _estimate_magnitude(duration),
            'max_elevation': _estimate_max_elevation(duration)
        })
    
    return {
        'location': {'latitude': lat, 'longitude': lon, 'altitude': 0},
        'request_time': datetime.now().isoformat(),
        'passes': passes,
        'demo_data': True
    }

def _determine_pass_type(hour: int) -> str:
    """Determine the type of pass based on time"""
    if 5 <= hour < 7:
        return "Dawn pass"
    elif 19 <= hour < 21:
        return "Dusk pass"
    elif 21 <= hour or hour < 5:
        return "Night pass"
    else:
        return "Daylight pass (not visible)"

def _estimate_magnitude(duration: int) -> float:
    """Estimate ISS brightness based on pass duration"""
    # Longer passes typically mean higher elevation and brighter appearance
    if duration > 360:
        return -3.5  # Very bright
    elif duration > 240:
        return -2.5  # Bright
    elif duration > 120:
        return -1.5  # Moderate
    else:
        return -0.5  # Dim

def _estimate_max_elevation(duration: int) -> float:
    """Estimate maximum elevation based on pass duration"""
    # Rough estimation: longer duration = higher pass
    if duration > 360:
        return 75  # Nearly overhead
    elif duration > 240:
        return 50  # High pass
    elif duration > 120:
        return 30  # Medium pass
    else:
        return 15  # Low pass

def get_iss_current_position() -> Dict[str, Any]:
    """
    Get current ISS position
    
    Returns:
        Dictionary with current ISS coordinates
    """
    
    try:
        url = "http://api.open-notify.org/iss-now.json"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            position = data.get('iss_position', {})
            lat = float(position.get('latitude', 0))
            lon = float(position.get('longitude', 0))
            
            # Determine what the ISS is flying over
            location_description = _describe_location(lat, lon)
            
            return {
                'timestamp': data.get('timestamp', datetime.now().timestamp()),
                'latitude': lat,
                'longitude': lon,
                'altitude_km': 408,  # Approximate ISS altitude
                'velocity_kmh': 27600,  # Approximate ISS velocity
                'location_description': location_description,
                'daylight': _is_daylight_below(lat, lon),
                'visibility_region_radius_km': 2200  # Approximate visibility radius
            }
        else:
            return _get_demo_position()
            
    except Exception as e:
        print(f"Error fetching ISS position: {e}")
        return _get_demo_position()

def _get_demo_position() -> Dict[str, Any]:
    """Return demo ISS position when API is unavailable"""
    
    # Simulate ISS movement
    import time
    t = time.time()
    
    # Simple orbit simulation (not accurate, just for demo)
    lat = 51.5 * math.sin(t / 1000)
    lon = ((t / 100) % 360) - 180
    
    return {
        'timestamp': t,
        'latitude': lat,
        'longitude': lon,
        'altitude_km': 408,
        'velocity_kmh': 27600,
        'location_description': _describe_location(lat, lon),
        'daylight': _is_daylight_below(lat, lon),
        'visibility_region_radius_km': 2200,
        'demo_data': True
    }

def _describe_location(lat: float, lon: float) -> str:
    """Describe what the ISS is flying over based on coordinates"""
    
    # Simplified location description based on coordinates
    # In production, would use reverse geocoding
    
    if -30 <= lat <= 30:
        if -30 <= lon <= 60:
            return "Over Africa or Middle East"
        elif 60 <= lon <= 150:
            return "Over Asia or Australia"
        elif -150 <= lon <= -30:
            return "Over the Americas"
        else:
            return "Over the Pacific Ocean"
    elif lat > 30:
        if -30 <= lon <= 60:
            return "Over Europe or North Africa"
        elif 60 <= lon <= 150:
            return "Over Northern Asia"
        elif -150 <= lon <= -30:
            return "Over North America"
        else:
            return "Over the North Pacific"
    else:
        if -30 <= lon <= 60:
            return "Over Southern Africa"
        elif 60 <= lon <= 150:
            return "Over Australia or Southern Ocean"
        elif -150 <= lon <= -30:
            return "Over South America"
        else:
            return "Over the South Pacific"

def _is_daylight_below(lat: float, lon: float) -> bool:
    """Determine if it's daylight at the location below ISS"""
    
    # Simplified calculation
    current_hour = datetime.now().hour
    
    # Adjust for longitude (very simplified)
    local_hour = (current_hour + int(lon / 15)) % 24
    
    # Consider daylight between 6 AM and 6 PM local time
    return 6 <= local_hour <= 18

def calculate_iss_visibility(observer_lat: float, observer_lon: float,
                           iss_lat: float, iss_lon: float) -> Dict[str, Any]:
    """
    Calculate if ISS is visible from observer location
    
    Args:
        observer_lat: Observer's latitude
        observer_lon: Observer's longitude
        iss_lat: ISS latitude
        iss_lon: ISS longitude
        
    Returns:
        Dictionary with visibility information
    """
    
    # Calculate distance between observer and ISS ground track
    distance_km = _calculate_distance(observer_lat, observer_lon, iss_lat, iss_lon)
    
    # ISS is visible up to ~2200km away under ideal conditions
    max_visibility_range = 2200
    
    is_visible = distance_km <= max_visibility_range
    
    if is_visible:
        # Calculate approximate elevation angle
        elevation = _calculate_elevation_angle(distance_km)
        
        # Calculate azimuth (simplified)
        azimuth = _calculate_azimuth(observer_lat, observer_lon, iss_lat, iss_lon)
        
        return {
            'visible': True,
            'distance_km': distance_km,
            'elevation_degrees': elevation,
            'azimuth_degrees': azimuth,
            'brightness_magnitude': _estimate_magnitude_from_distance(distance_km),
            'visibility_quality': _rate_visibility(elevation)
        }
    else:
        return {
            'visible': False,
            'distance_km': distance_km,
            'next_opportunity': "Check pass predictions"
        }

def _calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points on Earth (Haversine formula)"""
    
    R = 6371  # Earth's radius in km
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_lat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) *
         math.sin(delta_lon / 2) ** 2)
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

def _calculate_elevation_angle(distance_km: float) -> float:
    """Calculate approximate elevation angle based on distance"""
    
    # ISS altitude
    h = 408  # km
    
    # Earth radius
    R = 6371  # km
    
    if distance_km == 0:
        return 90  # Directly overhead
    
    # Simplified calculation
    # In reality, would use more complex spherical geometry
    angle = math.degrees(math.atan(h / distance_km))
    
    return min(90, max(0, angle))

def _calculate_azimuth(observer_lat: float, observer_lon: float,
                      target_lat: float, target_lon: float) -> float:
    """Calculate azimuth from observer to target"""
    
    lat1 = math.radians(observer_lat)
    lat2 = math.radians(target_lat)
    lon1 = math.radians(observer_lon)
    lon2 = math.radians(target_lon)
    
    dlon = lon2 - lon1
    
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    
    azimuth = math.degrees(math.atan2(x, y))
    
    return (azimuth + 360) % 360

def _estimate_magnitude_from_distance(distance_km: float) -> float:
    """Estimate ISS brightness based on distance"""
    
    if distance_km < 500:
        return -3.5  # Very bright, nearly overhead
    elif distance_km < 1000:
        return -2.5  # Bright
    elif distance_km < 1500:
        return -1.5  # Moderate
    else:
        return -0.5  # Dim, near horizon

def _rate_visibility(elevation: float) -> str:
    """Rate visibility quality based on elevation"""
    
    if elevation > 60:
        return "Excellent - Nearly overhead"
    elif elevation > 40:
        return "Very good - High in sky"
    elif elevation > 25:
        return "Good - Clearly visible"
    elif elevation > 10:
        return "Fair - Low but visible"
    else:
        return "Poor - Very low on horizon"

def get_crew_info() -> Dict[str, Any]:
    """
    Get information about current ISS crew
    
    Returns:
        Dictionary with crew information
    """
    
    try:
        url = "http://api.open-notify.org/astros.json"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            # Filter for ISS crew only
            iss_crew = [person for person in data.get('people', [])
                       if person.get('craft') == 'ISS']
            
            return {
                'number': len(iss_crew),
                'crew': iss_crew,
                'last_updated': datetime.now().isoformat()
            }
        else:
            return _get_demo_crew()
            
    except Exception as e:
        print(f"Error fetching crew data: {e}")
        return _get_demo_crew()

def _get_demo_crew() -> Dict[str, Any]:
    """Return demo crew data when API is unavailable"""
    
    return {
        'number': 7,
        'crew': [
            {'name': 'Demo Astronaut 1', 'craft': 'ISS'},
            {'name': 'Demo Astronaut 2', 'craft': 'ISS'},
            {'name': 'Demo Astronaut 3', 'craft': 'ISS'},
            {'name': 'Demo Astronaut 4', 'craft': 'ISS'},
            {'name': 'Demo Astronaut 5', 'craft': 'ISS'},
            {'name': 'Demo Astronaut 6', 'craft': 'ISS'},
            {'name': 'Demo Astronaut 7', 'craft': 'ISS'}
        ],
        'last_updated': datetime.now().isoformat(),
        'demo_data': True
    }

def format_pass_for_display(pass_data: Dict[str, Any]) -> str:
    """
    Format ISS pass data for user-friendly display
    
    Args:
        pass_data: Dictionary with pass information
        
    Returns:
        Formatted string for display
    """
    
    rise_time = pass_data.get('risetime', 'Unknown')
    duration = pass_data.get('duration_minutes', 0)
    magnitude = pass_data.get('magnitude', 0)
    pass_type = pass_data.get('pass_type', 'Unknown')
    
    return (f"🛸 ISS Pass on {rise_time}\n"
            f"   Duration: {duration:.1f} minutes\n"
            f"   Brightness: {magnitude:.1f} magnitude\n"
            f"   Type: {pass_type}")

# Helper function for testing
def main():
    """Test the ISS tracker functionality"""
    
    print("🛸 ISS Tracker Test\n")
    
    # Test location (Austin, TX)
    lat, lon = 30.2672, -97.7431
    
    print(f"Location: {lat}°N, {lon}°W\n")
    
    # Get current position
    print("Current ISS Position:")
    position = get_iss_current_position()
    print(f"  Latitude: {position['latitude']:.2f}°")
    print(f"  Longitude: {position['longitude']:.2f}°")
    print(f"  Location: {position['location_description']}")
    print()
    
    # Get upcoming passes
    print("Upcoming ISS Passes:")
    passes = get_iss_passes(lat, lon, n=3)
    for i, pass_info in enumerate(passes['passes'], 1):
        print(f"\nPass #{i}:")
        print(format_pass_for_display(pass_info))
    
    # Get crew info
    print("\n\nCurrent ISS Crew:")
    crew = get_crew_info()
    print(f"  Total: {crew['number']} astronauts")
    for person in crew['crew'][:3]:
        print(f"  - {person['name']}")

if __name__ == "__main__":
    main()