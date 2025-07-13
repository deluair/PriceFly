"""Data generators for specific airline industry datasets."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import random
from dataclasses import dataclass

from ..models.passenger import TravelPurpose, BookingChannel, LoyaltyTier
from ..models.airline import CabinClass


class AirportGenerator:
    """Generate airport data for simulation."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = np.random.RandomState(random_state)
    
    def generate_airports(self, num_airports: int = 100) -> List[Dict[str, Any]]:
        """Generate realistic airport data."""
        airports = []
        
        # Major airports with realistic data
        major_airports = [
            {"iata_code": "JFK", "name": "John F. Kennedy International", "city": "New York", "country": "USA", "lat": 40.6413, "lon": -73.7781},
            {"iata_code": "LAX", "name": "Los Angeles International", "city": "Los Angeles", "country": "USA", "lat": 33.9425, "lon": -118.4081},
            {"iata_code": "LHR", "name": "London Heathrow", "city": "London", "country": "UK", "lat": 51.4700, "lon": -0.4543},
            {"iata_code": "CDG", "name": "Charles de Gaulle", "city": "Paris", "country": "France", "lat": 49.0097, "lon": 2.5479},
            {"iata_code": "NRT", "name": "Narita International", "city": "Tokyo", "country": "Japan", "lat": 35.7720, "lon": 140.3929}
        ]
        
        airports.extend(major_airports)
        
        # Generate additional airports
        for i in range(len(major_airports), num_airports):
            airport = {
                "iata_code": f"A{i:02d}",
                "name": f"Airport {i}",
                "city": f"City {i}",
                "country": self.random_state.choice(["USA", "UK", "France", "Germany", "Japan"]),
                "lat": self.random_state.uniform(-60, 60),
                "lon": self.random_state.uniform(-180, 180)
            }
            airports.append(airport)
        
        return airports


class AircraftGenerator:
    """Generate aircraft data for simulation."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = np.random.RandomState(random_state)
    
    def generate_aircraft(self, num_aircraft: int = 50) -> List[Dict[str, Any]]:
        """Generate realistic aircraft data."""
        aircraft_types = [
            {"type": "Boeing 737-800", "seats": 189, "range_km": 5765, "fuel_burn": 2500},
            {"type": "Airbus A320", "seats": 180, "range_km": 6150, "fuel_burn": 2400},
            {"type": "Boeing 777-300ER", "seats": 396, "range_km": 14685, "fuel_burn": 9600},
            {"type": "Airbus A350-900", "seats": 325, "range_km": 15000, "fuel_burn": 8000}
        ]
        
        aircraft = []
        for i in range(num_aircraft):
            aircraft_type = self.random_state.choice(aircraft_types)
            aircraft.append({
                "aircraft_id": f"AC{i:03d}",
                "type": aircraft_type["type"],
                "total_seats": aircraft_type["seats"],
                "range_km": aircraft_type["range_km"],
                "fuel_burn_per_hour": aircraft_type["fuel_burn"]
            })
        
        return aircraft


class RouteGenerator:
    """Generate route data for simulation."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = np.random.RandomState(random_state)
    
    def generate_routes(self, airports: List[Dict], num_routes: int = 200) -> List[Dict[str, Any]]:
        """Generate realistic route data."""
        routes = []
        
        for i in range(num_routes):
            origin = self.random_state.choice(airports)
            destination = self.random_state.choice(airports)
            
            # Ensure origin != destination
            while destination["iata_code"] == origin["iata_code"]:
                destination = self.random_state.choice(airports)
            
            # Calculate distance (simplified)
            distance = self._calculate_distance(origin, destination)
            
            route = {
                "route_code": f"{origin['iata_code']}-{destination['iata_code']}",
                "origin": origin["iata_code"],
                "destination": destination["iata_code"],
                "distance_km": distance,
                "flight_time_minutes": int(distance / 800 * 60),  # Approx 800 km/h
                "frequency_per_day": self.random_state.randint(1, 6),
                "average_fare_economy": self.random_state.uniform(200, 1500),
                "average_fare_business": self.random_state.uniform(800, 5000),
                "average_load_factor": self.random_state.uniform(0.6, 0.9)
            }
            routes.append(route)
        
        return routes
    
    def _calculate_distance(self, origin: Dict, destination: Dict) -> float:
        """Calculate distance between two airports (simplified)."""
        # Simplified distance calculation
        lat_diff = abs(origin["lat"] - destination["lat"])
        lon_diff = abs(origin["lon"] - destination["lon"])
        return (lat_diff + lon_diff) * 111  # Rough km conversion


class PassengerGenerator:
    """Generate passenger data for simulation."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = np.random.RandomState(random_state)
    
    def generate_passengers(self, num_passengers: int = 1000) -> List[Dict[str, Any]]:
        """Generate realistic passenger data."""
        passengers = []
        
        for i in range(num_passengers):
            passenger = {
                "passenger_id": f"PAX{i:06d}",
                "age": self.random_state.randint(18, 80),
                "travel_purpose": self.random_state.choice(["Business", "Leisure", "VFR"]),
                "booking_channel": self.random_state.choice(["Online", "Agent", "Mobile"]),
                "loyalty_tier": self.random_state.choice(["None", "Silver", "Gold", "Platinum"]),
                "price_sensitivity": self.random_state.uniform(0.1, 1.0),
                "advance_booking_days": self.random_state.randint(1, 365)
            }
            passengers.append(passenger)
        
        return passengers


class AirlineGenerator:
    """Generate airline data for simulation."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = np.random.RandomState(random_state)
    
    def generate_airlines(self, num_airlines: int = 10) -> List[Dict[str, Any]]:
        """Generate realistic airline data."""
        airlines = []
        
        airline_names = ["SkyWings", "AeroFly", "CloudJet", "StarAir", "BlueSkies", 
                        "FastWings", "GlobalAir", "PremiumFly", "EcoJet", "LuxuryAir"]
        
        for i in range(num_airlines):
            airline = {
                "airline_code": f"A{i:02d}",
                "name": airline_names[i] if i < len(airline_names) else f"Airline {i}",
                "country": self.random_state.choice(["USA", "UK", "France", "Germany", "Japan"]),
                "fleet_size": self.random_state.randint(10, 200),
                "price_competitiveness": self.random_state.uniform(0.8, 1.2),
                "service_quality_score": self.random_state.uniform(3.0, 5.0),
                "on_time_performance": self.random_state.uniform(0.7, 0.95)
            }
            airlines.append(airline)
        
        return airlines


class MarketDataGenerator:
    """Generate market data for simulation."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = np.random.RandomState(random_state)
    
    def generate_market_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate market conditions data."""
        days = (end_date - start_date).days
        dates = [start_date + timedelta(days=i) for i in range(days)]
        
        market_data = {
            "dates": dates,
            "fuel_prices": [self.random_state.uniform(0.8, 1.2) for _ in dates],
            "economic_index": [self.random_state.uniform(0.9, 1.1) for _ in dates],
            "seasonal_factor": [1.0 + 0.3 * np.sin(2 * np.pi * i / 365) for i in range(days)],
            "competition_intensity": [self.random_state.uniform(0.7, 1.3) for _ in dates]
        }
        
        return market_data


class BookingDataGenerator:
    """Generate realistic booking transaction data."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = np.random.RandomState(random_state)
    
    def generate_booking_transactions(
        self, 
        num_bookings: int,
        start_date: datetime,
        end_date: datetime,
        routes: List[Any],
        customer_segments: List[Any]
    ) -> pd.DataFrame:
        """Generate realistic booking transaction data."""
        
        bookings = []
        
        for _ in range(num_bookings):
            # Select random route and segment
            route = self.random_state.choice(routes)
            segment = self.random_state.choice(customer_segments)
            
            # Generate booking date
            booking_date = self._random_date(start_date, end_date)
            
            # Generate travel date (lead time based on segment)
            lead_time = segment.generate_booking_lead_time()
            travel_date = booking_date + timedelta(days=lead_time)
            
            # Determine cabin class
            cabin_class = self._select_cabin_class(segment.cabin_class_preference)
            
            # Calculate fare
            base_fare = self._calculate_base_fare(route, cabin_class, lead_time)
            
            # Add booking
            booking = {
                'booking_id': f"BK{_:08d}",
                'booking_date': booking_date,
                'travel_date': travel_date,
                'lead_time_days': lead_time,
                'origin': route.origin.iata_code,
                'destination': route.destination.iata_code,
                'route_code': route.route_code,
                'cabin_class': cabin_class.value,
                'passenger_segment': segment.segment_name,
                'travel_purpose': segment.primary_travel_purpose.value,
                'base_fare': base_fare,
                'taxes_fees': base_fare * 0.15,
                'total_fare': base_fare * 1.15,
                'booking_channel': self._select_booking_channel(),
                'loyalty_tier': self._select_loyalty_tier(segment.loyalty_tier_distribution),
                'party_size': self._generate_party_size(segment.primary_travel_purpose),
                'is_round_trip': self.random_state.choice([True, False], p=[0.7, 0.3]),
                'advance_purchase': lead_time,
                'day_of_week': travel_date.weekday(),
                'month': travel_date.month,
                'is_weekend': travel_date.weekday() >= 5,
                'is_holiday_period': self._is_holiday_period(travel_date)
            }
            
            bookings.append(booking)
        
        return pd.DataFrame(bookings)
    
    def _random_date(self, start: datetime, end: datetime) -> datetime:
        """Generate random date between start and end."""
        delta = end - start
        random_days = self.random_state.randint(0, delta.days)
        return start + timedelta(days=random_days)
    
    def _select_cabin_class(self, preferences: Dict[str, float]) -> CabinClass:
        """Select cabin class based on preferences."""
        classes = [CabinClass.ECONOMY, CabinClass.PREMIUM_ECONOMY, CabinClass.BUSINESS]
        probs = [preferences.get('economy', 0.8), 
                preferences.get('premium_economy', 0.15), 
                preferences.get('business', 0.05)]
        
        # Normalize probabilities
        total = sum(probs)
        probs = [p/total for p in probs]
        
        return self.random_state.choice(classes, p=probs)
    
    def _calculate_base_fare(self, route: Any, cabin_class: CabinClass, lead_time: int) -> float:
        """Calculate base fare with dynamic pricing factors."""
        if cabin_class == CabinClass.ECONOMY:
            base = route.average_fare_economy
        elif cabin_class == CabinClass.PREMIUM_ECONOMY:
            base = route.average_fare_economy * 1.5
        else:  # Business
            base = route.average_fare_business
        
        # Lead time adjustment
        if lead_time < 7:
            multiplier = 1.5  # Last minute premium
        elif lead_time < 21:
            multiplier = 1.2
        elif lead_time > 90:
            multiplier = 0.8  # Early bird discount
        else:
            multiplier = 1.0
        
        # Add some randomness
        noise = self.random_state.uniform(0.9, 1.1)
        
        return base * multiplier * noise
    
    def _select_booking_channel(self) -> str:
        """Select booking channel."""
        channels = ['Online Direct', 'OTA', 'Travel Agent', 'Mobile App', 'Call Center']
        probs = [0.4, 0.3, 0.1, 0.15, 0.05]
        return self.random_state.choice(channels, p=probs)
    
    def _select_loyalty_tier(self, distribution: Dict[LoyaltyTier, float]) -> str:
        """Select loyalty tier based on distribution."""
        tiers = list(distribution.keys())
        probs = list(distribution.values())
        return self.random_state.choice(tiers, p=probs).value
    
    def _generate_party_size(self, purpose: TravelPurpose) -> int:
        """Generate party size based on travel purpose."""
        if purpose == TravelPurpose.BUSINESS:
            return self.random_state.choice([1, 2], p=[0.8, 0.2])
        elif purpose == TravelPurpose.LEISURE:
            return self.random_state.choice([1, 2, 3, 4], p=[0.3, 0.4, 0.2, 0.1])
        else:  # VFR
            return self.random_state.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
    
    def _is_holiday_period(self, date: datetime) -> bool:
        """Check if date is in holiday period."""
        # Simplified holiday detection
        month = date.month
        day = date.day
        
        # Major holiday periods
        if month == 12 and day >= 20:  # Christmas/New Year
            return True
        elif month == 1 and day <= 5:  # New Year
            return True
        elif month == 7:  # Summer vacation
            return True
        elif month == 11 and 20 <= day <= 30:  # Thanksgiving
            return True
        
        return False


class PricingDataGenerator:
    """Generate pricing and revenue management data."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = np.random.RandomState(random_state)
    
    def generate_pricing_history(
        self,
        routes: List[Any],
        start_date: datetime,
        end_date: datetime,
        observations_per_day: int = 4
    ) -> pd.DataFrame:
        """Generate historical pricing data."""
        
        pricing_data = []
        current_date = start_date
        
        while current_date <= end_date:
            for route in routes:
                for obs in range(observations_per_day):
                    timestamp = current_date + timedelta(hours=obs * 6)
                    
                    # Generate prices for different booking classes
                    for cabin in [CabinClass.ECONOMY, CabinClass.BUSINESS]:
                        for booking_class in ['Y', 'B', 'M', 'H', 'Q', 'V', 'W']:
                            price = self._generate_price(
                                route, cabin, booking_class, timestamp
                            )
                            
                            availability = self._generate_availability(
                                booking_class, timestamp
                            )
                            
                            pricing_data.append({
                                'timestamp': timestamp,
                                'route_code': route.route_code,
                                'origin': route.origin.iata_code,
                                'destination': route.destination.iata_code,
                                'cabin_class': cabin.value,
                                'booking_class': booking_class,
                                'price': price,
                                'availability': availability,
                                'days_to_departure': self._days_to_departure(timestamp),
                                'day_of_week': timestamp.weekday(),
                                'hour_of_day': timestamp.hour
                            })
            
            current_date += timedelta(days=1)
        
        return pd.DataFrame(pricing_data)
    
    def _generate_price(
        self, 
        route: Any, 
        cabin: CabinClass, 
        booking_class: str, 
        timestamp: datetime
    ) -> float:
        """Generate price for specific booking class."""
        
        # Base price
        if cabin == CabinClass.ECONOMY:
            base_price = route.average_fare_economy
        else:
            base_price = route.average_fare_business
        
        # Booking class multipliers
        class_multipliers = {
            'Y': 1.0,   # Full fare
            'B': 0.9,   # Business discount
            'M': 0.8,   # Mid-tier
            'H': 0.7,   # Advance purchase
            'Q': 0.6,   # Restricted
            'V': 0.5,   # Deep discount
            'W': 0.4    # Web special
        }
        
        multiplier = class_multipliers.get(booking_class, 1.0)
        
        # Time-based adjustments
        days_out = self._days_to_departure(timestamp)
        if days_out < 7:
            time_multiplier = 1.3
        elif days_out < 21:
            time_multiplier = 1.1
        elif days_out > 90:
            time_multiplier = 0.9
        else:
            time_multiplier = 1.0
        
        # Add noise
        noise = self.random_state.uniform(0.95, 1.05)
        
        return base_price * multiplier * time_multiplier * noise
    
    def _generate_availability(self, booking_class: str, timestamp: datetime) -> int:
        """Generate seat availability for booking class."""
        # Higher availability for discount classes
        class_availability = {
            'Y': (5, 20),
            'B': (3, 15),
            'M': (5, 25),
            'H': (8, 30),
            'Q': (10, 40),
            'V': (5, 20),
            'W': (2, 10)
        }
        
        min_avail, max_avail = class_availability.get(booking_class, (0, 10))
        return self.random_state.randint(min_avail, max_avail + 1)
    
    def _days_to_departure(self, timestamp: datetime) -> int:
        """Calculate days to departure (simplified)."""
        # Assume departure is always 30-365 days in the future
        return self.random_state.randint(1, 365)


class CompetitorDataGenerator:
    """Generate competitive intelligence data."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = np.random.RandomState(random_state)
    
    def generate_competitor_prices(
        self,
        routes: List[Any],
        airlines: List[Any],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Generate competitor pricing data."""
        
        competitor_data = []
        current_date = start_date
        
        while current_date <= end_date:
            for route in routes:
                # Find airlines operating this route
                operating_airlines = [a for a in airlines 
                                    if any(r.route_code == route.route_code 
                                          for r in a.routes_operated)]
                
                if len(operating_airlines) < 2:
                    continue  # Skip routes with no competition
                
                for airline in operating_airlines:
                    # Generate competitive position
                    market_share = self._calculate_route_market_share(
                        airline, route, operating_airlines
                    )
                    
                    price_position = self._calculate_price_position(
                        airline, route, operating_airlines
                    )
                    
                    competitor_data.append({
                        'date': current_date,
                        'route_code': route.route_code,
                        'airline_code': airline.airline_code,
                        'competitor_count': len(operating_airlines) - 1,
                        'market_share': market_share,
                        'price_position': price_position,
                        'capacity_share': self._calculate_capacity_share(
                            airline, route, operating_airlines
                        ),
                        'frequency_share': self._calculate_frequency_share(
                            airline, route, operating_airlines
                        ),
                        'service_differential': self._calculate_service_differential(
                            airline, operating_airlines
                        )
                    })
            
            current_date += timedelta(days=1)
        
        return pd.DataFrame(competitor_data)
    
    def _calculate_route_market_share(
        self, 
        airline: Any, 
        route: Any, 
        competitors: List[Any]
    ) -> float:
        """Calculate airline's market share on route."""
        # Simplified market share calculation
        total_capacity = sum(len(a.fleet.aircraft) for a in competitors)
        airline_capacity = len(airline.fleet.aircraft)
        
        base_share = airline_capacity / total_capacity if total_capacity > 0 else 0
        
        # Adjust for airline competitiveness
        competitiveness_factor = airline.price_competitiveness
        adjusted_share = base_share * competitiveness_factor
        
        # Add some randomness
        noise = self.random_state.uniform(0.8, 1.2)
        
        return min(1.0, adjusted_share * noise)
    
    def _calculate_price_position(
        self, 
        airline: Any, 
        route: Any, 
        competitors: List[Any]
    ) -> float:
        """Calculate relative price position (1.0 = market average)."""
        # Base on airline's price competitiveness
        return airline.price_competitiveness * self.random_state.uniform(0.9, 1.1)
    
    def _calculate_capacity_share(self, airline: Any, route: Any, competitors: List[Any]) -> float:
        """Calculate capacity share on route."""
        return self._calculate_route_market_share(airline, route, competitors)
    
    def _calculate_frequency_share(self, airline: Any, route: Any, competitors: List[Any]) -> float:
        """Calculate frequency share on route."""
        # Assume frequency correlates with capacity
        return self._calculate_route_market_share(airline, route, competitors)
    
    def _calculate_service_differential(
        self, 
        airline: Any, 
        competitors: List[Any]
    ) -> float:
        """Calculate service quality differential vs competitors."""
        avg_competitor_quality = np.mean([a.service_quality_score for a in competitors])
        return airline.service_quality_score - avg_competitor_quality


class OperationalDataGenerator:
    """Generate operational performance data."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = np.random.RandomState(random_state)
    
    def generate_flight_operations(
        self,
        airlines: List[Any],
        routes: List[Any],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Generate flight operational data."""
        
        operations_data = []
        flight_id = 1
        
        current_date = start_date
        while current_date <= end_date:
            for airline in airlines:
                for route in airline.routes_operated:
                    # Generate flights based on frequency
                    daily_flights = route.frequency_per_day
                    
                    for flight_num in range(int(daily_flights)):
                        # Select aircraft
                        available_aircraft = [
                            ac for ac in airline.fleet.aircraft 
                            if ac.can_operate_route(route.distance_km)
                        ]
                        
                        if not available_aircraft:
                            continue
                        
                        aircraft = self.random_state.choice(available_aircraft)
                        
                        # Generate operational metrics
                        operations_data.append({
                            'flight_id': f"{airline.airline_code}{flight_id:04d}",
                            'date': current_date,
                            'airline_code': airline.airline_code,
                            'aircraft_id': aircraft.aircraft_id,
                            'aircraft_type': aircraft.aircraft_type.value,
                            'route_code': route.route_code,
                            'origin': route.origin.iata_code,
                            'destination': route.destination.iata_code,
                            'scheduled_departure': current_date + timedelta(
                                hours=6 + flight_num * 4
                            ),
                            'actual_departure': self._generate_actual_departure(
                                current_date + timedelta(hours=6 + flight_num * 4),
                                airline.on_time_performance
                            ),
                            'flight_time_minutes': route.flight_time_minutes,
                            'distance_km': route.distance_km,
                            'passengers_boarded': self._generate_passenger_count(
                                aircraft, route.average_load_factor
                            ),
                            'load_factor': self._generate_load_factor(
                                route.average_load_factor
                            ),
                            'fuel_consumed_liters': self._calculate_fuel_consumption(
                                aircraft, route.distance_km
                            ),
                            'crew_cost': aircraft.crew_cost_per_hour * (
                                route.flight_time_minutes / 60
                            ),
                            'fuel_cost': self._calculate_fuel_cost(
                                aircraft, route.distance_km, current_date
                            ),
                            'airport_fees': route.origin.landing_fee_base + 
                                          route.destination.landing_fee_base
                        })
                        
                        flight_id += 1
            
            current_date += timedelta(days=1)
        
        return pd.DataFrame(operations_data)
    
    def _generate_actual_departure(
        self, 
        scheduled: datetime, 
        on_time_performance: float
    ) -> datetime:
        """Generate actual departure time based on OTP."""
        if self.random_state.random() < on_time_performance:
            # On time (within 15 minutes)
            delay = self.random_state.randint(-5, 16)
        else:
            # Delayed
            delay = self.random_state.exponential(30) + 15
        
        return scheduled + timedelta(minutes=delay)
    
    def _generate_passenger_count(self, aircraft: Any, target_load_factor: float) -> int:
        """Generate passenger count based on load factor."""
        expected_passengers = int(aircraft.total_seats * target_load_factor)
        # Add some variation
        variation = self.random_state.randint(-10, 11)
        return max(0, min(aircraft.total_seats, expected_passengers + variation))
    
    def _generate_load_factor(self, target_load_factor: float) -> float:
        """Generate load factor with variation."""
        variation = self.random_state.normal(0, 0.05)
        return max(0.0, min(1.0, target_load_factor + variation))
    
    def _calculate_fuel_consumption(
        self, 
        aircraft: Any, 
        distance_km: float
    ) -> float:
        """Calculate fuel consumption for flight."""
        flight_time_hours = distance_km / aircraft.cruise_speed_kmh
        return aircraft.fuel_burn_per_hour * flight_time_hours
    
    def _calculate_fuel_cost(
        self, 
        aircraft: Any, 
        distance_km: float, 
        date: datetime
    ) -> float:
        """Calculate fuel cost for flight."""
        fuel_consumed = self._calculate_fuel_consumption(aircraft, distance_km)
        # Simplified fuel price (would normally lookup from fuel price series)
        fuel_price_per_liter = 0.85  # USD
        return fuel_consumed * fuel_price_per_liter


class EventDataGenerator:
    """Generate event and external factor data."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = np.random.RandomState(random_state)
    
    def generate_market_events(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Generate market events and external factors."""
        
        events = []
        
        # Generate regular events
        current_date = start_date
        while current_date <= end_date:
            # Check for various event types
            if self._is_holiday(current_date):
                events.append({
                    'date': current_date,
                    'event_type': 'Holiday',
                    'event_name': self._get_holiday_name(current_date),
                    'impact_magnitude': self.random_state.uniform(1.2, 2.0),
                    'affected_regions': self._get_affected_regions(current_date),
                    'duration_days': self._get_event_duration('Holiday')
                })
            
            # Random events
            if self.random_state.random() < 0.01:  # 1% chance per day
                event_type = self.random_state.choice([
                    'Weather Disruption', 'Economic News', 'Geopolitical Event',
                    'Industry News', 'Fuel Price Shock'
                ])
                
                events.append({
                    'date': current_date,
                    'event_type': event_type,
                    'event_name': self._generate_event_name(event_type),
                    'impact_magnitude': self._get_event_impact(event_type),
                    'affected_regions': self._get_random_regions(),
                    'duration_days': self._get_event_duration(event_type)
                })
            
            current_date += timedelta(days=1)
        
        return pd.DataFrame(events)
    
    def _is_holiday(self, date: datetime) -> bool:
        """Check if date is a major holiday."""
        # Simplified holiday detection
        return date.month == 12 and date.day == 25  # Christmas
    
    def _get_holiday_name(self, date: datetime) -> str:
        """Get holiday name for date."""
        if date.month == 12 and date.day == 25:
            return "Christmas"
        return "Holiday"
    
    def _get_affected_regions(self, date: datetime) -> List[str]:
        """Get regions affected by holiday."""
        return ["North America", "Europe"]  # Simplified
    
    def _get_event_duration(self, event_type: str) -> int:
        """Get typical duration for event type."""
        durations = {
            'Holiday': 3,
            'Weather Disruption': 2,
            'Economic News': 5,
            'Geopolitical Event': 10,
            'Industry News': 3,
            'Fuel Price Shock': 7
        }
        return durations.get(event_type, 1)
    
    def _generate_event_name(self, event_type: str) -> str:
        """Generate event name based on type."""
        names = {
            'Weather Disruption': ['Hurricane', 'Blizzard', 'Fog', 'Thunderstorms'],
            'Economic News': ['GDP Report', 'Inflation Data', 'Employment Report'],
            'Geopolitical Event': ['Trade Tensions', 'Border Restrictions', 'Sanctions'],
            'Industry News': ['Merger Announcement', 'New Route Launch', 'Capacity Changes'],
            'Fuel Price Shock': ['Oil Supply Disruption', 'Refinery Issues', 'OPEC Decision']
        }
        
        event_names = names.get(event_type, ['Generic Event'])
        return self.random_state.choice(event_names)
    
    def _get_event_impact(self, event_type: str) -> float:
        """Get impact magnitude for event type."""
        impacts = {
            'Weather Disruption': (0.5, 0.8),
            'Economic News': (0.9, 1.1),
            'Geopolitical Event': (0.7, 1.3),
            'Industry News': (0.95, 1.05),
            'Fuel Price Shock': (0.8, 1.2)
        }
        
        min_impact, max_impact = impacts.get(event_type, (0.9, 1.1))
        return self.random_state.uniform(min_impact, max_impact)
    
    def _get_random_regions(self) -> List[str]:
        """Get random affected regions."""
        regions = ["North America", "Europe", "Asia Pacific", "Latin America"]
        num_regions = self.random_state.randint(1, 3)
        return self.random_state.choice(regions, num_regions, replace=False).tolist()