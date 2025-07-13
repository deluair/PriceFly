"""Synthetic data generation engine for airline pricing simulation."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import random
import json
from pathlib import Path

from ..models.aircraft import Aircraft, Fleet, AircraftType, AircraftStatus
from ..models.airport import Airport, Route, AirportType, RouteType
from ..models.airline import Airline, BookingClass, AirlineType, AllianceType
from ..models.passenger import Passenger, CustomerSegment, TravelPurpose, LoyaltyTier


@dataclass
class DataGenerationConfig:
    """Configuration for synthetic data generation."""
    
    # Scale parameters
    num_airports: int = 200
    num_airlines: int = 15
    num_aircraft_per_airline: int = 50
    num_routes_per_airline: int = 100
    num_passengers_per_day: int = 10000
    
    # Geographic distribution
    regions: List[str] = None
    countries: List[str] = None
    
    # Time parameters
    simulation_start_date: datetime = None
    simulation_duration_days: int = 365
    
    # Market parameters
    market_concentration: float = 0.3  # HHI-like measure
    competition_intensity: float = 0.7
    
    # Economic parameters
    base_fuel_price: float = 0.85  # USD per liter
    fuel_volatility: float = 0.15
    economic_growth_rate: float = 0.02
    
    def __post_init__(self):
        if self.regions is None:
            self.regions = ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East", "Africa"]
        
        if self.countries is None:
            self.countries = [
                "United States", "Canada", "United Kingdom", "Germany", "France", "Spain",
                "China", "Japan", "India", "Australia", "Brazil", "Mexico", "UAE", "Singapore"
            ]
        
        if self.simulation_start_date is None:
            self.simulation_start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)


class SyntheticDataEngine:
    """Comprehensive synthetic data generation engine for airline simulation."""
    
    def __init__(self, config: DataGenerationConfig = None):
        self.config = config or DataGenerationConfig()
        self.random_state = np.random.RandomState(42)  # For reproducibility
        
        # Generated data storage
        self.airports: List[Airport] = []
        self.airlines: List[Airline] = []
        self.routes: List[Route] = []
        self.aircraft: List[Aircraft] = []
        self.customer_segments: List[CustomerSegment] = []
        
        # Reference data
        self.airport_codes = set()
        self.airline_codes = set()
        
    def generate_all_data(self) -> Dict[str, Any]:
        """Generate complete synthetic dataset."""
        print("Generating synthetic airline industry data...")
        
        # Generate in dependency order
        self.generate_airports()
        self.generate_customer_segments()
        self.generate_airlines()
        self.generate_aircraft()
        self.generate_routes()
        
        # Generate time-series data
        fuel_prices = self.generate_fuel_price_series()
        economic_indicators = self.generate_economic_indicators()
        demand_patterns = self.generate_demand_patterns()
        
        return {
            "airports": self.airports,
            "airlines": self.airlines,
            "routes": self.routes,
            "aircraft": self.aircraft,
            "customer_segments": self.customer_segments,
            "fuel_prices": fuel_prices,
            "economic_indicators": economic_indicators,
            "demand_patterns": demand_patterns,
            "generation_metadata": {
                "generated_at": datetime.now().isoformat(),
                "config": self.config.__dict__,
                "total_airports": len(self.airports),
                "total_airlines": len(self.airlines),
                "total_routes": len(self.routes),
                "total_aircraft": len(self.aircraft)
            }
        }
    
    def generate_airports(self) -> List[Airport]:
        """Generate realistic airport data."""
        print(f"Generating {self.config.num_airports} airports...")
        
        # Major hub airports (real data for realism)
        major_hubs = [
            {"iata": "ATL", "name": "Hartsfield-Jackson Atlanta International", "city": "Atlanta", "country": "United States", "lat": 33.6407, "lon": -84.4277, "pax": 110000000},
            {"iata": "LAX", "name": "Los Angeles International", "city": "Los Angeles", "country": "United States", "lat": 33.9425, "lon": -118.4081, "pax": 88000000},
            {"iata": "ORD", "name": "O'Hare International", "city": "Chicago", "country": "United States", "lat": 41.9742, "lon": -87.9073, "pax": 84000000},
            {"iata": "LHR", "name": "London Heathrow", "city": "London", "country": "United Kingdom", "lat": 51.4700, "lon": -0.4543, "pax": 80000000},
            {"iata": "HND", "name": "Tokyo Haneda", "city": "Tokyo", "country": "Japan", "lat": 35.5494, "lon": 139.7798, "pax": 85000000},
            {"iata": "CDG", "name": "Charles de Gaulle", "city": "Paris", "country": "France", "lat": 49.0097, "lon": 2.5479, "pax": 76000000},
            {"iata": "DXB", "name": "Dubai International", "city": "Dubai", "country": "UAE", "lat": 25.2532, "lon": 55.3657, "pax": 89000000},
            {"iata": "FRA", "name": "Frankfurt am Main", "city": "Frankfurt", "country": "Germany", "lat": 50.0379, "lon": 8.5622, "pax": 70000000},
            {"iata": "SIN", "name": "Singapore Changi", "city": "Singapore", "country": "Singapore", "lat": 1.3644, "lon": 103.9915, "pax": 68000000},
            {"iata": "AMS", "name": "Amsterdam Schiphol", "city": "Amsterdam", "country": "Netherlands", "lat": 52.3105, "lon": 4.7683, "pax": 71000000}
        ]
        
        # Create major hub airports
        for hub_data in major_hubs:
            airport = Airport(
                iata_code=hub_data["iata"],
                icao_code=f"K{hub_data['iata']}",  # Simplified ICAO
                name=hub_data["name"],
                city=hub_data["city"],
                country=hub_data["country"],
                latitude=hub_data["lat"],
                longitude=hub_data["lon"],
                airport_type=AirportType.MAJOR_HUB,
                annual_passengers=hub_data["pax"],
                runway_count=self.random_state.randint(2, 5),
                slots_per_hour=self.random_state.randint(40, 80),
                landing_fee_base=self.random_state.uniform(2000, 5000),
                terminal_fee_per_passenger=self.random_state.uniform(20, 40),
                catchment_population=self.random_state.randint(2000000, 10000000),
                average_income_usd=self.random_state.uniform(40000, 80000),
                business_index=self.random_state.uniform(1.2, 2.0),
                tourism_index=self.random_state.uniform(1.0, 1.8)
            )
            self.airports.append(airport)
            self.airport_codes.add(hub_data["iata"])
        
        # Generate additional airports
        cities = [
            "New York", "Miami", "San Francisco", "Boston", "Seattle", "Denver", "Phoenix",
            "Madrid", "Rome", "Barcelona", "Munich", "Zurich", "Vienna", "Stockholm",
            "Beijing", "Shanghai", "Hong Kong", "Seoul", "Bangkok", "Kuala Lumpur",
            "Sydney", "Melbourne", "Auckland", "Toronto", "Vancouver", "Montreal",
            "São Paulo", "Rio de Janeiro", "Buenos Aires", "Lima", "Bogotá",
            "Cairo", "Johannesburg", "Lagos", "Nairobi", "Casablanca"
        ]
        
        remaining_airports = self.config.num_airports - len(major_hubs)
        
        for i in range(remaining_airports):
            # Generate unique IATA code
            while True:
                iata_code = ''.join(self.random_state.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 3))
                if iata_code not in self.airport_codes:
                    self.airport_codes.add(iata_code)
                    break
            
            city = self.random_state.choice(cities)
            country = self.random_state.choice(self.config.countries)
            
            # Determine airport type based on size
            airport_types = [AirportType.REGIONAL_HUB, AirportType.FOCUS_CITY, AirportType.DESTINATION]
            weights = [0.2, 0.3, 0.5]
            airport_type = self.random_state.choice(airport_types, p=weights)
            
            # Scale characteristics based on type
            if airport_type == AirportType.REGIONAL_HUB:
                pax_range = (5000000, 25000000)
                runway_range = (2, 4)
                slots_range = (20, 50)
            elif airport_type == AirportType.FOCUS_CITY:
                pax_range = (1000000, 8000000)
                runway_range = (1, 3)
                slots_range = (10, 30)
            else:  # DESTINATION
                pax_range = (100000, 3000000)
                runway_range = (1, 2)
                slots_range = (5, 20)
            
            airport = Airport(
                iata_code=iata_code,
                icao_code=f"X{iata_code}",
                name=f"{city} {self.random_state.choice(['International', 'Airport', 'Regional'])}",
                city=city,
                country=country,
                latitude=self.random_state.uniform(-60, 70),  # Realistic latitude range
                longitude=self.random_state.uniform(-180, 180),
                airport_type=airport_type,
                annual_passengers=self.random_state.randint(*pax_range),
                runway_count=self.random_state.randint(*runway_range),
                slots_per_hour=self.random_state.randint(*slots_range),
                landing_fee_base=self.random_state.uniform(500, 3000),
                terminal_fee_per_passenger=self.random_state.uniform(10, 30),
                catchment_population=self.random_state.randint(200000, 5000000),
                average_income_usd=self.random_state.uniform(20000, 70000),
                business_index=self.random_state.uniform(0.5, 1.5),
                tourism_index=self.random_state.uniform(0.3, 2.0)
            )
            
            self.airports.append(airport)
        
        print(f"Generated {len(self.airports)} airports")
        return self.airports
    
    def generate_customer_segments(self) -> List[CustomerSegment]:
        """Generate customer segments with realistic behavioral patterns."""
        print("Generating customer segments...")
        
        segments_config = [
            {
                "name": "Business Frequent Flyer",
                "description": "High-income business travelers with low price sensitivity",
                "purpose": TravelPurpose.BUSINESS,
                "income_range": (80000, 200000),
                "price_elasticity": -0.3,
                "wtp_multiplier": 2.0,
                "price_sensitivity": 0.2,
                "booking_lead_time": (1, 30),
                "cabin_preference": {"economy": 0.2, "premium_economy": 0.3, "business": 0.5},
                "loyalty_distribution": {
                    LoyaltyTier.NONE: 0.1, LoyaltyTier.BASIC: 0.2, LoyaltyTier.SILVER: 0.3,
                    LoyaltyTier.GOLD: 0.3, LoyaltyTier.PLATINUM: 0.1
                },
                "segment_size": 0.15
            },
            {
                "name": "Leisure Price-Sensitive",
                "description": "Leisure travelers highly sensitive to price changes",
                "purpose": TravelPurpose.LEISURE,
                "income_range": (30000, 70000),
                "price_elasticity": -2.0,
                "wtp_multiplier": 0.7,
                "price_sensitivity": 0.9,
                "booking_lead_time": (30, 120),
                "cabin_preference": {"economy": 0.95, "premium_economy": 0.05, "business": 0.0},
                "loyalty_distribution": {
                    LoyaltyTier.NONE: 0.7, LoyaltyTier.BASIC: 0.25, LoyaltyTier.SILVER: 0.05,
                    LoyaltyTier.GOLD: 0.0, LoyaltyTier.PLATINUM: 0.0
                },
                "segment_size": 0.35
            },
            {
                "name": "Premium Leisure",
                "description": "Affluent leisure travelers willing to pay for comfort",
                "purpose": TravelPurpose.LEISURE,
                "income_range": (70000, 150000),
                "price_elasticity": -0.8,
                "wtp_multiplier": 1.4,
                "price_sensitivity": 0.4,
                "booking_lead_time": (14, 90),
                "cabin_preference": {"economy": 0.5, "premium_economy": 0.4, "business": 0.1},
                "loyalty_distribution": {
                    LoyaltyTier.NONE: 0.4, LoyaltyTier.BASIC: 0.3, LoyaltyTier.SILVER: 0.2,
                    LoyaltyTier.GOLD: 0.1, LoyaltyTier.PLATINUM: 0.0
                },
                "segment_size": 0.20
            },
            {
                "name": "VFR Budget",
                "description": "Visiting friends and relatives, budget-conscious",
                "purpose": TravelPurpose.VFR,
                "income_range": (25000, 60000),
                "price_elasticity": -1.5,
                "wtp_multiplier": 0.8,
                "price_sensitivity": 0.8,
                "booking_lead_time": (7, 60),
                "cabin_preference": {"economy": 0.98, "premium_economy": 0.02, "business": 0.0},
                "loyalty_distribution": {
                    LoyaltyTier.NONE: 0.8, LoyaltyTier.BASIC: 0.18, LoyaltyTier.SILVER: 0.02,
                    LoyaltyTier.GOLD: 0.0, LoyaltyTier.PLATINUM: 0.0
                },
                "segment_size": 0.15
            },
            {
                "name": "Corporate Managed",
                "description": "Corporate travelers with company booking policies",
                "purpose": TravelPurpose.BUSINESS,
                "income_range": (50000, 120000),
                "price_elasticity": -0.5,
                "wtp_multiplier": 1.3,
                "price_sensitivity": 0.3,
                "booking_lead_time": (3, 21),
                "cabin_preference": {"economy": 0.6, "premium_economy": 0.3, "business": 0.1},
                "loyalty_distribution": {
                    LoyaltyTier.NONE: 0.3, LoyaltyTier.BASIC: 0.4, LoyaltyTier.SILVER: 0.2,
                    LoyaltyTier.GOLD: 0.1, LoyaltyTier.PLATINUM: 0.0
                },
                "segment_size": 0.15
            }
        ]
        
        for config in segments_config:
            segment = CustomerSegment(
                segment_name=config["name"],
                description=config["description"],
                primary_travel_purpose=config["purpose"],
                income_range=config["income_range"],
                price_elasticity=config["price_elasticity"],
                willingness_to_pay_multiplier=config["wtp_multiplier"],
                price_sensitivity_score=config["price_sensitivity"],
                booking_lead_time_days=config["booking_lead_time"],
                cabin_class_preference=config["cabin_preference"],
                loyalty_tier_distribution=config["loyalty_distribution"],
                segment_size_percentage=config["segment_size"]
            )
            
            self.customer_segments.append(segment)
        
        print(f"Generated {len(self.customer_segments)} customer segments")
        return self.customer_segments
    
    def generate_airlines(self) -> List[Airline]:
        """Generate realistic airline data."""
        print(f"Generating {self.config.num_airlines} airlines...")
        
        # Major airlines (based on real carriers)
        major_airlines = [
            {"code": "AA", "name": "American Airlines", "country": "United States", "type": AirlineType.FULL_SERVICE, "alliance": AllianceType.ONEWORLD},
            {"code": "DL", "name": "Delta Air Lines", "country": "United States", "type": AirlineType.FULL_SERVICE, "alliance": AllianceType.SKYTEAM},
            {"code": "UA", "name": "United Airlines", "country": "United States", "type": AirlineType.FULL_SERVICE, "alliance": AllianceType.STAR_ALLIANCE},
            {"code": "LH", "name": "Lufthansa", "country": "Germany", "type": AirlineType.FULL_SERVICE, "alliance": AllianceType.STAR_ALLIANCE},
            {"code": "BA", "name": "British Airways", "country": "United Kingdom", "type": AirlineType.FULL_SERVICE, "alliance": AllianceType.ONEWORLD},
            {"code": "AF", "name": "Air France", "country": "France", "type": AirlineType.FULL_SERVICE, "alliance": AllianceType.SKYTEAM},
            {"code": "EK", "name": "Emirates", "country": "UAE", "type": AirlineType.FULL_SERVICE, "alliance": AllianceType.NONE},
            {"code": "SQ", "name": "Singapore Airlines", "country": "Singapore", "type": AirlineType.FULL_SERVICE, "alliance": AllianceType.STAR_ALLIANCE},
            {"code": "WN", "name": "Southwest Airlines", "country": "United States", "type": AirlineType.LOW_COST, "alliance": AllianceType.NONE},
            {"code": "FR", "name": "Ryanair", "country": "Ireland", "type": AirlineType.ULTRA_LOW_COST, "alliance": AllianceType.NONE}
        ]
        
        for airline_data in major_airlines[:min(len(major_airlines), self.config.num_airlines)]:
            # Select hub airports based on country
            country_airports = [a for a in self.airports if a.country == airline_data["country"]]
            if not country_airports:
                country_airports = [a for a in self.airports if a.airport_type == AirportType.MAJOR_HUB][:2]
            
            hubs = [a.iata_code for a in country_airports[:2]]
            
            airline = Airline(
                airline_code=airline_data["code"],
                icao_code=f"{airline_data['code']}L",
                airline_name=airline_data["name"],
                country=airline_data["country"],
                airline_type=airline_data["type"],
                alliance=airline_data["alliance"],
                hub_airports=hubs,
                annual_revenue_usd=self.random_state.uniform(5e9, 50e9),
                annual_passengers=self.random_state.randint(10000000, 200000000),
                load_factor=self.random_state.uniform(0.75, 0.90),
                on_time_performance=self.random_state.uniform(0.75, 0.95),
                service_quality_score=self.random_state.uniform(6.0, 9.5),
                market_share_percentage=self.random_state.uniform(2.0, 15.0),
                price_competitiveness=self.random_state.uniform(0.8, 1.2)
            )
            
            self.airlines.append(airline)
            self.airline_codes.add(airline_data["code"])
        
        # Generate additional airlines if needed
        remaining_airlines = self.config.num_airlines - len(major_airlines)
        
        airline_names = [
            "Pacific Airways", "Continental Express", "Global Wings", "Skyline Airlines",
            "Horizon Air", "Summit Airlines", "Coastal Airways", "Mountain Air",
            "Valley Express", "Metro Airlines", "Regional Connect", "City Hopper"
        ]
        
        for i in range(remaining_airlines):
            # Generate unique airline code
            while True:
                code = ''.join(self.random_state.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 2))
                if code not in self.airline_codes:
                    self.airline_codes.add(code)
                    break
            
            airline_type = self.random_state.choice([
                AirlineType.FULL_SERVICE, AirlineType.LOW_COST, 
                AirlineType.REGIONAL, AirlineType.ULTRA_LOW_COST
            ], p=[0.4, 0.3, 0.2, 0.1])
            
            country = self.random_state.choice(self.config.countries)
            name = self.random_state.choice(airline_names)
            
            # Select hubs
            potential_hubs = [a for a in self.airports 
                            if a.country == country or a.airport_type == AirportType.MAJOR_HUB]
            hubs = [a.iata_code for a in self.random_state.choice(potential_hubs, 
                                                                 min(2, len(potential_hubs)), 
                                                                 replace=False)]
            
            airline = Airline(
                airline_code=code,
                icao_code=f"{code}X",
                airline_name=name,
                country=country,
                airline_type=airline_type,
                alliance=self.random_state.choice(list(AllianceType), p=[0.3, 0.2, 0.2, 0.25, 0.05]),
                hub_airports=hubs,
                annual_revenue_usd=self.random_state.uniform(1e8, 10e9),
                annual_passengers=self.random_state.randint(1000000, 50000000),
                load_factor=self.random_state.uniform(0.70, 0.88),
                on_time_performance=self.random_state.uniform(0.70, 0.90),
                service_quality_score=self.random_state.uniform(5.0, 8.5),
                market_share_percentage=self.random_state.uniform(0.5, 5.0),
                price_competitiveness=self.random_state.uniform(0.7, 1.3)
            )
            
            self.airlines.append(airline)
        
        print(f"Generated {len(self.airlines)} airlines")
        return self.airlines
    
    def generate_aircraft(self) -> List[Aircraft]:
        """Generate aircraft fleet data for all airlines."""
        print("Generating aircraft fleets...")
        
        # Aircraft type templates
        aircraft_templates = [
            {"model": "Boeing 737-800", "manufacturer": "Boeing", "type": AircraftType.NARROW_BODY,
             "seats": 180, "range": 5765, "speed": 842, "fuel_burn": 2400},
            {"model": "Airbus A320", "manufacturer": "Airbus", "type": AircraftType.NARROW_BODY,
             "seats": 180, "range": 6150, "speed": 840, "fuel_burn": 2200},
            {"model": "Boeing 777-300ER", "manufacturer": "Boeing", "type": AircraftType.WIDE_BODY,
             "seats": 350, "range": 13649, "speed": 905, "fuel_burn": 8500},
            {"model": "Airbus A350-900", "manufacturer": "Airbus", "type": AircraftType.WIDE_BODY,
             "seats": 325, "range": 15000, "speed": 903, "fuel_burn": 7500},
            {"model": "Embraer E190", "manufacturer": "Embraer", "type": AircraftType.REGIONAL,
             "seats": 100, "range": 4537, "speed": 829, "fuel_burn": 1200},
            {"model": "Boeing 787-9", "manufacturer": "Boeing", "type": AircraftType.WIDE_BODY,
             "seats": 290, "range": 14140, "speed": 903, "fuel_burn": 6800}
        ]
        
        for airline in self.airlines:
            fleet_size = self.random_state.randint(10, self.config.num_aircraft_per_airline)
            
            # Determine fleet composition based on airline type
            if airline.airline_type == AirlineType.REGIONAL:
                preferred_types = [AircraftType.REGIONAL, AircraftType.NARROW_BODY]
                type_weights = [0.7, 0.3]
            elif airline.airline_type in [AirlineType.LOW_COST, AirlineType.ULTRA_LOW_COST]:
                preferred_types = [AircraftType.NARROW_BODY]
                type_weights = [1.0]
            else:  # Full service
                preferred_types = [AircraftType.NARROW_BODY, AircraftType.WIDE_BODY, AircraftType.REGIONAL]
                type_weights = [0.6, 0.3, 0.1]
            
            for _ in range(fleet_size):
                # Select aircraft type
                aircraft_type = self.random_state.choice(preferred_types, p=type_weights)
                
                # Select template
                type_templates = [t for t in aircraft_templates if t["type"] == aircraft_type]
                template = self.random_state.choice(type_templates)
                
                # Add some variation to the template
                seats_variation = self.random_state.randint(-20, 21)
                total_seats = max(50, template["seats"] + seats_variation)
                
                # Distribute seats by class
                if airline.airline_type == AirlineType.FULL_SERVICE:
                    if aircraft_type == AircraftType.WIDE_BODY:
                        business_ratio = 0.15
                        premium_ratio = 0.10
                    else:
                        business_ratio = 0.10
                        premium_ratio = 0.05
                elif airline.airline_type == AirlineType.LOW_COST:
                    business_ratio = 0.05
                    premium_ratio = 0.0
                else:  # Ultra low cost or regional
                    business_ratio = 0.0
                    premium_ratio = 0.0
                
                business_seats = int(total_seats * business_ratio)
                premium_seats = int(total_seats * premium_ratio)
                economy_seats = total_seats - business_seats - premium_seats
                
                aircraft = Aircraft(
                    model=template["model"],
                    manufacturer=template["manufacturer"],
                    aircraft_type=aircraft_type,
                    total_seats=total_seats,
                    business_seats=business_seats,
                    premium_economy_seats=premium_seats,
                    economy_seats=economy_seats,
                    max_range_km=template["range"] + self.random_state.randint(-500, 501),
                    cruise_speed_kmh=template["speed"] + self.random_state.randint(-50, 51),
                    fuel_burn_per_hour=template["fuel_burn"] + self.random_state.randint(-200, 201),
                    crew_cost_per_hour=self.random_state.uniform(200, 800),
                    maintenance_cost_per_hour=self.random_state.uniform(300, 1200),
                    insurance_cost_per_hour=self.random_state.uniform(50, 200),
                    depreciation_per_hour=self.random_state.uniform(100, 500),
                    manufacture_year=self.random_state.randint(2010, 2024),
                    utilization_hours_per_day=self.random_state.uniform(8, 14)
                )
                
                airline.fleet.add_aircraft(aircraft)
                self.aircraft.append(aircraft)
        
        print(f"Generated {len(self.aircraft)} aircraft across all airlines")
        return self.aircraft
    
    def generate_routes(self) -> List[Route]:
        """Generate route network for all airlines."""
        print("Generating route networks...")
        
        for airline in self.airlines:
            routes_to_generate = min(self.config.num_routes_per_airline, 
                                   len(self.airports) * (len(self.airports) - 1) // 2)
            
            # Start with hub routes
            hub_airports = [a for a in self.airports if a.iata_code in airline.hub_airports]
            
            routes_generated = 0
            
            # Generate hub-to-hub routes
            for i, hub1 in enumerate(hub_airports):
                for hub2 in hub_airports[i+1:]:
                    if routes_generated >= routes_to_generate:
                        break
                    
                    route = self._create_route(hub1, hub2, airline)
                    airline.add_route(route)
                    self.routes.append(route)
                    routes_generated += 1
            
            # Generate hub-to-spoke routes
            for hub in hub_airports:
                potential_destinations = [a for a in self.airports 
                                        if a.iata_code not in airline.hub_airports 
                                        and a != hub]
                
                # Select destinations based on airport importance and distance
                destinations = self.random_state.choice(
                    potential_destinations,
                    min(len(potential_destinations), routes_to_generate - routes_generated),
                    replace=False
                )
                
                for dest in destinations:
                    if routes_generated >= routes_to_generate:
                        break
                    
                    route = self._create_route(hub, dest, airline)
                    airline.add_route(route)
                    self.routes.append(route)
                    routes_generated += 1
            
            # Generate some point-to-point routes
            remaining_routes = routes_to_generate - routes_generated
            for _ in range(remaining_routes):
                origin = self.random_state.choice(self.airports)
                destination = self.random_state.choice(
                    [a for a in self.airports if a != origin]
                )
                
                # Check if route already exists
                existing_route = airline.find_route(origin.iata_code, destination.iata_code)
                if existing_route is None:
                    route = self._create_route(origin, destination, airline)
                    airline.add_route(route)
                    self.routes.append(route)
        
        print(f"Generated {len(self.routes)} routes across all airlines")
        return self.routes
    
    def _create_route(self, origin: Airport, destination: Airport, airline: Airline) -> Route:
        """Create a route with realistic characteristics."""
        distance = origin.calculate_distance_to(destination)
        
        # Determine route type
        if origin.country != destination.country:
            if distance > 5000:
                route_type = RouteType.TRANSCONTINENTAL
            else:
                route_type = RouteType.INTERNATIONAL
        else:
            if distance > 2000:
                route_type = RouteType.TRANSCONTINENTAL
            elif distance < 500:
                route_type = RouteType.REGIONAL
            else:
                route_type = RouteType.DOMESTIC
        
        # Calculate demand based on airport sizes and distance
        demand_factor = (
            (origin.annual_passengers + destination.annual_passengers) / 2000000
        ) * (1 / (1 + distance / 5000))  # Distance decay
        
        annual_demand = int(demand_factor * self.random_state.uniform(50000, 500000))
        
        # Frequency based on demand and distance
        if route_type == RouteType.REGIONAL:
            frequency = self.random_state.randint(2, 8)
        elif route_type == RouteType.DOMESTIC:
            frequency = self.random_state.randint(1, 4)
        else:  # International/Transcontinental
            frequency = self.random_state.randint(1, 2)
        
        # Pricing based on distance and route type
        base_fare = 50 + (distance * 0.1)  # $0.10 per km base
        
        if route_type == RouteType.INTERNATIONAL:
            base_fare *= 1.3
        elif route_type == RouteType.TRANSCONTINENTAL:
            base_fare *= 1.5
        
        # Adjust for airline type
        if airline.airline_type == AirlineType.LOW_COST:
            base_fare *= 0.8
        elif airline.airline_type == AirlineType.ULTRA_LOW_COST:
            base_fare *= 0.6
        
        route = Route(
            origin=origin,
            destination=destination,
            route_type=route_type,
            annual_demand=annual_demand,
            frequency_per_day=frequency,
            average_fare_economy=base_fare,
            average_fare_business=base_fare * 3.5,
            business_class_ratio=0.15 if airline.airline_type == AirlineType.FULL_SERVICE else 0.05,
            competitor_count=self.random_state.randint(1, 5),
            price_elasticity=self.random_state.uniform(-2.0, -0.5),
            average_load_factor=self.random_state.uniform(0.65, 0.85)
        )
        
        return route
    
    def generate_fuel_price_series(self) -> pd.DataFrame:
        """Generate realistic fuel price time series."""
        print("Generating fuel price series...")
        
        dates = pd.date_range(
            start=self.config.simulation_start_date,
            periods=self.config.simulation_duration_days,
            freq='D'
        )
        
        # Generate fuel prices with volatility and trends
        prices = []
        current_price = self.config.base_fuel_price
        
        for i, date in enumerate(dates):
            # Add random walk with mean reversion
            daily_change = self.random_state.normal(0, self.config.fuel_volatility * 0.02)
            mean_reversion = (self.config.base_fuel_price - current_price) * 0.001
            
            # Add seasonal effects (higher in summer)
            seasonal_effect = 0.05 * np.sin(2 * np.pi * date.dayofyear / 365.25)
            
            current_price += daily_change + mean_reversion + seasonal_effect
            current_price = max(0.3, current_price)  # Floor price
            
            prices.append(current_price)
        
        fuel_df = pd.DataFrame({
            'date': dates,
            'fuel_price_usd_per_liter': prices
        })
        
        return fuel_df
    
    def generate_economic_indicators(self) -> pd.DataFrame:
        """Generate economic indicators time series."""
        print("Generating economic indicators...")
        
        dates = pd.date_range(
            start=self.config.simulation_start_date,
            periods=self.config.simulation_duration_days,
            freq='D'
        )
        
        # Generate correlated economic indicators
        gdp_growth = []
        inflation_rate = []
        unemployment_rate = []
        consumer_confidence = []
        
        base_gdp = self.config.economic_growth_rate
        base_inflation = 0.025  # 2.5%
        base_unemployment = 0.05  # 5%
        base_confidence = 100
        
        for i, date in enumerate(dates):
            # GDP growth with business cycle
            cycle_effect = 0.01 * np.sin(2 * np.pi * i / (365.25 * 7))  # 7-year cycle
            gdp_growth.append(base_gdp + cycle_effect + self.random_state.normal(0, 0.005))
            
            # Inflation with some persistence
            inflation_change = self.random_state.normal(0, 0.001)
            base_inflation += inflation_change
            inflation_rate.append(max(0, base_inflation))
            
            # Unemployment (inversely related to GDP growth)
            unemployment_change = -gdp_growth[-1] * 0.5 + self.random_state.normal(0, 0.002)
            base_unemployment += unemployment_change
            unemployment_rate.append(max(0.02, min(0.15, base_unemployment)))
            
            # Consumer confidence
            confidence_change = gdp_growth[-1] * 10 - unemployment_rate[-1] * 5
            base_confidence += confidence_change + self.random_state.normal(0, 2)
            consumer_confidence.append(max(50, min(150, base_confidence)))
        
        econ_df = pd.DataFrame({
            'date': dates,
            'gdp_growth_rate': gdp_growth,
            'inflation_rate': inflation_rate,
            'unemployment_rate': unemployment_rate,
            'consumer_confidence_index': consumer_confidence
        })
        
        return econ_df
    
    def generate_demand_patterns(self) -> pd.DataFrame:
        """Generate demand patterns for routes."""
        print("Generating demand patterns...")
        
        demand_data = []
        
        for route in self.routes[:100]:  # Sample of routes
            for month in range(1, 13):
                base_demand = route.annual_demand / 12
                seasonal_factor = route.seasonality_factor.get(month, 1.0)
                monthly_demand = int(base_demand * seasonal_factor)
                
                demand_data.append({
                    'route_code': route.route_code,
                    'month': month,
                    'base_demand': base_demand,
                    'seasonal_factor': seasonal_factor,
                    'monthly_demand': monthly_demand,
                    'route_type': route.route_type.value
                })
        
        return pd.DataFrame(demand_data)
    
    def save_data(self, output_dir: str) -> None:
        """Save generated data to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving data to {output_path}...")
        
        # Generate all data
        data = self.generate_all_data()
        
        # Save to JSON files
        with open(output_path / 'airports.json', 'w') as f:
            json.dump([airport.__dict__ for airport in data['airports']], f, indent=2, default=str)
        
        with open(output_path / 'airlines.json', 'w') as f:
            airline_data = []
            for airline in data['airlines']:
                airline_dict = airline.__dict__.copy()
                airline_dict['fleet'] = [aircraft.__dict__ for aircraft in airline.fleet.aircraft]
                airline_dict['routes_operated'] = [route.__dict__ for route in airline.routes_operated]
                airline_data.append(airline_dict)
            json.dump(airline_data, f, indent=2, default=str)
        
        # Save time series data as CSV
        data['fuel_prices'].to_csv(output_path / 'fuel_prices.csv', index=False)
        data['economic_indicators'].to_csv(output_path / 'economic_indicators.csv', index=False)
        data['demand_patterns'].to_csv(output_path / 'demand_patterns.csv', index=False)
        
        # Save metadata
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(data['generation_metadata'], f, indent=2, default=str)
        
        print("Data saved successfully!")