"""Airport and Route models for airline network simulation."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import math
from datetime import datetime, time


class AirportType(Enum):
    """Airport classification types."""
    MAJOR_HUB = "major_hub"
    REGIONAL_HUB = "regional_hub"
    FOCUS_CITY = "focus_city"
    DESTINATION = "destination"
    CARGO_HUB = "cargo_hub"


class RouteType(Enum):
    """Route classification types."""
    DOMESTIC = "domestic"
    INTERNATIONAL = "international"
    TRANSCONTINENTAL = "transcontinental"
    REGIONAL = "regional"


@dataclass
class Airport:
    """Represents an airport with operational characteristics."""
    
    # Basic Information
    iata_code: str = ""
    icao_code: str = ""
    name: str = ""
    city: str = ""
    country: str = ""
    
    # Geographic Information
    latitude: float = 0.0
    longitude: float = 0.0
    elevation_meters: float = 0.0
    timezone: str = "UTC"
    
    # Airport Classification
    airport_type: AirportType = AirportType.DESTINATION
    
    # Operational Characteristics
    runway_count: int = 1
    max_aircraft_size: str = "narrow_body"  # narrow_body, wide_body, super_jumbo
    slots_per_hour: int = 20
    operating_hours_start: time = field(default_factory=lambda: time(5, 0))
    operating_hours_end: time = field(default_factory=lambda: time(23, 0))
    
    # Cost Structure
    landing_fee_base: float = 1000.0  # Base landing fee
    landing_fee_per_tonne: float = 5.0  # Per tonne of aircraft weight
    terminal_fee_per_passenger: float = 15.0
    ground_handling_fee_per_flight: float = 500.0
    fuel_price_per_liter: float = 0.85
    
    # Passenger Characteristics
    annual_passengers: int = 1000000
    business_passenger_ratio: float = 0.25
    connecting_passenger_ratio: float = 0.30
    
    # Market Information
    catchment_population: int = 500000
    average_income_usd: float = 50000.0
    tourism_index: float = 1.0  # 1.0 = average, >1.0 = high tourism
    business_index: float = 1.0  # 1.0 = average, >1.0 = high business travel
    
    # Operational Constraints
    slot_restricted: bool = False
    noise_restrictions: bool = False
    curfew_start: Optional[time] = None
    curfew_end: Optional[time] = None
    weather_delay_factor: float = 1.0  # 1.0 = average, >1.0 = more delays
    
    def calculate_distance_to(self, other_airport: 'Airport') -> float:
        """Calculate great circle distance to another airport in kilometers."""
        # Haversine formula
        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(other_airport.latitude), math.radians(other_airport.longitude)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth's radius in kilometers
        earth_radius = 6371.0
        return earth_radius * c
    
    def calculate_flight_time(self, other_airport: 'Airport', 
                            cruise_speed_kmh: float = 850.0) -> float:
        """Calculate estimated flight time to another airport in hours."""
        distance = self.calculate_distance_to(other_airport)
        # Add 30 minutes for taxi, takeoff, and landing
        return (distance / cruise_speed_kmh) + 0.5
    
    def calculate_airport_fees(self, aircraft_weight_tonnes: float, 
                             passenger_count: int) -> Dict[str, float]:
        """Calculate total airport fees for a flight."""
        fees = {
            "landing_fee": self.landing_fee_base + (aircraft_weight_tonnes * self.landing_fee_per_tonne),
            "terminal_fee": passenger_count * self.terminal_fee_per_passenger,
            "ground_handling_fee": self.ground_handling_fee_per_flight
        }
        fees["total_fees"] = sum(fees.values())
        return fees
    
    def is_operational_at_time(self, check_time: time) -> bool:
        """Check if airport is operational at given time."""
        # Handle curfew restrictions
        if self.curfew_start and self.curfew_end:
            if self.curfew_start <= self.curfew_end:
                # Same day curfew
                if self.curfew_start <= check_time <= self.curfew_end:
                    return False
            else:
                # Overnight curfew
                if check_time >= self.curfew_start or check_time <= self.curfew_end:
                    return False
        
        # Check operating hours
        return self.operating_hours_start <= check_time <= self.operating_hours_end
    
    @property
    def daily_slot_capacity(self) -> int:
        """Calculate total daily slot capacity."""
        operating_hours = (
            datetime.combine(datetime.today(), self.operating_hours_end) -
            datetime.combine(datetime.today(), self.operating_hours_start)
        ).total_seconds() / 3600
        
        return int(operating_hours * self.slots_per_hour)


@dataclass
class Route:
    """Represents a flight route between two airports."""
    
    # Route Identification
    origin: Airport = field(default_factory=Airport)
    destination: Airport = field(default_factory=Airport)
    route_type: RouteType = RouteType.DOMESTIC
    
    # Route Characteristics
    distance_km: float = field(init=False)
    flight_time_hours: float = field(init=False)
    
    # Market Characteristics
    annual_demand: int = 100000  # Annual passenger demand
    seasonality_factor: Dict[int, float] = field(default_factory=lambda: {
        1: 0.8, 2: 0.7, 3: 0.9, 4: 1.0, 5: 1.1, 6: 1.3,
        7: 1.4, 8: 1.3, 9: 1.1, 10: 1.0, 11: 0.9, 12: 1.2
    })
    
    # Competition
    competitor_count: int = 2
    market_concentration: float = 0.4  # HHI-like measure
    
    # Demand Elasticity
    price_elasticity: float = -1.2  # Price elasticity of demand
    income_elasticity: float = 1.5   # Income elasticity of demand
    
    # Operational Factors
    average_load_factor: float = 0.80
    frequency_per_day: int = 2
    
    # Revenue Characteristics
    average_fare_economy: float = 300.0
    average_fare_business: float = 1200.0
    business_class_ratio: float = 0.15
    
    # Cost Factors
    fuel_cost_factor: float = 1.0  # Multiplier for base fuel costs
    airport_fee_factor: float = 1.0  # Multiplier for airport fees
    
    def __post_init__(self):
        """Calculate derived fields after initialization."""
        self.distance_km = self.origin.calculate_distance_to(self.destination)
        self.flight_time_hours = self.origin.calculate_flight_time(self.destination)
        
        # Determine route type based on countries
        if self.origin.country != self.destination.country:
            if self.distance_km > 5000:
                self.route_type = RouteType.TRANSCONTINENTAL
            else:
                self.route_type = RouteType.INTERNATIONAL
        else:
            if self.distance_km > 2000:
                self.route_type = RouteType.TRANSCONTINENTAL
            elif self.distance_km < 500:
                self.route_type = RouteType.REGIONAL
            else:
                self.route_type = RouteType.DOMESTIC
    
    @property
    def route_code(self) -> str:
        """Generate a unique route code."""
        return f"{self.origin.iata_code}-{self.destination.iata_code}"
    
    @property
    def reverse_route_code(self) -> str:
        """Generate reverse route code."""
        return f"{self.destination.iata_code}-{self.origin.iata_code}"
    
    def get_seasonal_demand(self, month: int) -> int:
        """Get demand for a specific month considering seasonality."""
        base_monthly_demand = self.annual_demand / 12
        seasonal_factor = self.seasonality_factor.get(month, 1.0)
        return int(base_monthly_demand * seasonal_factor)
    
    def calculate_route_economics(self, aircraft_seats: int, 
                                fuel_price_per_liter: float) -> Dict[str, float]:
        """Calculate basic route economics."""
        # Revenue calculation
        economy_passengers = aircraft_seats * (1 - self.business_class_ratio)
        business_passengers = aircraft_seats * self.business_class_ratio
        
        revenue_per_flight = (
            economy_passengers * self.average_fare_economy +
            business_passengers * self.average_fare_business
        ) * self.average_load_factor
        
        # Basic cost estimation (simplified)
        fuel_cost_estimate = self.distance_km * 3.5 * fuel_price_per_liter * self.fuel_cost_factor
        airport_fees_estimate = (self.origin.landing_fee_base + 
                               self.destination.landing_fee_base) * self.airport_fee_factor
        
        total_cost_estimate = fuel_cost_estimate + airport_fees_estimate
        
        return {
            "revenue_per_flight": revenue_per_flight,
            "cost_per_flight": total_cost_estimate,
            "profit_per_flight": revenue_per_flight - total_cost_estimate,
            "revenue_per_passenger": revenue_per_flight / (aircraft_seats * self.average_load_factor),
            "cost_per_passenger": total_cost_estimate / (aircraft_seats * self.average_load_factor),
            "load_factor": self.average_load_factor,
            "passengers_per_flight": aircraft_seats * self.average_load_factor
        }
    
    def estimate_demand_response(self, price_change_percent: float) -> float:
        """Estimate demand response to price changes using elasticity."""
        return 1 + (self.price_elasticity * price_change_percent / 100)
    
    def is_viable_route(self, min_load_factor: float = 0.6, 
                       min_frequency: int = 1) -> bool:
        """Check if route meets viability criteria."""
        return (self.average_load_factor >= min_load_factor and 
                self.frequency_per_day >= min_frequency and
                self.annual_demand > 10000)
    
    def get_competition_intensity(self) -> str:
        """Classify competition intensity on the route."""
        if self.competitor_count <= 1:
            return "monopoly"
        elif self.competitor_count <= 2:
            return "duopoly"
        elif self.competitor_count <= 4:
            return "moderate"
        else:
            return "intense"
    
    def calculate_market_potential(self) -> Dict[str, float]:
        """Calculate market potential metrics."""
        catchment_score = (
            (self.origin.catchment_population + self.destination.catchment_population) / 2000000
        )
        
        income_score = (
            (self.origin.average_income_usd + self.destination.average_income_usd) / 100000
        )
        
        tourism_score = (
            (self.origin.tourism_index + self.destination.tourism_index) / 2
        )
        
        business_score = (
            (self.origin.business_index + self.destination.business_index) / 2
        )
        
        # Distance penalty for very short or very long routes
        distance_score = 1.0
        if self.distance_km < 300:
            distance_score = 0.7  # Too short, ground transport competition
        elif self.distance_km > 8000:
            distance_score = 0.8  # Very long, limited market
        
        overall_potential = (
            catchment_score * 0.3 +
            income_score * 0.25 +
            tourism_score * 0.2 +
            business_score * 0.15 +
            distance_score * 0.1
        )
        
        return {
            "catchment_score": catchment_score,
            "income_score": income_score,
            "tourism_score": tourism_score,
            "business_score": business_score,
            "distance_score": distance_score,
            "overall_potential": overall_potential,
            "market_size_estimate": int(overall_potential * 200000)  # Estimated annual PAX
        }