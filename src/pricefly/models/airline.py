"""Airline and Booking Class models for airline operations simulation."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum
from datetime import datetime
import uuid

from .aircraft import Fleet
from .airport import Airport, Route


class AirlineType(Enum):
    """Airline business model classification."""
    FULL_SERVICE = "full_service"
    LOW_COST = "low_cost"
    ULTRA_LOW_COST = "ultra_low_cost"
    REGIONAL = "regional"
    CARGO = "cargo"
    CHARTER = "charter"


class AllianceType(Enum):
    """Airline alliance memberships."""
    STAR_ALLIANCE = "star_alliance"
    ONEWORLD = "oneworld"
    SKYTEAM = "skyteam"
    NONE = "none"
    OTHER = "other"


class CabinClass(Enum):
    """Aircraft cabin classes."""
    FIRST = "first"
    BUSINESS = "business"
    PREMIUM_ECONOMY = "premium_economy"
    ECONOMY = "economy"


@dataclass
class BookingClass:
    """Represents a booking class with fare rules and restrictions."""
    
    # Basic Information
    booking_code: str = ""  # Single letter code (Y, B, M, H, etc.)
    class_name: str = ""
    cabin_class: CabinClass = CabinClass.ECONOMY
    
    # Fare Information
    base_fare: float = 0.0
    fare_basis: str = ""  # Fare basis code
    
    # Restrictions
    advance_purchase_days: int = 0  # Minimum days in advance
    minimum_stay_days: int = 0
    maximum_stay_days: int = 365
    saturday_night_stay_required: bool = False
    
    # Flexibility
    refundable: bool = False
    changeable: bool = True
    change_fee: float = 0.0
    cancellation_fee: float = 0.0
    
    # Availability Control
    seats_available: int = 0
    total_seats_allocated: int = 0
    booking_limit: int = 0  # Maximum bookings allowed
    
    # Revenue Management
    protection_level: int = 0  # Seats protected for higher classes
    authorization_level: int = 1  # Required authorization level
    
    # Seasonal Adjustments
    seasonal_multipliers: Dict[int, float] = field(default_factory=lambda: {
        month: 1.0 for month in range(1, 13)
    })
    
    # Ancillary Services Included
    checked_baggage_included: int = 0  # Number of bags included
    seat_selection_included: bool = False
    meal_included: bool = False
    priority_boarding: bool = False
    lounge_access: bool = False
    
    def __post_init__(self):
        """Initialize derived fields."""
        if not self.class_name:
            self.class_name = f"{self.cabin_class.value.title()} {self.booking_code}"
        
        if self.total_seats_allocated == 0:
            self.total_seats_allocated = self.seats_available
    
    @property
    def availability_rate(self) -> float:
        """Calculate current availability rate."""
        if self.total_seats_allocated == 0:
            return 0.0
        return self.seats_available / self.total_seats_allocated
    
    @property
    def is_available(self) -> bool:
        """Check if booking class has available seats."""
        return self.seats_available > 0
    
    def calculate_fare_with_restrictions(self, booking_date: datetime, 
                                       travel_date: datetime) -> Optional[float]:
        """Calculate fare considering booking restrictions."""
        days_in_advance = (travel_date - booking_date).days
        
        # Check advance purchase requirement
        if days_in_advance < self.advance_purchase_days:
            return None  # Not eligible
        
        # Apply seasonal multiplier
        month = travel_date.month
        seasonal_multiplier = self.seasonal_multipliers.get(month, 1.0)
        
        return self.base_fare * seasonal_multiplier
    
    def book_seat(self, quantity: int = 1) -> bool:
        """Attempt to book seats in this class."""
        if self.seats_available >= quantity:
            self.seats_available -= quantity
            return True
        return False
    
    def release_seat(self, quantity: int = 1) -> None:
        """Release seats back to inventory."""
        self.seats_available = min(
            self.seats_available + quantity,
            self.total_seats_allocated
        )
    
    def calculate_change_cost(self, new_fare: float) -> Dict[str, float]:
        """Calculate cost of changing to a different fare."""
        fare_difference = max(0, new_fare - self.base_fare)
        total_change_cost = self.change_fee + fare_difference
        
        return {
            "change_fee": self.change_fee,
            "fare_difference": fare_difference,
            "total_cost": total_change_cost
        }


@dataclass
class Airline:
    """Represents an airline with operational characteristics."""
    
    # Basic Information
    airline_code: str = ""  # IATA code (e.g., "AA")
    icao_code: str = ""     # ICAO code (e.g., "AAL")
    airline_name: str = ""
    country: str = ""
    
    # Business Model
    airline_type: AirlineType = AirlineType.FULL_SERVICE
    alliance: AllianceType = AllianceType.NONE
    
    # Network Information
    hub_airports: List[str] = field(default_factory=list)  # IATA codes
    focus_cities: List[str] = field(default_factory=list)
    destinations_served: Set[str] = field(default_factory=set)
    
    # Fleet
    fleet: Fleet = field(default_factory=Fleet)
    
    # Financial Information
    annual_revenue_usd: float = 1000000000.0  # $1B default
    market_capitalization_usd: float = 5000000000.0  # $5B default
    debt_to_equity_ratio: float = 0.6
    
    # Operational Metrics
    annual_passengers: int = 10000000  # 10M default
    load_factor: float = 0.82
    on_time_performance: float = 0.85
    
    # Pricing Strategy
    pricing_strategy: str = "dynamic"  # dynamic, fixed, hybrid
    price_competitiveness: float = 1.0  # 1.0 = market average
    
    # Service Levels
    service_quality_score: float = 7.5  # Out of 10
    customer_satisfaction_score: float = 7.0
    
    # Cost Structure
    cost_per_available_seat_mile: float = 0.12  # CASM in USD
    fuel_hedging_percentage: float = 0.3  # 30% hedged
    
    # Loyalty Program
    loyalty_program_name: str = ""
    loyalty_members: int = 1000000
    
    # Routes and Booking Classes
    routes_operated: List[Route] = field(default_factory=list)
    booking_classes: Dict[str, BookingClass] = field(default_factory=dict)
    
    # Competitive Position
    market_share_percentage: float = 5.0  # 5% default market share
    brand_strength_score: float = 7.0  # Out of 10
    
    def __post_init__(self):
        """Initialize derived fields and default booking classes."""
        if not self.booking_classes:
            self._create_default_booking_classes()
        
        if not self.loyalty_program_name:
            self.loyalty_program_name = f"{self.airline_name} Miles"
    
    def _create_default_booking_classes(self):
        """Create default booking class structure."""
        if self.airline_type == AirlineType.FULL_SERVICE:
            # Full service carrier booking classes
            classes = [
                ("F", "First Class", CabinClass.FIRST, 2000, True, True),
                ("J", "Business Class", CabinClass.BUSINESS, 1200, True, True),
                ("W", "Premium Economy", CabinClass.PREMIUM_ECONOMY, 600, False, True),
                ("Y", "Economy Flexible", CabinClass.ECONOMY, 400, True, True),
                ("B", "Economy Standard", CabinClass.ECONOMY, 350, False, True),
                ("M", "Economy Saver", CabinClass.ECONOMY, 300, False, False),
                ("H", "Economy Basic", CabinClass.ECONOMY, 250, False, False),
            ]
        elif self.airline_type == AirlineType.LOW_COST:
            # Low cost carrier booking classes
            classes = [
                ("Y", "Flex", CabinClass.ECONOMY, 200, False, True),
                ("M", "Standard", CabinClass.ECONOMY, 150, False, False),
                ("L", "Basic", CabinClass.ECONOMY, 100, False, False),
            ]
        else:
            # Ultra low cost carrier
            classes = [
                ("Y", "Bundle", CabinClass.ECONOMY, 120, False, True),
                ("L", "Basic", CabinClass.ECONOMY, 80, False, False),
            ]
        
        for code, name, cabin, fare, refundable, changeable in classes:
            self.booking_classes[code] = BookingClass(
                booking_code=code,
                class_name=name,
                cabin_class=cabin,
                base_fare=fare,
                refundable=refundable,
                changeable=changeable,
                change_fee=0 if changeable else 999999,
                seats_available=20,  # Default allocation
                total_seats_allocated=20
            )
    
    def add_route(self, route: Route) -> None:
        """Add a route to the airline's network."""
        self.routes_operated.append(route)
        self.destinations_served.add(route.origin.iata_code)
        self.destinations_served.add(route.destination.iata_code)
    
    def get_routes_from_airport(self, airport_code: str) -> List[Route]:
        """Get all routes departing from a specific airport."""
        return [r for r in self.routes_operated 
                if r.origin.iata_code == airport_code]
    
    def get_routes_to_airport(self, airport_code: str) -> List[Route]:
        """Get all routes arriving at a specific airport."""
        return [r for r in self.routes_operated 
                if r.destination.iata_code == airport_code]
    
    def find_route(self, origin: str, destination: str) -> Optional[Route]:
        """Find a specific route between two airports."""
        for route in self.routes_operated:
            if (route.origin.iata_code == origin and 
                route.destination.iata_code == destination):
                return route
        return None
    
    def get_available_booking_classes(self, route: Route) -> List[BookingClass]:
        """Get available booking classes for a specific route."""
        return [bc for bc in self.booking_classes.values() if bc.is_available]
    
    def calculate_route_profitability(self, route: Route) -> Dict[str, float]:
        """Calculate profitability metrics for a route."""
        # Get capable aircraft for this route
        capable_aircraft = self.fleet.get_capable_aircraft(route.distance_km)
        
        if not capable_aircraft:
            return {"profitable": False, "reason": "No capable aircraft"}
        
        # Use most efficient aircraft
        aircraft = min(capable_aircraft, 
                      key=lambda a: a.fuel_efficiency_per_seat_km)
        
        # Calculate costs and revenues
        trip_costs = aircraft.calculate_trip_cost(route.distance_km, 0.85)  # $0.85/liter fuel
        route_economics = route.calculate_route_economics(
            aircraft.total_seats, 0.85
        )
        
        profit_per_flight = route_economics["profit_per_flight"]
        annual_flights = route.frequency_per_day * 365
        annual_profit = profit_per_flight * annual_flights
        
        return {
            "profitable": profit_per_flight > 0,
            "profit_per_flight": profit_per_flight,
            "annual_profit": annual_profit,
            "load_factor_breakeven": trip_costs["total_cost"] / (
                aircraft.total_seats * route.average_fare_economy
            ) if route.average_fare_economy > 0 else 1.0,
            "aircraft_used": aircraft.model
        }
    
    def optimize_booking_class_allocation(self, route: Route, 
                                        aircraft_seats: int) -> Dict[str, int]:
        """Optimize seat allocation across booking classes."""
        allocation = {}
        
        if self.airline_type == AirlineType.FULL_SERVICE:
            # Full service allocation
            allocation = {
                "F": max(1, int(aircraft_seats * 0.05)),  # 5% First
                "J": max(2, int(aircraft_seats * 0.15)),  # 15% Business
                "W": max(3, int(aircraft_seats * 0.10)),  # 10% Premium Economy
                "Y": max(5, int(aircraft_seats * 0.25)),  # 25% Economy Flex
                "B": max(8, int(aircraft_seats * 0.25)),  # 25% Economy Standard
                "M": max(10, int(aircraft_seats * 0.15)), # 15% Economy Saver
                "H": max(5, int(aircraft_seats * 0.05)),  # 5% Economy Basic
            }
        elif self.airline_type == AirlineType.LOW_COST:
            # Low cost allocation
            allocation = {
                "Y": max(10, int(aircraft_seats * 0.30)),  # 30% Flex
                "M": max(15, int(aircraft_seats * 0.40)),  # 40% Standard
                "L": max(20, int(aircraft_seats * 0.30)),  # 30% Basic
            }
        else:
            # Ultra low cost allocation
            allocation = {
                "Y": max(5, int(aircraft_seats * 0.20)),   # 20% Bundle
                "L": max(25, int(aircraft_seats * 0.80)),  # 80% Basic
            }
        
        # Ensure total doesn't exceed aircraft capacity
        total_allocated = sum(allocation.values())
        if total_allocated > aircraft_seats:
            # Scale down proportionally
            scale_factor = aircraft_seats / total_allocated
            allocation = {k: max(1, int(v * scale_factor)) 
                         for k, v in allocation.items()}
        
        return allocation
    
    def update_competitive_position(self, market_data: Dict) -> None:
        """Update airline's competitive position based on market data."""
        # Update market share based on performance
        performance_factor = (
            self.on_time_performance * 0.3 +
            self.service_quality_score / 10 * 0.3 +
            self.customer_satisfaction_score / 10 * 0.2 +
            (1 / self.price_competitiveness) * 0.2  # Lower prices = better
        )
        
        # Adjust market share (simplified model)
        market_adjustment = (performance_factor - 0.75) * 0.1  # ±10% max adjustment
        self.market_share_percentage = max(
            0.1, 
            self.market_share_percentage * (1 + market_adjustment)
        )
    
    @property
    def network_size(self) -> int:
        """Number of destinations served."""
        return len(self.destinations_served)
    
    @property
    def is_hub_carrier(self) -> bool:
        """Check if airline operates hub-and-spoke model."""
        return len(self.hub_airports) > 0
    
    @property
    def revenue_per_passenger(self) -> float:
        """Calculate average revenue per passenger."""
        if self.annual_passengers == 0:
            return 0.0
        return self.annual_revenue_usd / self.annual_passengers
    
    def get_competitive_advantages(self) -> List[str]:
        """Identify competitive advantages."""
        advantages = []
        
        if self.on_time_performance > 0.9:
            advantages.append("Excellent punctuality")
        
        if self.service_quality_score > 8.5:
            advantages.append("Superior service quality")
        
        if self.price_competitiveness < 0.9:
            advantages.append("Competitive pricing")
        
        if len(self.hub_airports) > 2:
            advantages.append("Extensive hub network")
        
        if self.load_factor > 0.85:
            advantages.append("High operational efficiency")
        
        if self.alliance != AllianceType.NONE:
            advantages.append(f"Global alliance membership ({self.alliance.value})")
        
        return advantages