"""Route model for airline pricing simulation.

This module defines the Route class and related data structures for modeling
airline routes and their operational characteristics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, date, time, timedelta
from enum import Enum
import numpy as np
from .aircraft import Aircraft
from .airport import Airport


class RouteType(Enum):
    """Types of airline routes."""
    DOMESTIC = "domestic"
    INTERNATIONAL = "international"
    REGIONAL = "regional"
    TRANSCONTINENTAL = "transcontinental"
    SHORT_HAUL = "short_haul"  # < 3 hours
    MEDIUM_HAUL = "medium_haul"  # 3-6 hours
    LONG_HAUL = "long_haul"  # > 6 hours


class RouteStatus(Enum):
    """Status of a route."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SEASONAL = "seasonal"
    SUSPENDED = "suspended"
    PLANNED = "planned"


class FlightFrequency(Enum):
    """Flight frequency patterns."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MULTIPLE_DAILY = "multiple_daily"
    SEASONAL = "seasonal"
    CHARTER = "charter"


@dataclass
class RouteSchedule:
    """Schedule information for a route."""
    departure_times: List[time]
    frequency: FlightFrequency
    days_of_week: Set[int] = field(default_factory=lambda: {0, 1, 2, 3, 4, 5, 6})  # 0=Monday
    seasonal_start: Optional[date] = None
    seasonal_end: Optional[date] = None
    flight_duration: timedelta = field(default_factory=lambda: timedelta(hours=2))
    
    def __post_init__(self):
        """Validate schedule data."""
        if not self.departure_times:
            raise ValueError("At least one departure time must be specified")
        
        if not all(0 <= day <= 6 for day in self.days_of_week):
            raise ValueError("Days of week must be between 0 (Monday) and 6 (Sunday)")
        
        if self.seasonal_start and self.seasonal_end:
            if self.seasonal_start >= self.seasonal_end:
                raise ValueError("Seasonal start must be before seasonal end")


@dataclass
class RouteMetrics:
    """Performance metrics for a route."""
    load_factor: float = 0.0
    revenue_per_flight: float = 0.0
    cost_per_flight: float = 0.0
    profit_per_flight: float = 0.0
    yield_per_passenger: float = 0.0
    passengers_per_day: int = 0
    flights_per_day: int = 0
    on_time_performance: float = 0.95
    cancellation_rate: float = 0.02
    average_fare: float = 0.0
    
    @property
    def profit_margin(self) -> float:
        """Calculate profit margin."""
        if self.revenue_per_flight == 0:
            return 0.0
        return self.profit_per_flight / self.revenue_per_flight
    
    @property
    def break_even_load_factor(self) -> float:
        """Calculate break-even load factor."""
        if self.yield_per_passenger == 0:
            return 1.0
        # Simplified calculation
        return min(1.0, self.cost_per_flight / (self.yield_per_passenger * 150))  # Assume 150 seats


@dataclass
class RouteOperationalData:
    """Operational data for a route."""
    fuel_consumption_per_flight: float = 0.0  # gallons
    crew_cost_per_flight: float = 0.0
    maintenance_cost_per_flight: float = 0.0
    airport_fees_per_flight: float = 0.0
    ground_handling_cost_per_flight: float = 0.0
    catering_cost_per_flight: float = 0.0
    average_delay_minutes: float = 15.0
    weather_delay_risk: float = 0.1  # 0-1 scale
    air_traffic_delay_risk: float = 0.15  # 0-1 scale
    
    @property
    def total_variable_cost_per_flight(self) -> float:
        """Calculate total variable cost per flight."""
        return (
            self.fuel_consumption_per_flight * 3.5 +  # Assume $3.5/gallon
            self.crew_cost_per_flight +
            self.maintenance_cost_per_flight +
            self.airport_fees_per_flight +
            self.ground_handling_cost_per_flight +
            self.catering_cost_per_flight
        )


class Route:
    """Represents an airline route between two airports."""
    
    def __init__(
        self,
        route_id: str,
        airline_code: str,
        origin: Airport,
        destination: Airport,
        aircraft: Aircraft,
        route_type: RouteType,
        schedule: RouteSchedule,
        status: RouteStatus = RouteStatus.ACTIVE
    ):
        """Initialize route.
        
        Args:
            route_id: Unique identifier for the route
            airline_code: IATA code of the operating airline
            origin: Origin airport
            destination: Destination airport
            aircraft: Aircraft used on this route
            route_type: Type of route
            schedule: Flight schedule
            status: Current status of the route
        """
        self.route_id = route_id
        self.airline_code = airline_code
        self.origin = origin
        self.destination = destination
        self.aircraft = aircraft
        self.route_type = route_type
        self.schedule = schedule
        self.status = status
        
        # Calculate basic route characteristics
        self.distance = self._calculate_distance()
        self.flight_time = self._estimate_flight_time()
        
        # Performance tracking
        self.metrics = RouteMetrics()
        self.operational_data = RouteOperationalData()
        self.historical_metrics: List[Tuple[date, RouteMetrics]] = []
        
        # Pricing and demand
        self.base_fare = 0.0
        self.fare_classes: Dict[str, float] = {}  # class -> price multiplier
        self.demand_forecast: Dict[date, int] = {}
        self.booking_curve: Dict[int, float] = {}  # days_before -> booking_percentage
        
        # Competition and market
        self.competitor_routes: List[str] = []  # Route IDs of competing routes
        self.market_share = 0.0
        self.price_elasticity = -1.2
        
        # Operational constraints
        self.slot_restrictions: Dict[str, int] = {}  # airport -> max_daily_flights
        self.seasonal_adjustments: Dict[int, float] = {}  # month -> adjustment_factor
        
        # Initialize default values
        self._initialize_defaults()
    
    def _calculate_distance(self) -> float:
        """Calculate great circle distance between airports."""
        # Simplified calculation using Haversine formula
        lat1, lon1 = np.radians(self.origin.latitude), np.radians(self.origin.longitude)
        lat2, lon2 = np.radians(self.destination.latitude), np.radians(self.destination.longitude)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth's radius in miles
        r = 3959
        
        return r * c
    
    def _estimate_flight_time(self) -> timedelta:
        """Estimate flight time based on distance and aircraft speed."""
        # Add taxi, takeoff, and landing time
        ground_time = timedelta(minutes=30)
        
        # Calculate flight time based on aircraft cruise speed
        if self.aircraft.cruise_speed > 0:
            flight_hours = self.distance / self.aircraft.cruise_speed
            flight_time = timedelta(hours=flight_hours)
        else:
            # Default estimate: 500 mph average speed
            flight_hours = self.distance / 500
            flight_time = timedelta(hours=flight_hours)
        
        return ground_time + flight_time
    
    def _initialize_defaults(self) -> None:
        """Initialize default values for the route."""
        # Set route type based on distance
        if self.distance < 500:
            self.route_type = RouteType.SHORT_HAUL
        elif self.distance < 1500:
            self.route_type = RouteType.MEDIUM_HAUL
        else:
            self.route_type = RouteType.LONG_HAUL
        
        # Set base fare based on distance
        self.base_fare = max(99, self.distance * 0.15 + 50)
        
        # Initialize fare classes
        self.fare_classes = {
            'economy': 1.0,
            'premium_economy': 1.5,
            'business': 3.0,
            'first': 5.0
        }
        
        # Initialize booking curve (simplified)
        self.booking_curve = {
            90: 0.05,  # 5% book 90+ days out
            60: 0.15,  # 15% book 60-90 days out
            30: 0.35,  # 35% book 30-60 days out
            14: 0.25,  # 25% book 14-30 days out
            7: 0.15,   # 15% book 7-14 days out
            0: 0.05    # 5% book within 7 days
        }
        
        # Initialize operational costs
        self._estimate_operational_costs()
    
    def _estimate_operational_costs(self) -> None:
        """Estimate operational costs for the route."""
        # Fuel cost based on distance and aircraft efficiency
        fuel_per_mile = self.aircraft.fuel_capacity / self.aircraft.range if self.aircraft.range > 0 else 5
        self.operational_data.fuel_consumption_per_flight = self.distance * fuel_per_mile
        
        # Crew costs (simplified)
        if self.distance < 500:
            self.operational_data.crew_cost_per_flight = 800
        elif self.distance < 1500:
            self.operational_data.crew_cost_per_flight = 1200
        else:
            self.operational_data.crew_cost_per_flight = 2000
        
        # Airport fees based on aircraft size and airport type
        base_fee = 200
        size_multiplier = self.aircraft.capacity / 150  # Normalize to 150 seats
        
        origin_multiplier = 1.5 if self.origin.airport_type.value in ['hub', 'international'] else 1.0
        dest_multiplier = 1.5 if self.destination.airport_type.value in ['hub', 'international'] else 1.0
        
        self.operational_data.airport_fees_per_flight = (
            base_fee * size_multiplier * (origin_multiplier + dest_multiplier) / 2
        )
        
        # Other costs
        self.operational_data.maintenance_cost_per_flight = self.distance * 0.5
        self.operational_data.ground_handling_cost_per_flight = 300 * size_multiplier
        self.operational_data.catering_cost_per_flight = self.aircraft.capacity * 8  # $8 per passenger
    
    def calculate_capacity(self) -> int:
        """Calculate daily passenger capacity."""
        daily_flights = len(self.schedule.departure_times)
        return daily_flights * self.aircraft.capacity
    
    def calculate_revenue_potential(self, load_factor: float = 0.8) -> float:
        """Calculate daily revenue potential."""
        daily_capacity = self.calculate_capacity()
        daily_passengers = int(daily_capacity * load_factor)
        
        # Apply fare class mix (simplified)
        economy_pax = int(daily_passengers * 0.8)
        business_pax = int(daily_passengers * 0.15)
        first_pax = int(daily_passengers * 0.05)
        
        revenue = (
            economy_pax * self.base_fare * self.fare_classes['economy'] +
            business_pax * self.base_fare * self.fare_classes['business'] +
            first_pax * self.base_fare * self.fare_classes['first']
        )
        
        return revenue
    
    def calculate_operating_cost(self) -> float:
        """Calculate daily operating cost."""
        daily_flights = len(self.schedule.departure_times)
        return daily_flights * self.operational_data.total_variable_cost_per_flight
    
    def update_metrics(self, new_metrics: RouteMetrics) -> None:
        """Update route metrics and store historical data."""
        # Store current metrics in history
        self.historical_metrics.append((date.today(), self.metrics))
        
        # Update current metrics
        self.metrics = new_metrics
        
        # Keep only last 365 days of history
        cutoff_date = date.today() - timedelta(days=365)
        self.historical_metrics = [
            (d, m) for d, m in self.historical_metrics if d >= cutoff_date
        ]
    
    def forecast_demand(
        self, 
        target_date: date, 
        base_price: float,
        external_factors: Optional[Dict[str, float]] = None
    ) -> int:
        """Forecast demand for a specific date."""
        external_factors = external_factors or {}
        
        # Base demand (simplified model)
        base_demand = self.calculate_capacity() * 0.8  # 80% base load factor
        
        # Seasonal adjustment
        month = target_date.month
        seasonal_factor = self.seasonal_adjustments.get(month, 1.0)
        
        # Price elasticity effect
        if self.base_fare > 0:
            price_ratio = base_price / self.base_fare
            price_effect = price_ratio ** self.price_elasticity
        else:
            price_effect = 1.0
        
        # Day of week effect
        dow = target_date.weekday()
        dow_factors = {0: 1.1, 1: 1.2, 2: 1.0, 3: 1.1, 4: 1.3, 5: 0.8, 6: 0.9}  # Mon-Sun
        dow_factor = dow_factors.get(dow, 1.0)
        
        # External factors
        economic_factor = external_factors.get('economic_index', 1.0)
        weather_factor = external_factors.get('weather_index', 1.0)
        event_factor = external_factors.get('event_factor', 1.0)
        
        # Calculate final demand
        forecasted_demand = (
            base_demand * 
            seasonal_factor * 
            price_effect * 
            dow_factor * 
            economic_factor * 
            weather_factor * 
            event_factor
        )
        
        return max(0, int(forecasted_demand))
    
    def optimize_schedule(self, demand_pattern: Dict[time, float]) -> RouteSchedule:
        """Optimize flight schedule based on demand patterns."""
        # Sort times by demand
        sorted_times = sorted(demand_pattern.items(), key=lambda x: x[1], reverse=True)
        
        # Select top times based on current frequency
        current_flights = len(self.schedule.departure_times)
        optimal_times = [t for t, _ in sorted_times[:current_flights]]
        
        # Create new schedule
        new_schedule = RouteSchedule(
            departure_times=optimal_times,
            frequency=self.schedule.frequency,
            days_of_week=self.schedule.days_of_week,
            seasonal_start=self.schedule.seasonal_start,
            seasonal_end=self.schedule.seasonal_end,
            flight_duration=self.flight_time
        )
        
        return new_schedule
    
    def add_competitor_route(self, route_id: str) -> None:
        """Add a competing route."""
        if route_id not in self.competitor_routes:
            self.competitor_routes.append(route_id)
    
    def remove_competitor_route(self, route_id: str) -> None:
        """Remove a competing route."""
        if route_id in self.competitor_routes:
            self.competitor_routes.remove(route_id)
    
    def calculate_profitability(self) -> Dict[str, float]:
        """Calculate various profitability metrics."""
        revenue = self.calculate_revenue_potential(self.metrics.load_factor)
        cost = self.calculate_operating_cost()
        
        return {
            'daily_revenue': revenue,
            'daily_cost': cost,
            'daily_profit': revenue - cost,
            'profit_margin': (revenue - cost) / revenue if revenue > 0 else 0,
            'revenue_per_passenger': revenue / (self.calculate_capacity() * self.metrics.load_factor) if self.metrics.load_factor > 0 else 0,
            'cost_per_passenger': cost / (self.calculate_capacity() * self.metrics.load_factor) if self.metrics.load_factor > 0 else 0,
            'break_even_load_factor': cost / (self.base_fare * self.calculate_capacity()) if self.base_fare > 0 else 1.0
        }
    
    def is_profitable(self, min_margin: float = 0.1) -> bool:
        """Check if route is profitable."""
        profitability = self.calculate_profitability()
        return profitability['profit_margin'] >= min_margin
    
    def get_performance_summary(self) -> Dict:
        """Get a comprehensive performance summary."""
        profitability = self.calculate_profitability()
        
        return {
            'route_info': {
                'route_id': self.route_id,
                'airline': self.airline_code,
                'origin': self.origin.iata_code,
                'destination': self.destination.iata_code,
                'distance': self.distance,
                'route_type': self.route_type.value,
                'status': self.status.value
            },
            'operational': {
                'aircraft': self.aircraft.model,
                'capacity': self.aircraft.capacity,
                'daily_flights': len(self.schedule.departure_times),
                'daily_capacity': self.calculate_capacity(),
                'flight_time': str(self.flight_time)
            },
            'performance': {
                'load_factor': self.metrics.load_factor,
                'on_time_performance': self.metrics.on_time_performance,
                'cancellation_rate': self.metrics.cancellation_rate,
                'average_fare': self.metrics.average_fare,
                'market_share': self.market_share
            },
            'financial': profitability,
            'competition': {
                'competitor_count': len(self.competitor_routes),
                'price_elasticity': self.price_elasticity
            }
        }
    
    def export_data(self) -> Dict:
        """Export route data for analysis."""
        return {
            'route_id': self.route_id,
            'airline_code': self.airline_code,
            'origin': self.origin.export_data(),
            'destination': self.destination.export_data(),
            'aircraft': self.aircraft.export_data(),
            'route_type': self.route_type.value,
            'status': self.status.value,
            'distance': self.distance,
            'flight_time': str(self.flight_time),
            'schedule': {
                'departure_times': [str(t) for t in self.schedule.departure_times],
                'frequency': self.schedule.frequency.value,
                'days_of_week': list(self.schedule.days_of_week),
                'flight_duration': str(self.schedule.flight_duration)
            },
            'metrics': self.metrics.__dict__,
            'operational_data': self.operational_data.__dict__,
            'base_fare': self.base_fare,
            'fare_classes': self.fare_classes,
            'market_share': self.market_share,
            'competitor_count': len(self.competitor_routes),
            'performance_summary': self.get_performance_summary()
        }
    
    def __str__(self) -> str:
        """String representation of the route."""
        return f"Route({self.origin.iata_code}-{self.destination.iata_code}, {self.airline_code})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"Route(id='{self.route_id}', {self.origin.iata_code}-{self.destination.iata_code}, "
            f"airline='{self.airline_code}', aircraft='{self.aircraft.model}', "
            f"distance={self.distance:.0f}mi, status={self.status.value})"
        )