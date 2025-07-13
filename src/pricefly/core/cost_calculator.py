"""Cost calculation engine for airline operations.

This module provides comprehensive cost calculation capabilities for airline
operations, including fuel costs, operational costs, maintenance costs,
and total cost of ownership calculations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
from enum import Enum
import math
import logging

from ..models.costs import CostStructure, FuelCosts, OperationalCosts, CostCategory, CostType
from ..models.aircraft import Aircraft
from ..models.route import Route


class CostModel(Enum):
    """Cost calculation models."""
    SIMPLE_LINEAR = "simple_linear"
    ACTIVITY_BASED = "activity_based"
    MARGINAL_COST = "marginal_cost"
    FULL_COST_ALLOCATION = "full_cost_allocation"
    VARIABLE_COST_ONLY = "variable_cost_only"


class CostAccuracy(Enum):
    """Cost calculation accuracy levels."""
    ROUGH_ESTIMATE = "rough_estimate"  # ±20%
    STANDARD = "standard"  # ±10%
    DETAILED = "detailed"  # ±5%
    PRECISE = "precise"  # ±2%


@dataclass
class CostBreakdown:
    """Detailed cost breakdown for analysis."""
    # Direct costs
    fuel_cost: float = 0.0
    crew_cost: float = 0.0
    maintenance_cost: float = 0.0
    airport_fees: float = 0.0
    navigation_fees: float = 0.0
    
    # Indirect costs
    overhead_cost: float = 0.0
    administrative_cost: float = 0.0
    insurance_cost: float = 0.0
    depreciation_cost: float = 0.0
    
    # Variable costs
    variable_cost_total: float = 0.0
    
    # Fixed costs
    fixed_cost_total: float = 0.0
    
    # Total costs
    total_cost: float = 0.0
    cost_per_seat: float = 0.0
    cost_per_passenger: float = 0.0
    cost_per_mile: float = 0.0
    
    # Metadata
    calculation_date: datetime = field(default_factory=datetime.now)
    model_used: CostModel = CostModel.ACTIVITY_BASED
    accuracy_level: CostAccuracy = CostAccuracy.STANDARD
    confidence_score: float = 0.8
    
    def __post_init__(self):
        """Calculate totals after initialization."""
        self._calculate_totals()
    
    def _calculate_totals(self):
        """Calculate total costs and derived metrics."""
        # Variable costs
        self.variable_cost_total = (
            self.fuel_cost + 
            self.crew_cost + 
            self.airport_fees + 
            self.navigation_fees
        )
        
        # Fixed costs
        self.fixed_cost_total = (
            self.maintenance_cost + 
            self.overhead_cost + 
            self.administrative_cost + 
            self.insurance_cost + 
            self.depreciation_cost
        )
        
        # Total cost
        self.total_cost = self.variable_cost_total + self.fixed_cost_total
    
    def calculate_per_unit_costs(
        self, 
        seats: int, 
        passengers: int, 
        distance_miles: float
    ):
        """Calculate per-unit cost metrics."""
        if seats > 0:
            self.cost_per_seat = self.total_cost / seats
        
        if passengers > 0:
            self.cost_per_passenger = self.total_cost / passengers
        
        if distance_miles > 0:
            self.cost_per_mile = self.total_cost / distance_miles
    
    def get_cost_summary(self) -> Dict:
        """Get summary of cost breakdown."""
        return {
            'total_cost': self.total_cost,
            'variable_costs': self.variable_cost_total,
            'fixed_costs': self.fixed_cost_total,
            'variable_percentage': (self.variable_cost_total / self.total_cost * 100) if self.total_cost > 0 else 0,
            'fixed_percentage': (self.fixed_cost_total / self.total_cost * 100) if self.total_cost > 0 else 0,
            'cost_per_seat': self.cost_per_seat,
            'cost_per_passenger': self.cost_per_passenger,
            'cost_per_mile': self.cost_per_mile,
            'model_used': self.model_used.value,
            'accuracy_level': self.accuracy_level.value,
            'confidence_score': self.confidence_score
        }


@dataclass
class CostScenario:
    """Cost calculation scenario parameters."""
    scenario_name: str
    fuel_price_per_gallon: float
    load_factor: float = 0.80
    crew_cost_multiplier: float = 1.0
    maintenance_cost_multiplier: float = 1.0
    overhead_allocation_rate: float = 0.15  # 15% of direct costs
    
    # Economic factors
    inflation_rate: float = 0.03
    currency_exchange_rate: float = 1.0
    fuel_hedging_percentage: float = 0.0
    
    # Operational factors
    delay_cost_per_minute: float = 50.0
    cancellation_cost: float = 25000.0
    weather_impact_factor: float = 1.0
    
    def apply_economic_adjustments(
        self, 
        base_cost: float, 
        calculation_date: datetime
    ) -> float:
        """Apply economic adjustments to base cost."""
        # Apply inflation
        years_from_base = (calculation_date.year - 2023)  # Assuming 2023 as base year
        inflation_adjusted = base_cost * ((1 + self.inflation_rate) ** years_from_base)
        
        # Apply currency exchange
        currency_adjusted = inflation_adjusted * self.currency_exchange_rate
        
        return currency_adjusted


class CostCalculator:
    """Advanced cost calculation engine for airline operations."""
    
    def __init__(
        self,
        model: CostModel = CostModel.ACTIVITY_BASED,
        accuracy: CostAccuracy = CostAccuracy.STANDARD
    ):
        self.model = model
        self.accuracy = accuracy
        
        # Cost databases
        self.fuel_price_history: Dict[date, float] = {}
        self.airport_fee_database: Dict[str, Dict[str, float]] = {}
        self.crew_cost_database: Dict[str, Dict[str, float]] = {}
        
        # Default cost parameters
        self.default_fuel_price = 3.50  # USD per gallon
        self.default_crew_cost_per_hour = 250.0  # USD
        self.default_maintenance_cost_per_hour = 150.0  # USD
        
        # Cost allocation factors
        self.overhead_allocation_rate = 0.15  # 15% of direct costs
        self.administrative_cost_rate = 0.08  # 8% of direct costs
        self.insurance_rate = 0.02  # 2% of aircraft value per year
        
        self.logger = logging.getLogger(__name__)
    
    def calculate_flight_cost(
        self,
        aircraft: Aircraft,
        route: Route,
        scenario: Optional[CostScenario] = None,
        passengers: int = 0,
        cargo_weight_lbs: float = 0.0
    ) -> CostBreakdown:
        """Calculate comprehensive flight cost."""
        
        self.logger.info(
            f"Calculating flight cost for {aircraft.registration} on route {route.route_id}"
        )
        
        if scenario is None:
            scenario = CostScenario(
                scenario_name="default",
                fuel_price_per_gallon=self.default_fuel_price
            )
        
        breakdown = CostBreakdown(
            model_used=self.model,
            accuracy_level=self.accuracy
        )
        
        # Calculate fuel cost
        breakdown.fuel_cost = self._calculate_fuel_cost(
            aircraft, route, scenario, passengers, cargo_weight_lbs
        )
        
        # Calculate crew cost
        breakdown.crew_cost = self._calculate_crew_cost(
            aircraft, route, scenario
        )
        
        # Calculate maintenance cost
        breakdown.maintenance_cost = self._calculate_maintenance_cost(
            aircraft, route, scenario
        )
        
        # Calculate airport and navigation fees
        breakdown.airport_fees = self._calculate_airport_fees(
            aircraft, route, scenario
        )
        breakdown.navigation_fees = self._calculate_navigation_fees(
            aircraft, route, scenario
        )
        
        # Calculate indirect costs
        direct_costs = (
            breakdown.fuel_cost + 
            breakdown.crew_cost + 
            breakdown.maintenance_cost + 
            breakdown.airport_fees + 
            breakdown.navigation_fees
        )
        
        breakdown.overhead_cost = direct_costs * self.overhead_allocation_rate
        breakdown.administrative_cost = direct_costs * self.administrative_cost_rate
        breakdown.insurance_cost = self._calculate_insurance_cost(aircraft, route)
        breakdown.depreciation_cost = self._calculate_depreciation_cost(aircraft, route)
        
        # Calculate per-unit costs
        breakdown.calculate_per_unit_costs(
            aircraft.seating_capacity,
            passengers if passengers > 0 else int(aircraft.seating_capacity * scenario.load_factor),
            route.distance_miles
        )
        
        # Calculate confidence score
        breakdown.confidence_score = self._calculate_confidence_score(
            aircraft, route, scenario
        )
        
        self.logger.info(
            f"Flight cost calculation completed. Total cost: ${breakdown.total_cost:.2f}"
        )
        
        return breakdown
    
    def _calculate_fuel_cost(
        self,
        aircraft: Aircraft,
        route: Route,
        scenario: CostScenario,
        passengers: int,
        cargo_weight_lbs: float
    ) -> float:
        """Calculate fuel cost for the flight."""
        
        # Base fuel consumption (gallons per hour)
        base_fuel_consumption = aircraft.fuel_consumption_per_hour
        
        # Adjust for payload
        total_payload = passengers * 180 + cargo_weight_lbs  # Assume 180 lbs per passenger
        payload_factor = 1.0 + (total_payload / aircraft.max_payload) * 0.15  # 15% increase at max payload
        
        # Adjust for distance (efficiency changes with flight length)
        if route.distance_miles < 500:
            distance_factor = 1.2  # Short flights are less efficient
        elif route.distance_miles > 2000:
            distance_factor = 0.9  # Long flights are more efficient
        else:
            distance_factor = 1.0
        
        # Calculate flight time
        flight_time_hours = route.estimated_flight_time_hours
        
        # Total fuel consumption
        total_fuel_gallons = (
            base_fuel_consumption * 
            flight_time_hours * 
            payload_factor * 
            distance_factor
        )
        
        # Apply fuel hedging if applicable
        effective_fuel_price = scenario.fuel_price_per_gallon
        if scenario.fuel_hedging_percentage > 0:
            hedged_price = self.default_fuel_price  # Assume hedged at default price
            effective_fuel_price = (
                hedged_price * scenario.fuel_hedging_percentage +
                scenario.fuel_price_per_gallon * (1 - scenario.fuel_hedging_percentage)
            )
        
        fuel_cost = total_fuel_gallons * effective_fuel_price
        
        self.logger.debug(
            f"Fuel calculation: {total_fuel_gallons:.1f} gallons × ${effective_fuel_price:.2f} = ${fuel_cost:.2f}"
        )
        
        return fuel_cost
    
    def _calculate_crew_cost(
        self,
        aircraft: Aircraft,
        route: Route,
        scenario: CostScenario
    ) -> float:
        """Calculate crew cost for the flight."""
        
        # Determine crew size based on aircraft type
        if aircraft.seating_capacity <= 50:
            pilots = 2
            flight_attendants = 1
        elif aircraft.seating_capacity <= 150:
            pilots = 2
            flight_attendants = 3
        elif aircraft.seating_capacity <= 300:
            pilots = 2
            flight_attendants = 5
        else:
            pilots = 3  # Long-haul wide-body
            flight_attendants = 8
        
        # Calculate flight time including turnaround
        flight_time_hours = route.estimated_flight_time_hours
        duty_time_hours = flight_time_hours + 1.5  # Add 1.5 hours for pre/post flight duties
        
        # Crew hourly rates
        pilot_hourly_rate = self.default_crew_cost_per_hour * scenario.crew_cost_multiplier
        fa_hourly_rate = pilot_hourly_rate * 0.6  # Flight attendants earn ~60% of pilot rate
        
        # Calculate total crew cost
        pilot_cost = pilots * pilot_hourly_rate * duty_time_hours
        fa_cost = flight_attendants * fa_hourly_rate * duty_time_hours
        
        # Add per diem and hotel costs for overnight trips
        if route.distance_miles > 1000:  # Assume overnight for long routes
            per_diem_cost = (pilots + flight_attendants) * 75  # $75 per person per day
            hotel_cost = (pilots + flight_attendants) * 120  # $120 per room per night
            overnight_costs = per_diem_cost + hotel_cost
        else:
            overnight_costs = 0
        
        total_crew_cost = pilot_cost + fa_cost + overnight_costs
        
        self.logger.debug(
            f"Crew cost: {pilots} pilots + {flight_attendants} FAs × {duty_time_hours:.1f}h = ${total_crew_cost:.2f}"
        )
        
        return total_crew_cost
    
    def _calculate_maintenance_cost(
        self,
        aircraft: Aircraft,
        route: Route,
        scenario: CostScenario
    ) -> float:
        """Calculate maintenance cost for the flight."""
        
        flight_time_hours = route.estimated_flight_time_hours
        
        # Base maintenance cost per flight hour
        base_maintenance_rate = self.default_maintenance_cost_per_hour
        
        # Adjust for aircraft age
        current_year = datetime.now().year
        aircraft_age = current_year - aircraft.year_manufactured
        age_factor = 1.0 + (aircraft_age * 0.02)  # 2% increase per year of age
        
        # Adjust for route characteristics
        if route.distance_miles < 500:
            # Short routes have more cycles, higher maintenance
            route_factor = 1.3
        else:
            route_factor = 1.0
        
        # Calculate maintenance cost
        maintenance_cost = (
            base_maintenance_rate * 
            flight_time_hours * 
            age_factor * 
            route_factor * 
            scenario.maintenance_cost_multiplier
        )
        
        self.logger.debug(
            f"Maintenance cost: ${base_maintenance_rate:.2f}/h × {flight_time_hours:.1f}h × {age_factor:.2f} = ${maintenance_cost:.2f}"
        )
        
        return maintenance_cost
    
    def _calculate_airport_fees(
        self,
        aircraft: Aircraft,
        route: Route,
        scenario: CostScenario
    ) -> float:
        """Calculate airport fees for departure and arrival."""
        
        # Base fees by airport size/type
        def get_airport_fees(airport_code: str, aircraft_weight: float) -> float:
            # Simplified fee structure
            if airport_code in ['JFK', 'LAX', 'LHR', 'CDG', 'NRT']:  # Major international
                landing_fee = aircraft_weight * 0.05  # $0.05 per lb
                terminal_fee = 150
                security_fee = 75
            elif len(airport_code) == 3:  # Assume major domestic
                landing_fee = aircraft_weight * 0.03
                terminal_fee = 100
                security_fee = 50
            else:  # Regional airports
                landing_fee = aircraft_weight * 0.02
                terminal_fee = 75
                security_fee = 25
            
            return landing_fee + terminal_fee + security_fee
        
        departure_fees = get_airport_fees(route.origin_airport, aircraft.max_takeoff_weight)
        arrival_fees = get_airport_fees(route.destination_airport, aircraft.max_takeoff_weight)
        
        total_airport_fees = departure_fees + arrival_fees
        
        self.logger.debug(
            f"Airport fees: {route.origin_airport} ${departure_fees:.2f} + {route.destination_airport} ${arrival_fees:.2f} = ${total_airport_fees:.2f}"
        )
        
        return total_airport_fees
    
    def _calculate_navigation_fees(
        self,
        aircraft: Aircraft,
        route: Route,
        scenario: CostScenario
    ) -> float:
        """Calculate air navigation service fees."""
        
        # Navigation fees are typically based on distance and aircraft weight
        distance_km = route.distance_miles * 1.60934  # Convert to kilometers
        weight_factor = math.sqrt(aircraft.max_takeoff_weight / 1000)  # Weight factor
        
        # Base rate per 100km
        base_rate_per_100km = 15.0
        
        # Calculate navigation fee
        navigation_fee = (distance_km / 100) * base_rate_per_100km * weight_factor
        
        # International flights have higher fees
        if route.origin_airport[:2] != route.destination_airport[:2]:  # Different countries
            navigation_fee *= 1.5
        
        self.logger.debug(
            f"Navigation fees: {distance_km:.0f}km × ${base_rate_per_100km:.2f}/100km × {weight_factor:.2f} = ${navigation_fee:.2f}"
        )
        
        return navigation_fee
    
    def _calculate_insurance_cost(
        self,
        aircraft: Aircraft,
        route: Route
    ) -> float:
        """Calculate insurance cost allocation for the flight."""
        
        # Annual insurance cost as percentage of aircraft value
        annual_insurance_cost = aircraft.current_value * self.insurance_rate
        
        # Allocate based on flight hours
        # Assume 3000 flight hours per year for utilization
        annual_flight_hours = 3000
        insurance_per_hour = annual_insurance_cost / annual_flight_hours
        
        flight_insurance_cost = insurance_per_hour * route.estimated_flight_time_hours
        
        return flight_insurance_cost
    
    def _calculate_depreciation_cost(
        self,
        aircraft: Aircraft,
        route: Route
    ) -> float:
        """Calculate depreciation cost allocation for the flight."""
        
        # Calculate annual depreciation
        aircraft_age = datetime.now().year - aircraft.year_manufactured
        useful_life_years = 25  # Typical aircraft useful life
        remaining_life = max(1, useful_life_years - aircraft_age)
        
        annual_depreciation = (aircraft.current_value * 0.8) / remaining_life  # 80% of value depreciates
        
        # Allocate based on flight hours
        annual_flight_hours = 3000
        depreciation_per_hour = annual_depreciation / annual_flight_hours
        
        flight_depreciation_cost = depreciation_per_hour * route.estimated_flight_time_hours
        
        return flight_depreciation_cost
    
    def _calculate_confidence_score(
        self,
        aircraft: Aircraft,
        route: Route,
        scenario: CostScenario
    ) -> float:
        """Calculate confidence score for cost calculation."""
        
        confidence = 1.0
        
        # Reduce confidence for older aircraft (less predictable costs)
        aircraft_age = datetime.now().year - aircraft.year_manufactured
        if aircraft_age > 15:
            confidence *= 0.9
        elif aircraft_age > 25:
            confidence *= 0.8
        
        # Reduce confidence for very long or very short routes
        if route.distance_miles < 200 or route.distance_miles > 5000:
            confidence *= 0.95
        
        # Reduce confidence based on accuracy level
        if self.accuracy == CostAccuracy.ROUGH_ESTIMATE:
            confidence *= 0.7
        elif self.accuracy == CostAccuracy.STANDARD:
            confidence *= 0.85
        elif self.accuracy == CostAccuracy.DETAILED:
            confidence *= 0.95
        
        return max(0.5, confidence)  # Minimum 50% confidence
    
    def calculate_break_even_load_factor(
        self,
        aircraft: Aircraft,
        route: Route,
        average_fare: float,
        scenario: Optional[CostScenario] = None
    ) -> float:
        """Calculate break-even load factor."""
        
        # Calculate total flight cost
        cost_breakdown = self.calculate_flight_cost(
            aircraft, route, scenario, passengers=0
        )
        
        # Calculate break-even passengers
        break_even_passengers = cost_breakdown.total_cost / average_fare
        
        # Calculate break-even load factor
        break_even_load_factor = break_even_passengers / aircraft.seating_capacity
        
        return min(1.0, break_even_load_factor)
    
    def calculate_marginal_cost(
        self,
        aircraft: Aircraft,
        route: Route,
        additional_passengers: int,
        scenario: Optional[CostScenario] = None
    ) -> float:
        """Calculate marginal cost of additional passengers."""
        
        # Calculate base cost
        base_cost = self.calculate_flight_cost(
            aircraft, route, scenario, passengers=0
        )
        
        # Calculate cost with additional passengers
        cost_with_passengers = self.calculate_flight_cost(
            aircraft, route, scenario, passengers=additional_passengers
        )
        
        # Marginal cost
        marginal_cost = cost_with_passengers.total_cost - base_cost.total_cost
        
        return marginal_cost
    
    def compare_scenarios(
        self,
        aircraft: Aircraft,
        route: Route,
        scenarios: List[CostScenario]
    ) -> Dict[str, CostBreakdown]:
        """Compare costs across multiple scenarios."""
        
        results = {}
        
        for scenario in scenarios:
            cost_breakdown = self.calculate_flight_cost(
                aircraft, route, scenario
            )
            results[scenario.scenario_name] = cost_breakdown
        
        return results
    
    def update_fuel_price(
        self,
        price_date: date,
        price_per_gallon: float
    ) -> None:
        """Update fuel price history."""
        self.fuel_price_history[price_date] = price_per_gallon
        
        # Keep only last 365 days
        cutoff_date = date.today() - timedelta(days=365)
        self.fuel_price_history = {
            d: p for d, p in self.fuel_price_history.items() 
            if d >= cutoff_date
        }
    
    def get_cost_trends(
        self,
        aircraft: Aircraft,
        route: Route,
        days: int = 30
    ) -> Dict:
        """Analyze cost trends over time."""
        
        if len(self.fuel_price_history) < 2:
            return {'error': 'Insufficient historical data'}
        
        # Get recent fuel prices
        recent_dates = sorted(self.fuel_price_history.keys())[-days:]
        recent_prices = [self.fuel_price_history[d] for d in recent_dates]
        
        if len(recent_prices) < 2:
            return {'error': 'Insufficient recent data'}
        
        # Calculate trend
        price_change = recent_prices[-1] - recent_prices[0]
        price_change_percent = (price_change / recent_prices[0]) * 100
        
        # Calculate cost impact
        base_scenario = CostScenario(
            scenario_name="base",
            fuel_price_per_gallon=recent_prices[0]
        )
        current_scenario = CostScenario(
            scenario_name="current",
            fuel_price_per_gallon=recent_prices[-1]
        )
        
        base_cost = self.calculate_flight_cost(aircraft, route, base_scenario)
        current_cost = self.calculate_flight_cost(aircraft, route, current_scenario)
        
        cost_impact = current_cost.total_cost - base_cost.total_cost
        cost_impact_percent = (cost_impact / base_cost.total_cost) * 100
        
        return {
            'period_days': days,
            'fuel_price_change': price_change,
            'fuel_price_change_percent': price_change_percent,
            'cost_impact': cost_impact,
            'cost_impact_percent': cost_impact_percent,
            'trend': 'increasing' if price_change > 0 else 'decreasing' if price_change < 0 else 'stable'
        }
    
    def export_cost_data(self) -> Dict:
        """Export cost calculator configuration and data."""
        
        return {
            'configuration': {
                'model': self.model.value,
                'accuracy': self.accuracy.value,
                'default_fuel_price': self.default_fuel_price,
                'default_crew_cost_per_hour': self.default_crew_cost_per_hour,
                'overhead_allocation_rate': self.overhead_allocation_rate,
                'insurance_rate': self.insurance_rate
            },
            'historical_data': {
                'fuel_price_records': len(self.fuel_price_history),
                'airport_fee_records': len(self.airport_fee_database),
                'crew_cost_records': len(self.crew_cost_database)
            }
        }