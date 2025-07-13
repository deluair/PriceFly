"""Cost models for airline operations and financial calculations.

This module defines cost-related data structures and classes for managing
operational costs, fuel costs, and other financial aspects of airline operations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
from enum import Enum
import numpy as np


class CostCategory(Enum):
    """Categories of airline costs."""
    FUEL = "fuel"
    CREW = "crew"
    MAINTENANCE = "maintenance"
    AIRPORT_FEES = "airport_fees"
    NAVIGATION = "navigation"
    INSURANCE = "insurance"
    DEPRECIATION = "depreciation"
    CATERING = "catering"
    GROUND_HANDLING = "ground_handling"
    PASSENGER_SERVICE = "passenger_service"
    MARKETING = "marketing"
    ADMINISTRATION = "administration"
    OTHER = "other"


class CostType(Enum):
    """Types of costs based on variability."""
    FIXED = "fixed"  # Costs that don't vary with flight operations
    VARIABLE = "variable"  # Costs that vary with flight operations
    SEMI_VARIABLE = "semi_variable"  # Costs with both fixed and variable components


class FuelType(Enum):
    """Types of aviation fuel."""
    JET_A1 = "jet_a1"
    JET_A = "jet_a"
    JET_B = "jet_b"
    AVGAS = "avgas"


@dataclass
class FuelCosts:
    """Represents fuel costs and consumption data."""
    fuel_type: FuelType = FuelType.JET_A1
    price_per_gallon: float = 0.0
    price_per_liter: float = 0.0
    currency: str = "USD"
    
    # Consumption data
    consumption_rate_per_hour: float = 0.0  # Gallons per hour
    consumption_rate_per_mile: float = 0.0  # Gallons per mile
    
    # Market data
    last_updated: datetime = field(default_factory=datetime.now)
    price_volatility: float = 0.0  # Standard deviation of price changes
    hedging_percentage: float = 0.0  # Percentage of fuel costs hedged
    
    # Environmental factors
    carbon_tax_per_gallon: float = 0.0
    environmental_surcharge: float = 0.0
    
    def __post_init__(self):
        """Validate fuel cost data."""
        if self.price_per_gallon < 0:
            raise ValueError("Fuel price per gallon must be non-negative")
        
        if self.price_per_liter < 0:
            raise ValueError("Fuel price per liter must be non-negative")
        
        if self.consumption_rate_per_hour < 0:
            raise ValueError("Consumption rate per hour must be non-negative")
        
        if self.hedging_percentage < 0 or self.hedging_percentage > 100:
            raise ValueError("Hedging percentage must be between 0 and 100")
        
        # Convert between gallons and liters if only one is provided
        if self.price_per_gallon > 0 and self.price_per_liter == 0:
            self.price_per_liter = self.price_per_gallon / 3.78541  # 1 gallon = 3.78541 liters
        elif self.price_per_liter > 0 and self.price_per_gallon == 0:
            self.price_per_gallon = self.price_per_liter * 3.78541
    
    def calculate_flight_fuel_cost(
        self, 
        flight_time_hours: float, 
        distance_miles: float = 0.0
    ) -> float:
        """Calculate fuel cost for a specific flight."""
        if flight_time_hours <= 0:
            return 0.0
        
        # Calculate fuel consumption
        fuel_consumed = 0.0
        
        if self.consumption_rate_per_hour > 0:
            fuel_consumed = self.consumption_rate_per_hour * flight_time_hours
        elif self.consumption_rate_per_mile > 0 and distance_miles > 0:
            fuel_consumed = self.consumption_rate_per_mile * distance_miles
        else:
            # Default consumption estimate (rough approximation)
            fuel_consumed = flight_time_hours * 500  # 500 gallons per hour average
        
        # Calculate base fuel cost
        base_cost = fuel_consumed * self.price_per_gallon
        
        # Add environmental costs
        environmental_cost = fuel_consumed * (self.carbon_tax_per_gallon + self.environmental_surcharge)
        
        # Apply hedging effect (hedged fuel is at a different price)
        hedged_portion = base_cost * (self.hedging_percentage / 100)
        unhedged_portion = base_cost * (1 - self.hedging_percentage / 100)
        
        # Assume hedged fuel is 5% cheaper on average
        hedged_cost = hedged_portion * 0.95
        
        total_cost = hedged_cost + unhedged_portion + environmental_cost
        
        return total_cost
    
    def update_price(self, new_price_per_gallon: float) -> None:
        """Update fuel price and track volatility."""
        if self.price_per_gallon > 0:
            price_change = abs(new_price_per_gallon - self.price_per_gallon) / self.price_per_gallon
            # Simple volatility calculation (exponential moving average)
            self.price_volatility = 0.9 * self.price_volatility + 0.1 * price_change
        
        self.price_per_gallon = new_price_per_gallon
        self.price_per_liter = new_price_per_gallon / 3.78541
        self.last_updated = datetime.now()
    
    def get_cost_breakdown(self, fuel_consumed: float) -> Dict[str, float]:
        """Get detailed breakdown of fuel costs."""
        base_cost = fuel_consumed * self.price_per_gallon
        carbon_tax = fuel_consumed * self.carbon_tax_per_gallon
        env_surcharge = fuel_consumed * self.environmental_surcharge
        
        hedged_savings = base_cost * (self.hedging_percentage / 100) * 0.05
        
        return {
            'base_fuel_cost': base_cost,
            'carbon_tax': carbon_tax,
            'environmental_surcharge': env_surcharge,
            'hedging_savings': -hedged_savings,
            'total_cost': base_cost + carbon_tax + env_surcharge - hedged_savings
        }


@dataclass
class OperationalCosts:
    """Represents operational costs for airline operations."""
    # Crew costs
    pilot_cost_per_hour: float = 0.0
    cabin_crew_cost_per_hour: float = 0.0
    crew_per_diem: float = 0.0
    crew_hotel_cost: float = 0.0
    
    # Maintenance costs
    maintenance_cost_per_hour: float = 0.0
    maintenance_cost_per_cycle: float = 0.0  # Per takeoff/landing cycle
    scheduled_maintenance_reserve: float = 0.0
    
    # Airport and navigation fees
    landing_fee_base: float = 0.0
    landing_fee_per_1000kg: float = 0.0
    parking_fee_per_hour: float = 0.0
    navigation_fee_per_km: float = 0.0
    terminal_fee_per_passenger: float = 0.0
    
    # Insurance and depreciation
    insurance_cost_per_hour: float = 0.0
    depreciation_cost_per_hour: float = 0.0
    
    # Service costs
    catering_cost_per_passenger: float = 0.0
    ground_handling_cost: float = 0.0
    cleaning_cost_per_flight: float = 0.0
    
    # Administrative costs
    reservation_system_cost_per_booking: float = 0.0
    credit_card_processing_fee_rate: float = 0.025  # 2.5%
    
    # Currency and metadata
    currency: str = "USD"
    last_updated: datetime = field(default_factory=datetime.now)
    cost_base_year: int = field(default_factory=lambda: datetime.now().year)
    inflation_rate: float = 0.03  # 3% annual inflation
    
    def __post_init__(self):
        """Validate operational cost data."""
        # Validate that all cost values are non-negative
        cost_fields = [
            'pilot_cost_per_hour', 'cabin_crew_cost_per_hour', 'crew_per_diem',
            'crew_hotel_cost', 'maintenance_cost_per_hour', 'maintenance_cost_per_cycle',
            'scheduled_maintenance_reserve', 'landing_fee_base', 'landing_fee_per_1000kg',
            'parking_fee_per_hour', 'navigation_fee_per_km', 'terminal_fee_per_passenger',
            'insurance_cost_per_hour', 'depreciation_cost_per_hour', 'catering_cost_per_passenger',
            'ground_handling_cost', 'cleaning_cost_per_flight', 'reservation_system_cost_per_booking'
        ]
        
        for field_name in cost_fields:
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")
        
        if self.credit_card_processing_fee_rate < 0 or self.credit_card_processing_fee_rate > 1:
            raise ValueError("Credit card processing fee rate must be between 0 and 1")
    
    def calculate_flight_operational_cost(
        self,
        flight_time_hours: float,
        distance_km: float,
        passengers: int,
        aircraft_weight_kg: float,
        crew_size: int = 6,  # 2 pilots + 4 cabin crew
        overnight_stay: bool = False
    ) -> Dict[str, float]:
        """Calculate operational costs for a specific flight."""
        costs = {}
        
        # Crew costs
        pilot_cost = 2 * self.pilot_cost_per_hour * flight_time_hours  # 2 pilots
        cabin_crew_cost = (crew_size - 2) * self.cabin_crew_cost_per_hour * flight_time_hours
        crew_per_diem_cost = crew_size * self.crew_per_diem if flight_time_hours > 4 else 0
        crew_hotel_cost = crew_size * self.crew_hotel_cost if overnight_stay else 0
        
        costs['crew_costs'] = pilot_cost + cabin_crew_cost + crew_per_diem_cost + crew_hotel_cost
        
        # Maintenance costs
        maintenance_hourly = self.maintenance_cost_per_hour * flight_time_hours
        maintenance_cycle = self.maintenance_cost_per_cycle  # One cycle per flight
        maintenance_reserve = self.scheduled_maintenance_reserve * flight_time_hours
        
        costs['maintenance_costs'] = maintenance_hourly + maintenance_cycle + maintenance_reserve
        
        # Airport and navigation fees
        landing_fee = self.landing_fee_base + (aircraft_weight_kg / 1000) * self.landing_fee_per_1000kg
        navigation_fee = self.navigation_fee_per_km * distance_km
        terminal_fee = self.terminal_fee_per_passenger * passengers
        # Assume 2 hours parking time
        parking_fee = self.parking_fee_per_hour * 2
        
        costs['airport_navigation_fees'] = landing_fee + navigation_fee + terminal_fee + parking_fee
        
        # Insurance and depreciation
        insurance_cost = self.insurance_cost_per_hour * flight_time_hours
        depreciation_cost = self.depreciation_cost_per_hour * flight_time_hours
        
        costs['insurance_depreciation'] = insurance_cost + depreciation_cost
        
        # Service costs
        catering_cost = self.catering_cost_per_passenger * passengers
        ground_handling = self.ground_handling_cost
        cleaning_cost = self.cleaning_cost_per_flight
        
        costs['service_costs'] = catering_cost + ground_handling + cleaning_cost
        
        # Calculate total
        costs['total_operational_cost'] = sum(costs.values())
        
        return costs
    
    def calculate_booking_processing_cost(
        self, 
        ticket_price: float, 
        payment_method: str = "credit_card"
    ) -> float:
        """Calculate cost of processing a booking."""
        reservation_cost = self.reservation_system_cost_per_booking
        
        if payment_method == "credit_card":
            processing_fee = ticket_price * self.credit_card_processing_fee_rate
        else:
            processing_fee = 0.0
        
        return reservation_cost + processing_fee
    
    def adjust_for_inflation(self, target_year: int) -> 'OperationalCosts':
        """Adjust costs for inflation to target year."""
        if target_year == self.cost_base_year:
            return self
        
        years_diff = target_year - self.cost_base_year
        inflation_multiplier = (1 + self.inflation_rate) ** years_diff
        
        # Create a new instance with adjusted costs
        adjusted_costs = OperationalCosts(
            pilot_cost_per_hour=self.pilot_cost_per_hour * inflation_multiplier,
            cabin_crew_cost_per_hour=self.cabin_crew_cost_per_hour * inflation_multiplier,
            crew_per_diem=self.crew_per_diem * inflation_multiplier,
            crew_hotel_cost=self.crew_hotel_cost * inflation_multiplier,
            maintenance_cost_per_hour=self.maintenance_cost_per_hour * inflation_multiplier,
            maintenance_cost_per_cycle=self.maintenance_cost_per_cycle * inflation_multiplier,
            scheduled_maintenance_reserve=self.scheduled_maintenance_reserve * inflation_multiplier,
            landing_fee_base=self.landing_fee_base * inflation_multiplier,
            landing_fee_per_1000kg=self.landing_fee_per_1000kg * inflation_multiplier,
            parking_fee_per_hour=self.parking_fee_per_hour * inflation_multiplier,
            navigation_fee_per_km=self.navigation_fee_per_km * inflation_multiplier,
            terminal_fee_per_passenger=self.terminal_fee_per_passenger * inflation_multiplier,
            insurance_cost_per_hour=self.insurance_cost_per_hour * inflation_multiplier,
            depreciation_cost_per_hour=self.depreciation_cost_per_hour * inflation_multiplier,
            catering_cost_per_passenger=self.catering_cost_per_passenger * inflation_multiplier,
            ground_handling_cost=self.ground_handling_cost * inflation_multiplier,
            cleaning_cost_per_flight=self.cleaning_cost_per_flight * inflation_multiplier,
            reservation_system_cost_per_booking=self.reservation_system_cost_per_booking * inflation_multiplier,
            credit_card_processing_fee_rate=self.credit_card_processing_fee_rate,  # Rate doesn't change
            currency=self.currency,
            cost_base_year=target_year,
            inflation_rate=self.inflation_rate
        )
        
        return adjusted_costs
    
    def get_cost_per_category(self) -> Dict[CostCategory, float]:
        """Get costs organized by category for a typical 1-hour flight."""
        return {
            CostCategory.CREW: (
                2 * self.pilot_cost_per_hour + 
                4 * self.cabin_crew_cost_per_hour
            ),
            CostCategory.MAINTENANCE: (
                self.maintenance_cost_per_hour + 
                self.maintenance_cost_per_cycle + 
                self.scheduled_maintenance_reserve
            ),
            CostCategory.AIRPORT_FEES: (
                self.landing_fee_base + 
                self.parking_fee_per_hour * 2 + 
                self.terminal_fee_per_passenger * 150  # Assume 150 passengers
            ),
            CostCategory.NAVIGATION: self.navigation_fee_per_km * 500,  # Assume 500km flight
            CostCategory.INSURANCE: self.insurance_cost_per_hour,
            CostCategory.DEPRECIATION: self.depreciation_cost_per_hour,
            CostCategory.CATERING: self.catering_cost_per_passenger * 150,  # Assume 150 passengers
            CostCategory.GROUND_HANDLING: self.ground_handling_cost,
            CostCategory.PASSENGER_SERVICE: self.cleaning_cost_per_flight,
            CostCategory.ADMINISTRATION: self.reservation_system_cost_per_booking * 150
        }
    
    def calculate_cost_per_seat_mile(
        self, 
        distance_miles: float, 
        seats: int,
        load_factor: float = 0.8
    ) -> float:
        """Calculate cost per available seat mile (CASM)."""
        if distance_miles <= 0 or seats <= 0:
            return 0.0
        
        # Estimate flight time (rough approximation)
        flight_time_hours = distance_miles / 500  # 500 mph average speed
        
        # Calculate operational costs for this flight
        costs = self.calculate_flight_operational_cost(
            flight_time_hours=flight_time_hours,
            distance_km=distance_miles * 1.60934,  # Convert to km
            passengers=int(seats * load_factor),
            aircraft_weight_kg=80000,  # Typical aircraft weight
            crew_size=6
        )
        
        total_cost = costs['total_operational_cost']
        available_seat_miles = seats * distance_miles
        
        return total_cost / available_seat_miles if available_seat_miles > 0 else 0.0
    
    def export_data(self) -> Dict:
        """Export operational costs data."""
        return {
            'crew_costs': {
                'pilot_cost_per_hour': self.pilot_cost_per_hour,
                'cabin_crew_cost_per_hour': self.cabin_crew_cost_per_hour,
                'crew_per_diem': self.crew_per_diem,
                'crew_hotel_cost': self.crew_hotel_cost
            },
            'maintenance_costs': {
                'maintenance_cost_per_hour': self.maintenance_cost_per_hour,
                'maintenance_cost_per_cycle': self.maintenance_cost_per_cycle,
                'scheduled_maintenance_reserve': self.scheduled_maintenance_reserve
            },
            'airport_navigation_fees': {
                'landing_fee_base': self.landing_fee_base,
                'landing_fee_per_1000kg': self.landing_fee_per_1000kg,
                'parking_fee_per_hour': self.parking_fee_per_hour,
                'navigation_fee_per_km': self.navigation_fee_per_km,
                'terminal_fee_per_passenger': self.terminal_fee_per_passenger
            },
            'insurance_depreciation': {
                'insurance_cost_per_hour': self.insurance_cost_per_hour,
                'depreciation_cost_per_hour': self.depreciation_cost_per_hour
            },
            'service_costs': {
                'catering_cost_per_passenger': self.catering_cost_per_passenger,
                'ground_handling_cost': self.ground_handling_cost,
                'cleaning_cost_per_flight': self.cleaning_cost_per_flight
            },
            'administrative_costs': {
                'reservation_system_cost_per_booking': self.reservation_system_cost_per_booking,
                'credit_card_processing_fee_rate': self.credit_card_processing_fee_rate
            },
            'metadata': {
                'currency': self.currency,
                'last_updated': self.last_updated.isoformat(),
                'cost_base_year': self.cost_base_year,
                'inflation_rate': self.inflation_rate
            }
        }
    
    def __str__(self) -> str:
        """String representation of operational costs."""
        return f"OperationalCosts(currency={self.currency}, base_year={self.cost_base_year})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        total_hourly_cost = (
            2 * self.pilot_cost_per_hour + 
            4 * self.cabin_crew_cost_per_hour + 
            self.maintenance_cost_per_hour + 
            self.insurance_cost_per_hour + 
            self.depreciation_cost_per_hour
        )
        
        return (
            f"OperationalCosts(currency='{self.currency}', "
            f"total_hourly_cost={total_hourly_cost:.2f}, "
            f"base_year={self.cost_base_year})"
        )


@dataclass
class CostStructure:
    """Complete cost structure combining all cost components."""
    operational_costs: OperationalCosts
    fuel_costs: FuelCosts
    
    # Route-specific cost adjustments
    route_cost_multiplier: float = 1.0
    seasonal_adjustment: float = 1.0
    
    # Cost allocation
    fixed_cost_allocation: float = 0.0  # Fixed costs allocated to this route
    
    # Metadata
    route_id: str = ""
    aircraft_type: str = ""
    last_updated: datetime = field(default_factory=datetime.now)
    
    def calculate_total_flight_cost(
        self,
        flight_time_hours: float,
        distance_miles: float,
        passengers: int,
        aircraft_weight_kg: float,
        crew_size: int = 6,
        overnight_stay: bool = False
    ) -> Dict[str, float]:
        """Calculate total cost for a flight including fuel and operational costs."""
        # Calculate operational costs
        operational_breakdown = self.operational_costs.calculate_flight_operational_cost(
            flight_time_hours=flight_time_hours,
            distance_km=distance_miles * 1.60934,
            passengers=passengers,
            aircraft_weight_kg=aircraft_weight_kg,
            crew_size=crew_size,
            overnight_stay=overnight_stay
        )
        
        # Calculate fuel costs
        fuel_cost = self.fuel_costs.calculate_flight_fuel_cost(
            flight_time_hours=flight_time_hours,
            distance_miles=distance_miles
        )
        
        # Apply route and seasonal adjustments
        adjusted_operational_cost = operational_breakdown['total_operational_cost'] * self.route_cost_multiplier * self.seasonal_adjustment
        adjusted_fuel_cost = fuel_cost * self.route_cost_multiplier * self.seasonal_adjustment
        
        # Add fixed cost allocation
        total_cost = adjusted_operational_cost + adjusted_fuel_cost + self.fixed_cost_allocation
        
        return {
            'operational_cost': adjusted_operational_cost,
            'fuel_cost': adjusted_fuel_cost,
            'fixed_cost_allocation': self.fixed_cost_allocation,
            'total_cost': total_cost,
            'cost_per_passenger': total_cost / passengers if passengers > 0 else 0,
            'operational_breakdown': operational_breakdown
        }
    
    def calculate_break_even_load_factor(
        self,
        flight_time_hours: float,
        distance_miles: float,
        aircraft_capacity: int,
        average_fare: float,
        aircraft_weight_kg: float
    ) -> float:
        """Calculate the load factor needed to break even."""
        if average_fare <= 0 or aircraft_capacity <= 0:
            return 1.0  # 100% load factor needed if no revenue
        
        # Calculate costs at full capacity
        cost_breakdown = self.calculate_total_flight_cost(
            flight_time_hours=flight_time_hours,
            distance_miles=distance_miles,
            passengers=aircraft_capacity,
            aircraft_weight_kg=aircraft_weight_kg
        )
        
        total_cost = cost_breakdown['total_cost']
        total_revenue_at_capacity = aircraft_capacity * average_fare
        
        # Break-even load factor
        break_even_load_factor = total_cost / total_revenue_at_capacity
        
        return min(break_even_load_factor, 1.0)  # Cap at 100%
    
    def export_data(self) -> Dict:
        """Export complete cost structure data."""
        return {
            'route_id': self.route_id,
            'aircraft_type': self.aircraft_type,
            'operational_costs': self.operational_costs.export_data(),
            'fuel_costs': {
                'fuel_type': self.fuel_costs.fuel_type.value,
                'price_per_gallon': self.fuel_costs.price_per_gallon,
                'consumption_rate_per_hour': self.fuel_costs.consumption_rate_per_hour,
                'hedging_percentage': self.fuel_costs.hedging_percentage,
                'carbon_tax_per_gallon': self.fuel_costs.carbon_tax_per_gallon,
                'last_updated': self.fuel_costs.last_updated.isoformat()
            },
            'adjustments': {
                'route_cost_multiplier': self.route_cost_multiplier,
                'seasonal_adjustment': self.seasonal_adjustment,
                'fixed_cost_allocation': self.fixed_cost_allocation
            },
            'metadata': {
                'last_updated': self.last_updated.isoformat()
            }
        }
    
    def __str__(self) -> str:
        """String representation of cost structure."""
        return f"CostStructure(route={self.route_id}, aircraft={self.aircraft_type})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"CostStructure(route='{self.route_id}', aircraft='{self.aircraft_type}', "
            f"fuel_price={self.fuel_costs.price_per_gallon:.2f}, "
            f"route_multiplier={self.route_cost_multiplier})"
        )