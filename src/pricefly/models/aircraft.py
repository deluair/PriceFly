"""Aircraft and Fleet models for airline operations simulation."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import uuid


class AircraftType(Enum):
    """Aircraft category types."""
    NARROW_BODY = "narrow_body"
    WIDE_BODY = "wide_body"
    REGIONAL = "regional"
    CARGO = "cargo"


class AircraftStatus(Enum):
    """Current status of aircraft."""
    ACTIVE = "active"
    MAINTENANCE = "maintenance"
    GROUNDED = "grounded"
    RETIRED = "retired"


@dataclass
class Aircraft:
    """Represents an individual aircraft with operational characteristics."""
    
    aircraft_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model: str = ""
    manufacturer: str = ""
    aircraft_type: AircraftType = AircraftType.NARROW_BODY
    
    # Capacity and Configuration
    total_seats: int = 0
    business_seats: int = 0
    premium_economy_seats: int = 0
    economy_seats: int = 0
    
    # Performance Characteristics
    max_range_km: float = 0.0
    cruise_speed_kmh: float = 0.0
    fuel_capacity_liters: float = 0.0
    fuel_burn_per_hour: float = 0.0  # liters per hour
    fuel_efficiency_per_seat_km: float = 0.0  # liters per seat per km
    
    # Operational Costs (per hour)
    crew_cost_per_hour: float = 0.0
    maintenance_cost_per_hour: float = 0.0
    insurance_cost_per_hour: float = 0.0
    depreciation_per_hour: float = 0.0
    
    # Current Status
    status: AircraftStatus = AircraftStatus.ACTIVE
    current_location: Optional[str] = None
    utilization_hours_per_day: float = 8.0
    
    # Age and Condition
    manufacture_year: int = 2020
    total_flight_hours: float = 0.0
    cycles_completed: int = 0
    
    def __post_init__(self):
        """Validate aircraft configuration after initialization."""
        if self.total_seats == 0:
            self.total_seats = (
                self.business_seats + 
                self.premium_economy_seats + 
                self.economy_seats
            )
        
        # Calculate fuel efficiency if not provided
        if self.fuel_efficiency_per_seat_km == 0.0 and self.total_seats > 0:
            # Rough estimate: fuel burn per hour / (cruise speed * seats)
            self.fuel_efficiency_per_seat_km = (
                self.fuel_burn_per_hour / 
                (self.cruise_speed_kmh * self.total_seats)
            )
    
    @property
    def age_years(self) -> int:
        """Calculate aircraft age in years."""
        from datetime import datetime
        return datetime.now().year - self.manufacture_year
    
    @property
    def total_operating_cost_per_hour(self) -> float:
        """Calculate total operating cost per hour excluding fuel."""
        return (
            self.crew_cost_per_hour +
            self.maintenance_cost_per_hour +
            self.insurance_cost_per_hour +
            self.depreciation_per_hour
        )
    
    def calculate_trip_cost(self, distance_km: float, fuel_price_per_liter: float) -> Dict[str, float]:
        """Calculate total cost for a specific trip."""
        flight_time_hours = distance_km / self.cruise_speed_kmh
        fuel_needed = self.fuel_burn_per_hour * flight_time_hours
        
        costs = {
            "fuel_cost": fuel_needed * fuel_price_per_liter,
            "crew_cost": self.crew_cost_per_hour * flight_time_hours,
            "maintenance_cost": self.maintenance_cost_per_hour * flight_time_hours,
            "insurance_cost": self.insurance_cost_per_hour * flight_time_hours,
            "depreciation_cost": self.depreciation_per_hour * flight_time_hours,
            "flight_time_hours": flight_time_hours,
            "fuel_liters": fuel_needed
        }
        
        costs["total_cost"] = sum(v for k, v in costs.items() 
                                 if k not in ["flight_time_hours", "fuel_liters"])
        costs["cost_per_seat"] = costs["total_cost"] / self.total_seats
        
        return costs
    
    def is_capable_of_route(self, distance_km: float) -> bool:
        """Check if aircraft can operate on a route of given distance."""
        return distance_km <= self.max_range_km * 0.9  # 10% safety margin


@dataclass
class Fleet:
    """Represents an airline's fleet of aircraft."""
    
    airline_code: str = ""
    aircraft: List[Aircraft] = field(default_factory=list)
    
    def add_aircraft(self, aircraft: Aircraft) -> None:
        """Add an aircraft to the fleet."""
        self.aircraft.append(aircraft)
    
    def remove_aircraft(self, aircraft_id: str) -> bool:
        """Remove an aircraft from the fleet."""
        for i, aircraft in enumerate(self.aircraft):
            if aircraft.aircraft_id == aircraft_id:
                self.aircraft.pop(i)
                return True
        return False
    
    def get_aircraft_by_id(self, aircraft_id: str) -> Optional[Aircraft]:
        """Get aircraft by ID."""
        for aircraft in self.aircraft:
            if aircraft.aircraft_id == aircraft_id:
                return aircraft
        return None
    
    def get_available_aircraft(self, location: Optional[str] = None) -> List[Aircraft]:
        """Get all available aircraft, optionally filtered by location."""
        available = [a for a in self.aircraft if a.status == AircraftStatus.ACTIVE]
        
        if location:
            available = [a for a in available if a.current_location == location]
        
        return available
    
    def get_aircraft_by_type(self, aircraft_type: AircraftType) -> List[Aircraft]:
        """Get all aircraft of a specific type."""
        return [a for a in self.aircraft if a.aircraft_type == aircraft_type]
    
    def get_capable_aircraft(self, distance_km: float, 
                           location: Optional[str] = None) -> List[Aircraft]:
        """Get aircraft capable of operating a route of given distance."""
        available = self.get_available_aircraft(location)
        return [a for a in available if a.is_capable_of_route(distance_km)]
    
    @property
    def total_seats(self) -> int:
        """Total seats across all active aircraft."""
        return sum(a.total_seats for a in self.aircraft 
                  if a.status == AircraftStatus.ACTIVE)
    
    @property
    def fleet_size(self) -> int:
        """Total number of aircraft in fleet."""
        return len(self.aircraft)
    
    @property
    def active_fleet_size(self) -> int:
        """Number of active aircraft in fleet."""
        return len([a for a in self.aircraft if a.status == AircraftStatus.ACTIVE])
    
    def fleet_utilization_stats(self) -> Dict[str, float]:
        """Calculate fleet utilization statistics."""
        active_aircraft = [a for a in self.aircraft if a.status == AircraftStatus.ACTIVE]
        
        if not active_aircraft:
            return {"avg_utilization": 0.0, "total_daily_hours": 0.0}
        
        total_utilization = sum(a.utilization_hours_per_day for a in active_aircraft)
        avg_utilization = total_utilization / len(active_aircraft)
        
        return {
            "avg_utilization": avg_utilization,
            "total_daily_hours": total_utilization,
            "max_possible_hours": len(active_aircraft) * 24,
            "utilization_rate": total_utilization / (len(active_aircraft) * 24)
        }
    
    def calculate_fleet_costs(self, fuel_price_per_liter: float) -> Dict[str, float]:
        """Calculate total fleet operating costs per day."""
        active_aircraft = [a for a in self.aircraft if a.status == AircraftStatus.ACTIVE]
        
        total_costs = {
            "crew_costs": 0.0,
            "maintenance_costs": 0.0,
            "insurance_costs": 0.0,
            "depreciation_costs": 0.0,
            "fuel_costs": 0.0
        }
        
        for aircraft in active_aircraft:
            daily_hours = aircraft.utilization_hours_per_day
            total_costs["crew_costs"] += aircraft.crew_cost_per_hour * daily_hours
            total_costs["maintenance_costs"] += aircraft.maintenance_cost_per_hour * daily_hours
            total_costs["insurance_costs"] += aircraft.insurance_cost_per_hour * daily_hours
            total_costs["depreciation_costs"] += aircraft.depreciation_per_hour * daily_hours
            total_costs["fuel_costs"] += (
                aircraft.fuel_burn_per_hour * daily_hours * fuel_price_per_liter
            )
        
        total_costs["total_daily_cost"] = sum(total_costs.values())
        total_costs["cost_per_seat_per_day"] = (
            total_costs["total_daily_cost"] / self.total_seats if self.total_seats > 0 else 0
        )
        
        return total_costs