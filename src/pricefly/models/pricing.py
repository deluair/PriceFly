"""Pricing models for airline fare structures and price points.

This module defines pricing-related data structures and classes for managing
airline fare structures, price points, and booking classes.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, date, timedelta
from enum import Enum
import numpy as np


class FareType(Enum):
    """Types of airline fares."""
    PUBLISHED = "published"
    PRIVATE = "private"
    NEGOTIATED = "negotiated"
    PROMOTIONAL = "promotional"
    DYNAMIC = "dynamic"


class PriceChangeReason(Enum):
    """Reasons for price changes."""
    DEMAND_ADJUSTMENT = "demand_adjustment"
    COMPETITION_RESPONSE = "competition_response"
    INVENTORY_MANAGEMENT = "inventory_management"
    SEASONAL_ADJUSTMENT = "seasonal_adjustment"
    PROMOTIONAL_CAMPAIGN = "promotional_campaign"
    COST_ADJUSTMENT = "cost_adjustment"
    REVENUE_OPTIMIZATION = "revenue_optimization"


class BookingRestriction(Enum):
    """Types of booking restrictions."""
    ADVANCE_PURCHASE = "advance_purchase"
    MINIMUM_STAY = "minimum_stay"
    MAXIMUM_STAY = "maximum_stay"
    SATURDAY_NIGHT_STAY = "saturday_night_stay"
    NON_REFUNDABLE = "non_refundable"
    CHANGE_FEE = "change_fee"
    BLACKOUT_DATES = "blackout_dates"


@dataclass
class PricePoint:
    """Represents a specific price point for a fare."""
    price: float
    currency: str = "USD"
    effective_date: datetime = field(default_factory=datetime.now)
    expiry_date: Optional[datetime] = None
    fare_type: FareType = FareType.PUBLISHED
    booking_class: str = "Y"
    
    # Pricing metadata
    base_fare: float = 0.0
    taxes_and_fees: float = 0.0
    fuel_surcharge: float = 0.0
    
    # Restrictions and conditions
    restrictions: Set[BookingRestriction] = field(default_factory=set)
    advance_purchase_days: Optional[int] = None
    minimum_stay_days: Optional[int] = None
    maximum_stay_days: Optional[int] = None
    
    # Availability
    seats_available: int = 0
    total_inventory: int = 0
    
    # Change tracking
    last_updated: datetime = field(default_factory=datetime.now)
    change_reason: Optional[PriceChangeReason] = None
    previous_price: Optional[float] = None
    
    def __post_init__(self):
        """Validate price point data."""
        if self.price < 0:
            raise ValueError("Price must be non-negative")
        
        if self.base_fare < 0:
            raise ValueError("Base fare must be non-negative")
        
        if self.taxes_and_fees < 0:
            raise ValueError("Taxes and fees must be non-negative")
        
        if self.seats_available < 0:
            raise ValueError("Seats available must be non-negative")
        
        if self.total_inventory < 0:
            raise ValueError("Total inventory must be non-negative")
        
        if self.seats_available > self.total_inventory:
            raise ValueError("Seats available cannot exceed total inventory")
        
        # Set base fare if not provided
        if self.base_fare == 0.0:
            self.base_fare = max(0, self.price - self.taxes_and_fees - self.fuel_surcharge)
    
    @property
    def is_active(self) -> bool:
        """Check if price point is currently active."""
        now = datetime.now()
        if now < self.effective_date:
            return False
        if self.expiry_date and now > self.expiry_date:
            return False
        return True
    
    @property
    def availability_percentage(self) -> float:
        """Calculate availability as percentage of total inventory."""
        if self.total_inventory == 0:
            return 0.0
        return (self.seats_available / self.total_inventory) * 100
    
    @property
    def total_price_breakdown(self) -> Dict[str, float]:
        """Get breakdown of total price components."""
        return {
            'base_fare': self.base_fare,
            'taxes_and_fees': self.taxes_and_fees,
            'fuel_surcharge': self.fuel_surcharge,
            'total': self.price
        }
    
    def update_price(
        self, 
        new_price: float, 
        reason: PriceChangeReason,
        effective_date: Optional[datetime] = None
    ) -> None:
        """Update the price with tracking."""
        self.previous_price = self.price
        self.price = new_price
        self.change_reason = reason
        self.last_updated = datetime.now()
        
        if effective_date:
            self.effective_date = effective_date
        
        # Update base fare proportionally
        if self.previous_price > 0:
            price_ratio = new_price / self.previous_price
            self.base_fare *= price_ratio
    
    def update_availability(self, seats_sold: int) -> bool:
        """Update seat availability after booking."""
        if seats_sold > self.seats_available:
            return False
        
        self.seats_available -= seats_sold
        self.last_updated = datetime.now()
        return True
    
    def add_restriction(self, restriction: BookingRestriction) -> None:
        """Add a booking restriction."""
        self.restrictions.add(restriction)
        self.last_updated = datetime.now()
    
    def remove_restriction(self, restriction: BookingRestriction) -> None:
        """Remove a booking restriction."""
        self.restrictions.discard(restriction)
        self.last_updated = datetime.now()
    
    def has_restriction(self, restriction: BookingRestriction) -> bool:
        """Check if a specific restriction applies."""
        return restriction in self.restrictions
    
    def is_eligible_for_booking(
        self, 
        booking_date: datetime,
        departure_date: datetime,
        return_date: Optional[datetime] = None
    ) -> Tuple[bool, List[str]]:
        """Check if booking is eligible based on restrictions."""
        issues = []
        
        # Check advance purchase requirement
        if self.advance_purchase_days:
            days_in_advance = (departure_date - booking_date).days
            if days_in_advance < self.advance_purchase_days:
                issues.append(f"Must book at least {self.advance_purchase_days} days in advance")
        
        # Check minimum stay requirement
        if return_date and self.minimum_stay_days:
            stay_days = (return_date - departure_date).days
            if stay_days < self.minimum_stay_days:
                issues.append(f"Minimum stay of {self.minimum_stay_days} days required")
        
        # Check maximum stay requirement
        if return_date and self.maximum_stay_days:
            stay_days = (return_date - departure_date).days
            if stay_days > self.maximum_stay_days:
                issues.append(f"Maximum stay of {self.maximum_stay_days} days exceeded")
        
        # Check Saturday night stay requirement
        if (return_date and 
            BookingRestriction.SATURDAY_NIGHT_STAY in self.restrictions):
            # Check if stay includes a Saturday night
            current_date = departure_date
            includes_saturday = False
            while current_date < return_date:
                if current_date.weekday() == 5:  # Saturday
                    includes_saturday = True
                    break
                current_date += timedelta(days=1)
            
            if not includes_saturday:
                issues.append("Saturday night stay required")
        
        # Check availability
        if self.seats_available <= 0:
            issues.append("No seats available at this fare")
        
        return len(issues) == 0, issues
    
    def calculate_change_fee(self, new_departure_date: datetime) -> float:
        """Calculate change fee if applicable."""
        if BookingRestriction.CHANGE_FEE not in self.restrictions:
            return 0.0
        
        # Simple change fee calculation
        # In reality, this would be more complex
        base_change_fee = 200.0  # Base change fee
        
        # Increase fee for last-minute changes
        days_to_departure = (new_departure_date - datetime.now()).days
        if days_to_departure < 7:
            return base_change_fee * 1.5
        elif days_to_departure < 14:
            return base_change_fee * 1.2
        else:
            return base_change_fee
    
    def export_data(self) -> Dict:
        """Export price point data."""
        return {
            'price': self.price,
            'currency': self.currency,
            'effective_date': self.effective_date.isoformat(),
            'expiry_date': self.expiry_date.isoformat() if self.expiry_date else None,
            'fare_type': self.fare_type.value,
            'booking_class': self.booking_class,
            'base_fare': self.base_fare,
            'taxes_and_fees': self.taxes_and_fees,
            'fuel_surcharge': self.fuel_surcharge,
            'restrictions': [r.value for r in self.restrictions],
            'advance_purchase_days': self.advance_purchase_days,
            'minimum_stay_days': self.minimum_stay_days,
            'maximum_stay_days': self.maximum_stay_days,
            'seats_available': self.seats_available,
            'total_inventory': self.total_inventory,
            'availability_percentage': self.availability_percentage,
            'is_active': self.is_active,
            'last_updated': self.last_updated.isoformat(),
            'change_reason': self.change_reason.value if self.change_reason else None,
            'previous_price': self.previous_price
        }


@dataclass
class FareStructure:
    """Represents the complete fare structure for a route/flight."""
    route_id: str
    flight_number: str
    departure_date: date
    
    # Fare classes and their price points
    fare_classes: Dict[str, PricePoint] = field(default_factory=dict)
    
    # Pricing strategy metadata
    pricing_strategy: str = "dynamic"
    last_optimization: datetime = field(default_factory=datetime.now)
    optimization_frequency_hours: int = 6
    
    # Market context
    competitor_prices: Dict[str, float] = field(default_factory=dict)
    demand_forecast: float = 0.0
    load_factor_target: float = 0.8
    
    # Revenue management
    revenue_target: float = 0.0
    current_revenue: float = 0.0
    bookings_count: int = 0
    
    def __post_init__(self):
        """Initialize default fare classes if none provided."""
        if not self.fare_classes:
            self._initialize_default_fare_classes()
    
    def _initialize_default_fare_classes(self) -> None:
        """Initialize default fare classes with basic pricing."""
        base_price = 300.0  # Default base price
        
        default_classes = {
            'L': {'multiplier': 0.8, 'name': 'Basic Economy', 'inventory': 30},
            'M': {'multiplier': 1.0, 'name': 'Economy', 'inventory': 50},
            'Y': {'multiplier': 1.3, 'name': 'Flexible Economy', 'inventory': 40},
            'W': {'multiplier': 1.8, 'name': 'Premium Economy', 'inventory': 20},
            'J': {'multiplier': 3.0, 'name': 'Business', 'inventory': 15},
            'F': {'multiplier': 5.0, 'name': 'First', 'inventory': 5}
        }
        
        for class_code, config in default_classes.items():
            price = base_price * config['multiplier']
            
            # Add restrictions based on fare class
            restrictions = set()
            if class_code in ['L', 'M']:
                restrictions.add(BookingRestriction.NON_REFUNDABLE)
                restrictions.add(BookingRestriction.CHANGE_FEE)
            
            if class_code == 'L':
                restrictions.add(BookingRestriction.ADVANCE_PURCHASE)
            
            self.fare_classes[class_code] = PricePoint(
                price=price,
                booking_class=class_code,
                base_fare=price * 0.8,  # 80% base fare, 20% taxes/fees
                taxes_and_fees=price * 0.2,
                seats_available=config['inventory'],
                total_inventory=config['inventory'],
                restrictions=restrictions,
                advance_purchase_days=7 if class_code == 'L' else None
            )
    
    def get_fare_class(self, class_code: str) -> Optional[PricePoint]:
        """Get price point for a specific fare class."""
        return self.fare_classes.get(class_code)
    
    def add_fare_class(self, class_code: str, price_point: PricePoint) -> None:
        """Add or update a fare class."""
        self.fare_classes[class_code] = price_point
    
    def remove_fare_class(self, class_code: str) -> bool:
        """Remove a fare class."""
        if class_code in self.fare_classes:
            del self.fare_classes[class_code]
            return True
        return False
    
    def get_available_classes(self) -> List[str]:
        """Get list of fare classes with available inventory."""
        return [
            class_code for class_code, price_point in self.fare_classes.items()
            if price_point.seats_available > 0 and price_point.is_active
        ]
    
    def get_lowest_fare(self) -> Optional[Tuple[str, PricePoint]]:
        """Get the lowest available fare."""
        available_fares = [
            (code, pp) for code, pp in self.fare_classes.items()
            if pp.seats_available > 0 and pp.is_active
        ]
        
        if not available_fares:
            return None
        
        return min(available_fares, key=lambda x: x[1].price)
    
    def get_highest_fare(self) -> Optional[Tuple[str, PricePoint]]:
        """Get the highest available fare."""
        available_fares = [
            (code, pp) for code, pp in self.fare_classes.items()
            if pp.seats_available > 0 and pp.is_active
        ]
        
        if not available_fares:
            return None
        
        return max(available_fares, key=lambda x: x[1].price)
    
    def calculate_total_inventory(self) -> int:
        """Calculate total seat inventory across all classes."""
        return sum(pp.total_inventory for pp in self.fare_classes.values())
    
    def calculate_available_inventory(self) -> int:
        """Calculate available seat inventory across all classes."""
        return sum(pp.seats_available for pp in self.fare_classes.values())
    
    def calculate_load_factor(self) -> float:
        """Calculate current load factor."""
        total_inventory = self.calculate_total_inventory()
        if total_inventory == 0:
            return 0.0
        
        sold_seats = total_inventory - self.calculate_available_inventory()
        return sold_seats / total_inventory
    
    def calculate_average_fare(self) -> float:
        """Calculate revenue-weighted average fare."""
        if self.bookings_count == 0:
            return 0.0
        
        return self.current_revenue / self.bookings_count
    
    def process_booking(
        self, 
        class_code: str, 
        passengers: int = 1
    ) -> Tuple[bool, float, List[str]]:
        """Process a booking and update inventory."""
        if class_code not in self.fare_classes:
            return False, 0.0, [f"Fare class {class_code} not available"]
        
        price_point = self.fare_classes[class_code]
        
        # Check availability
        if price_point.seats_available < passengers:
            return False, 0.0, [f"Only {price_point.seats_available} seats available"]
        
        # Check if fare is active
        if not price_point.is_active:
            return False, 0.0, ["Fare is not currently active"]
        
        # Process the booking
        success = price_point.update_availability(passengers)
        if success:
            booking_revenue = price_point.price * passengers
            self.current_revenue += booking_revenue
            self.bookings_count += passengers
            
            return True, booking_revenue, []
        else:
            return False, 0.0, ["Failed to update availability"]
    
    def optimize_pricing(
        self, 
        demand_multiplier: float = 1.0,
        competition_factor: float = 1.0
    ) -> Dict[str, float]:
        """Optimize pricing across all fare classes."""
        price_changes = {}
        
        current_load_factor = self.calculate_load_factor()
        
        for class_code, price_point in self.fare_classes.items():
            old_price = price_point.price
            
            # Calculate new price based on demand and competition
            demand_adjustment = 1.0
            
            # Adjust based on load factor vs target
            if current_load_factor > self.load_factor_target:
                demand_adjustment = 1.1  # Increase prices
            elif current_load_factor < self.load_factor_target * 0.7:
                demand_adjustment = 0.9  # Decrease prices
            
            # Apply demand multiplier
            demand_adjustment *= demand_multiplier
            
            # Apply competition factor
            new_price = old_price * demand_adjustment * competition_factor
            
            # Update price
            price_point.update_price(
                new_price, 
                PriceChangeReason.REVENUE_OPTIMIZATION
            )
            
            price_changes[class_code] = {
                'old_price': old_price,
                'new_price': new_price,
                'change_percent': ((new_price - old_price) / old_price) * 100
            }
        
        self.last_optimization = datetime.now()
        return price_changes
    
    def needs_optimization(self) -> bool:
        """Check if pricing needs optimization based on frequency."""
        time_since_last = datetime.now() - self.last_optimization
        return time_since_last.total_seconds() / 3600 >= self.optimization_frequency_hours
    
    def get_pricing_summary(self) -> Dict:
        """Get comprehensive pricing summary."""
        lowest_fare = self.get_lowest_fare()
        highest_fare = self.get_highest_fare()
        
        return {
            'route_id': self.route_id,
            'flight_number': self.flight_number,
            'departure_date': self.departure_date.isoformat(),
            'total_inventory': self.calculate_total_inventory(),
            'available_inventory': self.calculate_available_inventory(),
            'load_factor': self.calculate_load_factor(),
            'bookings_count': self.bookings_count,
            'current_revenue': self.current_revenue,
            'average_fare': self.calculate_average_fare(),
            'lowest_fare': {
                'class': lowest_fare[0] if lowest_fare else None,
                'price': lowest_fare[1].price if lowest_fare else None
            },
            'highest_fare': {
                'class': highest_fare[0] if highest_fare else None,
                'price': highest_fare[1].price if highest_fare else None
            },
            'available_classes': self.get_available_classes(),
            'pricing_strategy': self.pricing_strategy,
            'last_optimization': self.last_optimization.isoformat(),
            'needs_optimization': self.needs_optimization()
        }
    
    def export_data(self) -> Dict:
        """Export complete fare structure data."""
        return {
            'route_id': self.route_id,
            'flight_number': self.flight_number,
            'departure_date': self.departure_date.isoformat(),
            'fare_classes': {
                code: pp.export_data() for code, pp in self.fare_classes.items()
            },
            'pricing_strategy': self.pricing_strategy,
            'last_optimization': self.last_optimization.isoformat(),
            'optimization_frequency_hours': self.optimization_frequency_hours,
            'competitor_prices': self.competitor_prices,
            'demand_forecast': self.demand_forecast,
            'load_factor_target': self.load_factor_target,
            'revenue_target': self.revenue_target,
            'current_revenue': self.current_revenue,
            'bookings_count': self.bookings_count,
            'summary': self.get_pricing_summary()
        }
    
    def __str__(self) -> str:
        """String representation of fare structure."""
        return f"FareStructure({self.route_id}, {self.flight_number}, {self.departure_date})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"FareStructure(route='{self.route_id}', flight='{self.flight_number}', "
            f"date={self.departure_date}, classes={len(self.fare_classes)}, "
            f"load_factor={self.calculate_load_factor():.2f})"
        )