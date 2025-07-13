"""Demand modeling and booking curve analysis for airline pricing.

This module defines demand patterns, booking curves, and passenger behavior
models for airline revenue management and pricing optimization.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime, date, timedelta
from enum import Enum
import numpy as np
import math


class DemandType(Enum):
    """Types of demand patterns."""
    BUSINESS = "business"
    LEISURE = "leisure"
    VFR = "visiting_friends_relatives"  # Visiting Friends and Relatives
    EMERGENCY = "emergency"
    GROUP = "group"
    SEASONAL = "seasonal"


class BookingBehavior(Enum):
    """Passenger booking behavior patterns."""
    EARLY_BOOKER = "early_booker"  # Books well in advance
    LAST_MINUTE = "last_minute"    # Books close to departure
    PRICE_SENSITIVE = "price_sensitive"  # Highly responsive to price changes
    SCHEDULE_SENSITIVE = "schedule_sensitive"  # Prioritizes convenient times
    BRAND_LOYAL = "brand_loyal"    # Prefers specific airlines
    FLEXIBLE = "flexible"          # Can adjust travel dates


class SeasonalityType(Enum):
    """Types of seasonal demand patterns."""
    NONE = "none"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    HOLIDAY = "holiday"
    EVENT_DRIVEN = "event_driven"


@dataclass
class BookingCurve:
    """Represents the booking curve for a flight or route.
    
    The booking curve shows how bookings accumulate over time
    from the initial booking window to departure.
    """
    # Curve parameters
    total_capacity: int = 0
    booking_window_days: int = 365  # Days before departure when bookings open
    
    # Curve shape parameters (S-curve model)
    early_booking_rate: float = 0.1   # Fraction booked in first 50% of window
    late_booking_rate: float = 0.3    # Fraction booked in last 20% of window
    peak_booking_period: int = 60     # Days before departure when bookings peak
    
    # Demand elasticity
    price_elasticity: float = -1.2   # Price elasticity of demand
    schedule_elasticity: float = -0.5 # Schedule convenience elasticity
    
    # Seasonal and external factors
    seasonality_factor: float = 1.0  # Multiplier for seasonal demand
    competition_factor: float = 1.0  # Impact of competitive pricing
    economic_factor: float = 1.0     # Economic conditions impact
    
    # Booking behavior distribution
    behavior_mix: Dict[BookingBehavior, float] = field(default_factory=lambda: {
        BookingBehavior.EARLY_BOOKER: 0.25,
        BookingBehavior.LAST_MINUTE: 0.15,
        BookingBehavior.PRICE_SENSITIVE: 0.35,
        BookingBehavior.SCHEDULE_SENSITIVE: 0.15,
        BookingBehavior.BRAND_LOYAL: 0.10
    })
    
    # Metadata
    route_id: str = ""
    demand_type: DemandType = DemandType.LEISURE
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate booking curve parameters."""
        if self.total_capacity <= 0:
            raise ValueError("Total capacity must be positive")
        
        if self.booking_window_days <= 0:
            raise ValueError("Booking window must be positive")
        
        if not (0 <= self.early_booking_rate <= 1):
            raise ValueError("Early booking rate must be between 0 and 1")
        
        if not (0 <= self.late_booking_rate <= 1):
            raise ValueError("Late booking rate must be between 0 and 1")
        
        # Normalize behavior mix to sum to 1
        total_mix = sum(self.behavior_mix.values())
        if total_mix > 0:
            self.behavior_mix = {
                behavior: fraction / total_mix 
                for behavior, fraction in self.behavior_mix.items()
            }
    
    def calculate_cumulative_bookings(
        self, 
        days_before_departure: int,
        current_price: float = 0.0,
        base_price: float = 0.0
    ) -> int:
        """Calculate cumulative bookings at a given point in time."""
        if days_before_departure > self.booking_window_days:
            return 0
        
        if days_before_departure <= 0:
            return self.total_capacity
        
        # Calculate base booking progression using S-curve
        progress = 1 - (days_before_departure / self.booking_window_days)
        
        # S-curve formula: f(x) = 1 / (1 + e^(-k*(x-0.5)))
        # Adjust k based on booking behavior
        k = 6.0  # Steepness parameter
        
        if self.demand_type == DemandType.BUSINESS:
            k = 4.0  # Less steep, more linear for business travel
        elif self.demand_type == DemandType.LEISURE:
            k = 8.0  # Steeper, more last-minute bookings
        
        base_booking_rate = 1 / (1 + math.exp(-k * (progress - 0.5)))
        
        # Apply price elasticity if prices are provided
        price_adjustment = 1.0
        if current_price > 0 and base_price > 0 and current_price != base_price:
            price_ratio = current_price / base_price
            price_adjustment = price_ratio ** self.price_elasticity
        
        # Apply external factors
        total_adjustment = (
            price_adjustment * 
            self.seasonality_factor * 
            self.competition_factor * 
            self.economic_factor
        )
        
        # Calculate adjusted bookings
        adjusted_booking_rate = base_booking_rate * total_adjustment
        adjusted_booking_rate = max(0, min(1, adjusted_booking_rate))  # Clamp to [0,1]
        
        cumulative_bookings = int(self.total_capacity * adjusted_booking_rate)
        
        return min(cumulative_bookings, self.total_capacity)
    
    def calculate_daily_bookings(
        self, 
        days_before_departure: int,
        current_price: float = 0.0,
        base_price: float = 0.0
    ) -> int:
        """Calculate expected bookings for a specific day."""
        if days_before_departure > self.booking_window_days or days_before_departure <= 0:
            return 0
        
        # Get cumulative bookings for today and tomorrow
        today_cumulative = self.calculate_cumulative_bookings(
            days_before_departure, current_price, base_price
        )
        tomorrow_cumulative = self.calculate_cumulative_bookings(
            days_before_departure + 1, current_price, base_price
        )
        
        daily_bookings = today_cumulative - tomorrow_cumulative
        return max(0, daily_bookings)
    
    def get_booking_velocity(
        self, 
        days_before_departure: int,
        window_days: int = 7
    ) -> float:
        """Calculate booking velocity (bookings per day) over a time window."""
        total_bookings = 0
        
        for day in range(days_before_departure, days_before_departure + window_days):
            total_bookings += self.calculate_daily_bookings(day)
        
        return total_bookings / window_days if window_days > 0 else 0
    
    def predict_final_load_factor(
        self, 
        current_bookings: int,
        days_before_departure: int,
        current_price: float = 0.0,
        base_price: float = 0.0
    ) -> float:
        """Predict final load factor based on current booking pace."""
        if self.total_capacity <= 0:
            return 0.0
        
        # Calculate expected total bookings at departure
        expected_total = self.calculate_cumulative_bookings(
            0, current_price, base_price
        )
        
        # Adjust based on current booking pace vs expected pace
        expected_current = self.calculate_cumulative_bookings(
            days_before_departure, current_price, base_price
        )
        
        if expected_current > 0:
            pace_ratio = current_bookings / expected_current
            adjusted_total = expected_total * pace_ratio
        else:
            adjusted_total = expected_total
        
        final_load_factor = min(adjusted_total / self.total_capacity, 1.0)
        return final_load_factor
    
    def optimize_for_target_load_factor(
        self, 
        target_load_factor: float,
        days_before_departure: int,
        base_price: float
    ) -> float:
        """Find the price that achieves target load factor."""
        if target_load_factor <= 0 or target_load_factor > 1:
            return base_price
        
        target_bookings = int(self.total_capacity * target_load_factor)
        
        # Binary search for optimal price
        min_price = base_price * 0.5
        max_price = base_price * 2.0
        tolerance = 0.01
        
        for _ in range(20):  # Max iterations
            test_price = (min_price + max_price) / 2
            predicted_bookings = self.calculate_cumulative_bookings(
                0, test_price, base_price
            )
            
            if abs(predicted_bookings - target_bookings) < tolerance * self.total_capacity:
                return test_price
            
            if predicted_bookings > target_bookings:
                min_price = test_price  # Price too low, increase
            else:
                max_price = test_price  # Price too high, decrease
        
        return (min_price + max_price) / 2
    
    def get_booking_curve_data(
        self, 
        price: float = 0.0,
        base_price: float = 0.0
    ) -> List[Tuple[int, int, int]]:
        """Get complete booking curve data.
        
        Returns:
            List of (days_before_departure, daily_bookings, cumulative_bookings)
        """
        curve_data = []
        
        for days in range(self.booking_window_days, -1, -1):
            daily = self.calculate_daily_bookings(days, price, base_price)
            cumulative = self.calculate_cumulative_bookings(days, price, base_price)
            curve_data.append((days, daily, cumulative))
        
        return curve_data
    
    def export_data(self) -> Dict:
        """Export booking curve data."""
        return {
            'route_id': self.route_id,
            'demand_type': self.demand_type.value,
            'capacity': self.total_capacity,
            'booking_window_days': self.booking_window_days,
            'curve_parameters': {
                'early_booking_rate': self.early_booking_rate,
                'late_booking_rate': self.late_booking_rate,
                'peak_booking_period': self.peak_booking_period
            },
            'elasticity': {
                'price_elasticity': self.price_elasticity,
                'schedule_elasticity': self.schedule_elasticity
            },
            'external_factors': {
                'seasonality_factor': self.seasonality_factor,
                'competition_factor': self.competition_factor,
                'economic_factor': self.economic_factor
            },
            'behavior_mix': {
                behavior.value: fraction 
                for behavior, fraction in self.behavior_mix.items()
            },
            'last_updated': self.last_updated.isoformat()
        }


@dataclass
class DemandPattern:
    """Represents demand patterns for a route or market segment."""
    # Basic demand characteristics
    base_demand: float = 0.0  # Base daily demand
    demand_type: DemandType = DemandType.LEISURE
    seasonality_type: SeasonalityType = SeasonalityType.NONE
    
    # Temporal patterns
    weekly_pattern: List[float] = field(default_factory=lambda: [1.0] * 7)  # Mon-Sun multipliers
    monthly_pattern: List[float] = field(default_factory=lambda: [1.0] * 12)  # Jan-Dec multipliers
    hourly_pattern: List[float] = field(default_factory=lambda: [1.0] * 24)  # 0-23 hour multipliers
    
    # Market characteristics
    market_size: int = 0  # Total addressable market
    market_penetration: float = 0.1  # Fraction of market captured
    growth_rate: float = 0.0  # Annual growth rate
    
    # Price sensitivity
    price_elasticity: float = -1.0
    income_elasticity: float = 1.0
    cross_elasticity: Dict[str, float] = field(default_factory=dict)  # Elasticity vs other routes
    
    # Booking behavior
    advance_booking_curve: Optional[BookingCurve] = None
    no_show_rate: float = 0.05
    cancellation_rate: float = 0.10
    
    # External factors
    economic_sensitivity: float = 1.0
    weather_sensitivity: float = 0.1
    event_impact_multiplier: float = 1.0
    
    # Metadata
    route_id: str = ""
    market_segment: str = ""
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate and initialize demand pattern."""
        if self.base_demand < 0:
            raise ValueError("Base demand must be non-negative")
        
        if not (0 <= self.market_penetration <= 1):
            raise ValueError("Market penetration must be between 0 and 1")
        
        if not (0 <= self.no_show_rate <= 1):
            raise ValueError("No-show rate must be between 0 and 1")
        
        if not (0 <= self.cancellation_rate <= 1):
            raise ValueError("Cancellation rate must be between 0 and 1")
        
        # Ensure pattern arrays have correct length
        if len(self.weekly_pattern) != 7:
            self.weekly_pattern = [1.0] * 7
        
        if len(self.monthly_pattern) != 12:
            self.monthly_pattern = [1.0] * 12
        
        if len(self.hourly_pattern) != 24:
            self.hourly_pattern = [1.0] * 24
        
        # Create default booking curve if not provided
        if self.advance_booking_curve is None:
            self.advance_booking_curve = BookingCurve(
                total_capacity=150,  # Default aircraft capacity
                demand_type=self.demand_type,
                route_id=self.route_id
            )
    
    def calculate_demand(
        self, 
        target_date: date,
        target_hour: int = 12,
        price: float = 0.0,
        base_price: float = 0.0,
        external_factors: Optional[Dict[str, float]] = None
    ) -> float:
        """Calculate demand for a specific date and time."""
        # Start with base demand
        demand = self.base_demand
        
        # Apply temporal patterns
        weekday = target_date.weekday()  # 0=Monday, 6=Sunday
        month = target_date.month - 1    # 0=January, 11=December
        hour = max(0, min(23, target_hour))  # Clamp to valid hour range
        
        demand *= self.weekly_pattern[weekday]
        demand *= self.monthly_pattern[month]
        demand *= self.hourly_pattern[hour]
        
        # Apply price elasticity
        if price > 0 and base_price > 0 and price != base_price:
            price_ratio = price / base_price
            price_effect = price_ratio ** self.price_elasticity
            demand *= price_effect
        
        # Apply external factors
        if external_factors:
            economic_factor = external_factors.get('economic_index', 1.0)
            weather_factor = external_factors.get('weather_index', 1.0)
            event_factor = external_factors.get('event_multiplier', 1.0)
            
            demand *= (economic_factor ** self.economic_sensitivity)
            demand *= (1 + (weather_factor - 1) * self.weather_sensitivity)
            demand *= (event_factor * self.event_impact_multiplier)
        
        # Apply market growth
        years_from_base = (target_date.year - self.last_updated.year)
        if years_from_base > 0:
            growth_multiplier = (1 + self.growth_rate) ** years_from_base
            demand *= growth_multiplier
        
        return max(0, demand)
    
    def forecast_demand(
        self, 
        start_date: date,
        end_date: date,
        price_schedule: Optional[Dict[date, float]] = None,
        base_price: float = 0.0
    ) -> Dict[date, float]:
        """Forecast demand over a date range."""
        forecast = {}
        current_date = start_date
        
        while current_date <= end_date:
            price = price_schedule.get(current_date, 0.0) if price_schedule else 0.0
            
            daily_demand = self.calculate_demand(
                target_date=current_date,
                price=price,
                base_price=base_price
            )
            
            forecast[current_date] = daily_demand
            current_date += timedelta(days=1)
        
        return forecast
    
    def calculate_price_sensitivity(
        self, 
        base_price: float,
        price_range: Tuple[float, float] = (0.5, 2.0)
    ) -> Dict[float, float]:
        """Calculate demand response to different price levels."""
        sensitivity_data = {}
        
        min_multiplier, max_multiplier = price_range
        price_points = np.linspace(
            base_price * min_multiplier,
            base_price * max_multiplier,
            20
        )
        
        base_demand = self.calculate_demand(
            target_date=date.today(),
            price=base_price,
            base_price=base_price
        )
        
        for price in price_points:
            demand = self.calculate_demand(
                target_date=date.today(),
                price=price,
                base_price=base_price
            )
            
            sensitivity_data[price] = demand
        
        return sensitivity_data
    
    def update_seasonality_pattern(
        self, 
        historical_data: Dict[date, float]
    ) -> None:
        """Update seasonality patterns based on historical data."""
        if not historical_data:
            return
        
        # Calculate monthly averages
        monthly_totals = [0.0] * 12
        monthly_counts = [0] * 12
        
        for date_key, demand_value in historical_data.items():
            month_idx = date_key.month - 1
            monthly_totals[month_idx] += demand_value
            monthly_counts[month_idx] += 1
        
        # Calculate monthly averages and normalize
        monthly_averages = [
            total / count if count > 0 else 1.0
            for total, count in zip(monthly_totals, monthly_counts)
        ]
        
        overall_average = sum(monthly_averages) / len(monthly_averages)
        
        if overall_average > 0:
            self.monthly_pattern = [
                avg / overall_average for avg in monthly_averages
            ]
        
        # Calculate weekly averages
        weekly_totals = [0.0] * 7
        weekly_counts = [0] * 7
        
        for date_key, demand_value in historical_data.items():
            weekday_idx = date_key.weekday()
            weekly_totals[weekday_idx] += demand_value
            weekly_counts[weekday_idx] += 1
        
        weekly_averages = [
            total / count if count > 0 else 1.0
            for total, count in zip(weekly_totals, weekly_counts)
        ]
        
        overall_weekly_average = sum(weekly_averages) / len(weekly_averages)
        
        if overall_weekly_average > 0:
            self.weekly_pattern = [
                avg / overall_weekly_average for avg in weekly_averages
            ]
        
        self.last_updated = datetime.now()
    
    def get_peak_demand_periods(
        self, 
        start_date: date,
        end_date: date,
        threshold_multiplier: float = 1.5
    ) -> List[Tuple[date, date, float]]:
        """Identify peak demand periods."""
        forecast = self.forecast_demand(start_date, end_date)
        average_demand = sum(forecast.values()) / len(forecast) if forecast else 0
        
        peak_threshold = average_demand * threshold_multiplier
        peak_periods = []
        
        current_peak_start = None
        current_peak_max = 0
        
        for date_key in sorted(forecast.keys()):
            demand = forecast[date_key]
            
            if demand >= peak_threshold:
                if current_peak_start is None:
                    current_peak_start = date_key
                    current_peak_max = demand
                else:
                    current_peak_max = max(current_peak_max, demand)
            else:
                if current_peak_start is not None:
                    # End of peak period
                    peak_periods.append((
                        current_peak_start,
                        date_key - timedelta(days=1),
                        current_peak_max
                    ))
                    current_peak_start = None
                    current_peak_max = 0
        
        # Handle case where peak period extends to end date
        if current_peak_start is not None:
            peak_periods.append((
                current_peak_start,
                end_date,
                current_peak_max
            ))
        
        return peak_periods
    
    def export_data(self) -> Dict:
        """Export demand pattern data."""
        return {
            'route_id': self.route_id,
            'market_segment': self.market_segment,
            'demand_type': self.demand_type.value,
            'seasonality_type': self.seasonality_type.value,
            'base_characteristics': {
                'base_demand': self.base_demand,
                'market_size': self.market_size,
                'market_penetration': self.market_penetration,
                'growth_rate': self.growth_rate
            },
            'temporal_patterns': {
                'weekly_pattern': self.weekly_pattern,
                'monthly_pattern': self.monthly_pattern,
                'hourly_pattern': self.hourly_pattern
            },
            'elasticity': {
                'price_elasticity': self.price_elasticity,
                'income_elasticity': self.income_elasticity,
                'cross_elasticity': self.cross_elasticity
            },
            'booking_behavior': {
                'no_show_rate': self.no_show_rate,
                'cancellation_rate': self.cancellation_rate,
                'advance_booking_curve': self.advance_booking_curve.export_data() if self.advance_booking_curve else None
            },
            'external_sensitivity': {
                'economic_sensitivity': self.economic_sensitivity,
                'weather_sensitivity': self.weather_sensitivity,
                'event_impact_multiplier': self.event_impact_multiplier
            },
            'last_updated': self.last_updated.isoformat()
        }
    
    def __str__(self) -> str:
        """String representation of demand pattern."""
        return f"DemandPattern({self.route_id}, {self.demand_type.value}, base={self.base_demand})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"DemandPattern(route='{self.route_id}', type={self.demand_type.value}, "
            f"base_demand={self.base_demand}, elasticity={self.price_elasticity})"
        )