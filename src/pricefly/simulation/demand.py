"""Demand simulation and forecasting for airline pricing."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import stats
from collections import defaultdict
import math

from ..models.passenger import CustomerSegment, TravelPurpose
from ..models.airport import Route
from ..models.airline import CabinClass


class SeasonalityType(Enum):
    """Types of seasonal patterns."""
    BUSINESS = "business"  # Weekday peaks, holiday lows
    LEISURE = "leisure"  # Weekend peaks, holiday highs
    MIXED = "mixed"  # Combination of business and leisure
    SPECIAL_EVENT = "special_event"  # Conference, sports, etc.


class DemandShock(Enum):
    """Types of demand shocks."""
    POSITIVE = "positive"  # Unexpected increase
    NEGATIVE = "negative"  # Unexpected decrease
    VOLATILITY = "volatility"  # Increased uncertainty


@dataclass
class DemandForecast:
    """Demand forecast for a specific route and time period."""
    route: str
    date: datetime
    forecast_horizon: int  # Days ahead
    
    # Base demand
    base_demand: float
    
    # Segment-specific demand
    business_demand: float
    leisure_demand: float
    vfr_demand: float  # Visiting friends/relatives
    
    # Confidence intervals
    lower_bound: float
    upper_bound: float
    confidence_level: float = 0.95
    
    # Factors affecting demand
    seasonal_factor: float = 1.0
    economic_factor: float = 1.0
    competitive_factor: float = 1.0
    event_factor: float = 1.0
    
    # Price elasticity
    price_elasticity: float = -1.2
    
    # Forecast accuracy metrics
    forecast_error: Optional[float] = None
    model_confidence: float = 0.8


@dataclass
class BookingCurve:
    """Booking pattern over time before departure."""
    route: str
    travel_purpose: TravelPurpose
    
    # Booking timeline (days before departure -> booking probability)
    booking_timeline: Dict[int, float] = field(default_factory=dict)
    
    # Peak booking periods
    early_booking_peak: int = 60  # Days before departure
    late_booking_peak: int = 14
    
    # Booking velocity parameters
    max_booking_rate: float = 0.15  # Max daily booking rate
    booking_acceleration: float = 1.5  # How quickly bookings accelerate
    
    # Price sensitivity over time
    price_sensitivity_timeline: Dict[int, float] = field(default_factory=dict)


@dataclass
class DemandScenario:
    """Scenario parameters for demand simulation."""
    scenario_name: str
    
    # Base demand parameters
    base_demand_multiplier: float = 1.0
    demand_volatility: float = 0.1
    
    # Economic factors
    economic_growth_rate: float = 0.025
    unemployment_rate: float = 0.05
    consumer_confidence: float = 0.8
    
    # Seasonal factors
    seasonality_amplitude: float = 0.3
    peak_season_multiplier: float = 1.5
    off_season_multiplier: float = 0.7
    
    # Price elasticity
    business_price_elasticity: float = -0.8
    leisure_price_elasticity: float = -1.5
    
    # External shocks
    demand_shock_probability: float = 0.05
    shock_magnitude_range: Tuple[float, float] = (-0.3, 0.3)
    
    # Competition effects
    competitive_response_factor: float = 0.5
    market_share_sensitivity: float = 0.3
    
    # Booking behavior
    advance_booking_trend: float = 0.0  # Shift towards earlier/later booking
    last_minute_booking_factor: float = 1.0
    
    # Special events
    special_events_impact: float = 0.2
    event_duration_days: int = 7
    
    def apply_scenario_adjustments(
        self, 
        base_forecast: DemandForecast
    ) -> DemandForecast:
        """Apply scenario adjustments to a base demand forecast."""
        adjusted_forecast = DemandForecast(
            route=base_forecast.route,
            date=base_forecast.date,
            forecast_horizon=base_forecast.forecast_horizon,
            base_demand=base_forecast.base_demand * self.base_demand_multiplier,
            business_demand=base_forecast.business_demand * self.base_demand_multiplier,
            leisure_demand=base_forecast.leisure_demand * self.base_demand_multiplier,
            vfr_demand=base_forecast.vfr_demand * self.base_demand_multiplier,
            lower_bound=base_forecast.lower_bound * self.base_demand_multiplier,
            upper_bound=base_forecast.upper_bound * self.base_demand_multiplier,
            confidence_level=base_forecast.confidence_level,
            seasonal_factor=base_forecast.seasonal_factor * self.seasonality_amplitude,
            economic_factor=base_forecast.economic_factor * self.consumer_confidence,
            competitive_factor=base_forecast.competitive_factor * self.competitive_response_factor,
            event_factor=base_forecast.event_factor,
            price_elasticity=base_forecast.price_elasticity
        )
        
        return adjusted_forecast


class DemandSimulator:
    """Simulates passenger demand patterns and forecasting."""
    
    def __init__(
        self,
        routes: List[Route],
        customer_segments: List[CustomerSegment],
        base_demand_params: Optional[Dict[str, Any]] = None
    ):
        self.routes = {f"{route.origin}-{route.destination}": route for route in routes}
        self.customer_segments = {segment.segment_id: segment for segment in customer_segments}
        
        # Base demand parameters
        self.base_demand_params = base_demand_params or {
            'daily_demand_mean': 150,
            'daily_demand_std': 30,
            'seasonality_amplitude': 0.3,
            'trend_factor': 0.02,  # Annual growth
            'noise_level': 0.1
        }
        
        # Booking curves for different segments
        self.booking_curves = self._initialize_booking_curves()
        
        # Historical demand data
        self.historical_demand: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Demand forecasting models
        self.forecast_models: Dict[str, Any] = {}
        
        # External factors
        self.economic_indicators = {
            'gdp_growth': 0.025,
            'unemployment_rate': 0.05,
            'consumer_confidence': 0.8,
            'fuel_price_index': 1.0
        }
        
        # Event calendar
        self.events_calendar: Dict[datetime, Dict[str, Any]] = {}
        
        self.logger = logging.getLogger(__name__)
    
    def _initialize_booking_curves(self) -> Dict[str, BookingCurve]:
        """Initialize booking curves for different travel purposes."""
        curves = {}
        
        # Business travel booking curve
        business_timeline = {}
        business_price_sensitivity = {}
        
        for days_ahead in range(1, 366):
            if days_ahead <= 7:
                # Last-minute business bookings
                booking_prob = 0.08 * (8 - days_ahead) / 7
                price_sensitivity = 0.3  # Less price sensitive
            elif days_ahead <= 30:
                # Normal business booking window
                booking_prob = 0.12 * np.exp(-(days_ahead - 14) ** 2 / 200)
                price_sensitivity = 0.5
            elif days_ahead <= 90:
                # Early business bookings
                booking_prob = 0.06 * np.exp(-(days_ahead - 60) ** 2 / 800)
                price_sensitivity = 0.7
            else:
                # Very early bookings
                booking_prob = 0.02
                price_sensitivity = 0.8
            
            business_timeline[days_ahead] = booking_prob
            business_price_sensitivity[days_ahead] = price_sensitivity
        
        curves['business'] = BookingCurve(
            route="all",
            travel_purpose=TravelPurpose.BUSINESS,
            booking_timeline=business_timeline,
            early_booking_peak=60,
            late_booking_peak=14,
            max_booking_rate=0.12,
            price_sensitivity_timeline=business_price_sensitivity
        )
        
        # Leisure travel booking curve
        leisure_timeline = {}
        leisure_price_sensitivity = {}
        
        for days_ahead in range(1, 366):
            if days_ahead <= 14:
                # Last-minute leisure bookings
                booking_prob = 0.03 * (15 - days_ahead) / 14
                price_sensitivity = 1.5  # Very price sensitive
            elif days_ahead <= 60:
                # Peak leisure booking window
                booking_prob = 0.15 * np.exp(-(days_ahead - 30) ** 2 / 300)
                price_sensitivity = 1.2
            elif days_ahead <= 120:
                # Early leisure bookings
                booking_prob = 0.10 * np.exp(-(days_ahead - 90) ** 2 / 600)
                price_sensitivity = 1.0
            else:
                # Very early leisure bookings
                booking_prob = 0.05
                price_sensitivity = 0.9
            
            leisure_timeline[days_ahead] = booking_prob
            leisure_price_sensitivity[days_ahead] = price_sensitivity
        
        curves['leisure'] = BookingCurve(
            route="all",
            travel_purpose=TravelPurpose.LEISURE,
            booking_timeline=leisure_timeline,
            early_booking_peak=90,
            late_booking_peak=30,
            max_booking_rate=0.15,
            price_sensitivity_timeline=leisure_price_sensitivity
        )
        
        # VFR (Visiting Friends/Relatives) booking curve
        vfr_timeline = {}
        vfr_price_sensitivity = {}
        
        for days_ahead in range(1, 366):
            if days_ahead <= 21:
                # VFR bookings often made closer to travel
                booking_prob = 0.08 * (22 - days_ahead) / 21
                price_sensitivity = 1.3
            elif days_ahead <= 90:
                # Normal VFR booking window
                booking_prob = 0.10 * np.exp(-(days_ahead - 45) ** 2 / 400)
                price_sensitivity = 1.1
            else:
                # Early VFR bookings
                booking_prob = 0.04
                price_sensitivity = 1.0
            
            vfr_timeline[days_ahead] = booking_prob
            vfr_price_sensitivity[days_ahead] = price_sensitivity
        
        curves['vfr'] = BookingCurve(
            route="all",
            travel_purpose=TravelPurpose.VFR,
            booking_timeline=vfr_timeline,
            early_booking_peak=45,
            late_booking_peak=21,
            max_booking_rate=0.10,
            price_sensitivity_timeline=vfr_price_sensitivity
        )
        
        return curves
    
    def generate_base_demand(
        self,
        route: Route,
        date: datetime,
        seasonality_type: SeasonalityType = SeasonalityType.MIXED
    ) -> Dict[str, float]:
        """Generate base demand for a route on a specific date."""
        route_key = f"{route.origin}-{route.destination}"
        
        # Base demand from route characteristics
        # Calculate daily demand estimate from annual demand
        base_daily_demand = route.annual_demand / 365.0
        
        # Apply seasonal factors
        seasonal_factor = self._calculate_seasonal_factor(date, seasonality_type)
        
        # Apply day-of-week factors
        dow_factor = self._calculate_day_of_week_factor(date, seasonality_type)
        
        # Apply economic factors
        economic_factor = self._calculate_economic_factor(route)
        
        # Apply route-specific factors
        route_factor = self._calculate_route_factor(route, date)
        
        # Calculate total demand
        total_demand = (
            base_daily_demand *
            seasonal_factor *
            dow_factor *
            economic_factor *
            route_factor
        )
        
        # Add noise
        noise = np.random.normal(0, total_demand * self.base_demand_params['noise_level'])
        total_demand = max(0, total_demand + noise)
        
        # Segment demand by travel purpose
        segment_distribution = self._get_segment_distribution(route, date)
        
        demand_by_segment = {
            'business': total_demand * segment_distribution['business'],
            'leisure': total_demand * segment_distribution['leisure'],
            'vfr': total_demand * segment_distribution['vfr']
        }
        
        return demand_by_segment
    
    def _calculate_seasonal_factor(self, date: datetime, seasonality_type: SeasonalityType) -> float:
        """Calculate seasonal demand factor."""
        day_of_year = date.timetuple().tm_yday
        
        if seasonality_type == SeasonalityType.BUSINESS:
            # Business travel peaks in spring/fall, low in summer/winter holidays
            seasonal_base = (
                0.8 * np.sin(2 * np.pi * day_of_year / 365) +  # Annual cycle
                0.3 * np.sin(4 * np.pi * day_of_year / 365)    # Semi-annual cycle
            )
            # Reduce during holiday periods
            if date.month in [7, 8, 12]:  # Summer and winter holidays
                seasonal_base -= 0.3
        
        elif seasonality_type == SeasonalityType.LEISURE:
            # Leisure travel peaks in summer and winter holidays
            seasonal_base = (
                0.6 * np.sin(2 * np.pi * (day_of_year - 90) / 365) +  # Summer peak
                0.4 * np.sin(2 * np.pi * (day_of_year - 350) / 365)   # Winter peak
            )
            # Boost during holiday periods
            if date.month in [6, 7, 8, 12]:  # Summer and winter holidays
                seasonal_base += 0.4
        
        else:  # MIXED or SPECIAL_EVENT
            # Balanced seasonal pattern
            seasonal_base = (
                0.4 * np.sin(2 * np.pi * day_of_year / 365) +
                0.2 * np.sin(4 * np.pi * day_of_year / 365)
            )
        
        # Normalize to positive factor around 1.0
        seasonal_factor = 1.0 + seasonal_base * self.base_demand_params['seasonality_amplitude']
        
        return max(0.3, seasonal_factor)
    
    def _calculate_day_of_week_factor(self, date: datetime, seasonality_type: SeasonalityType) -> float:
        """Calculate day-of-week demand factor."""
        weekday = date.weekday()  # 0 = Monday, 6 = Sunday
        
        if seasonality_type == SeasonalityType.BUSINESS:
            # Business travel peaks Tuesday-Thursday
            dow_factors = [0.9, 1.2, 1.3, 1.2, 1.0, 0.6, 0.4]  # Mon-Sun
        elif seasonality_type == SeasonalityType.LEISURE:
            # Leisure travel peaks Friday-Sunday
            dow_factors = [0.8, 0.7, 0.8, 0.9, 1.3, 1.4, 1.2]  # Mon-Sun
        else:  # MIXED
            # Balanced pattern
            dow_factors = [0.9, 1.0, 1.1, 1.1, 1.2, 1.1, 0.9]  # Mon-Sun
        
        return dow_factors[weekday]
    
    def _calculate_economic_factor(self, route: Route) -> float:
        """Calculate economic impact on demand."""
        # GDP growth impact
        gdp_impact = 1.0 + (self.economic_indicators['gdp_growth'] - 0.02) * 2.0
        
        # Unemployment impact
        unemployment_impact = 1.0 - (self.economic_indicators['unemployment_rate'] - 0.05) * 1.5
        
        # Consumer confidence impact
        confidence_impact = 0.5 + 0.5 * self.economic_indicators['consumer_confidence']
        
        # Fuel price impact (affects ticket prices)
        fuel_impact = 1.0 - (self.economic_indicators['fuel_price_index'] - 1.0) * 0.3
        
        # Route-specific economic sensitivity
        if route.route_type.value in ['domestic_short', 'domestic_medium']:
            economic_sensitivity = 0.8  # Domestic routes less sensitive
        else:
            economic_sensitivity = 1.2  # International routes more sensitive
        
        economic_factor = (
            gdp_impact * 0.3 +
            unemployment_impact * 0.2 +
            confidence_impact * 0.3 +
            fuel_impact * 0.2
        )
        
        # Apply route sensitivity
        economic_factor = 1.0 + (economic_factor - 1.0) * economic_sensitivity
        
        return max(0.5, economic_factor)
    
    def _calculate_route_factor(self, route: Route, date: datetime) -> float:
        """Calculate route-specific demand factors."""
        route_factor = 1.0
        
        # Hub vs. point-to-point
        if hasattr(route, 'is_hub_route') and route.is_hub_route:
            route_factor *= 1.1  # Hub routes have higher demand
        
        # Competition factor based on competitor count
        if route.competitor_count > 4:
            route_factor *= 1.05  # High competition routes have more stimulated demand
        elif route.competitor_count <= 1:
            route_factor *= 0.95  # Low competition may have suppressed demand
        
        # Distance factor
        if route.distance_km < 500:
            route_factor *= 0.9  # Short routes compete with ground transport
        elif route.distance_km > 3000:
            route_factor *= 1.1  # Long routes have less competition
        
        # Check for special events
        event_factor = self._get_event_factor(route, date)
        route_factor *= event_factor
        
        return route_factor
    
    def _get_segment_distribution(self, route: Route, date: datetime) -> Dict[str, float]:
        """Get distribution of demand across customer segments."""
        # Base distribution
        if route.route_type.value in ['domestic_short', 'domestic_medium']:
            # Domestic routes have more business travel
            base_distribution = {
                'business': 0.35,
                'leisure': 0.50,
                'vfr': 0.15
            }
        else:
            # International routes have more leisure travel
            base_distribution = {
                'business': 0.25,
                'leisure': 0.60,
                'vfr': 0.15
            }
        
        # Adjust based on day of week
        weekday = date.weekday()
        if weekday < 5:  # Weekday
            base_distribution['business'] *= 1.2
            base_distribution['leisure'] *= 0.8
        else:  # Weekend
            base_distribution['business'] *= 0.6
            base_distribution['leisure'] *= 1.3
        
        # Normalize
        total = sum(base_distribution.values())
        return {k: v / total for k, v in base_distribution.items()}
    
    def _get_event_factor(self, route: Route, date: datetime) -> float:
        """Get demand factor for special events."""
        event_factor = 1.0
        
        # Check events calendar
        for event_date, event_info in self.events_calendar.items():
            if abs((date - event_date).days) <= event_info.get('impact_days', 3):
                if route.destination in event_info.get('affected_cities', []):
                    event_factor *= event_info.get('demand_multiplier', 1.0)
        
        # Add some random events
        if np.random.random() < 0.05:  # 5% chance of random event
            event_factor *= np.random.uniform(0.8, 1.5)
        
        return event_factor
    
    def forecast_demand(
        self,
        route: Route,
        forecast_date: datetime,
        forecast_horizon: int = 30,
        current_prices: Optional[Dict[str, float]] = None
    ) -> DemandForecast:
        """Generate demand forecast for a route."""
        route_key = f"{route.origin}-{route.destination}"
        
        # Generate base demand
        base_demand_segments = self.generate_base_demand(route, forecast_date)
        base_demand = sum(base_demand_segments.values())
        
        # Apply price elasticity if prices provided
        if current_prices:
            price_factor = self._calculate_price_elasticity_factor(route, current_prices)
            base_demand *= price_factor
            for segment in base_demand_segments:
                base_demand_segments[segment] *= price_factor
        
        # Calculate confidence intervals
        forecast_std = base_demand * 0.2  # 20% standard deviation
        confidence_level = 0.95
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        lower_bound = max(0, base_demand - z_score * forecast_std)
        upper_bound = base_demand + z_score * forecast_std
        
        # Calculate various factors
        seasonal_factor = self._calculate_seasonal_factor(forecast_date, SeasonalityType.MIXED)
        economic_factor = self._calculate_economic_factor(route)
        competitive_factor = 1.0  # Would be calculated based on competitor analysis
        event_factor = self._get_event_factor(route, forecast_date)
        
        # Estimate price elasticity
        price_elasticity = self._estimate_price_elasticity(route, base_demand_segments)
        
        # Model confidence based on historical accuracy
        model_confidence = self._calculate_model_confidence(route_key, forecast_horizon)
        
        forecast = DemandForecast(
            route=route_key,
            date=forecast_date,
            forecast_horizon=forecast_horizon,
            base_demand=base_demand,
            business_demand=base_demand_segments['business'],
            leisure_demand=base_demand_segments['leisure'],
            vfr_demand=base_demand_segments['vfr'],
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=confidence_level,
            seasonal_factor=seasonal_factor,
            economic_factor=economic_factor,
            competitive_factor=competitive_factor,
            event_factor=event_factor,
            price_elasticity=price_elasticity,
            model_confidence=model_confidence
        )
        
        return forecast
    
    def _calculate_price_elasticity_factor(self, route: Route, current_prices: Dict[str, float]) -> float:
        """Calculate demand adjustment factor based on current prices."""
        # Get historical average price for route
        historical_avg_price = self._get_historical_average_price(route)
        
        if not historical_avg_price:
            return 1.0
        
        # Calculate price change
        current_avg_price = np.mean(list(current_prices.values()))
        price_change = (current_avg_price - historical_avg_price) / historical_avg_price
        
        # Apply elasticity (typically negative)
        elasticity = -1.2  # Base elasticity
        
        # Adjust elasticity based on route characteristics
        if route.route_type.value in ['domestic_short']:
            elasticity *= 1.5  # Short routes more elastic (more substitutes)
        elif route.route_type.value in ['international_long']:
            elasticity *= 0.7  # Long international routes less elastic
        
        # Calculate demand factor
        demand_factor = (1 + price_change) ** elasticity
        
        return max(0.3, demand_factor)  # Minimum 30% of base demand
    
    def _get_historical_average_price(self, route: Route) -> Optional[float]:
        """Get historical average price for a route."""
        route_key = f"{route.origin}-{route.destination}"
        
        if route_key in self.historical_demand:
            prices = []
            for record in self.historical_demand[route_key][-30:]:  # Last 30 records
                if 'average_price' in record:
                    prices.append(record['average_price'])
            
            if prices:
                return np.mean(prices)
        
        # Fallback to route-based estimate
        base_price = 100 + route.distance_km * 0.15  # Simple distance-based pricing
        return base_price
    
    def _estimate_price_elasticity(self, route: Route, demand_segments: Dict[str, float]) -> float:
        """Estimate price elasticity for the route."""
        # Base elasticity by segment
        segment_elasticities = {
            'business': -0.8,  # Less price sensitive
            'leisure': -1.5,   # More price sensitive
            'vfr': -1.2        # Moderately price sensitive
        }
        
        # Weight by segment demand
        total_demand = sum(demand_segments.values())
        if total_demand == 0:
            return -1.2
        
        weighted_elasticity = sum(
            segment_elasticities[segment] * demand / total_demand
            for segment, demand in demand_segments.items()
        )
        
        # Adjust for route characteristics based on competitor count
        if route.competitor_count > 4:
            weighted_elasticity *= 1.3  # More elastic with high competition
        elif route.competitor_count <= 1:
            weighted_elasticity *= 0.7  # Less elastic with low competition
        
        return weighted_elasticity
    
    def _calculate_model_confidence(self, route_key: str, forecast_horizon: int) -> float:
        """Calculate model confidence based on historical accuracy."""
        base_confidence = 0.8
        
        # Reduce confidence for longer horizons
        horizon_penalty = min(0.3, forecast_horizon * 0.01)
        
        # Adjust based on historical data availability
        if route_key in self.historical_demand:
            data_points = len(self.historical_demand[route_key])
            if data_points > 100:
                data_bonus = 0.1
            elif data_points > 50:
                data_bonus = 0.05
            else:
                data_bonus = -0.1
        else:
            data_bonus = -0.2
        
        confidence = base_confidence - horizon_penalty + data_bonus
        return max(0.3, min(0.95, confidence))
    
    def simulate_booking_behavior(
        self,
        route: Route,
        departure_date: datetime,
        current_date: datetime,
        prices: Dict[str, float],
        available_seats: Dict[str, int]
    ) -> Dict[str, Dict[str, Any]]:
        """Simulate booking behavior for different customer segments."""
        days_ahead = (departure_date - current_date).days
        
        if days_ahead <= 0:
            return {}
        
        route_key = f"{route.origin}-{route.destination}"
        
        # Get base demand for the departure date
        base_demand_segments = self.generate_base_demand(route, departure_date)
        
        # Simulate bookings for each segment
        booking_results = {}
        
        for segment_name, segment_demand in base_demand_segments.items():
            # Get booking curve for this segment
            if segment_name == 'business':
                curve = self.booking_curves['business']
            elif segment_name == 'leisure':
                curve = self.booking_curves['leisure']
            else:  # vfr
                curve = self.booking_curves['vfr']
            
            # Calculate booking probability for this time point
            booking_prob = curve.booking_timeline.get(days_ahead, 0.01)
            
            # Calculate price sensitivity
            price_sensitivity = curve.price_sensitivity_timeline.get(days_ahead, 1.0)
            
            # Simulate bookings for each cabin class
            segment_bookings = {}
            
            for cabin_class, price in prices.items():
                available = available_seats.get(cabin_class, 0)
                
                if available <= 0:
                    segment_bookings[cabin_class] = {
                        'bookings': 0,
                        'revenue': 0,
                        'load_factor': 0,
                        'price_paid': price
                    }
                    continue
                
                # Calculate demand for this cabin class
                if cabin_class == 'economy':
                    class_demand = segment_demand * 0.7
                elif cabin_class == 'business':
                    class_demand = segment_demand * 0.25 if segment_name == 'business' else segment_demand * 0.05
                elif cabin_class == 'first':
                    class_demand = segment_demand * 0.05 if segment_name == 'business' else segment_demand * 0.01
                else:
                    class_demand = segment_demand * 0.1
                
                # Apply booking probability
                expected_bookings = class_demand * booking_prob
                
                # Apply price sensitivity
                reference_price = self._get_reference_price(route, cabin_class)
                price_factor = (reference_price / price) ** price_sensitivity if price > 0 else 0
                expected_bookings *= price_factor
                
                # Add randomness
                actual_bookings = np.random.poisson(max(0, expected_bookings))
                actual_bookings = min(actual_bookings, available)
                
                # Calculate metrics
                revenue = actual_bookings * price
                load_factor = actual_bookings / available if available > 0 else 0
                
                segment_bookings[cabin_class] = {
                    'bookings': actual_bookings,
                    'revenue': revenue,
                    'load_factor': load_factor,
                    'price_paid': price,
                    'demand_factor': price_factor,
                    'booking_probability': booking_prob
                }
            
            booking_results[segment_name] = segment_bookings
        
        return booking_results
    
    def _get_reference_price(self, route: Route, cabin_class: str) -> float:
        """Get reference price for price sensitivity calculations."""
        base_price = 100 + route.distance_km * 0.15
        
        if cabin_class == 'economy':
            return base_price
        elif cabin_class == 'business':
            return base_price * 3.0
        elif cabin_class == 'first':
            return base_price * 5.0
        else:
            return base_price * 1.5
    
    def add_event(
        self,
        event_date: datetime,
        event_name: str,
        affected_cities: List[str],
        demand_multiplier: float = 1.5,
        impact_days: int = 3
    ):
        """Add a special event that affects demand."""
        self.events_calendar[event_date] = {
            'name': event_name,
            'affected_cities': affected_cities,
            'demand_multiplier': demand_multiplier,
            'impact_days': impact_days
        }
        
        self.logger.info(
            f"Added event '{event_name}' on {event_date.date()} "
            f"affecting {affected_cities} with {demand_multiplier:.1f}x demand multiplier"
        )
    
    def update_economic_indicators(self, indicators: Dict[str, float]):
        """Update economic indicators affecting demand."""
        self.economic_indicators.update(indicators)
        self.logger.info(f"Updated economic indicators: {indicators}")
    
    def record_actual_demand(
        self,
        route: Route,
        date: datetime,
        actual_bookings: Dict[str, int],
        prices: Dict[str, float]
    ):
        """Record actual demand data for model improvement."""
        route_key = f"{route.origin}-{route.destination}"
        
        total_bookings = sum(actual_bookings.values())
        total_revenue = sum(bookings * prices.get(cabin, 0) for cabin, bookings in actual_bookings.items())
        average_price = total_revenue / total_bookings if total_bookings > 0 else 0
        
        record = {
            'date': date,
            'total_bookings': total_bookings,
            'total_revenue': total_revenue,
            'average_price': average_price,
            'bookings_by_class': actual_bookings.copy(),
            'prices_by_class': prices.copy()
        }
        
        self.historical_demand[route_key].append(record)
        
        # Keep only last 365 days of data
        cutoff_date = date - timedelta(days=365)
        self.historical_demand[route_key] = [
            r for r in self.historical_demand[route_key]
            if r['date'] >= cutoff_date
        ]
    
    def get_demand_analytics(self, route: Route, days_back: int = 30) -> Dict[str, Any]:
        """Get demand analytics for a route."""
        route_key = f"{route.origin}-{route.destination}"
        
        if route_key not in self.historical_demand:
            return {'error': 'No historical data available'}
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_data = [
            r for r in self.historical_demand[route_key]
            if r['date'] >= cutoff_date
        ]
        
        if not recent_data:
            return {'error': 'No recent data available'}
        
        # Calculate analytics
        total_bookings = [r['total_bookings'] for r in recent_data]
        total_revenues = [r['total_revenue'] for r in recent_data]
        average_prices = [r['average_price'] for r in recent_data if r['average_price'] > 0]
        
        analytics = {
            'period_days': days_back,
            'data_points': len(recent_data),
            'avg_daily_bookings': np.mean(total_bookings),
            'avg_daily_revenue': np.mean(total_revenues),
            'avg_price': np.mean(average_prices) if average_prices else 0,
            'booking_volatility': np.std(total_bookings) / np.mean(total_bookings) if np.mean(total_bookings) > 0 else 0,
            'revenue_trend': self._calculate_trend(total_revenues),
            'price_trend': self._calculate_trend(average_prices) if average_prices else 0,
            'peak_booking_day': max(recent_data, key=lambda x: x['total_bookings'])['date'].strftime('%Y-%m-%d'),
            'peak_revenue_day': max(recent_data, key=lambda x: x['total_revenue'])['date'].strftime('%Y-%m-%d')
        }
        
        return analytics
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (slope) in a series of values."""
        if len(values) < 2:
            return 0
        
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return slope
    
    def generate_demand_shock(
        self,
        shock_type: DemandShock,
        magnitude: float = 0.3,
        duration_days: int = 7,
        affected_routes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate a demand shock event."""
        shock_start = datetime.now()
        shock_end = shock_start + timedelta(days=duration_days)
        
        if affected_routes is None:
            affected_routes = list(self.routes.keys())
        
        shock_info = {
            'type': shock_type.value,
            'magnitude': magnitude,
            'start_date': shock_start,
            'end_date': shock_end,
            'affected_routes': affected_routes,
            'description': self._generate_shock_description(shock_type, magnitude)
        }
        
        # Apply shock to economic indicators
        if shock_type == DemandShock.NEGATIVE:
            self.economic_indicators['consumer_confidence'] *= (1 - magnitude)
        elif shock_type == DemandShock.POSITIVE:
            self.economic_indicators['consumer_confidence'] *= (1 + magnitude)
        elif shock_type == DemandShock.VOLATILITY:
            # Increase noise level
            self.base_demand_params['noise_level'] *= (1 + magnitude)
        
        self.logger.warning(f"Demand shock generated: {shock_info['description']}")
        
        return shock_info
    
    def _generate_shock_description(self, shock_type: DemandShock, magnitude: float) -> str:
        """Generate description for demand shock."""
        magnitude_pct = magnitude * 100
        
        if shock_type == DemandShock.POSITIVE:
            return f"Positive demand shock: {magnitude_pct:.1f}% increase in travel demand"
        elif shock_type == DemandShock.NEGATIVE:
            return f"Negative demand shock: {magnitude_pct:.1f}% decrease in travel demand"
        else:
            return f"Volatility shock: {magnitude_pct:.1f}% increase in demand uncertainty"