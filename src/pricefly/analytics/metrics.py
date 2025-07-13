"""Performance metrics calculation for airline pricing simulation."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict
import statistics


class MetricType(Enum):
    """Types of metrics."""
    REVENUE = "revenue"
    PERFORMANCE = "performance"
    COMPETITIVE = "competitive"
    OPERATIONAL = "operational"
    CUSTOMER = "customer"
    FINANCIAL = "financial"


class MetricPeriod(Enum):
    """Time periods for metric calculation."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


@dataclass
class MetricValue:
    """Represents a calculated metric value."""
    name: str
    value: float
    unit: str
    period: MetricPeriod
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    trend: Optional[str] = None  # "up", "down", "stable"
    benchmark: Optional[float] = None
    target: Optional[float] = None


@dataclass
class PerformanceMetrics:
    """Core performance metrics for airline operations."""
    # Revenue metrics
    total_revenue: float = 0.0
    revenue_per_available_seat_mile: float = 0.0  # RASM
    revenue_per_passenger_mile: float = 0.0  # RPM
    yield_per_passenger: float = 0.0
    ancillary_revenue_per_passenger: float = 0.0
    
    # Load factor and capacity metrics
    load_factor: float = 0.0
    available_seat_miles: float = 0.0  # ASM
    revenue_passenger_miles: float = 0.0  # RPM
    passengers_carried: int = 0
    
    # Pricing metrics
    average_fare: float = 0.0
    fare_variance: float = 0.0
    price_elasticity: float = 0.0
    booking_class_mix: Dict[str, float] = field(default_factory=dict)
    
    # Operational metrics
    on_time_performance: float = 0.0
    cancellation_rate: float = 0.0
    aircraft_utilization: float = 0.0
    
    # Cost metrics
    cost_per_available_seat_mile: float = 0.0  # CASM
    fuel_cost_per_asm: float = 0.0
    labor_cost_per_asm: float = 0.0
    
    # Profitability metrics
    operating_margin: float = 0.0
    profit_per_passenger: float = 0.0
    return_on_invested_capital: float = 0.0


@dataclass
class RevenueMetrics:
    """Revenue-specific metrics and analysis."""
    # Total revenue breakdown
    passenger_revenue: float = 0.0
    cargo_revenue: float = 0.0
    ancillary_revenue: float = 0.0
    
    # Revenue by segment
    business_revenue: float = 0.0
    leisure_revenue: float = 0.0
    vfr_revenue: float = 0.0  # Visiting friends and relatives
    
    # Revenue by booking channel
    direct_booking_revenue: float = 0.0
    ota_revenue: float = 0.0
    travel_agent_revenue: float = 0.0
    
    # Revenue by route type
    domestic_revenue: float = 0.0
    international_revenue: float = 0.0
    hub_revenue: float = 0.0
    point_to_point_revenue: float = 0.0
    
    # Revenue optimization metrics
    revenue_opportunity: float = 0.0  # Potential additional revenue
    price_optimization_impact: float = 0.0
    demand_forecast_accuracy: float = 0.0
    
    # Revenue quality metrics
    revenue_concentration: float = 0.0  # HHI for route revenue
    revenue_volatility: float = 0.0
    seasonal_revenue_variation: float = 0.0


@dataclass
class CompetitiveMetrics:
    """Competitive position and market share metrics."""
    # Market share metrics
    market_share_passengers: float = 0.0
    market_share_revenue: float = 0.0
    market_share_capacity: float = 0.0
    
    # Competitive positioning
    price_position: str = "neutral"  # "premium", "discount", "neutral"
    price_index: float = 1.0  # Relative to market average
    service_quality_index: float = 1.0
    
    # Competitive response metrics
    price_response_time: float = 0.0  # Hours to respond to competitor changes
    competitive_moves_initiated: int = 0
    competitive_moves_responded: int = 0
    
    # Market dynamics
    market_concentration_hhi: float = 0.0  # Herfindahl-Hirschman Index
    competitive_intensity: float = 0.0
    new_entrant_impact: float = 0.0
    
    # Route-level competition
    monopoly_routes: int = 0
    duopoly_routes: int = 0
    highly_competitive_routes: int = 0
    
    # Competitive intelligence
    competitor_price_tracking_accuracy: float = 0.0
    market_intelligence_score: float = 0.0


@dataclass
class OperationalMetrics:
    """Operational efficiency and performance metrics."""
    # Fleet utilization
    aircraft_utilization_hours: float = 0.0
    aircraft_utilization_rate: float = 0.0
    fleet_productivity: float = 0.0
    
    # Schedule performance
    on_time_departure: float = 0.0
    on_time_arrival: float = 0.0
    schedule_reliability: float = 0.0
    
    # Operational disruptions
    weather_cancellations: int = 0
    mechanical_cancellations: int = 0
    crew_related_delays: int = 0
    air_traffic_delays: int = 0
    
    # Cost efficiency
    fuel_efficiency: float = 0.0  # Miles per gallon equivalent
    maintenance_cost_efficiency: float = 0.0
    crew_productivity: float = 0.0
    
    # Customer service
    baggage_handling_performance: float = 0.0
    customer_complaint_rate: float = 0.0
    customer_satisfaction_score: float = 0.0
    
    # Technology performance
    booking_system_uptime: float = 0.0
    mobile_app_performance: float = 0.0
    check_in_automation_rate: float = 0.0


class MetricsCalculator:
    """Calculates various metrics from simulation data."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metric_history: Dict[str, List[MetricValue]] = defaultdict(list)
        self.benchmarks: Dict[str, float] = self._initialize_benchmarks()
    
    def _initialize_benchmarks(self) -> Dict[str, float]:
        """Initialize industry benchmark values."""
        return {
            'load_factor': 0.82,
            'on_time_performance': 0.85,
            'cancellation_rate': 0.02,
            'operating_margin': 0.08,
            'rasm': 0.14,  # Revenue per ASM in dollars
            'casm': 0.13,  # Cost per ASM in dollars
            'fuel_cost_share': 0.30,
            'labor_cost_share': 0.31,
            'customer_satisfaction': 0.75,
            'price_elasticity': -1.2,
            'revenue_concentration': 0.15,
            'market_concentration': 0.60
        }
    
    def calculate_performance_metrics(
        self,
        booking_data: pd.DataFrame,
        flight_data: pd.DataFrame,
        cost_data: pd.DataFrame,
        period: MetricPeriod = MetricPeriod.MONTHLY
    ) -> PerformanceMetrics:
        """Calculate core performance metrics."""
        metrics = PerformanceMetrics()
        
        # Revenue metrics
        metrics.total_revenue = booking_data['fare'].sum()
        metrics.passengers_carried = len(booking_data)
        
        if metrics.passengers_carried > 0:
            metrics.average_fare = metrics.total_revenue / metrics.passengers_carried
            metrics.fare_variance = booking_data['fare'].var()
        
        # Calculate ASM and RPM
        flight_data['asm'] = flight_data['capacity'] * flight_data['distance']
        metrics.available_seat_miles = flight_data['asm'].sum()
        
        # Calculate RPM from bookings
        booking_with_distance = booking_data.merge(
            flight_data[['flight_id', 'distance']], 
            on='flight_id', 
            how='left'
        )
        metrics.revenue_passenger_miles = (booking_with_distance['distance']).sum()
        
        # Load factor
        if metrics.available_seat_miles > 0:
            metrics.load_factor = metrics.revenue_passenger_miles / metrics.available_seat_miles
        
        # RASM and yield
        if metrics.available_seat_miles > 0:
            metrics.revenue_per_available_seat_mile = metrics.total_revenue / metrics.available_seat_miles
        
        if metrics.revenue_passenger_miles > 0:
            metrics.revenue_per_passenger_mile = metrics.total_revenue / metrics.revenue_passenger_miles
        
        if metrics.passengers_carried > 0:
            metrics.yield_per_passenger = metrics.total_revenue / metrics.passengers_carried
        
        # Operational metrics
        if 'on_time' in flight_data.columns:
            metrics.on_time_performance = flight_data['on_time'].mean()
        
        if 'cancelled' in flight_data.columns:
            metrics.cancellation_rate = flight_data['cancelled'].mean()
        
        # Cost metrics
        total_costs = cost_data['total_cost'].sum() if 'total_cost' in cost_data.columns else 0
        if metrics.available_seat_miles > 0 and total_costs > 0:
            metrics.cost_per_available_seat_mile = total_costs / metrics.available_seat_miles
        
        # Operating margin
        if total_costs > 0:
            metrics.operating_margin = (metrics.total_revenue - total_costs) / metrics.total_revenue
        
        # Profit per passenger
        if metrics.passengers_carried > 0 and total_costs > 0:
            metrics.profit_per_passenger = (metrics.total_revenue - total_costs) / metrics.passengers_carried
        
        return metrics
    
    def calculate_revenue_metrics(
        self,
        booking_data: pd.DataFrame,
        route_data: pd.DataFrame,
        period: MetricPeriod = MetricPeriod.MONTHLY
    ) -> RevenueMetrics:
        """Calculate revenue-specific metrics."""
        metrics = RevenueMetrics()
        
        # Total revenue breakdown
        metrics.passenger_revenue = booking_data['fare'].sum()
        
        if 'ancillary_revenue' in booking_data.columns:
            metrics.ancillary_revenue = booking_data['ancillary_revenue'].sum()
        
        # Revenue by customer segment
        if 'segment' in booking_data.columns:
            segment_revenue = booking_data.groupby('segment')['fare'].sum()
            metrics.business_revenue = segment_revenue.get('business', 0)
            metrics.leisure_revenue = segment_revenue.get('leisure', 0)
            metrics.vfr_revenue = segment_revenue.get('vfr', 0)
        
        # Revenue by booking channel
        if 'booking_channel' in booking_data.columns:
            channel_revenue = booking_data.groupby('booking_channel')['fare'].sum()
            metrics.direct_booking_revenue = channel_revenue.get('direct', 0)
            metrics.ota_revenue = channel_revenue.get('ota', 0)
            metrics.travel_agent_revenue = channel_revenue.get('agent', 0)
        
        # Revenue by route type
        booking_with_routes = booking_data.merge(
            route_data[['route_id', 'route_type', 'domestic']], 
            on='route_id', 
            how='left'
        )
        
        if 'domestic' in booking_with_routes.columns:
            domestic_mask = booking_with_routes['domestic'] == True
            metrics.domestic_revenue = booking_with_routes[domestic_mask]['fare'].sum()
            metrics.international_revenue = booking_with_routes[~domestic_mask]['fare'].sum()
        
        # Revenue concentration (HHI by route)
        route_revenue = booking_data.groupby('route_id')['fare'].sum()
        if len(route_revenue) > 0:
            total_revenue = route_revenue.sum()
            route_shares = route_revenue / total_revenue
            metrics.revenue_concentration = (route_shares ** 2).sum()
        
        # Revenue volatility
        if len(booking_data) > 1:
            daily_revenue = booking_data.groupby(booking_data['booking_date'].dt.date)['fare'].sum()
            if len(daily_revenue) > 1:
                metrics.revenue_volatility = daily_revenue.std() / daily_revenue.mean()
        
        return metrics
    
    def calculate_competitive_metrics(
        self,
        market_data: pd.DataFrame,
        competitor_data: pd.DataFrame,
        pricing_data: pd.DataFrame,
        period: MetricPeriod = MetricPeriod.MONTHLY
    ) -> CompetitiveMetrics:
        """Calculate competitive position metrics."""
        metrics = CompetitiveMetrics()
        
        # Market share calculations
        if 'passengers' in market_data.columns:
            total_passengers = market_data['passengers'].sum()
            our_passengers = market_data[market_data['airline'] == 'our_airline']['passengers'].sum()
            if total_passengers > 0:
                metrics.market_share_passengers = our_passengers / total_passengers
        
        if 'revenue' in market_data.columns:
            total_revenue = market_data['revenue'].sum()
            our_revenue = market_data[market_data['airline'] == 'our_airline']['revenue'].sum()
            if total_revenue > 0:
                metrics.market_share_revenue = our_revenue / total_revenue
        
        if 'capacity' in market_data.columns:
            total_capacity = market_data['capacity'].sum()
            our_capacity = market_data[market_data['airline'] == 'our_airline']['capacity'].sum()
            if total_capacity > 0:
                metrics.market_share_capacity = our_capacity / total_capacity
        
        # Price positioning
        if 'price' in competitor_data.columns and len(competitor_data) > 0:
            market_avg_price = competitor_data['price'].mean()
            our_avg_price = competitor_data[competitor_data['airline'] == 'our_airline']['price'].mean()
            
            if market_avg_price > 0:
                metrics.price_index = our_avg_price / market_avg_price
                
                if metrics.price_index > 1.1:
                    metrics.price_position = "premium"
                elif metrics.price_index < 0.9:
                    metrics.price_position = "discount"
                else:
                    metrics.price_position = "neutral"
        
        # Market concentration (HHI)
        if 'market_share' in market_data.columns:
            metrics.market_concentration_hhi = (market_data['market_share'] ** 2).sum()
        
        # Competitive response metrics
        if 'response_time' in pricing_data.columns:
            metrics.price_response_time = pricing_data['response_time'].mean()
        
        # Route competition analysis
        if 'competitors_count' in market_data.columns:
            route_competition = market_data.groupby('route_id')['competitors_count'].first()
            metrics.monopoly_routes = (route_competition == 1).sum()
            metrics.duopoly_routes = (route_competition == 2).sum()
            metrics.highly_competitive_routes = (route_competition >= 4).sum()
        
        return metrics
    
    def calculate_operational_metrics(
        self,
        flight_data: pd.DataFrame,
        aircraft_data: pd.DataFrame,
        service_data: pd.DataFrame,
        period: MetricPeriod = MetricPeriod.MONTHLY
    ) -> OperationalMetrics:
        """Calculate operational efficiency metrics."""
        metrics = OperationalMetrics()
        
        # Schedule performance
        if 'on_time_departure' in flight_data.columns:
            metrics.on_time_departure = flight_data['on_time_departure'].mean()
        
        if 'on_time_arrival' in flight_data.columns:
            metrics.on_time_arrival = flight_data['on_time_arrival'].mean()
        
        # Aircraft utilization
        if 'flight_hours' in aircraft_data.columns:
            metrics.aircraft_utilization_hours = aircraft_data['flight_hours'].mean()
            
            # Assuming 24 hours available per day
            metrics.aircraft_utilization_rate = metrics.aircraft_utilization_hours / 24.0
        
        # Operational disruptions
        if 'cancellation_reason' in flight_data.columns:
            cancellation_reasons = flight_data['cancellation_reason'].value_counts()
            metrics.weather_cancellations = cancellation_reasons.get('weather', 0)
            metrics.mechanical_cancellations = cancellation_reasons.get('mechanical', 0)
            metrics.crew_related_delays = cancellation_reasons.get('crew', 0)
        
        # Customer service metrics
        if 'customer_satisfaction' in service_data.columns:
            metrics.customer_satisfaction_score = service_data['customer_satisfaction'].mean()
        
        if 'baggage_performance' in service_data.columns:
            metrics.baggage_handling_performance = service_data['baggage_performance'].mean()
        
        # Fuel efficiency
        if 'fuel_consumed' in flight_data.columns and 'distance' in flight_data.columns:
            flight_data['fuel_efficiency'] = flight_data['distance'] / flight_data['fuel_consumed']
            metrics.fuel_efficiency = flight_data['fuel_efficiency'].mean()
        
        return metrics
    
    def calculate_trend_analysis(
        self,
        metric_name: str,
        values: List[float],
        periods: List[datetime]
    ) -> Dict[str, Any]:
        """Calculate trend analysis for a metric."""
        if len(values) < 2:
            return {'trend': 'insufficient_data', 'slope': 0, 'r_squared': 0}
        
        # Convert periods to numeric values for regression
        x = np.array([(p - periods[0]).days for p in periods])
        y = np.array(values)
        
        # Linear regression
        if len(x) > 1:
            slope, intercept = np.polyfit(x, y, 1)
            
            # Calculate R-squared
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Determine trend direction
            if abs(slope) < 0.01 * np.mean(y):  # Less than 1% change per period
                trend = 'stable'
            elif slope > 0:
                trend = 'up'
            else:
                trend = 'down'
            
            return {
                'trend': trend,
                'slope': slope,
                'r_squared': r_squared,
                'recent_change': (values[-1] - values[-2]) / values[-2] if len(values) >= 2 else 0
            }
        
        return {'trend': 'stable', 'slope': 0, 'r_squared': 0}
    
    def calculate_metric_percentiles(
        self,
        values: List[float],
        percentiles: List[int] = [25, 50, 75, 90, 95]
    ) -> Dict[int, float]:
        """Calculate percentiles for a metric."""
        if not values:
            return {p: 0.0 for p in percentiles}
        
        return {p: np.percentile(values, p) for p in percentiles}
    
    def calculate_correlation_matrix(
        self,
        metrics_data: Dict[str, List[float]]
    ) -> pd.DataFrame:
        """Calculate correlation matrix between metrics."""
        df = pd.DataFrame(metrics_data)
        return df.corr()
    
    def calculate_metric_volatility(
        self,
        values: List[float],
        method: str = 'std'
    ) -> float:
        """Calculate volatility of a metric."""
        if len(values) < 2:
            return 0.0
        
        if method == 'std':
            return np.std(values)
        elif method == 'cv':  # Coefficient of variation
            mean_val = np.mean(values)
            return np.std(values) / mean_val if mean_val != 0 else 0
        elif method == 'range':
            return (max(values) - min(values)) / np.mean(values) if np.mean(values) != 0 else 0
        else:
            return np.std(values)
    
    def calculate_seasonality_index(
        self,
        values: List[float],
        periods: List[datetime],
        season_length: int = 12  # months
    ) -> Dict[int, float]:
        """Calculate seasonality index for a metric."""
        if len(values) < season_length:
            return {}
        
        # Group by month (or other seasonal period)
        seasonal_data = defaultdict(list)
        
        for value, period in zip(values, periods):
            season_key = period.month  # Use month as seasonal key
            seasonal_data[season_key].append(value)
        
        # Calculate seasonal indices
        overall_mean = np.mean(values)
        seasonal_indices = {}
        
        for season, season_values in seasonal_data.items():
            if season_values:
                season_mean = np.mean(season_values)
                seasonal_indices[season] = season_mean / overall_mean if overall_mean != 0 else 1.0
        
        return seasonal_indices
    
    def benchmark_metric(
        self,
        metric_name: str,
        value: float,
        custom_benchmark: Optional[float] = None
    ) -> Dict[str, Any]:
        """Compare metric against benchmark."""
        benchmark = custom_benchmark or self.benchmarks.get(metric_name)
        
        if benchmark is None:
            return {
                'benchmark_available': False,
                'performance': 'unknown'
            }
        
        difference = value - benchmark
        percentage_diff = (difference / benchmark) * 100 if benchmark != 0 else 0
        
        # Determine performance level
        if abs(percentage_diff) < 5:
            performance = 'on_target'
        elif percentage_diff > 10:
            performance = 'above_target'
        elif percentage_diff > 0:
            performance = 'slightly_above'
        elif percentage_diff < -10:
            performance = 'below_target'
        else:
            performance = 'slightly_below'
        
        return {
            'benchmark_available': True,
            'benchmark_value': benchmark,
            'actual_value': value,
            'difference': difference,
            'percentage_difference': percentage_diff,
            'performance': performance
        }
    
    def create_metric_summary(
        self,
        metrics: Dict[str, float],
        period: MetricPeriod,
        timestamp: datetime
    ) -> Dict[str, MetricValue]:
        """Create a summary of metrics with metadata."""
        summary = {}
        
        for name, value in metrics.items():
            # Get trend if historical data exists
            trend = None
            if name in self.metric_history and len(self.metric_history[name]) > 1:
                recent_values = [mv.value for mv in self.metric_history[name][-5:]]  # Last 5 periods
                recent_periods = [mv.timestamp for mv in self.metric_history[name][-5:]]
                trend_analysis = self.calculate_trend_analysis(name, recent_values, recent_periods)
                trend = trend_analysis['trend']
            
            # Get benchmark
            benchmark_info = self.benchmark_metric(name, value)
            
            # Determine unit
            unit = self._get_metric_unit(name)
            
            metric_value = MetricValue(
                name=name,
                value=value,
                unit=unit,
                period=period,
                timestamp=timestamp,
                trend=trend,
                benchmark=benchmark_info.get('benchmark_value'),
                metadata={
                    'benchmark_performance': benchmark_info.get('performance', 'unknown'),
                    'benchmark_difference': benchmark_info.get('percentage_difference', 0)
                }
            )
            
            summary[name] = metric_value
            
            # Store in history
            self.metric_history[name].append(metric_value)
        
        return summary
    
    def _get_metric_unit(self, metric_name: str) -> str:
        """Get appropriate unit for a metric."""
        unit_mapping = {
            'total_revenue': '$',
            'average_fare': '$',
            'rasm': '$/ASM',
            'casm': '$/ASM',
            'load_factor': '%',
            'on_time_performance': '%',
            'cancellation_rate': '%',
            'operating_margin': '%',
            'market_share': '%',
            'fuel_efficiency': 'mpg',
            'aircraft_utilization': 'hours',
            'passengers_carried': 'count',
            'price_response_time': 'hours'
        }
        
        return unit_mapping.get(metric_name, 'value')
    
    def export_metrics_history(
        self,
        metric_names: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Export metrics history as DataFrame."""
        data = []
        
        for metric_name, history in self.metric_history.items():
            if metric_names and metric_name not in metric_names:
                continue
            
            for metric_value in history:
                if start_date and metric_value.timestamp < start_date:
                    continue
                if end_date and metric_value.timestamp > end_date:
                    continue
                
                data.append({
                    'metric_name': metric_name,
                    'value': metric_value.value,
                    'unit': metric_value.unit,
                    'period': metric_value.period.value,
                    'timestamp': metric_value.timestamp,
                    'trend': metric_value.trend,
                    'benchmark': metric_value.benchmark,
                    'confidence': metric_value.confidence
                })
        
        return pd.DataFrame(data)