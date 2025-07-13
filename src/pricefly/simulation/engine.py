"""Main simulation engine for airline pricing dynamics."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict

from ..models.airline import Airline
from ..models.airport import Airport, Route
from ..models.passenger import CustomerSegment
from ..core.pricing_engine import DynamicPricingEngine, PricingContext
from .market import MarketSimulator, CompetitorAgent
from .demand import DemandSimulator
from .events import EventSimulator, MarketEvent


class SimulationState(Enum):
    """Simulation execution states."""
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


class TimeResolution(Enum):
    """Time resolution for simulation steps."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"


@dataclass
class SimulationConfig:
    """Configuration for simulation parameters."""
    # Time parameters
    start_date: datetime
    end_date: datetime
    time_resolution: TimeResolution = TimeResolution.DAILY
    
    # Market parameters
    enable_competition: bool = True
    enable_demand_elasticity: bool = True
    enable_external_events: bool = True
    
    # Simulation parameters
    random_seed: Optional[int] = None
    parallel_execution: bool = True
    max_workers: int = 4
    
    # Output parameters
    save_intermediate_results: bool = True
    output_frequency: int = 7  # Save every N time steps
    detailed_logging: bool = False
    
    # Economic parameters
    base_fuel_price: float = 0.85  # USD per liter
    fuel_volatility: float = 0.15
    economic_growth_rate: float = 0.02
    inflation_rate: float = 0.025
    
    # Demand parameters
    seasonal_amplitude: float = 0.3
    business_travel_ratio: float = 0.25
    price_elasticity_economy: float = -1.2
    price_elasticity_business: float = -0.8
    
    # Competition parameters
    competitor_response_speed: float = 0.7  # How quickly competitors respond
    market_share_sensitivity: float = 0.5
    
    def __post_init__(self):
        if self.random_seed is not None:
            np.random.seed(self.random_seed)


@dataclass
class SimulationMetrics:
    """Metrics collected during simulation."""
    total_revenue: float = 0.0
    total_passengers: int = 0
    average_load_factor: float = 0.0
    average_fare: float = 0.0
    market_share: float = 0.0
    profit_margin: float = 0.0
    
    # Time series data
    daily_revenue: List[float] = field(default_factory=list)
    daily_passengers: List[int] = field(default_factory=list)
    daily_load_factor: List[float] = field(default_factory=list)
    daily_average_fare: List[float] = field(default_factory=list)
    
    # Route-level metrics
    route_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Competitive metrics
    competitor_market_share: Dict[str, float] = field(default_factory=dict)
    price_competitiveness: Dict[str, float] = field(default_factory=dict)


class SimulationEngine:
    """Main simulation engine for airline pricing dynamics."""
    
    def __init__(
        self,
        config: SimulationConfig,
        airlines: List[Airline],
        airports: List[Airport],
        routes: List[Route],
        customer_segments: List[CustomerSegment]
    ):
        self.config = config
        self.airlines = {airline.airline_code: airline for airline in airlines}
        self.airports = {airport.iata_code: airport for airport in airports}
        self.routes = routes
        self.customer_segments = customer_segments
        
        # Simulation state
        self.state = SimulationState.INITIALIZED
        self.current_time = config.start_date
        self.step_count = 0
        
        # Components
        self.pricing_engines = {}
        self.market_simulator = None
        self.demand_simulator = None
        self.event_simulator = None
        
        # Results storage
        self.metrics = {code: SimulationMetrics() for code in self.airlines.keys()}
        self.simulation_history = []
        self.event_log = []
        
        # Threading
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        
        # Logging
        self.logger = logging.getLogger(__name__)
        if config.detailed_logging:
            self.logger.setLevel(logging.DEBUG)
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize simulation components."""
        # Initialize pricing engines for each airline
        for airline_code, airline in self.airlines.items():
            self.pricing_engines[airline_code] = DynamicPricingEngine(
                airline=airline
            )
        
        # Initialize market simulator
        if self.config.enable_competition:
            competitor_agents = [
                CompetitorAgent(
                    airline=airline,
                    response_speed=self.config.competitor_response_speed,
                    market_share_sensitivity=self.config.market_share_sensitivity
                )
                for airline in self.airlines.values()
            ]
            
            self.market_simulator = MarketSimulator(
                competitors=competitor_agents,
                routes=self.routes
            )
        
        # Initialize demand simulator
        self.demand_simulator = DemandSimulator(
            routes=self.routes,
            customer_segments=self.customer_segments,
            base_demand_params={
                'daily_demand_mean': 150,
                'daily_demand_std': 30,
                'seasonality_amplitude': self.config.seasonal_amplitude,
                'trend_factor': 0.02,
                'noise_level': 0.1
            }
        )
        
        # Initialize event simulator
        if self.config.enable_external_events:
            self.event_simulator = EventSimulator(
                simulation_start_date=self.config.start_date,
                simulation_end_date=self.config.end_date
            )
        
        self.logger.info("Simulation components initialized")
    
    def run(self) -> Dict[str, Any]:
        """Run the complete simulation."""
        try:
            self.state = SimulationState.RUNNING
            self.logger.info(f"Starting simulation from {self.config.start_date} to {self.config.end_date}")
            
            while self.current_time <= self.config.end_date and not self._stop_event.is_set():
                self._execute_time_step()
                self._advance_time()
                
                # Save intermediate results
                if (self.step_count % self.config.output_frequency == 0 and 
                    self.config.save_intermediate_results):
                    self._save_intermediate_results()
            
            self.state = SimulationState.COMPLETED
            self.logger.info("Simulation completed successfully")
            
            return self._generate_final_results()
            
        except Exception as e:
            self.state = SimulationState.ERROR
            self.logger.error(f"Simulation error: {str(e)}")
            raise
    
    def _execute_time_step(self):
        """Execute a single time step of the simulation."""
        step_start_time = datetime.now()
        
        # Process external events
        active_events = []
        if self.event_simulator:
            active_events = self.event_simulator.simulate_daily_events(self.current_time)
            for event in active_events:
                self._process_event(event)
        
        # Update demand patterns
        demand_forecasts = {}
        for route in self.routes:
            route_key = f"{route.origin}-{route.destination}"
            forecast = self.demand_simulator.forecast_demand(
                route=route,
                forecast_date=self.current_time,
                forecast_horizon=30
            )
            demand_forecasts[route_key] = forecast
        
        # Execute pricing decisions for each airline
        if self.config.parallel_execution:
            self._execute_pricing_parallel(demand_forecasts, active_events)
        else:
            self._execute_pricing_sequential(demand_forecasts, active_events)
        
        # Update market dynamics
        if self.market_simulator:
            market_state = self.market_simulator.update_market_state(
                current_time=self.current_time,
                pricing_decisions=self._get_current_pricing_decisions()
            )
            self._update_competitive_metrics(market_state)
        
        # Update metrics
        self._update_metrics()
        
        # Log step completion
        step_duration = (datetime.now() - step_start_time).total_seconds()
        self.logger.debug(f"Step {self.step_count} completed in {step_duration:.2f}s")
        
        self.step_count += 1
    
    def _execute_pricing_parallel(self, demand_forecasts: Dict, active_events: List[MarketEvent]):
        """Execute pricing decisions in parallel for all airlines."""
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {}
            
            for airline_code, airline in self.airlines.items():
                future = executor.submit(
                    self._execute_airline_pricing,
                    airline_code,
                    demand_forecasts,
                    active_events
                )
                futures[future] = airline_code
            
            for future in as_completed(futures):
                airline_code = futures[future]
                try:
                    pricing_results = future.result()
                    self._store_pricing_results(airline_code, pricing_results)
                except Exception as e:
                    self.logger.error(f"Pricing execution failed for {airline_code}: {str(e)}")
    
    def _execute_pricing_sequential(self, demand_forecasts: Dict, active_events: List[MarketEvent]):
        """Execute pricing decisions sequentially for all airlines."""
        for airline_code in self.airlines.keys():
            try:
                pricing_results = self._execute_airline_pricing(
                    airline_code,
                    demand_forecasts,
                    active_events
                )
                self._store_pricing_results(airline_code, pricing_results)
            except Exception as e:
                self.logger.error(f"Pricing execution failed for {airline_code}: {str(e)}")
    
    def _execute_airline_pricing(
        self,
        airline_code: str,
        demand_forecasts: Dict,
        active_events: List[MarketEvent]
    ) -> Dict[str, Any]:
        """Execute pricing decisions for a specific airline."""
        airline = self.airlines[airline_code]
        pricing_engine = self.pricing_engines[airline_code]
        
        pricing_results = {}
        
        # Process each route operated by the airline
        for route in airline.routes_operated:
            route_key = f"{route.origin}-{route.destination}"
            
            # Get competitor prices
            competitor_prices = self._get_competitor_prices(route, airline_code)
            
            # Create pricing context
            context = PricingContext(
                route=route,
                current_date=self.current_time,
                competitor_prices=competitor_prices,
                fuel_price_per_liter=self._get_current_fuel_price(),
                economic_indicators=self._get_economic_indicators(),
                historical_demand=self._get_historical_bookings(route, airline_code)
            )
            
            # Generate pricing recommendations for different booking classes
            recommendations = {}
            # Create a simple booking class for economy
            from ..models.airline import BookingClass
            economy_class = BookingClass(booking_code='Y', base_fare=route.average_fare_economy)
            recommendation = pricing_engine.optimize_price(context, economy_class)
            recommendations['economy'] = recommendation
            pricing_results[route_key] = recommendations
            
            # Simulate bookings based on pricing decisions
            demand_forecast = demand_forecasts.get(route_key, None)
            bookings = self._simulate_bookings(route, recommendations, demand_forecast)
            pricing_results[route_key]['simulated_bookings'] = bookings
        
        return pricing_results
    
    def _process_event(self, event):
        """Process an external market event."""
        # Handle different event types
        if hasattr(event, 'to_dict'):
            event_dict = event.to_dict()
        else:
            # Handle ExternalEvent or other event types
            event_dict = {
                'event_id': getattr(event, 'event_id', 'unknown'),
                'event_type': getattr(event, 'event_type', 'unknown'),
                'name': getattr(event, 'name', 'Unknown Event'),
                'description': getattr(event, 'description', ''),
                'start_date': getattr(event, 'start_date', self.current_time).isoformat() if hasattr(getattr(event, 'start_date', None), 'isoformat') else str(getattr(event, 'start_date', self.current_time))
            }
        
        self.event_log.append({
            'timestamp': self.current_time,
            'event': event_dict
        })
        
        # Apply event effects to relevant components
        if event.event_type == 'fuel_price_shock':
            self._apply_fuel_price_shock(event)
        elif event.event_type == 'demand_shock':
            self._apply_demand_shock(event)
        elif event.event_type == 'regulatory_change':
            self._apply_regulatory_change(event)
        elif event.event_type == 'competitor_entry':
            self._apply_competitor_entry(event)
        
        # Log event with impact details
        impact_summary = ", ".join([f"{impact.impact_type.value}: {impact.magnitude}" for impact in event.impacts[:3]])
        self.logger.info(f"Processed event: {event.event_type} with impacts: {impact_summary}")
    
    def _get_competitor_prices(self, route: Route, exclude_airline: str) -> Dict[str, float]:
        """Get current competitor prices for a route."""
        competitor_prices = {}
        
        for airline_code, airline in self.airlines.items():
            if airline_code != exclude_airline and route in airline.routes_operated:
                # Get latest pricing from this competitor
                # This would typically come from the market simulator
                if self.market_simulator:
                    price = self.market_simulator.get_competitor_price(airline_code, route)
                    if price is not None:
                        competitor_prices[airline_code] = price
        
        return competitor_prices
    
    def _get_current_fuel_price(self) -> float:
        """Get current fuel price with volatility."""
        # Simple fuel price model with random walk
        days_from_start = (self.current_time - self.config.start_date).days
        volatility_factor = np.random.normal(1.0, self.config.fuel_volatility)
        trend_factor = 1.0 + (days_from_start * 0.0001)  # Small upward trend
        
        return self.config.base_fuel_price * volatility_factor * trend_factor
    
    def _get_economic_indicators(self) -> Dict[str, float]:
        """Get current economic indicators."""
        days_from_start = (self.current_time - self.config.start_date).days
        
        return {
            'gdp_growth': self.config.economic_growth_rate + np.random.normal(0, 0.005),
            'inflation_rate': self.config.inflation_rate + np.random.normal(0, 0.002),
            'consumer_confidence': 0.7 + np.random.normal(0, 0.1),
            'business_investment': 0.8 + np.random.normal(0, 0.05),
            'days_from_start': days_from_start
        }
    
    def _get_seasonal_factors(self) -> Dict[str, float]:
        """Get seasonal adjustment factors."""
        day_of_year = self.current_time.timetuple().tm_yday
        
        # Simple seasonal model
        summer_peak = np.sin(2 * np.pi * (day_of_year - 150) / 365) * self.config.seasonal_amplitude
        winter_holiday = np.sin(2 * np.pi * (day_of_year - 350) / 365) * 0.2
        
        return {
            'seasonal_demand_factor': 1.0 + summer_peak + winter_holiday,
            'business_travel_factor': 1.0 - 0.3 * max(0, np.sin(2 * np.pi * (day_of_year - 350) / 365)),
            'leisure_travel_factor': 1.0 + 0.4 * max(0, np.sin(2 * np.pi * (day_of_year - 200) / 365))
        }
    
    def _get_historical_bookings(self, route: Route, airline_code: str) -> List[Dict]:
        """Get historical booking data for a route."""
        # This would typically query a database or data store
        # For simulation, we'll return synthetic historical data
        historical_data = []
        
        for i in range(30):  # Last 30 days
            date = self.current_time - timedelta(days=i+1)
            bookings = np.random.poisson(route.annual_demand / 365)
            avg_fare = np.random.normal(route.average_fare_economy, route.average_fare_economy * 0.2)
            
            historical_data.append({
                'date': date,
                'bookings': bookings,
                'average_fare': avg_fare,
                'load_factor': min(1.0, bookings / (route.frequency_per_day * 150))
            })
        
        return historical_data
    
    def _simulate_bookings(self, route: Route, pricing_recommendations: Dict, demand_forecast) -> Dict:
        """Simulate booking behavior based on pricing decisions."""
        bookings_by_class = {}
        
        for cabin_class, recommendation in pricing_recommendations.items():
            if cabin_class == 'metadata':
                continue
                
            # Handle DemandForecast object or fallback to route data
            if demand_forecast and hasattr(demand_forecast, 'base_demand'):
                if cabin_class == 'economy':
                    base_demand = demand_forecast.leisure_demand + demand_forecast.vfr_demand
                elif cabin_class == 'business':
                    base_demand = demand_forecast.business_demand
                else:
                    base_demand = demand_forecast.base_demand / 365
            else:
                base_demand = route.annual_demand / 365
                
            price = recommendation.recommended_price
            
            # Apply price elasticity
            if cabin_class == 'economy':
                elasticity = self.config.price_elasticity_economy
            else:
                elasticity = self.config.price_elasticity_business
            
            # Simple demand response model
            price_factor = (price / route.average_fare_economy) ** elasticity
            adjusted_demand = base_demand * price_factor
            
            # Add randomness
            actual_bookings = max(0, np.random.poisson(adjusted_demand))
            
            bookings_by_class[cabin_class] = {
                'bookings': actual_bookings,
                'revenue': actual_bookings * price,
                'load_factor': actual_bookings / ((route.frequency_per_day * 150) * 0.5)  # Assume 50% capacity per class
            }
        
        return bookings_by_class
    
    def _store_pricing_results(self, airline_code: str, pricing_results: Dict):
        """Store pricing results for an airline."""
        with self._lock:
            # Store in simulation history
            self.simulation_history.append({
                'timestamp': self.current_time,
                'airline_code': airline_code,
                'pricing_results': pricing_results
            })
    
    def _get_current_pricing_decisions(self) -> Dict[str, Dict]:
        """Get current pricing decisions for all airlines."""
        current_decisions = {}
        
        # Get latest pricing decisions from simulation history
        for entry in reversed(self.simulation_history):
            if entry['timestamp'] == self.current_time:
                airline_code = entry['airline_code']
                if airline_code not in current_decisions:
                    current_decisions[airline_code] = entry['pricing_results']
        
        return current_decisions
    
    def _update_competitive_metrics(self, market_state: Dict):
        """Update competitive metrics based on market state."""
        for airline_code in self.airlines.keys():
            if airline_code in market_state:
                metrics = self.metrics[airline_code]
                airline_state = market_state[airline_code]
                
                metrics.market_share = airline_state.get('market_share', 0.0)
                metrics.competitor_market_share = airline_state.get('competitor_shares', {})
                metrics.price_competitiveness = airline_state.get('price_competitiveness', {})
    
    def _update_metrics(self):
        """Update simulation metrics for all airlines."""
        for airline_code in self.airlines.keys():
            metrics = self.metrics[airline_code]
            
            # Calculate daily metrics from recent pricing results
            daily_revenue = 0.0
            daily_passengers = 0
            total_capacity = 0
            total_fare_revenue = 0.0
            
            # Get today's results
            for entry in self.simulation_history:
                if (entry['timestamp'] == self.current_time and 
                    entry['airline_code'] == airline_code):
                    
                    for route_key, route_results in entry['pricing_results'].items():
                        if 'simulated_bookings' in route_results:
                            bookings = route_results['simulated_bookings']
                            
                            for cabin_class, booking_data in bookings.items():
                                daily_revenue += booking_data['revenue']
                                daily_passengers += booking_data['bookings']
                                total_fare_revenue += booking_data['revenue']
            
            # Update cumulative metrics
            metrics.total_revenue += daily_revenue
            metrics.total_passengers += daily_passengers
            
            # Update time series
            metrics.daily_revenue.append(daily_revenue)
            metrics.daily_passengers.append(daily_passengers)
            
            # Calculate averages
            if daily_passengers > 0:
                daily_avg_fare = daily_revenue / daily_passengers
                metrics.daily_average_fare.append(daily_avg_fare)
                
                if metrics.total_passengers > 0:
                    metrics.average_fare = metrics.total_revenue / metrics.total_passengers
            else:
                metrics.daily_average_fare.append(0.0)
            
            # Calculate load factor (simplified)
            airline = self.airlines[airline_code]
            # Calculate total daily capacity using frequency and estimated aircraft capacity
            total_daily_capacity = sum(route.frequency_per_day * 150 for route in airline.routes_operated)  # Assume 150 seats per aircraft
            if total_daily_capacity > 0:
                daily_load_factor = daily_passengers / total_daily_capacity
                metrics.daily_load_factor.append(daily_load_factor)
                
                # Update average load factor
                if len(metrics.daily_load_factor) > 0:
                    metrics.average_load_factor = np.mean(metrics.daily_load_factor)
    
    def _advance_time(self):
        """Advance simulation time by one step."""
        if self.config.time_resolution == TimeResolution.HOURLY:
            self.current_time += timedelta(hours=1)
        elif self.config.time_resolution == TimeResolution.DAILY:
            self.current_time += timedelta(days=1)
        elif self.config.time_resolution == TimeResolution.WEEKLY:
            self.current_time += timedelta(weeks=1)
    
    def _save_intermediate_results(self):
        """Save intermediate simulation results."""
        results = {
            'timestamp': self.current_time.isoformat(),
            'step_count': self.step_count,
            'metrics': {code: self._serialize_metrics(metrics) 
                       for code, metrics in self.metrics.items()},
            'recent_events': self.event_log[-10:] if len(self.event_log) > 10 else self.event_log
        }
        
        # In a real implementation, this would save to a file or database
        self.logger.debug(f"Intermediate results saved at step {self.step_count}")
    
    def _serialize_metrics(self, metrics: SimulationMetrics) -> Dict:
        """Serialize metrics for storage."""
        return {
            'total_revenue': metrics.total_revenue,
            'total_passengers': metrics.total_passengers,
            'average_load_factor': metrics.average_load_factor,
            'average_fare': metrics.average_fare,
            'market_share': metrics.market_share,
            'profit_margin': metrics.profit_margin,
            'daily_metrics_count': len(metrics.daily_revenue)
        }
    
    def _generate_final_results(self) -> Dict[str, Any]:
        """Generate final simulation results."""
        total_steps = self.step_count
        simulation_duration = (self.current_time - self.config.start_date).total_seconds()
        
        results = {
            'simulation_config': {
                'start_date': self.config.start_date.isoformat(),
                'end_date': self.config.end_date.isoformat(),
                'time_resolution': self.config.time_resolution.value,
                'total_steps': total_steps,
                'duration_seconds': simulation_duration
            },
            'airline_metrics': {},
            'market_summary': self._generate_market_summary(),
            'event_summary': self._generate_event_summary(),
            'performance_summary': self._generate_performance_summary()
        }
        
        # Add detailed metrics for each airline
        for airline_code, metrics in self.metrics.items():
            results['airline_metrics'][airline_code] = {
                'total_revenue': metrics.total_revenue,
                'total_passengers': metrics.total_passengers,
                'average_load_factor': metrics.average_load_factor,
                'average_fare': metrics.average_fare,
                'market_share': metrics.market_share,
                'profit_margin': metrics.profit_margin,
                'revenue_per_passenger': metrics.total_revenue / max(metrics.total_passengers, 1),
                'daily_metrics': {
                    'revenue': metrics.daily_revenue,
                    'passengers': metrics.daily_passengers,
                    'load_factor': metrics.daily_load_factor,
                    'average_fare': metrics.daily_average_fare
                },
                'route_performance': metrics.route_performance,
                'competitive_position': {
                    'market_share': metrics.market_share,
                    'competitor_shares': metrics.competitor_market_share,
                    'price_competitiveness': metrics.price_competitiveness
                }
            }
        
        return results
    
    def _generate_market_summary(self) -> Dict[str, Any]:
        """Generate market-level summary statistics."""
        total_market_revenue = sum(metrics.total_revenue for metrics in self.metrics.values())
        total_market_passengers = sum(metrics.total_passengers for metrics in self.metrics.values())
        
        market_shares = {}
        for airline_code, metrics in self.metrics.items():
            if total_market_revenue > 0:
                market_shares[airline_code] = metrics.total_revenue / total_market_revenue
            else:
                market_shares[airline_code] = 0.0
        
        return {
            'total_market_revenue': total_market_revenue,
            'total_market_passengers': total_market_passengers,
            'average_market_fare': total_market_revenue / max(total_market_passengers, 1),
            'market_shares': market_shares,
            'market_concentration': self._calculate_hhi(market_shares),
            'number_of_competitors': len(self.airlines)
        }
    
    def _generate_event_summary(self) -> Dict[str, Any]:
        """Generate summary of events that occurred during simulation."""
        event_counts = defaultdict(int)
        total_events = len(self.event_log)
        
        for event_entry in self.event_log:
            event_type = event_entry['event']['event_type']
            event_counts[event_type] += 1
        
        return {
            'total_events': total_events,
            'event_types': dict(event_counts),
            'events_per_day': total_events / max((self.current_time - self.config.start_date).days, 1)
        }
    
    def _generate_performance_summary(self) -> Dict[str, Any]:
        """Generate simulation performance summary."""
        return {
            'total_simulation_steps': self.step_count,
            'average_step_duration': 'Not tracked',  # Would need to implement timing
            'parallel_execution': self.config.parallel_execution,
            'max_workers': self.config.max_workers if self.config.parallel_execution else 1,
            'memory_usage': 'Not tracked',  # Would need to implement memory monitoring
            'data_points_generated': len(self.simulation_history)
        }
    
    def _calculate_hhi(self, market_shares: Dict[str, float]) -> float:
        """Calculate Herfindahl-Hirschman Index for market concentration."""
        return sum(share ** 2 for share in market_shares.values()) * 10000
    
    def _apply_fuel_price_shock(self, event: MarketEvent):
        """Apply fuel price shock event."""
        # This would modify fuel price parameters
        shock_magnitude = event.impact_magnitude
        self.config.base_fuel_price *= (1 + shock_magnitude)
        self.logger.info(f"Applied fuel price shock: {shock_magnitude:.2%}")
    
    def _apply_demand_shock(self, event: MarketEvent):
        """Apply demand shock event."""
        # This would modify demand parameters in the demand simulator
        if self.demand_simulator:
            self.demand_simulator.apply_demand_shock(event.impact_magnitude, event.duration_days)
        self.logger.info(f"Applied demand shock: {event.impact_magnitude:.2%}")
    
    def _apply_regulatory_change(self, event: MarketEvent):
        """Apply regulatory change event."""
        # This would modify operational constraints or costs
        self.logger.info(f"Applied regulatory change: {event.description}")
    
    def _apply_competitor_entry(self, event: MarketEvent):
        """Apply competitor entry event."""
        # This would add a new competitor to the market
        self.logger.info(f"Applied competitor entry: {event.description}")
    
    def pause(self):
        """Pause the simulation."""
        if self.state == SimulationState.RUNNING:
            self.state = SimulationState.PAUSED
            self.logger.info("Simulation paused")
    
    def resume(self):
        """Resume the simulation."""
        if self.state == SimulationState.PAUSED:
            self.state = SimulationState.RUNNING
            self.logger.info("Simulation resumed")
    
    def stop(self):
        """Stop the simulation."""
        self._stop_event.set()
        self.state = SimulationState.COMPLETED
        self.logger.info("Simulation stopped")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current simulation status."""
        return {
            'state': self.state.value,
            'current_time': self.current_time.isoformat(),
            'step_count': self.step_count,
            'progress': (self.current_time - self.config.start_date).total_seconds() / 
                       (self.config.end_date - self.config.start_date).total_seconds(),
            'airlines_count': len(self.airlines),
            'routes_count': len(self.routes),
            'events_processed': len(self.event_log)
        }
    
    def export_results(self, filepath: str, format: str = 'json'):
        """Export simulation results to file."""
        results = self._generate_final_results()
        
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        elif format.lower() == 'csv':
            # Export key metrics to CSV
            metrics_df = pd.DataFrame({
                airline_code: {
                    'total_revenue': metrics.total_revenue,
                    'total_passengers': metrics.total_passengers,
                    'average_load_factor': metrics.average_load_factor,
                    'average_fare': metrics.average_fare,
                    'market_share': metrics.market_share
                }
                for airline_code, metrics in self.metrics.items()
            }).T
            metrics_df.to_csv(filepath)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Results exported to {filepath}")