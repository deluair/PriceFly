"""Market simulation for competitive airline dynamics."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict

from ..models.airline import Airline
from ..models.airport import Route
from ..core.pricing_engine import PricingRecommendation


class CompetitorStrategy(Enum):
    """Competitive strategy types."""
    PRICE_LEADER = "price_leader"  # Sets prices first, others follow
    PRICE_FOLLOWER = "price_follower"  # Follows market prices
    PREMIUM_POSITIONING = "premium_positioning"  # Maintains price premium
    LOW_COST = "low_cost"  # Aggressive low pricing
    DYNAMIC_RESPONSE = "dynamic_response"  # Responds based on market conditions


class MarketCondition(Enum):
    """Market condition states."""
    STABLE = "stable"
    PRICE_WAR = "price_war"
    CAPACITY_CONSTRAINED = "capacity_constrained"
    DEMAND_SURGE = "demand_surge"
    ECONOMIC_DOWNTURN = "economic_downturn"


@dataclass
class CompetitiveAction:
    """Represents a competitive action taken by an airline."""
    airline_code: str
    action_type: str  # 'price_change', 'capacity_change', 'route_entry', etc.
    route: str
    magnitude: float
    timestamp: datetime
    reasoning: str
    expected_impact: Dict[str, float]


@dataclass
class MarketState:
    """Current state of the market."""
    timestamp: datetime
    market_condition: MarketCondition
    total_capacity: int
    total_demand: int
    average_load_factor: float
    price_dispersion: float  # Coefficient of variation in prices
    market_concentration: float  # HHI
    
    # Airline-specific states
    airline_market_shares: Dict[str, float] = field(default_factory=dict)
    airline_load_factors: Dict[str, float] = field(default_factory=dict)
    airline_average_fares: Dict[str, float] = field(default_factory=dict)
    competitive_positions: Dict[str, str] = field(default_factory=dict)


class CompetitorAgent:
    """Represents a competing airline with specific behavior patterns."""
    
    def __init__(
        self,
        airline: Airline,
        strategy: CompetitorStrategy = CompetitorStrategy.DYNAMIC_RESPONSE,
        response_speed: float = 0.7,
        market_share_sensitivity: float = 0.5,
        price_sensitivity: float = 0.3
    ):
        self.airline = airline
        self.strategy = strategy
        self.response_speed = response_speed  # How quickly to respond (0-1)
        self.market_share_sensitivity = market_share_sensitivity
        self.price_sensitivity = price_sensitivity
        
        # State tracking
        self.current_prices: Dict[str, Dict[str, float]] = {}  # route -> cabin_class -> price
        self.historical_actions: List[CompetitiveAction] = []
        self.market_share_history: List[float] = []
        self.profitability_history: List[float] = []
        
        # Strategy parameters
        self.price_premium_target = 0.05 if strategy == CompetitorStrategy.PREMIUM_POSITIONING else 0.0
        self.price_discount_target = -0.15 if strategy == CompetitorStrategy.LOW_COST else 0.0
        
        self.logger = logging.getLogger(f"{__name__}.{airline.airline_code}")
    
    def decide_competitive_response(
        self,
        market_state: MarketState,
        competitor_actions: List[CompetitiveAction],
        route_performance: Dict[str, Dict[str, float]]
    ) -> List[CompetitiveAction]:
        """Decide how to respond to market conditions and competitor actions."""
        actions = []
        
        # Analyze recent competitor actions
        recent_actions = [
            action for action in competitor_actions
            if (datetime.now() - action.timestamp).days <= 7
        ]
        
        # Determine response based on strategy
        if self.strategy == CompetitorStrategy.PRICE_LEADER:
            actions.extend(self._price_leader_strategy(market_state, route_performance))
        elif self.strategy == CompetitorStrategy.PRICE_FOLLOWER:
            actions.extend(self._price_follower_strategy(market_state, recent_actions))
        elif self.strategy == CompetitorStrategy.PREMIUM_POSITIONING:
            actions.extend(self._premium_positioning_strategy(market_state, recent_actions))
        elif self.strategy == CompetitorStrategy.LOW_COST:
            actions.extend(self._low_cost_strategy(market_state, recent_actions))
        elif self.strategy == CompetitorStrategy.DYNAMIC_RESPONSE:
            actions.extend(self._dynamic_response_strategy(market_state, recent_actions, route_performance))
        
        # Store actions in history
        self.historical_actions.extend(actions)
        
        return actions
    
    def _price_leader_strategy(
        self,
        market_state: MarketState,
        route_performance: Dict[str, Dict[str, float]]
    ) -> List[CompetitiveAction]:
        """Implement price leadership strategy."""
        actions = []
        
        for route_key, performance in route_performance.items():
            # Price leaders set prices based on demand and capacity utilization
            current_load_factor = performance.get('load_factor', 0.7)
            
            if current_load_factor > 0.85:  # High demand - increase prices
                price_change = 0.05 + np.random.normal(0, 0.02)
                action = CompetitiveAction(
                    airline_code=self.airline.airline_code,
                    action_type='price_increase',
                    route=route_key,
                    magnitude=price_change,
                    timestamp=datetime.now(),
                    reasoning=f"High load factor ({current_load_factor:.2f}) - increasing prices",
                    expected_impact={'revenue': 0.04, 'demand': -0.02}
                )
                actions.append(action)
            
            elif current_load_factor < 0.65:  # Low demand - decrease prices
                price_change = -0.03 + np.random.normal(0, 0.01)
                action = CompetitiveAction(
                    airline_code=self.airline.airline_code,
                    action_type='price_decrease',
                    route=route_key,
                    magnitude=price_change,
                    timestamp=datetime.now(),
                    reasoning=f"Low load factor ({current_load_factor:.2f}) - decreasing prices",
                    expected_impact={'revenue': -0.02, 'demand': 0.05}
                )
                actions.append(action)
        
        return actions
    
    def _price_follower_strategy(
        self,
        market_state: MarketState,
        recent_actions: List[CompetitiveAction]
    ) -> List[CompetitiveAction]:
        """Implement price following strategy."""
        actions = []
        
        # Group actions by route
        route_actions = defaultdict(list)
        for action in recent_actions:
            if action.airline_code != self.airline.airline_code:
                route_actions[action.route].append(action)
        
        for route_key, route_actions_list in route_actions.items():
            if not route_actions_list:
                continue
            
            # Calculate average competitor price change
            price_changes = [
                action.magnitude for action in route_actions_list
                if action.action_type in ['price_increase', 'price_decrease']
            ]
            
            if price_changes:
                avg_change = np.mean(price_changes)
                
                # Follow with some delay and dampening
                follow_magnitude = avg_change * self.response_speed * 0.8
                
                if abs(follow_magnitude) > 0.01:  # Only act if change is significant
                    action_type = 'price_increase' if follow_magnitude > 0 else 'price_decrease'
                    
                    action = CompetitiveAction(
                        airline_code=self.airline.airline_code,
                        action_type=action_type,
                        route=route_key,
                        magnitude=abs(follow_magnitude),
                        timestamp=datetime.now(),
                        reasoning=f"Following competitor price changes (avg: {avg_change:.3f})",
                        expected_impact={'market_share': 0.01}
                    )
                    actions.append(action)
        
        return actions
    
    def _premium_positioning_strategy(
        self,
        market_state: MarketState,
        recent_actions: List[CompetitiveAction]
    ) -> List[CompetitiveAction]:
        """Implement premium positioning strategy."""
        actions = []
        
        # Maintain price premium regardless of competitor actions
        for route_key in self.current_prices.keys():
            market_avg_price = market_state.airline_average_fares.get(route_key, 500)
            target_price = market_avg_price * (1 + self.price_premium_target)
            
            current_price = self.current_prices[route_key].get('economy', market_avg_price)
            price_gap = (target_price - current_price) / current_price
            
            if abs(price_gap) > 0.02:  # Adjust if gap is significant
                action_type = 'price_increase' if price_gap > 0 else 'price_decrease'
                
                action = CompetitiveAction(
                    airline_code=self.airline.airline_code,
                    action_type=action_type,
                    route=route_key,
                    magnitude=abs(price_gap),
                    timestamp=datetime.now(),
                    reasoning=f"Maintaining premium positioning (target: {self.price_premium_target:.1%})",
                    expected_impact={'brand_value': 0.02, 'demand': -0.01}
                )
                actions.append(action)
        
        return actions
    
    def _low_cost_strategy(
        self,
        market_state: MarketState,
        recent_actions: List[CompetitiveAction]
    ) -> List[CompetitiveAction]:
        """Implement low-cost strategy."""
        actions = []
        
        # Maintain price discount to market
        for route_key in self.current_prices.keys():
            market_avg_price = market_state.airline_average_fares.get(route_key, 500)
            target_price = market_avg_price * (1 + self.price_discount_target)
            
            current_price = self.current_prices[route_key].get('economy', market_avg_price)
            price_gap = (target_price - current_price) / current_price
            
            if price_gap < -0.02:  # Only decrease prices, never increase above target
                action = CompetitiveAction(
                    airline_code=self.airline.airline_code,
                    action_type='price_decrease',
                    route=route_key,
                    magnitude=abs(price_gap),
                    timestamp=datetime.now(),
                    reasoning=f"Maintaining low-cost positioning (target: {self.price_discount_target:.1%})",
                    expected_impact={'market_share': 0.03, 'revenue': -0.01}
                )
                actions.append(action)
        
        return actions
    
    def _dynamic_response_strategy(
        self,
        market_state: MarketState,
        recent_actions: List[CompetitiveAction],
        route_performance: Dict[str, Dict[str, float]]
    ) -> List[CompetitiveAction]:
        """Implement dynamic response strategy based on market conditions."""
        actions = []
        
        # Adapt strategy based on market condition
        if market_state.market_condition == MarketCondition.PRICE_WAR:
            # In price war, focus on maintaining market share
            actions.extend(self._respond_to_price_war(market_state, recent_actions))
        
        elif market_state.market_condition == MarketCondition.CAPACITY_CONSTRAINED:
            # When capacity constrained, increase prices
            actions.extend(self._respond_to_capacity_constraint(route_performance))
        
        elif market_state.market_condition == MarketCondition.DEMAND_SURGE:
            # During demand surge, optimize for revenue
            actions.extend(self._respond_to_demand_surge(route_performance))
        
        elif market_state.market_condition == MarketCondition.ECONOMIC_DOWNTURN:
            # During downturn, focus on maintaining load factors
            actions.extend(self._respond_to_economic_downturn(route_performance))
        
        else:  # STABLE market
            # In stable conditions, optimize based on performance
            actions.extend(self._optimize_stable_market(route_performance))
        
        return actions
    
    def _respond_to_price_war(self, market_state: MarketState, recent_actions: List[CompetitiveAction]) -> List[CompetitiveAction]:
        """Respond to price war conditions."""
        actions = []
        
        # Match aggressive competitor pricing
        competitor_decreases = [
            action for action in recent_actions
            if action.action_type == 'price_decrease' and action.airline_code != self.airline.airline_code
        ]
        
        for action in competitor_decreases:
            # Match 80% of competitor decrease to avoid race to bottom
            response_magnitude = action.magnitude * 0.8
            
            response_action = CompetitiveAction(
                airline_code=self.airline.airline_code,
                action_type='price_decrease',
                route=action.route,
                magnitude=response_magnitude,
                timestamp=datetime.now(),
                reasoning=f"Responding to price war - matching {action.airline_code} decrease",
                expected_impact={'market_share': 0.02, 'revenue': -0.03}
            )
            actions.append(response_action)
        
        return actions
    
    def _respond_to_capacity_constraint(self, route_performance: Dict[str, Dict[str, float]]) -> List[CompetitiveAction]:
        """Respond to capacity-constrained market."""
        actions = []
        
        for route_key, performance in route_performance.items():
            load_factor = performance.get('load_factor', 0.7)
            
            if load_factor > 0.9:  # Very high utilization
                price_increase = 0.08 + np.random.normal(0, 0.02)
                
                action = CompetitiveAction(
                    airline_code=self.airline.airline_code,
                    action_type='price_increase',
                    route=route_key,
                    magnitude=price_increase,
                    timestamp=datetime.now(),
                    reasoning=f"Capacity constrained market - load factor {load_factor:.2f}",
                    expected_impact={'revenue': 0.06, 'demand': -0.03}
                )
                actions.append(action)
        
        return actions
    
    def _respond_to_demand_surge(self, route_performance: Dict[str, Dict[str, float]]) -> List[CompetitiveAction]:
        """Respond to demand surge conditions."""
        actions = []
        
        for route_key, performance in route_performance.items():
            # Increase prices to maximize revenue during surge
            price_increase = 0.12 + np.random.normal(0, 0.03)
            
            action = CompetitiveAction(
                airline_code=self.airline.airline_code,
                action_type='price_increase',
                route=route_key,
                magnitude=price_increase,
                timestamp=datetime.now(),
                reasoning="Demand surge - maximizing revenue opportunity",
                expected_impact={'revenue': 0.10, 'demand': -0.05}
            )
            actions.append(action)
        
        return actions
    
    def _respond_to_economic_downturn(self, route_performance: Dict[str, Dict[str, float]]) -> List[CompetitiveAction]:
        """Respond to economic downturn conditions."""
        actions = []
        
        for route_key, performance in route_performance.items():
            load_factor = performance.get('load_factor', 0.7)
            
            if load_factor < 0.6:  # Low utilization during downturn
                price_decrease = 0.06 + np.random.normal(0, 0.02)
                
                action = CompetitiveAction(
                    airline_code=self.airline.airline_code,
                    action_type='price_decrease',
                    route=route_key,
                    magnitude=price_decrease,
                    timestamp=datetime.now(),
                    reasoning=f"Economic downturn - stimulating demand (LF: {load_factor:.2f})",
                    expected_impact={'demand': 0.08, 'revenue': -0.04}
                )
                actions.append(action)
        
        return actions
    
    def _optimize_stable_market(self, route_performance: Dict[str, Dict[str, float]]) -> List[CompetitiveAction]:
        """Optimize pricing in stable market conditions."""
        actions = []
        
        for route_key, performance in route_performance.items():
            load_factor = performance.get('load_factor', 0.7)
            revenue_per_passenger = performance.get('revenue_per_passenger', 400)
            
            # Optimize based on load factor and revenue
            if load_factor > 0.85 and revenue_per_passenger < 450:
                # High demand, low revenue - increase prices
                price_increase = 0.03 + np.random.normal(0, 0.01)
                
                action = CompetitiveAction(
                    airline_code=self.airline.airline_code,
                    action_type='price_increase',
                    route=route_key,
                    magnitude=price_increase,
                    timestamp=datetime.now(),
                    reasoning=f"Optimizing revenue - high LF ({load_factor:.2f}), low RPP ({revenue_per_passenger:.0f})",
                    expected_impact={'revenue': 0.025, 'demand': -0.01}
                )
                actions.append(action)
            
            elif load_factor < 0.65:
                # Low demand - decrease prices to stimulate
                price_decrease = 0.02 + np.random.normal(0, 0.01)
                
                action = CompetitiveAction(
                    airline_code=self.airline.airline_code,
                    action_type='price_decrease',
                    route=route_key,
                    magnitude=price_decrease,
                    timestamp=datetime.now(),
                    reasoning=f"Stimulating demand - low LF ({load_factor:.2f})",
                    expected_impact={'demand': 0.04, 'revenue': -0.015}
                )
                actions.append(action)
        
        return actions
    
    def update_prices(self, route: str, cabin_class: str, new_price: float):
        """Update current price for a route and cabin class."""
        if route not in self.current_prices:
            self.current_prices[route] = {}
        self.current_prices[route][cabin_class] = new_price
    
    def get_current_price(self, route: str, cabin_class: str = 'economy') -> Optional[float]:
        """Get current price for a route and cabin class."""
        return self.current_prices.get(route, {}).get(cabin_class)


class MarketSimulator:
    """Simulates competitive market dynamics between airlines."""
    
    def __init__(self, competitors: List[CompetitorAgent], routes: List[Route]):
        self.competitors = {agent.airline.airline_code: agent for agent in competitors}
        self.routes = {f"{route.origin}-{route.destination}": route for route in routes}
        
        # Market state tracking
        self.current_market_state = MarketState(
            timestamp=datetime.now(),
            market_condition=MarketCondition.STABLE,
            total_capacity=0,
            total_demand=0,
            average_load_factor=0.7,
            price_dispersion=0.15,
            market_concentration=0.25
        )
        
        # Historical data
        self.market_history: List[MarketState] = []
        self.all_competitive_actions: List[CompetitiveAction] = []
        
        # Market dynamics parameters
        self.price_war_threshold = 0.3  # Price dispersion threshold for price war
        self.capacity_constraint_threshold = 0.9  # Load factor threshold
        self.demand_surge_threshold = 1.2  # Demand multiplier threshold
        
        self.logger = logging.getLogger(__name__)
    
    def update_market_state(
        self,
        current_time: datetime,
        pricing_decisions: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Update market state based on current pricing decisions."""
        
        # Update competitor prices
        self._update_competitor_prices(pricing_decisions)
        
        # Calculate market metrics
        market_metrics = self._calculate_market_metrics(pricing_decisions)
        
        # Determine market condition
        market_condition = self._determine_market_condition(market_metrics)
        
        # Update market state
        self.current_market_state = MarketState(
            timestamp=current_time,
            market_condition=market_condition,
            total_capacity=market_metrics['total_capacity'],
            total_demand=market_metrics['total_demand'],
            average_load_factor=market_metrics['average_load_factor'],
            price_dispersion=market_metrics['price_dispersion'],
            market_concentration=market_metrics['market_concentration'],
            airline_market_shares=market_metrics['market_shares'],
            airline_load_factors=market_metrics['load_factors'],
            airline_average_fares=market_metrics['average_fares'],
            competitive_positions=market_metrics['competitive_positions']
        )
        
        # Generate competitive responses
        competitive_actions = self._generate_competitive_responses(pricing_decisions)
        
        # Store in history
        self.market_history.append(self.current_market_state)
        self.all_competitive_actions.extend(competitive_actions)
        
        # Return market state for each airline
        return self._format_airline_market_states()
    
    def _update_competitor_prices(self, pricing_decisions: Dict[str, Dict[str, Any]]):
        """Update competitor price tracking."""
        for airline_code, decisions in pricing_decisions.items():
            if airline_code in self.competitors:
                competitor = self.competitors[airline_code]
                
                for route_key, route_decisions in decisions.items():
                    if route_key == 'metadata':
                        continue
                    
                    for cabin_class, recommendation in route_decisions.items():
                        if isinstance(recommendation, dict) and 'recommended_price' in recommendation:
                            price = recommendation['recommended_price']
                            competitor.update_prices(route_key, cabin_class, price)
    
    def _calculate_market_metrics(self, pricing_decisions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate current market metrics."""
        metrics = {
            'total_capacity': 0,
            'total_demand': 0,
            'market_shares': {},
            'load_factors': {},
            'average_fares': {},
            'competitive_positions': {}
        }
        
        # Collect pricing and booking data
        all_prices = []
        total_revenue = 0
        total_passengers = 0
        
        for airline_code, decisions in pricing_decisions.items():
            airline_revenue = 0
            airline_passengers = 0
            airline_prices = []
            
            for route_key, route_decisions in decisions.items():
                if route_key == 'metadata':
                    continue
                
                route = self.routes.get(route_key)
                if route:
                    metrics['total_capacity'] += route.frequency_per_day * 150  # Assume 150 seats per aircraft
                
                for cabin_class, recommendation in route_decisions.items():
                    if isinstance(recommendation, dict):
                        price = recommendation.get('recommended_price', 0)
                        bookings = recommendation.get('simulated_bookings', {})
                        
                        if isinstance(bookings, dict):
                            for class_bookings in bookings.values():
                                if isinstance(class_bookings, dict):
                                    revenue = class_bookings.get('revenue', 0)
                                    passengers = class_bookings.get('bookings', 0)
                                    
                                    airline_revenue += revenue
                                    airline_passengers += passengers
                                    total_revenue += revenue
                                    total_passengers += passengers
                        
                        if price > 0:
                            airline_prices.append(price)
                            all_prices.append(price)
            
            # Calculate airline-specific metrics
            if total_revenue > 0:
                metrics['market_shares'][airline_code] = airline_revenue / total_revenue
            else:
                metrics['market_shares'][airline_code] = 0
            
            if airline_passengers > 0 and metrics['total_capacity'] > 0:
                metrics['load_factors'][airline_code] = airline_passengers / (metrics['total_capacity'] / len(pricing_decisions))
            else:
                metrics['load_factors'][airline_code] = 0
            
            if airline_passengers > 0:
                metrics['average_fares'][airline_code] = airline_revenue / airline_passengers
            else:
                metrics['average_fares'][airline_code] = 0
        
        # Calculate market-level metrics
        metrics['total_demand'] = total_passengers
        
        if metrics['total_capacity'] > 0:
            metrics['average_load_factor'] = total_passengers / metrics['total_capacity']
        else:
            metrics['average_load_factor'] = 0
        
        # Calculate price dispersion (coefficient of variation)
        if all_prices and len(all_prices) > 1:
            price_mean = np.mean(all_prices)
            price_std = np.std(all_prices)
            metrics['price_dispersion'] = price_std / price_mean if price_mean > 0 else 0
        else:
            metrics['price_dispersion'] = 0
        
        # Calculate market concentration (HHI)
        market_shares = list(metrics['market_shares'].values())
        metrics['market_concentration'] = sum(share ** 2 for share in market_shares)
        
        # Determine competitive positions
        for airline_code in pricing_decisions.keys():
            market_share = metrics['market_shares'].get(airline_code, 0)
            avg_fare = metrics['average_fares'].get(airline_code, 0)
            market_avg_fare = np.mean(list(metrics['average_fares'].values())) if metrics['average_fares'] else 0
            
            if market_share > 0.3:
                position = "market_leader"
            elif market_share > 0.15:
                position = "major_competitor"
            elif avg_fare > market_avg_fare * 1.1:
                position = "premium_player"
            elif avg_fare < market_avg_fare * 0.9:
                position = "low_cost_competitor"
            else:
                position = "niche_player"
            
            metrics['competitive_positions'][airline_code] = position
        
        return metrics
    
    def _determine_market_condition(self, metrics: Dict[str, Any]) -> MarketCondition:
        """Determine current market condition based on metrics."""
        price_dispersion = metrics['price_dispersion']
        average_load_factor = metrics['average_load_factor']
        market_concentration = metrics['market_concentration']
        
        # Check for price war
        if price_dispersion > self.price_war_threshold:
            return MarketCondition.PRICE_WAR
        
        # Check for capacity constraints
        if average_load_factor > self.capacity_constraint_threshold:
            return MarketCondition.CAPACITY_CONSTRAINED
        
        # Check for demand surge (would need historical comparison)
        # For now, use high load factor as proxy
        if average_load_factor > 0.85 and price_dispersion < 0.1:
            return MarketCondition.DEMAND_SURGE
        
        # Check for economic downturn (low load factors across market)
        if average_load_factor < 0.5:
            return MarketCondition.ECONOMIC_DOWNTURN
        
        return MarketCondition.STABLE
    
    def _generate_competitive_responses(self, pricing_decisions: Dict[str, Dict[str, Any]]) -> List[CompetitiveAction]:
        """Generate competitive responses from all competitors."""
        all_actions = []
        
        # Calculate route performance for each airline
        route_performance = self._calculate_route_performance(pricing_decisions)
        
        for airline_code, competitor in self.competitors.items():
            airline_performance = route_performance.get(airline_code, {})
            
            # Get recent actions from other competitors
            recent_actions = [
                action for action in self.all_competitive_actions[-50:]  # Last 50 actions
                if action.airline_code != airline_code
            ]
            
            # Generate competitive response
            actions = competitor.decide_competitive_response(
                market_state=self.current_market_state,
                competitor_actions=recent_actions,
                route_performance=airline_performance
            )
            
            all_actions.extend(actions)
        
        return all_actions
    
    def _calculate_route_performance(self, pricing_decisions: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Calculate route performance metrics for each airline."""
        performance = {}
        
        for airline_code, decisions in pricing_decisions.items():
            performance[airline_code] = {}
            
            for route_key, route_decisions in decisions.items():
                if route_key == 'metadata':
                    continue
                
                route_metrics = {
                    'revenue': 0,
                    'passengers': 0,
                    'load_factor': 0,
                    'average_fare': 0,
                    'revenue_per_passenger': 0
                }
                
                total_revenue = 0
                total_passengers = 0
                
                for cabin_class, recommendation in route_decisions.items():
                    if isinstance(recommendation, dict):
                        bookings = recommendation.get('simulated_bookings', {})
                        
                        if isinstance(bookings, dict):
                            for class_bookings in bookings.values():
                                if isinstance(class_bookings, dict):
                                    revenue = class_bookings.get('revenue', 0)
                                    passengers = class_bookings.get('bookings', 0)
                                    load_factor = class_bookings.get('load_factor', 0)
                                    
                                    total_revenue += revenue
                                    total_passengers += passengers
                                    route_metrics['load_factor'] = max(route_metrics['load_factor'], load_factor)
                
                route_metrics['revenue'] = total_revenue
                route_metrics['passengers'] = total_passengers
                
                if total_passengers > 0:
                    route_metrics['average_fare'] = total_revenue / total_passengers
                    route_metrics['revenue_per_passenger'] = total_revenue / total_passengers
                
                performance[airline_code][route_key] = route_metrics
        
        return performance
    
    def _format_airline_market_states(self) -> Dict[str, Any]:
        """Format market state information for each airline."""
        airline_states = {}
        
        for airline_code in self.competitors.keys():
            airline_states[airline_code] = {
                'market_share': self.current_market_state.airline_market_shares.get(airline_code, 0),
                'load_factor': self.current_market_state.airline_load_factors.get(airline_code, 0),
                'average_fare': self.current_market_state.airline_average_fares.get(airline_code, 0),
                'competitive_position': self.current_market_state.competitive_positions.get(airline_code, 'unknown'),
                'market_condition': self.current_market_state.market_condition.value,
                'competitor_shares': {
                    code: share for code, share in self.current_market_state.airline_market_shares.items()
                    if code != airline_code
                },
                'price_competitiveness': self._calculate_price_competitiveness(airline_code)
            }
        
        return airline_states
    
    def _calculate_price_competitiveness(self, airline_code: str) -> Dict[str, float]:
        """Calculate price competitiveness metrics for an airline."""
        competitiveness = {}
        
        if airline_code not in self.competitors:
            return competitiveness
        
        competitor = self.competitors[airline_code]
        
        for route_key in competitor.current_prices.keys():
            airline_price = competitor.get_current_price(route_key, 'economy')
            
            if airline_price is None:
                continue
            
            # Get competitor prices for the same route
            competitor_prices = []
            for other_code, other_competitor in self.competitors.items():
                if other_code != airline_code:
                    other_price = other_competitor.get_current_price(route_key, 'economy')
                    if other_price is not None:
                        competitor_prices.append(other_price)
            
            if competitor_prices:
                avg_competitor_price = np.mean(competitor_prices)
                min_competitor_price = min(competitor_prices)
                
                competitiveness[route_key] = {
                    'price_index': airline_price / avg_competitor_price if avg_competitor_price > 0 else 1.0,
                    'price_rank': sum(1 for p in competitor_prices if p < airline_price) + 1,
                    'price_advantage': (avg_competitor_price - airline_price) / avg_competitor_price if avg_competitor_price > 0 else 0
                }
        
        return competitiveness
    
    def get_competitor_price(self, airline_code: str, route: Route, cabin_class: str = 'economy') -> Optional[float]:
        """Get current price for a specific competitor on a route."""
        if airline_code in self.competitors:
            route_key = f"{route.origin}-{route.destination}"
            return self.competitors[airline_code].get_current_price(route_key, cabin_class)
        return None
    
    def get_market_summary(self) -> Dict[str, Any]:
        """Get summary of current market state."""
        return {
            'timestamp': self.current_market_state.timestamp.isoformat(),
            'market_condition': self.current_market_state.market_condition.value,
            'total_capacity': self.current_market_state.total_capacity,
            'total_demand': self.current_market_state.total_demand,
            'average_load_factor': self.current_market_state.average_load_factor,
            'price_dispersion': self.current_market_state.price_dispersion,
            'market_concentration': self.current_market_state.market_concentration,
            'number_of_competitors': len(self.competitors),
            'recent_actions_count': len([a for a in self.all_competitive_actions if (datetime.now() - a.timestamp).days <= 7])
        }
    
    def get_competitive_intelligence(self, airline_code: str) -> Dict[str, Any]:
        """Get competitive intelligence for a specific airline."""
        if airline_code not in self.competitors:
            return {}
        
        competitor = self.competitors[airline_code]
        
        # Analyze competitor actions
        recent_actions = [
            action for action in self.all_competitive_actions
            if action.airline_code != airline_code and (datetime.now() - action.timestamp).days <= 30
        ]
        
        action_analysis = defaultdict(list)
        for action in recent_actions:
            action_analysis[action.airline_code].append(action)
        
        competitor_analysis = {}
        for comp_code, actions in action_analysis.items():
            price_changes = [a.magnitude for a in actions if 'price' in a.action_type]
            
            competitor_analysis[comp_code] = {
                'total_actions': len(actions),
                'avg_price_change': np.mean(price_changes) if price_changes else 0,
                'action_frequency': len(actions) / 30,  # Actions per day
                'most_common_action': max(set(a.action_type for a in actions), key=lambda x: sum(1 for a in actions if a.action_type == x)) if actions else None,
                'strategy_assessment': self._assess_competitor_strategy(actions)
            }
        
        return {
            'market_position': self.current_market_state.competitive_positions.get(airline_code, 'unknown'),
            'market_share': self.current_market_state.airline_market_shares.get(airline_code, 0),
            'competitor_analysis': competitor_analysis,
            'market_threats': self._identify_market_threats(airline_code),
            'opportunities': self._identify_market_opportunities(airline_code)
        }
    
    def _assess_competitor_strategy(self, actions: List[CompetitiveAction]) -> str:
        """Assess competitor strategy based on their actions."""
        if not actions:
            return "inactive"
        
        price_increases = sum(1 for a in actions if a.action_type == 'price_increase')
        price_decreases = sum(1 for a in actions if a.action_type == 'price_decrease')
        
        if price_increases > price_decreases * 2:
            return "premium_positioning"
        elif price_decreases > price_increases * 2:
            return "aggressive_pricing"
        elif len(actions) > 10:  # High activity
            return "dynamic_competitor"
        else:
            return "stable_competitor"
    
    def _identify_market_threats(self, airline_code: str) -> List[str]:
        """Identify potential market threats for an airline."""
        threats = []
        
        # Check for aggressive competitors
        for comp_code, competitor in self.competitors.items():
            if comp_code != airline_code:
                recent_actions = [
                    a for a in self.all_competitive_actions
                    if a.airline_code == comp_code and (datetime.now() - a.timestamp).days <= 14
                ]
                
                price_decreases = [a for a in recent_actions if a.action_type == 'price_decrease']
                if len(price_decreases) > 3:
                    threats.append(f"Aggressive pricing from {comp_code}")
        
        # Check market condition threats
        if self.current_market_state.market_condition == MarketCondition.PRICE_WAR:
            threats.append("Market-wide price war")
        elif self.current_market_state.market_condition == MarketCondition.ECONOMIC_DOWNTURN:
            threats.append("Economic downturn affecting demand")
        
        return threats
    
    def _identify_market_opportunities(self, airline_code: str) -> List[str]:
        """Identify potential market opportunities for an airline."""
        opportunities = []
        
        # Check for market gaps
        if self.current_market_state.market_condition == MarketCondition.DEMAND_SURGE:
            opportunities.append("High demand - opportunity for premium pricing")
        
        if self.current_market_state.average_load_factor > 0.85:
            opportunities.append("Capacity constrained market - pricing power")
        
        # Check for weak competitors
        weak_competitors = [
            code for code, share in self.current_market_state.airline_market_shares.items()
            if share < 0.05 and code != airline_code
        ]
        
        if weak_competitors:
            opportunities.append(f"Market share opportunity from weak competitors: {weak_competitors}")
        
        return opportunities