"""Revenue management system for airline pricing optimization.

This module implements revenue management strategies, inventory control,
and yield optimization for airline operations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, date, timedelta
from enum import Enum
import numpy as np
import logging
from collections import defaultdict

from ..models.pricing import FareStructure, PricePoint, BookingRestriction
from ..models.demand import DemandPattern, BookingCurve
from ..models.costs import CostStructure


class RevenueStrategy(Enum):
    """Revenue management strategies."""
    YIELD_MANAGEMENT = "yield_management"
    LOAD_FACTOR_OPTIMIZATION = "load_factor_optimization"
    REVENUE_MAXIMIZATION = "revenue_maximization"
    COMPETITIVE_PRICING = "competitive_pricing"
    DYNAMIC_PRICING = "dynamic_pricing"
    SEGMENTATION_BASED = "segmentation_based"


class InventoryControl(Enum):
    """Inventory control methods."""
    NESTED_BOOKING_LIMITS = "nested_booking_limits"
    VIRTUAL_NESTING = "virtual_nesting"
    BID_PRICE_CONTROL = "bid_price_control"
    HYBRID_CONTROL = "hybrid_control"


class OptimizationObjective(Enum):
    """Optimization objectives for revenue management."""
    MAXIMIZE_REVENUE = "maximize_revenue"
    MAXIMIZE_PROFIT = "maximize_profit"
    MAXIMIZE_LOAD_FACTOR = "maximize_load_factor"
    MINIMIZE_SPILL = "minimize_spill"
    BALANCE_REVENUE_LOAD = "balance_revenue_load"


@dataclass
class BookingLimit:
    """Booking limits for fare classes."""
    fare_class: str
    limit: int
    protection_level: int = 0  # Seats protected for higher fare classes
    authorization_level: int = 1  # Required authorization level for overrides
    
    # Dynamic adjustments
    original_limit: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    update_reason: str = ""
    
    def __post_init__(self):
        if self.original_limit == 0:
            self.original_limit = self.limit
    
    def adjust_limit(self, new_limit: int, reason: str) -> None:
        """Adjust booking limit with tracking."""
        self.limit = max(0, new_limit)
        self.last_updated = datetime.now()
        self.update_reason = reason
    
    def get_available_inventory(self, current_bookings: int) -> int:
        """Get available inventory considering protection levels."""
        available = max(0, self.limit - current_bookings)
        return max(0, available - self.protection_level)


@dataclass
class RevenueMetrics:
    """Revenue performance metrics."""
    # Revenue metrics
    total_revenue: float = 0.0
    passenger_revenue: float = 0.0
    ancillary_revenue: float = 0.0
    
    # Yield metrics
    revenue_per_passenger: float = 0.0
    revenue_per_available_seat_mile: float = 0.0
    yield_per_mile: float = 0.0
    
    # Load factor metrics
    load_factor: float = 0.0
    revenue_load_factor: float = 0.0  # Revenue-weighted load factor
    
    # Booking metrics
    total_bookings: int = 0
    bookings_by_class: Dict[str, int] = field(default_factory=dict)
    average_booking_value: float = 0.0
    
    # Optimization metrics
    spill_cost: float = 0.0  # Revenue lost due to capacity constraints
    spoilage_cost: float = 0.0  # Revenue lost due to unsold seats
    dilution_cost: float = 0.0  # Revenue lost due to down-selling
    
    # Time-based metrics
    booking_curve_performance: float = 1.0  # Actual vs expected booking pace
    price_elasticity_observed: float = 0.0
    
    # Metadata
    calculation_date: datetime = field(default_factory=datetime.now)
    route_id: str = ""
    flight_number: str = ""
    
    def calculate_yield_metrics(
        self, 
        distance_miles: float, 
        capacity: int
    ) -> None:
        """Calculate yield-related metrics."""
        if self.total_bookings > 0:
            self.revenue_per_passenger = self.total_revenue / self.total_bookings
            self.average_booking_value = self.passenger_revenue / self.total_bookings
        
        if capacity > 0:
            self.load_factor = self.total_bookings / capacity
        
        if distance_miles > 0 and capacity > 0:
            available_seat_miles = capacity * distance_miles
            if available_seat_miles > 0:
                self.revenue_per_available_seat_mile = self.total_revenue / available_seat_miles
            
            if self.total_bookings > 0:
                passenger_miles = self.total_bookings * distance_miles
                self.yield_per_mile = self.passenger_revenue / passenger_miles
    
    def calculate_opportunity_costs(
        self, 
        demand_spill: int, 
        average_fare: float,
        unsold_seats: int
    ) -> None:
        """Calculate opportunity costs."""
        self.spill_cost = demand_spill * average_fare
        # Spoilage cost is more complex - simplified here
        self.spoilage_cost = unsold_seats * (average_fare * 0.3)  # Assume 30% of average fare


@dataclass
class RevenueOptimizationResult:
    """Result of revenue optimization."""
    # Optimization results
    optimized_fare_structure: FareStructure
    booking_limits: Dict[str, BookingLimit]
    expected_revenue: float
    expected_load_factor: float
    
    # Optimization details
    strategy_used: RevenueStrategy
    objective: OptimizationObjective
    optimization_score: float  # Quality score of optimization
    
    # Recommendations
    price_recommendations: Dict[str, float]
    inventory_recommendations: Dict[str, int]
    
    # Risk assessment
    revenue_risk: float = 0.0
    demand_uncertainty: float = 0.0
    competitive_risk: float = 0.0
    
    # Metadata
    optimization_timestamp: datetime = field(default_factory=datetime.now)
    model_confidence: float = 0.8
    
    def get_summary(self) -> Dict:
        """Get optimization summary."""
        return {
            'expected_revenue': self.expected_revenue,
            'expected_load_factor': self.expected_load_factor,
            'strategy': self.strategy_used.value,
            'objective': self.objective.value,
            'optimization_score': self.optimization_score,
            'model_confidence': self.model_confidence,
            'total_risk': self.revenue_risk + self.demand_uncertainty + self.competitive_risk,
            'timestamp': self.optimization_timestamp.isoformat()
        }


class RevenueManager:
    """Advanced revenue management system for airline pricing."""
    
    def __init__(
        self,
        strategy: RevenueStrategy = RevenueStrategy.DYNAMIC_PRICING,
        inventory_control: InventoryControl = InventoryControl.BID_PRICE_CONTROL,
        objective: OptimizationObjective = OptimizationObjective.MAXIMIZE_REVENUE
    ):
        self.strategy = strategy
        self.inventory_control = inventory_control
        self.objective = objective
        
        # Historical data storage
        self.booking_history: Dict[str, List[Dict]] = defaultdict(list)
        self.revenue_history: Dict[str, List[RevenueMetrics]] = defaultdict(list)
        self.demand_history: Dict[str, List[float]] = defaultdict(list)
        
        # Current state
        self.active_fare_structures: Dict[str, FareStructure] = {}
        self.booking_limits: Dict[str, Dict[str, BookingLimit]] = defaultdict(dict)
        self.bid_prices: Dict[str, float] = {}
        
        # Configuration
        self.optimization_frequency_hours: int = 6
        self.min_fare_adjustment: float = 0.05  # 5% minimum adjustment
        self.max_fare_adjustment: float = 0.50  # 50% maximum adjustment
        self.load_factor_target: float = 0.80
        self.revenue_target_multiplier: float = 1.1  # 10% above break-even
        
        # Risk management
        self.max_revenue_risk: float = 0.20
        self.diversification_threshold: float = 0.30
        
        self.logger = logging.getLogger(__name__)
    
    def optimize_revenue(
        self,
        route_id: str,
        fare_structure: FareStructure,
        demand_pattern: DemandPattern,
        cost_structure: CostStructure,
        days_before_departure: int,
        current_bookings: Dict[str, int],
        competitor_prices: Optional[Dict[str, float]] = None
    ) -> RevenueOptimizationResult:
        """Optimize revenue for a specific flight."""
        
        self.logger.info(f"Starting revenue optimization for route {route_id}")
        
        # Calculate current metrics
        current_metrics = self._calculate_current_metrics(
            fare_structure, current_bookings, cost_structure
        )
        
        # Forecast demand
        demand_forecast = self._forecast_demand(
            demand_pattern, days_before_departure, fare_structure
        )
        
        # Optimize based on strategy
        if self.strategy == RevenueStrategy.YIELD_MANAGEMENT:
            result = self._optimize_yield_management(
                route_id, fare_structure, demand_forecast, cost_structure,
                days_before_departure, current_bookings
            )
        elif self.strategy == RevenueStrategy.DYNAMIC_PRICING:
            result = self._optimize_dynamic_pricing(
                route_id, fare_structure, demand_pattern, cost_structure,
                days_before_departure, current_bookings, competitor_prices
            )
        elif self.strategy == RevenueStrategy.LOAD_FACTOR_OPTIMIZATION:
            result = self._optimize_load_factor(
                route_id, fare_structure, demand_forecast, cost_structure,
                days_before_departure, current_bookings
            )
        else:
            # Default to revenue maximization
            result = self._optimize_revenue_maximization(
                route_id, fare_structure, demand_forecast, cost_structure,
                days_before_departure, current_bookings
            )
        
        # Store optimization result
        self.active_fare_structures[route_id] = result.optimized_fare_structure
        self.booking_limits[route_id] = result.booking_limits
        
        self.logger.info(
            f"Revenue optimization completed for {route_id}. "
            f"Expected revenue: ${result.expected_revenue:.2f}, "
            f"Load factor: {result.expected_load_factor:.2%}"
        )
        
        return result
    
    def _optimize_dynamic_pricing(
        self,
        route_id: str,
        fare_structure: FareStructure,
        demand_pattern: DemandPattern,
        cost_structure: CostStructure,
        days_before_departure: int,
        current_bookings: Dict[str, int],
        competitor_prices: Optional[Dict[str, float]] = None
    ) -> RevenueOptimizationResult:
        """Implement dynamic pricing optimization."""
        
        optimized_structure = FareStructure(
            route_id=fare_structure.route_id,
            flight_number=fare_structure.flight_number,
            departure_date=fare_structure.departure_date
        )
        
        # Copy existing fare classes
        for class_code, price_point in fare_structure.fare_classes.items():
            optimized_structure.fare_classes[class_code] = PricePoint(
                price=price_point.price,
                currency=price_point.currency,
                fare_type=price_point.fare_type,
                booking_class=class_code,
                base_fare=price_point.base_fare,
                taxes_and_fees=price_point.taxes_and_fees,
                seats_available=price_point.seats_available,
                total_inventory=price_point.total_inventory,
                restrictions=price_point.restrictions.copy()
            )
        
        # Calculate demand elasticity adjustments
        booking_curve = demand_pattern.advance_booking_curve
        if booking_curve:
            # Adjust prices based on booking pace
            expected_bookings = booking_curve.calculate_cumulative_bookings(
                days_before_departure
            )
            actual_bookings = sum(current_bookings.values())
            
            pace_ratio = actual_bookings / expected_bookings if expected_bookings > 0 else 1.0
            
            # Price adjustment based on booking pace
            if pace_ratio > 1.2:  # Booking ahead of pace
                price_multiplier = 1.1  # Increase prices
            elif pace_ratio < 0.8:  # Booking behind pace
                price_multiplier = 0.95  # Decrease prices
            else:
                price_multiplier = 1.0  # No change
            
            # Apply competitive adjustments
            if competitor_prices:
                avg_competitor_price = sum(competitor_prices.values()) / len(competitor_prices)
                current_avg_price = sum(
                    pp.price for pp in fare_structure.fare_classes.values()
                ) / len(fare_structure.fare_classes)
                
                if current_avg_price > avg_competitor_price * 1.1:
                    price_multiplier *= 0.98  # Slight decrease to stay competitive
                elif current_avg_price < avg_competitor_price * 0.9:
                    price_multiplier *= 1.02  # Slight increase to capture value
            
            # Apply price adjustments
            for class_code, price_point in optimized_structure.fare_classes.items():
                new_price = price_point.price * price_multiplier
                
                # Ensure price stays within reasonable bounds
                min_price = price_point.price * (1 - self.max_fare_adjustment)
                max_price = price_point.price * (1 + self.max_fare_adjustment)
                new_price = max(min_price, min(max_price, new_price))
                
                price_point.update_price(
                    new_price,
                    reason=f"Dynamic pricing adjustment: {price_multiplier:.3f}"
                )
        
        # Calculate booking limits
        booking_limits = self._calculate_booking_limits(
            optimized_structure, demand_pattern, days_before_departure
        )
        
        # Calculate expected outcomes
        expected_revenue = self._calculate_expected_revenue(
            optimized_structure, demand_pattern, days_before_departure
        )
        expected_load_factor = self._calculate_expected_load_factor(
            optimized_structure, demand_pattern, days_before_departure
        )
        
        # Calculate optimization score
        optimization_score = self._calculate_optimization_score(
            expected_revenue, expected_load_factor, cost_structure
        )
        
        return RevenueOptimizationResult(
            optimized_fare_structure=optimized_structure,
            booking_limits=booking_limits,
            expected_revenue=expected_revenue,
            expected_load_factor=expected_load_factor,
            strategy_used=RevenueStrategy.DYNAMIC_PRICING,
            objective=self.objective,
            optimization_score=optimization_score,
            price_recommendations={
                code: pp.price for code, pp in optimized_structure.fare_classes.items()
            },
            inventory_recommendations={
                code: limit.limit for code, limit in booking_limits.items()
            }
        )
    
    def _optimize_yield_management(
        self,
        route_id: str,
        fare_structure: FareStructure,
        demand_forecast: Dict[str, float],
        cost_structure: CostStructure,
        days_before_departure: int,
        current_bookings: Dict[str, int]
    ) -> RevenueOptimizationResult:
        """Implement traditional yield management optimization."""
        
        # This is a simplified yield management implementation
        # In practice, this would involve complex mathematical optimization
        
        optimized_structure = FareStructure(
            route_id=fare_structure.route_id,
            flight_number=fare_structure.flight_number,
            departure_date=fare_structure.departure_date
        )
        
        # Copy and optimize fare classes
        total_capacity = fare_structure.calculate_total_inventory()
        current_load_factor = fare_structure.calculate_load_factor()
        
        for class_code, price_point in fare_structure.fare_classes.items():
            # Calculate yield for this class
            class_yield = price_point.price
            class_demand = demand_forecast.get(class_code, 0)
            
            # Adjust inventory allocation based on yield and demand
            if current_load_factor < self.load_factor_target:
                # Increase lower fare class availability
                if class_code in ['L', 'M', 'Y']:
                    inventory_multiplier = 1.1
                else:
                    inventory_multiplier = 0.9
            else:
                # Restrict lower fare classes, favor higher yield
                if class_code in ['L', 'M']:
                    inventory_multiplier = 0.8
                else:
                    inventory_multiplier = 1.2
            
            new_inventory = int(price_point.total_inventory * inventory_multiplier)
            new_inventory = max(1, min(total_capacity // 2, new_inventory))
            
            optimized_structure.fare_classes[class_code] = PricePoint(
                price=price_point.price,
                currency=price_point.currency,
                fare_type=price_point.fare_type,
                booking_class=class_code,
                base_fare=price_point.base_fare,
                taxes_and_fees=price_point.taxes_and_fees,
                seats_available=new_inventory,
                total_inventory=new_inventory,
                restrictions=price_point.restrictions.copy()
            )
        
        # Calculate booking limits
        booking_limits = self._calculate_booking_limits(
            optimized_structure, None, days_before_departure
        )
        
        # Calculate expected outcomes
        expected_revenue = sum(
            pp.price * pp.seats_available * 0.8  # Assume 80% of inventory sells
            for pp in optimized_structure.fare_classes.values()
        )
        
        expected_load_factor = min(1.0, sum(
            pp.seats_available * 0.8 for pp in optimized_structure.fare_classes.values()
        ) / total_capacity)
        
        optimization_score = self._calculate_optimization_score(
            expected_revenue, expected_load_factor, cost_structure
        )
        
        return RevenueOptimizationResult(
            optimized_fare_structure=optimized_structure,
            booking_limits=booking_limits,
            expected_revenue=expected_revenue,
            expected_load_factor=expected_load_factor,
            strategy_used=RevenueStrategy.YIELD_MANAGEMENT,
            objective=self.objective,
            optimization_score=optimization_score,
            price_recommendations={
                code: pp.price for code, pp in optimized_structure.fare_classes.items()
            },
            inventory_recommendations={
                code: limit.limit for code, limit in booking_limits.items()
            }
        )
    
    def _optimize_load_factor(
        self,
        route_id: str,
        fare_structure: FareStructure,
        demand_forecast: Dict[str, float],
        cost_structure: CostStructure,
        days_before_departure: int,
        current_bookings: Dict[str, int]
    ) -> RevenueOptimizationResult:
        """Optimize for load factor target."""
        
        # Similar structure to other optimization methods
        # Focus on achieving target load factor
        
        optimized_structure = FareStructure(
            route_id=fare_structure.route_id,
            flight_number=fare_structure.flight_number,
            departure_date=fare_structure.departure_date
        )
        
        current_load_factor = fare_structure.calculate_load_factor()
        load_factor_gap = self.load_factor_target - current_load_factor
        
        for class_code, price_point in fare_structure.fare_classes.items():
            # Adjust prices to achieve load factor target
            if load_factor_gap > 0.1:  # Need more bookings
                price_multiplier = 0.95  # Reduce prices
            elif load_factor_gap < -0.1:  # Too many bookings
                price_multiplier = 1.05  # Increase prices
            else:
                price_multiplier = 1.0  # No change needed
            
            new_price = price_point.price * price_multiplier
            
            optimized_structure.fare_classes[class_code] = PricePoint(
                price=new_price,
                currency=price_point.currency,
                fare_type=price_point.fare_type,
                booking_class=class_code,
                base_fare=price_point.base_fare * price_multiplier,
                taxes_and_fees=price_point.taxes_and_fees,
                seats_available=price_point.seats_available,
                total_inventory=price_point.total_inventory,
                restrictions=price_point.restrictions.copy()
            )
        
        booking_limits = self._calculate_booking_limits(
            optimized_structure, None, days_before_departure
        )
        
        expected_revenue = self._calculate_expected_revenue(
            optimized_structure, None, days_before_departure
        )
        expected_load_factor = self.load_factor_target  # Target achieved
        
        optimization_score = self._calculate_optimization_score(
            expected_revenue, expected_load_factor, cost_structure
        )
        
        return RevenueOptimizationResult(
            optimized_fare_structure=optimized_structure,
            booking_limits=booking_limits,
            expected_revenue=expected_revenue,
            expected_load_factor=expected_load_factor,
            strategy_used=RevenueStrategy.LOAD_FACTOR_OPTIMIZATION,
            objective=self.objective,
            optimization_score=optimization_score,
            price_recommendations={
                code: pp.price for code, pp in optimized_structure.fare_classes.items()
            },
            inventory_recommendations={
                code: limit.limit for code, limit in booking_limits.items()
            }
        )
    
    def _optimize_revenue_maximization(
        self,
        route_id: str,
        fare_structure: FareStructure,
        demand_forecast: Dict[str, float],
        cost_structure: CostStructure,
        days_before_departure: int,
        current_bookings: Dict[str, int]
    ) -> RevenueOptimizationResult:
        """Optimize for maximum revenue."""
        
        # Implement revenue maximization logic
        # This would typically involve complex optimization algorithms
        
        optimized_structure = FareStructure(
            route_id=fare_structure.route_id,
            flight_number=fare_structure.flight_number,
            departure_date=fare_structure.departure_date
        )
        
        # Simple revenue maximization: increase prices for high-demand classes
        for class_code, price_point in fare_structure.fare_classes.items():
            class_demand = demand_forecast.get(class_code, 0)
            
            # Increase prices for classes with high demand
            if class_demand > price_point.seats_available * 1.2:
                price_multiplier = 1.1
            elif class_demand < price_point.seats_available * 0.8:
                price_multiplier = 0.95
            else:
                price_multiplier = 1.0
            
            new_price = price_point.price * price_multiplier
            
            optimized_structure.fare_classes[class_code] = PricePoint(
                price=new_price,
                currency=price_point.currency,
                fare_type=price_point.fare_type,
                booking_class=class_code,
                base_fare=price_point.base_fare * price_multiplier,
                taxes_and_fees=price_point.taxes_and_fees,
                seats_available=price_point.seats_available,
                total_inventory=price_point.total_inventory,
                restrictions=price_point.restrictions.copy()
            )
        
        booking_limits = self._calculate_booking_limits(
            optimized_structure, None, days_before_departure
        )
        
        expected_revenue = self._calculate_expected_revenue(
            optimized_structure, None, days_before_departure
        )
        expected_load_factor = self._calculate_expected_load_factor(
            optimized_structure, None, days_before_departure
        )
        
        optimization_score = self._calculate_optimization_score(
            expected_revenue, expected_load_factor, cost_structure
        )
        
        return RevenueOptimizationResult(
            optimized_fare_structure=optimized_structure,
            booking_limits=booking_limits,
            expected_revenue=expected_revenue,
            expected_load_factor=expected_load_factor,
            strategy_used=RevenueStrategy.REVENUE_MAXIMIZATION,
            objective=self.objective,
            optimization_score=optimization_score,
            price_recommendations={
                code: pp.price for code, pp in optimized_structure.fare_classes.items()
            },
            inventory_recommendations={
                code: limit.limit for code, limit in booking_limits.items()
            }
        )
    
    def _calculate_booking_limits(
        self,
        fare_structure: FareStructure,
        demand_pattern: Optional[DemandPattern],
        days_before_departure: int
    ) -> Dict[str, BookingLimit]:
        """Calculate optimal booking limits for fare classes."""
        
        booking_limits = {}
        
        for class_code, price_point in fare_structure.fare_classes.items():
            # Simple booking limit calculation
            # In practice, this would use more sophisticated algorithms
            
            base_limit = price_point.total_inventory
            
            # Adjust based on time to departure
            if days_before_departure > 90:
                # Far out - conservative limits for low fare classes
                if class_code in ['L', 'M']:
                    limit_multiplier = 0.6
                else:
                    limit_multiplier = 1.0
            elif days_before_departure > 30:
                # Medium term - moderate limits
                if class_code in ['L']:
                    limit_multiplier = 0.8
                else:
                    limit_multiplier = 1.0
            else:
                # Close to departure - open up inventory
                limit_multiplier = 1.0
            
            final_limit = int(base_limit * limit_multiplier)
            
            booking_limits[class_code] = BookingLimit(
                fare_class=class_code,
                limit=final_limit,
                protection_level=0,  # Simplified - no protection
                authorization_level=1
            )
        
        return booking_limits
    
    def _calculate_expected_revenue(
        self,
        fare_structure: FareStructure,
        demand_pattern: Optional[DemandPattern],
        days_before_departure: int
    ) -> float:
        """Calculate expected revenue from fare structure."""
        
        total_revenue = 0.0
        
        for class_code, price_point in fare_structure.fare_classes.items():
            # Simple expected revenue calculation
            # Assume 70% of available inventory will sell
            expected_sales = price_point.seats_available * 0.7
            class_revenue = expected_sales * price_point.price
            total_revenue += class_revenue
        
        return total_revenue
    
    def _calculate_expected_load_factor(
        self,
        fare_structure: FareStructure,
        demand_pattern: Optional[DemandPattern],
        days_before_departure: int
    ) -> float:
        """Calculate expected load factor."""
        
        total_capacity = fare_structure.calculate_total_inventory()
        if total_capacity == 0:
            return 0.0
        
        # Simple calculation - assume 70% of inventory sells
        expected_passengers = sum(
            pp.seats_available * 0.7 for pp in fare_structure.fare_classes.values()
        )
        
        return min(1.0, expected_passengers / total_capacity)
    
    def _calculate_optimization_score(
        self,
        expected_revenue: float,
        expected_load_factor: float,
        cost_structure: CostStructure
    ) -> float:
        """Calculate optimization quality score."""
        
        # Simple scoring based on revenue and load factor
        revenue_score = min(1.0, expected_revenue / 50000)  # Normalize to $50k
        load_factor_score = expected_load_factor
        
        # Weighted combination
        if self.objective == OptimizationObjective.MAXIMIZE_REVENUE:
            score = 0.8 * revenue_score + 0.2 * load_factor_score
        elif self.objective == OptimizationObjective.MAXIMIZE_LOAD_FACTOR:
            score = 0.2 * revenue_score + 0.8 * load_factor_score
        else:
            score = 0.5 * revenue_score + 0.5 * load_factor_score
        
        return score
    
    def _forecast_demand(
        self,
        demand_pattern: DemandPattern,
        days_before_departure: int,
        fare_structure: FareStructure
    ) -> Dict[str, float]:
        """Forecast demand by fare class."""
        
        demand_forecast = {}
        
        # Simple demand forecasting
        base_demand = demand_pattern.base_demand
        
        for class_code, price_point in fare_structure.fare_classes.items():
            # Distribute demand across fare classes based on price
            if class_code in ['L', 'M']:
                class_share = 0.4  # 40% for low fare classes
            elif class_code in ['Y', 'W']:
                class_share = 0.35  # 35% for mid fare classes
            else:
                class_share = 0.25  # 25% for high fare classes
            
            class_demand = base_demand * class_share
            demand_forecast[class_code] = class_demand
        
        return demand_forecast
    
    def _calculate_current_metrics(
        self,
        fare_structure: FareStructure,
        current_bookings: Dict[str, int],
        cost_structure: CostStructure
    ) -> RevenueMetrics:
        """Calculate current revenue metrics."""
        
        metrics = RevenueMetrics(
            route_id=fare_structure.route_id,
            flight_number=fare_structure.flight_number
        )
        
        # Calculate revenue
        total_revenue = 0.0
        total_bookings = 0
        
        for class_code, bookings in current_bookings.items():
            if class_code in fare_structure.fare_classes:
                price = fare_structure.fare_classes[class_code].price
                class_revenue = bookings * price
                total_revenue += class_revenue
                total_bookings += bookings
        
        metrics.total_revenue = total_revenue
        metrics.passenger_revenue = total_revenue  # Simplified
        metrics.total_bookings = total_bookings
        metrics.bookings_by_class = current_bookings.copy()
        
        if total_bookings > 0:
            metrics.revenue_per_passenger = total_revenue / total_bookings
        
        # Calculate load factor
        total_capacity = fare_structure.calculate_total_inventory()
        if total_capacity > 0:
            metrics.load_factor = total_bookings / total_capacity
        
        return metrics
    
    def get_revenue_summary(
        self, 
        route_id: str,
        time_period_days: int = 30
    ) -> Dict:
        """Get revenue performance summary."""
        
        if route_id not in self.revenue_history:
            return {'error': f'No revenue history for route {route_id}'}
        
        recent_metrics = self.revenue_history[route_id][-time_period_days:]
        
        if not recent_metrics:
            return {'error': f'No recent metrics for route {route_id}'}
        
        # Calculate summary statistics
        total_revenue = sum(m.total_revenue for m in recent_metrics)
        avg_load_factor = sum(m.load_factor for m in recent_metrics) / len(recent_metrics)
        avg_revenue_per_passenger = sum(m.revenue_per_passenger for m in recent_metrics) / len(recent_metrics)
        
        return {
            'route_id': route_id,
            'period_days': time_period_days,
            'total_revenue': total_revenue,
            'average_load_factor': avg_load_factor,
            'average_revenue_per_passenger': avg_revenue_per_passenger,
            'total_flights': len(recent_metrics),
            'revenue_trend': 'increasing' if len(recent_metrics) > 1 and recent_metrics[-1].total_revenue > recent_metrics[0].total_revenue else 'stable'
        }
    
    def update_booking_history(
        self,
        route_id: str,
        booking_data: Dict
    ) -> None:
        """Update booking history for learning."""
        
        booking_data['timestamp'] = datetime.now().isoformat()
        self.booking_history[route_id].append(booking_data)
        
        # Keep only recent history (last 1000 entries)
        if len(self.booking_history[route_id]) > 1000:
            self.booking_history[route_id] = self.booking_history[route_id][-1000:]
    
    def update_revenue_metrics(
        self,
        route_id: str,
        metrics: RevenueMetrics
    ) -> None:
        """Update revenue metrics history."""
        
        self.revenue_history[route_id].append(metrics)
        
        # Keep only recent history (last 365 days)
        if len(self.revenue_history[route_id]) > 365:
            self.revenue_history[route_id] = self.revenue_history[route_id][-365:]
    
    def export_data(self) -> Dict:
        """Export revenue manager data."""
        
        return {
            'configuration': {
                'strategy': self.strategy.value,
                'inventory_control': self.inventory_control.value,
                'objective': self.objective.value,
                'optimization_frequency_hours': self.optimization_frequency_hours,
                'load_factor_target': self.load_factor_target,
                'revenue_target_multiplier': self.revenue_target_multiplier
            },
            'active_routes': list(self.active_fare_structures.keys()),
            'historical_routes': list(self.revenue_history.keys()),
            'total_booking_records': sum(len(history) for history in self.booking_history.values()),
            'total_revenue_records': sum(len(history) for history in self.revenue_history.values())
        }