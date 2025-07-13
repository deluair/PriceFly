"""Dynamic Pricing Engine for airline fare optimization."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from ..models.airline import Airline, BookingClass
from ..models.airport import Route
from ..models.passenger import CustomerSegment, Passenger


class PricingStrategy(Enum):
    """Different pricing strategy approaches."""
    DEMAND_BASED = "demand_based"
    COMPETITION_BASED = "competition_based"
    COST_PLUS = "cost_plus"
    REVENUE_OPTIMIZATION = "revenue_optimization"
    DYNAMIC_HYBRID = "dynamic_hybrid"


class MarketCondition(Enum):
    """Market condition classifications."""
    LOW_DEMAND = "low_demand"
    NORMAL_DEMAND = "normal_demand"
    HIGH_DEMAND = "high_demand"
    PEAK_DEMAND = "peak_demand"


@dataclass
class PricingContext:
    """Context information for pricing decisions."""
    
    # Time Context
    current_date: datetime = field(default_factory=datetime.now)
    departure_date: datetime = field(default_factory=datetime.now)
    booking_horizon_days: int = 0
    
    # Market Context
    route: Optional[Route] = None
    market_condition: MarketCondition = MarketCondition.NORMAL_DEMAND
    competitor_prices: Dict[str, float] = field(default_factory=dict)
    
    # Demand Context
    current_bookings: int = 0
    total_capacity: int = 180
    historical_demand: List[int] = field(default_factory=list)
    
    # Cost Context
    fuel_price_per_liter: float = 0.85
    operational_cost_per_flight: float = 15000.0
    
    # External Factors
    economic_indicators: Dict[str, float] = field(default_factory=dict)
    seasonal_factor: float = 1.0
    event_factor: float = 1.0  # Special events impact
    
    def __post_init__(self):
        """Calculate derived fields."""
        if self.departure_date and self.current_date:
            self.booking_horizon_days = (self.departure_date - self.current_date).days
    
    @property
    def load_factor(self) -> float:
        """Current load factor."""
        if self.total_capacity == 0:
            return 0.0
        return self.current_bookings / self.total_capacity
    
    @property
    def days_to_departure(self) -> int:
        """Days until departure."""
        return max(0, self.booking_horizon_days)


@dataclass
class PricingRecommendation:
    """Pricing recommendation output."""
    
    booking_class: str = ""
    recommended_price: float = 0.0
    confidence_score: float = 0.0
    
    # Supporting Information
    base_price: float = 0.0
    demand_adjustment: float = 1.0
    competition_adjustment: float = 1.0
    time_adjustment: float = 1.0
    
    # Revenue Projections
    expected_bookings: float = 0.0
    expected_revenue: float = 0.0
    revenue_risk: float = 0.0
    
    # Market Intelligence
    price_position: str = ""  # "below_market", "at_market", "above_market"
    demand_forecast: str = ""  # "low", "normal", "high"
    
    reasoning: List[str] = field(default_factory=list)


class DynamicPricingEngine:
    """Advanced dynamic pricing engine for airline revenue optimization."""
    
    def __init__(self, airline: Airline, strategy: PricingStrategy = PricingStrategy.DYNAMIC_HYBRID):
        self.airline = airline
        self.strategy = strategy
        self.logger = logging.getLogger(__name__)
        
        # ML Models for demand forecasting
        self.demand_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.price_elasticity_model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        
        # Pricing parameters
        self.pricing_params = {
            "min_markup": 0.1,  # Minimum 10% markup over cost
            "max_markup": 3.0,  # Maximum 300% markup
            "demand_sensitivity": 0.5,
            "competition_sensitivity": 0.3,
            "time_sensitivity": 0.4,
            "risk_tolerance": 0.2
        }
        
        # Historical data storage
        self.pricing_history: List[Dict] = []
        self.demand_history: List[Dict] = []
        
        # Model training status
        self.models_trained = False
    
    def calculate_base_price(self, context: PricingContext, 
                           booking_class: BookingClass) -> float:
        """Calculate base price considering costs and market position."""
        if not context.route:
            return booking_class.base_fare
        
        # Cost-based pricing foundation
        cost_per_passenger = (
            context.operational_cost_per_flight / context.total_capacity
        )
        
        # Add fuel costs
        if context.route:
            fuel_cost_per_passenger = (
                context.route.distance_km * 3.5 * context.fuel_price_per_liter / 
                context.total_capacity
            )
            cost_per_passenger += fuel_cost_per_passenger
        
        # Apply markup based on booking class
        markup_multipliers = {
            "F": 4.0,  # First class
            "J": 2.5,  # Business class
            "W": 1.8,  # Premium economy
            "Y": 1.5,  # Economy flexible
            "B": 1.3,  # Economy standard
            "M": 1.2,  # Economy saver
            "H": 1.1,  # Economy basic
            "L": 1.05  # Ultra basic
        }
        
        markup = markup_multipliers.get(booking_class.booking_code, 1.3)
        base_price = cost_per_passenger * markup
        
        # Adjust for airline type
        if self.airline.airline_type.value == "low_cost":
            base_price *= 0.8
        elif self.airline.airline_type.value == "ultra_low_cost":
            base_price *= 0.6
        
        return max(base_price, booking_class.base_fare * 0.5)
    
    def calculate_demand_adjustment(self, context: PricingContext) -> float:
        """Calculate price adjustment based on demand conditions."""
        adjustments = []
        
        # Load factor adjustment
        if context.load_factor < 0.3:
            adjustments.append(0.8)  # Low demand, reduce prices
        elif context.load_factor < 0.6:
            adjustments.append(0.9)
        elif context.load_factor > 0.8:
            adjustments.append(1.3)  # High demand, increase prices
        elif context.load_factor > 0.9:
            adjustments.append(1.5)
        else:
            adjustments.append(1.0)
        
        # Market condition adjustment
        condition_multipliers = {
            MarketCondition.LOW_DEMAND: 0.85,
            MarketCondition.NORMAL_DEMAND: 1.0,
            MarketCondition.HIGH_DEMAND: 1.2,
            MarketCondition.PEAK_DEMAND: 1.4
        }
        adjustments.append(condition_multipliers[context.market_condition])
        
        # Seasonal adjustment
        adjustments.append(context.seasonal_factor)
        
        # Event adjustment
        adjustments.append(context.event_factor)
        
        # Calculate weighted average
        weights = [0.4, 0.3, 0.2, 0.1]  # Load factor has highest weight
        return sum(adj * weight for adj, weight in zip(adjustments, weights))
    
    def calculate_competition_adjustment(self, context: PricingContext, 
                                       base_price: float) -> float:
        """Calculate price adjustment based on competitive positioning."""
        if not context.competitor_prices:
            return 1.0
        
        competitor_prices = list(context.competitor_prices.values())
        avg_competitor_price = np.mean(competitor_prices)
        min_competitor_price = np.min(competitor_prices)
        
        # Calculate our position relative to competition
        if avg_competitor_price == 0:
            return 1.0
        
        price_ratio = base_price / avg_competitor_price
        
        # Adjust based on airline's competitive strategy
        if self.airline.price_competitiveness < 0.9:  # Aggressive pricing
            target_ratio = 0.95  # Aim to be 5% below market
        elif self.airline.price_competitiveness > 1.1:  # Premium pricing
            target_ratio = 1.1   # Aim to be 10% above market
        else:
            target_ratio = 1.0   # Match market
        
        # Calculate adjustment to reach target ratio
        adjustment = target_ratio / price_ratio if price_ratio > 0 else 1.0
        
        # Limit adjustment range
        return np.clip(adjustment, 0.8, 1.3)
    
    def calculate_time_adjustment(self, context: PricingContext) -> float:
        """Calculate price adjustment based on time to departure."""
        days_out = context.days_to_departure
        
        # Booking curve adjustments
        if days_out > 90:
            return 0.8   # Early bird discount
        elif days_out > 60:
            return 0.9
        elif days_out > 30:
            return 1.0   # Base price
        elif days_out > 14:
            return 1.1
        elif days_out > 7:
            return 1.2
        elif days_out > 3:
            return 1.4   # Last minute premium
        else:
            return 1.6   # Very last minute
    
    def forecast_demand(self, context: PricingContext, price: float) -> float:
        """Forecast demand at a given price point."""
        if not self.models_trained:
            # Simple elasticity-based model
            if context.route:
                base_demand = context.route.annual_demand / 365 / context.route.frequency_per_day
                price_elasticity = context.route.price_elasticity
                
                if context.route.average_fare_economy > 0:
                    price_ratio = price / context.route.average_fare_economy
                    demand_multiplier = price_ratio ** price_elasticity
                    return base_demand * demand_multiplier * context.seasonal_factor
            
            # Fallback calculation
            base_demand = context.total_capacity * 0.8  # 80% base load factor
            price_sensitivity = -0.5  # Default elasticity
            
            # Assume $300 as reference price
            price_ratio = price / 300.0
            demand_multiplier = price_ratio ** price_sensitivity
            
            return base_demand * demand_multiplier
        
        # Use trained ML model (placeholder for now)
        # In a real implementation, this would use the trained model
        return self._ml_demand_forecast(context, price)
    
    def _ml_demand_forecast(self, context: PricingContext, price: float) -> float:
        """Machine learning-based demand forecast (placeholder)."""
        # This would use the trained demand_model
        # For now, return a simple calculation
        base_demand = context.total_capacity * 0.8
        price_effect = max(0.1, 1 - (price - 200) / 1000)  # Simple price response
        return base_demand * price_effect * context.seasonal_factor
    
    def optimize_price(self, context: PricingContext, 
                      booking_class: BookingClass) -> PricingRecommendation:
        """Optimize price for maximum revenue."""
        base_price = self.calculate_base_price(context, booking_class)
        
        # Calculate adjustment factors
        demand_adj = self.calculate_demand_adjustment(context)
        competition_adj = self.calculate_competition_adjustment(context, base_price)
        time_adj = self.calculate_time_adjustment(context)
        
        # Apply strategy-specific logic
        if self.strategy == PricingStrategy.DEMAND_BASED:
            final_price = base_price * demand_adj * time_adj
        elif self.strategy == PricingStrategy.COMPETITION_BASED:
            final_price = base_price * competition_adj
        elif self.strategy == PricingStrategy.REVENUE_OPTIMIZATION:
            final_price = self._revenue_optimization(context, booking_class, base_price)
        else:  # DYNAMIC_HYBRID
            final_price = base_price * demand_adj * competition_adj * time_adj
        
        # Apply bounds
        min_price = base_price * (1 + self.pricing_params["min_markup"])
        max_price = base_price * (1 + self.pricing_params["max_markup"])
        final_price = np.clip(final_price, min_price, max_price)
        
        # Forecast demand and revenue
        expected_demand = self.forecast_demand(context, final_price)
        expected_revenue = final_price * expected_demand
        
        # Calculate confidence and risk
        confidence = self._calculate_confidence(context, final_price)
        risk = self._calculate_revenue_risk(context, final_price, expected_revenue)
        
        # Determine market position
        market_position = self._determine_market_position(context, final_price)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            context, base_price, final_price, demand_adj, competition_adj, time_adj
        )
        
        return PricingRecommendation(
            booking_class=booking_class.booking_code,
            recommended_price=final_price,
            confidence_score=confidence,
            base_price=base_price,
            demand_adjustment=demand_adj,
            competition_adjustment=competition_adj,
            time_adjustment=time_adj,
            expected_bookings=expected_demand,
            expected_revenue=expected_revenue,
            revenue_risk=risk,
            price_position=market_position,
            reasoning=reasoning
        )
    
    def _revenue_optimization(self, context: PricingContext, 
                            booking_class: BookingClass, base_price: float) -> float:
        """Optimize price for maximum revenue using mathematical optimization."""
        def revenue_function(price):
            demand = self.forecast_demand(context, price[0])
            return -(price[0] * demand)  # Negative because we minimize
        
        # Set bounds
        min_price = base_price * 0.5
        max_price = base_price * 3.0
        
        # Optimize
        result = minimize(
            revenue_function,
            x0=[base_price],
            bounds=[(min_price, max_price)],
            method='L-BFGS-B'
        )
        
        return result.x[0] if result.success else base_price
    
    def _calculate_confidence(self, context: PricingContext, price: float) -> float:
        """Calculate confidence score for pricing recommendation."""
        confidence_factors = []
        
        # Data quality
        if context.competitor_prices:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.4)
        
        # Historical data availability
        if len(self.pricing_history) > 100:
            confidence_factors.append(0.9)
        elif len(self.pricing_history) > 20:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        # Market stability
        if context.market_condition == MarketCondition.NORMAL_DEMAND:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.6)
        
        # Time to departure
        if 7 <= context.days_to_departure <= 60:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.6)
        
        return np.mean(confidence_factors)
    
    def _calculate_revenue_risk(self, context: PricingContext, 
                              price: float, expected_revenue: float) -> float:
        """Calculate revenue risk score."""
        # Risk factors
        demand_volatility = 0.2  # Assume 20% demand volatility
        price_sensitivity = abs(context.route.price_elasticity) if context.route else 1.2
        
        # Calculate potential revenue variance
        revenue_variance = expected_revenue * demand_volatility * price_sensitivity
        
        # Normalize to 0-1 scale
        risk_score = min(1.0, revenue_variance / expected_revenue)
        
        return risk_score
    
    def _determine_market_position(self, context: PricingContext, price: float) -> str:
        """Determine price position relative to market."""
        if not context.competitor_prices:
            return "unknown"
        
        avg_competitor_price = np.mean(list(context.competitor_prices.values()))
        
        if price < avg_competitor_price * 0.95:
            return "below_market"
        elif price > avg_competitor_price * 1.05:
            return "above_market"
        else:
            return "at_market"
    
    def _generate_reasoning(self, context: PricingContext, base_price: float,
                          final_price: float, demand_adj: float,
                          competition_adj: float, time_adj: float) -> List[str]:
        """Generate human-readable reasoning for pricing decision."""
        reasoning = []
        
        # Base price explanation
        reasoning.append(f"Base price calculated at ${base_price:.2f} considering operational costs")
        
        # Demand adjustment
        if demand_adj > 1.1:
            reasoning.append(f"Price increased {(demand_adj-1)*100:.1f}% due to high demand conditions")
        elif demand_adj < 0.9:
            reasoning.append(f"Price reduced {(1-demand_adj)*100:.1f}% due to low demand conditions")
        
        # Competition adjustment
        if competition_adj > 1.05:
            reasoning.append("Price adjusted upward to maintain premium positioning")
        elif competition_adj < 0.95:
            reasoning.append("Price adjusted downward to remain competitive")
        
        # Time adjustment
        if time_adj > 1.2:
            reasoning.append(f"Last-minute premium applied ({(time_adj-1)*100:.1f}% increase)")
        elif time_adj < 0.9:
            reasoning.append(f"Early booking discount applied ({(1-time_adj)*100:.1f}% reduction)")
        
        # Load factor consideration
        if context.load_factor > 0.8:
            reasoning.append("High load factor supports premium pricing")
        elif context.load_factor < 0.4:
            reasoning.append("Low load factor necessitates aggressive pricing")
        
        return reasoning
    
    def update_pricing_history(self, context: PricingContext, 
                             recommendation: PricingRecommendation,
                             actual_bookings: Optional[int] = None) -> None:
        """Update pricing history for model learning."""
        history_entry = {
            "timestamp": context.current_date,
            "route": context.route.route_code if context.route else "unknown",
            "booking_class": recommendation.booking_class,
            "recommended_price": recommendation.recommended_price,
            "base_price": recommendation.base_price,
            "load_factor": context.load_factor,
            "days_to_departure": context.days_to_departure,
            "market_condition": context.market_condition.value,
            "expected_bookings": recommendation.expected_bookings,
            "actual_bookings": actual_bookings,
            "confidence": recommendation.confidence_score
        }
        
        self.pricing_history.append(history_entry)
        
        # Limit history size
        if len(self.pricing_history) > 10000:
            self.pricing_history = self.pricing_history[-5000:]
    
    def train_models(self) -> bool:
        """Train ML models using historical data."""
        if len(self.pricing_history) < 50:
            self.logger.warning("Insufficient data for model training")
            return False
        
        try:
            # Prepare training data
            df = pd.DataFrame(self.pricing_history)
            df = df.dropna(subset=['actual_bookings'])
            
            if len(df) < 20:
                return False
            
            # Feature engineering
            features = [
                'recommended_price', 'base_price', 'load_factor',
                'days_to_departure', 'confidence'
            ]
            
            X = df[features].values
            y = df['actual_bookings'].values
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train demand model
            self.demand_model.fit(X_scaled, y)
            
            self.models_trained = True
            self.logger.info("Models trained successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            return False
    
    def get_pricing_analytics(self) -> Dict[str, Any]:
        """Get analytics on pricing performance."""
        if not self.pricing_history:
            return {"error": "No pricing history available"}
        
        df = pd.DataFrame(self.pricing_history)
        
        analytics = {
            "total_recommendations": len(df),
            "average_confidence": df['confidence'].mean(),
            "price_range": {
                "min": df['recommended_price'].min(),
                "max": df['recommended_price'].max(),
                "mean": df['recommended_price'].mean()
            },
            "accuracy_metrics": {},
            "route_performance": {},
            "booking_class_performance": {}
        }
        
        # Calculate accuracy where actual bookings are available
        actual_data = df.dropna(subset=['actual_bookings'])
        if len(actual_data) > 0:
            mape = np.mean(np.abs(
                (actual_data['actual_bookings'] - actual_data['expected_bookings']) /
                actual_data['actual_bookings']
            )) * 100
            
            analytics["accuracy_metrics"] = {
                "mean_absolute_percentage_error": mape,
                "predictions_with_actuals": len(actual_data)
            }
        
        # Route-level performance
        route_stats = df.groupby('route').agg({
            'recommended_price': ['mean', 'std'],
            'confidence': 'mean',
            'expected_bookings': 'sum'
        }).round(2)
        
        analytics["route_performance"] = route_stats.to_dict()
        
        return analytics