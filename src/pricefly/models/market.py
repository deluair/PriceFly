"""Market model for airline pricing simulation.

This module defines the Market class and related data structures for modeling
competitive market dynamics in the airline industry.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from datetime import datetime, date
from enum import Enum
import numpy as np


class MarketType(Enum):
    """Types of airline markets."""
    DOMESTIC = "domestic"
    INTERNATIONAL = "international"
    REGIONAL = "regional"
    TRANSCONTINENTAL = "transcontinental"


class CompetitionLevel(Enum):
    """Levels of market competition."""
    MONOPOLY = "monopoly"
    DUOPOLY = "duopoly"
    OLIGOPOLY = "oligopoly"
    COMPETITIVE = "competitive"
    HIGHLY_COMPETITIVE = "highly_competitive"


@dataclass
class CompetitorData:
    """Data about a competitor in the market."""
    airline_code: str
    market_share: float
    average_fare: float
    capacity: int
    load_factor: float
    service_quality: float
    brand_strength: float
    cost_structure: Dict[str, float] = field(default_factory=dict)
    pricing_strategy: str = "follower"
    last_price_change: Optional[datetime] = None
    price_change_frequency: int = 7  # days
    
    def __post_init__(self):
        """Validate competitor data."""
        if not 0 <= self.market_share <= 1:
            raise ValueError("Market share must be between 0 and 1")
        if not 0 <= self.load_factor <= 1:
            raise ValueError("Load factor must be between 0 and 1")
        if self.average_fare < 0:
            raise ValueError("Average fare must be non-negative")
        if self.capacity < 0:
            raise ValueError("Capacity must be non-negative")


@dataclass
class MarketMetrics:
    """Key metrics for a market."""
    total_capacity: int
    total_demand: int
    average_load_factor: float
    market_concentration_hhi: float  # Herfindahl-Hirschman Index
    price_dispersion: float
    average_fare: float
    revenue_per_passenger: float
    yield_per_mile: float
    competition_level: CompetitionLevel
    
    @classmethod
    def calculate_hhi(cls, market_shares: List[float]) -> float:
        """Calculate Herfindahl-Hirschman Index."""
        return sum(share ** 2 for share in market_shares) * 10000
    
    @classmethod
    def determine_competition_level(cls, hhi: float) -> CompetitionLevel:
        """Determine competition level based on HHI."""
        if hhi >= 2500:
            return CompetitionLevel.MONOPOLY
        elif hhi >= 1800:
            return CompetitionLevel.DUOPOLY
        elif hhi >= 1500:
            return CompetitionLevel.OLIGOPOLY
        elif hhi >= 1000:
            return CompetitionLevel.COMPETITIVE
        else:
            return CompetitionLevel.HIGHLY_COMPETITIVE


class Market:
    """Represents an airline market (route or city pair)."""
    
    def __init__(
        self,
        market_id: str,
        origin: str,
        destination: str,
        market_type: MarketType,
        distance: float,
        competitors: Optional[List[CompetitorData]] = None
    ):
        """Initialize market.
        
        Args:
            market_id: Unique identifier for the market
            origin: Origin airport code
            destination: Destination airport code
            market_type: Type of market (domestic, international, etc.)
            distance: Distance in miles
            competitors: List of competitors in this market
        """
        self.market_id = market_id
        self.origin = origin
        self.destination = destination
        self.market_type = market_type
        self.distance = distance
        self.competitors = competitors or []
        
        # Market state
        self.last_updated = datetime.now()
        self.historical_metrics: List[MarketMetrics] = []
        self.seasonal_factors: Dict[int, float] = {}  # month -> factor
        self.economic_factors: Dict[str, float] = {}
        
        # Demand characteristics
        self.base_demand = 0
        self.price_elasticity = -1.2
        self.business_mix = 0.3  # Percentage of business travelers
        self.leisure_mix = 0.7   # Percentage of leisure travelers
        
        # Market dynamics
        self.entry_barriers = 0.5  # 0-1 scale
        self.switching_costs = 0.3  # 0-1 scale
        self.brand_loyalty = 0.4   # 0-1 scale
    
    def add_competitor(self, competitor: CompetitorData) -> None:
        """Add a competitor to the market."""
        # Check if competitor already exists
        existing = next(
            (c for c in self.competitors if c.airline_code == competitor.airline_code),
            None
        )
        
        if existing:
            # Update existing competitor
            idx = self.competitors.index(existing)
            self.competitors[idx] = competitor
        else:
            self.competitors.append(competitor)
        
        self.last_updated = datetime.now()
    
    def remove_competitor(self, airline_code: str) -> bool:
        """Remove a competitor from the market."""
        original_count = len(self.competitors)
        self.competitors = [
            c for c in self.competitors if c.airline_code != airline_code
        ]
        
        if len(self.competitors) < original_count:
            self.last_updated = datetime.now()
            return True
        return False
    
    def get_competitor(self, airline_code: str) -> Optional[CompetitorData]:
        """Get competitor data by airline code."""
        return next(
            (c for c in self.competitors if c.airline_code == airline_code),
            None
        )
    
    def calculate_market_metrics(self) -> MarketMetrics:
        """Calculate current market metrics."""
        if not self.competitors:
            return MarketMetrics(
                total_capacity=0,
                total_demand=0,
                average_load_factor=0,
                market_concentration_hhi=0,
                price_dispersion=0,
                average_fare=0,
                revenue_per_passenger=0,
                yield_per_mile=0,
                competition_level=CompetitionLevel.MONOPOLY
            )
        
        # Calculate totals
        total_capacity = sum(c.capacity for c in self.competitors)
        market_shares = [c.market_share for c in self.competitors]
        fares = [c.average_fare for c in self.competitors]
        load_factors = [c.load_factor for c in self.competitors]
        
        # Calculate metrics
        hhi = MarketMetrics.calculate_hhi(market_shares)
        competition_level = MarketMetrics.determine_competition_level(hhi)
        
        # Weighted averages
        total_demand = int(total_capacity * np.mean(load_factors))
        average_load_factor = np.mean(load_factors)
        average_fare = np.average(fares, weights=market_shares)
        
        # Price dispersion (coefficient of variation)
        price_dispersion = np.std(fares) / np.mean(fares) if np.mean(fares) > 0 else 0
        
        # Revenue metrics
        revenue_per_passenger = average_fare
        yield_per_mile = average_fare / self.distance if self.distance > 0 else 0
        
        metrics = MarketMetrics(
            total_capacity=total_capacity,
            total_demand=total_demand,
            average_load_factor=average_load_factor,
            market_concentration_hhi=hhi,
            price_dispersion=price_dispersion,
            average_fare=average_fare,
            revenue_per_passenger=revenue_per_passenger,
            yield_per_mile=yield_per_mile,
            competition_level=competition_level
        )
        
        # Store historical data
        self.historical_metrics.append(metrics)
        
        return metrics
    
    def get_market_leader(self) -> Optional[CompetitorData]:
        """Get the market leader by market share."""
        if not self.competitors:
            return None
        
        return max(self.competitors, key=lambda c: c.market_share)
    
    def get_price_leader(self) -> Optional[CompetitorData]:
        """Get the price leader (highest fare)."""
        if not self.competitors:
            return None
        
        return max(self.competitors, key=lambda c: c.average_fare)
    
    def get_low_cost_leader(self) -> Optional[CompetitorData]:
        """Get the low-cost leader (lowest fare)."""
        if not self.competitors:
            return None
        
        return min(self.competitors, key=lambda c: c.average_fare)
    
    def calculate_demand_forecast(
        self,
        base_price: float,
        seasonality_factor: float = 1.0,
        economic_factor: float = 1.0
    ) -> int:
        """Calculate demand forecast based on price and external factors."""
        if self.base_demand == 0:
            # Estimate base demand from current market
            metrics = self.calculate_market_metrics()
            self.base_demand = metrics.total_demand
        
        # Apply price elasticity
        if self.competitors:
            current_avg_price = np.mean([c.average_fare for c in self.competitors])
            price_ratio = base_price / current_avg_price if current_avg_price > 0 else 1.0
            price_effect = price_ratio ** self.price_elasticity
        else:
            price_effect = 1.0
        
        # Calculate forecasted demand
        forecasted_demand = (
            self.base_demand * 
            price_effect * 
            seasonality_factor * 
            economic_factor
        )
        
        return max(0, int(forecasted_demand))
    
    def update_seasonal_factors(self, factors: Dict[int, float]) -> None:
        """Update seasonal factors for demand forecasting."""
        self.seasonal_factors.update(factors)
    
    def update_economic_factors(self, factors: Dict[str, float]) -> None:
        """Update economic factors affecting demand."""
        self.economic_factors.update(factors)
    
    def get_competitive_position(self, airline_code: str) -> Dict[str, float]:
        """Get competitive position metrics for a specific airline."""
        competitor = self.get_competitor(airline_code)
        if not competitor:
            return {}
        
        metrics = self.calculate_market_metrics()
        
        # Calculate relative positions
        market_leader = self.get_market_leader()
        price_leader = self.get_price_leader()
        low_cost_leader = self.get_low_cost_leader()
        
        position = {
            'market_share_rank': sorted(
                self.competitors, 
                key=lambda c: c.market_share, 
                reverse=True
            ).index(competitor) + 1,
            'price_rank': sorted(
                self.competitors, 
                key=lambda c: c.average_fare, 
                reverse=True
            ).index(competitor) + 1,
            'market_share_vs_leader': (
                competitor.market_share / market_leader.market_share 
                if market_leader and market_leader.market_share > 0 else 0
            ),
            'price_vs_average': (
                competitor.average_fare / metrics.average_fare 
                if metrics.average_fare > 0 else 0
            ),
            'load_factor_vs_average': (
                competitor.load_factor / metrics.average_load_factor 
                if metrics.average_load_factor > 0 else 0
            )
        }
        
        return position
    
    def simulate_price_change(
        self, 
        airline_code: str, 
        new_price: float
    ) -> Dict[str, float]:
        """Simulate the impact of a price change."""
        competitor = self.get_competitor(airline_code)
        if not competitor:
            return {}
        
        # Store original price
        original_price = competitor.average_fare
        
        # Calculate price change impact
        price_change_pct = (new_price - original_price) / original_price
        
        # Estimate demand impact using price elasticity
        demand_change_pct = price_change_pct * self.price_elasticity
        
        # Estimate market share impact (simplified model)
        market_share_change = -price_change_pct * 0.5  # Negative correlation
        
        # Estimate competitive response
        competitive_response = self._estimate_competitive_response(
            airline_code, price_change_pct
        )
        
        return {
            'price_change_pct': price_change_pct,
            'demand_change_pct': demand_change_pct,
            'market_share_change': market_share_change,
            'competitive_response': competitive_response,
            'revenue_impact': demand_change_pct + price_change_pct
        }
    
    def _estimate_competitive_response(
        self, 
        initiating_airline: str, 
        price_change_pct: float
    ) -> float:
        """Estimate how competitors will respond to a price change."""
        # Simplified competitive response model
        # In reality, this would be much more sophisticated
        
        response_intensity = 0.3  # How much competitors respond
        
        if price_change_pct < 0:  # Price decrease
            # Competitors likely to follow with price cuts
            return price_change_pct * response_intensity
        else:  # Price increase
            # Competitors may not follow immediately
            return price_change_pct * response_intensity * 0.5
    
    def export_data(self) -> Dict:
        """Export market data for analysis."""
        return {
            'market_id': self.market_id,
            'origin': self.origin,
            'destination': self.destination,
            'market_type': self.market_type.value,
            'distance': self.distance,
            'competitors': [
                {
                    'airline_code': c.airline_code,
                    'market_share': c.market_share,
                    'average_fare': c.average_fare,
                    'capacity': c.capacity,
                    'load_factor': c.load_factor,
                    'service_quality': c.service_quality,
                    'brand_strength': c.brand_strength,
                    'pricing_strategy': c.pricing_strategy
                }
                for c in self.competitors
            ],
            'metrics': self.calculate_market_metrics().__dict__,
            'base_demand': self.base_demand,
            'price_elasticity': self.price_elasticity,
            'business_mix': self.business_mix,
            'leisure_mix': self.leisure_mix,
            'last_updated': self.last_updated.isoformat()
        }
    
    def __str__(self) -> str:
        """String representation of the market."""
        return f"Market({self.origin}-{self.destination}, {len(self.competitors)} competitors)"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"Market(id='{self.market_id}', route='{self.origin}-{self.destination}', "
            f"type={self.market_type.value}, competitors={len(self.competitors)})"
        )