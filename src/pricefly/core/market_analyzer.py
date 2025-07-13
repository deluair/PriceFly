"""Market analysis engine for competitive intelligence and market dynamics.

This module provides comprehensive market analysis capabilities including
competitive analysis, market positioning, pricing intelligence, and
market trend analysis for airline revenue management.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, date, timedelta
from enum import Enum
import numpy as np
import logging
from collections import defaultdict
import statistics

from ..models.market import Market, MarketType, CompetitionLevel, CompetitorData, MarketMetrics
from ..models.route import Route
from ..models.pricing import FareStructure


class AnalysisType(Enum):
    """Types of market analysis."""
    COMPETITIVE_POSITIONING = "competitive_positioning"
    PRICE_INTELLIGENCE = "price_intelligence"
    MARKET_SHARE_ANALYSIS = "market_share_analysis"
    DEMAND_ELASTICITY = "demand_elasticity"
    ROUTE_PROFITABILITY = "route_profitability"
    SEASONAL_TRENDS = "seasonal_trends"
    CAPACITY_ANALYSIS = "capacity_analysis"


class MarketSegment(Enum):
    """Market segments for analysis."""
    BUSINESS = "business"
    LEISURE = "leisure"
    PREMIUM = "premium"
    BUDGET = "budget"
    CORPORATE = "corporate"
    GOVERNMENT = "government"
    CARGO = "cargo"


class CompetitivePosition(Enum):
    """Competitive position classifications."""
    MARKET_LEADER = "market_leader"
    STRONG_COMPETITOR = "strong_competitor"
    AVERAGE_COMPETITOR = "average_competitor"
    WEAK_COMPETITOR = "weak_competitor"
    NICHE_PLAYER = "niche_player"
    NEW_ENTRANT = "new_entrant"


@dataclass
class CompetitiveAnalysis:
    """Competitive analysis results."""
    # Market position
    market_position: CompetitivePosition
    market_share: float  # 0-1 scale
    relative_market_share: float  # Relative to largest competitor
    
    # Pricing analysis
    price_position: str  # 'premium', 'competitive', 'discount'
    price_index: float  # Relative to market average (1.0 = average)
    price_gap_to_leader: float  # Price difference to market leader
    
    # Competitive metrics
    number_of_competitors: int
    market_concentration: float  # HHI index
    competitive_intensity: float  # 0-1 scale
    
    # Performance metrics
    load_factor_vs_market: float  # Relative to market average
    revenue_per_passenger_vs_market: float
    frequency_advantage: float  # Schedule frequency vs competitors
    
    # Strategic insights
    key_competitors: List[str]
    competitive_advantages: List[str]
    competitive_threats: List[str]
    market_opportunities: List[str]
    
    # Metadata
    analysis_date: datetime = field(default_factory=datetime.now)
    route_id: str = ""
    confidence_score: float = 0.8
    
    def get_competitive_summary(self) -> Dict:
        """Get summary of competitive analysis."""
        return {
            'market_position': self.market_position.value,
            'market_share_percent': self.market_share * 100,
            'price_position': self.price_position,
            'price_index': self.price_index,
            'number_of_competitors': self.number_of_competitors,
            'competitive_intensity': self.competitive_intensity,
            'key_advantages': self.competitive_advantages[:3],  # Top 3
            'main_threats': self.competitive_threats[:3],  # Top 3
            'confidence_score': self.confidence_score
        }


@dataclass
class PriceIntelligence:
    """Price intelligence analysis results."""
    # Market pricing
    market_average_price: float
    market_price_range: Tuple[float, float]  # (min, max)
    price_standard_deviation: float
    
    # Competitor pricing
    competitor_prices: Dict[str, float]
    price_leader: str  # Competitor with lowest price
    premium_leader: str  # Competitor with highest price
    
    # Price trends
    price_trend_direction: str  # 'increasing', 'decreasing', 'stable'
    price_volatility: float  # Coefficient of variation
    seasonal_price_patterns: Dict[int, float]  # Month -> average price
    
    # Price elasticity
    estimated_price_elasticity: float
    demand_sensitivity: str  # 'high', 'medium', 'low'
    
    # Recommendations
    optimal_price_range: Tuple[float, float]
    pricing_strategy_recommendation: str
    
    # Metadata
    analysis_date: datetime = field(default_factory=datetime.now)
    route_id: str = ""
    data_quality: float = 0.8
    
    def get_pricing_summary(self) -> Dict:
        """Get summary of pricing intelligence."""
        return {
            'market_average_price': self.market_average_price,
            'price_range': {
                'min': self.price_range[0],
                'max': self.price_range[1]
            },
            'price_trend': self.price_trend_direction,
            'price_volatility': self.price_volatility,
            'demand_sensitivity': self.demand_sensitivity,
            'optimal_price_range': {
                'min': self.optimal_price_range[0],
                'max': self.optimal_price_range[1]
            },
            'strategy_recommendation': self.pricing_strategy_recommendation,
            'data_quality': self.data_quality
        }


@dataclass
class MarketTrend:
    """Market trend analysis."""
    trend_name: str
    trend_direction: str  # 'growing', 'declining', 'stable'
    trend_strength: float  # 0-1 scale
    trend_duration_months: int
    
    # Trend drivers
    primary_drivers: List[str]
    supporting_factors: List[str]
    
    # Impact assessment
    revenue_impact: float  # Estimated % impact on revenue
    demand_impact: float  # Estimated % impact on demand
    competitive_impact: str  # 'positive', 'negative', 'neutral'
    
    # Forecast
    expected_duration_months: int
    confidence_level: float
    
    def get_trend_summary(self) -> Dict:
        """Get summary of market trend."""
        return {
            'trend_name': self.trend_name,
            'direction': self.trend_direction,
            'strength': self.trend_strength,
            'duration_months': self.trend_duration_months,
            'revenue_impact_percent': self.revenue_impact,
            'demand_impact_percent': self.demand_impact,
            'competitive_impact': self.competitive_impact,
            'primary_drivers': self.primary_drivers,
            'confidence_level': self.confidence_level
        }


@dataclass
class MarketAnalysisResult:
    """Comprehensive market analysis result."""
    # Analysis components
    competitive_analysis: Optional[CompetitiveAnalysis] = None
    price_intelligence: Optional[PriceIntelligence] = None
    market_trends: List[MarketTrend] = field(default_factory=list)
    
    # Market characteristics
    market_size: float = 0.0  # Annual passengers
    market_growth_rate: float = 0.0  # Annual growth %
    market_maturity: str = "mature"  # 'emerging', 'growing', 'mature', 'declining'
    
    # Risk assessment
    market_risk_score: float = 0.5  # 0-1 scale
    regulatory_risk: float = 0.3
    competitive_risk: float = 0.4
    economic_risk: float = 0.3
    
    # Strategic recommendations
    strategic_recommendations: List[str] = field(default_factory=list)
    tactical_actions: List[str] = field(default_factory=list)
    
    # Metadata
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    route_id: str = ""
    analysis_types: List[AnalysisType] = field(default_factory=list)
    
    def get_executive_summary(self) -> Dict:
        """Get executive summary of market analysis."""
        summary = {
            'route_id': self.route_id,
            'analysis_date': self.analysis_timestamp.isoformat(),
            'market_size': self.market_size,
            'market_growth_rate': self.market_growth_rate,
            'market_maturity': self.market_maturity,
            'overall_risk_score': self.market_risk_score,
            'analysis_types': [at.value for at in self.analysis_types]
        }
        
        if self.competitive_analysis:
            summary['competitive_position'] = self.competitive_analysis.market_position.value
            summary['market_share'] = self.competitive_analysis.market_share
        
        if self.price_intelligence:
            summary['price_position'] = self.price_intelligence.pricing_strategy_recommendation
            summary['market_average_price'] = self.price_intelligence.market_average_price
        
        if self.market_trends:
            summary['key_trends'] = [trend.trend_name for trend in self.market_trends[:3]]
        
        summary['top_recommendations'] = self.strategic_recommendations[:3]
        
        return summary


class MarketAnalyzer:
    """Advanced market analysis engine for airline revenue management."""
    
    def __init__(self):
        # Historical data storage
        self.market_data: Dict[str, List[MarketMetrics]] = defaultdict(list)
        self.competitor_data: Dict[str, Dict[str, List[CompetitorData]]] = defaultdict(lambda: defaultdict(list))
        self.price_history: Dict[str, List[Tuple[date, Dict[str, float]]]] = defaultdict(list)
        
        # Analysis cache
        self.analysis_cache: Dict[str, MarketAnalysisResult] = {}
        self.cache_expiry_hours = 6
        
        # Configuration
        self.min_data_points = 10
        self.competitor_threshold = 0.05  # 5% market share to be considered significant
        self.price_change_threshold = 0.02  # 2% price change to be considered significant
        
        # Market intelligence sources
        self.data_sources: Set[str] = set()
        
        self.logger = logging.getLogger(__name__)
    
    def analyze_market(
        self,
        route_id: str,
        market: Market,
        analysis_types: List[AnalysisType],
        own_fare_structure: Optional[FareStructure] = None
    ) -> MarketAnalysisResult:
        """Perform comprehensive market analysis."""
        
        self.logger.info(
            f"Starting market analysis for route {route_id} with {len(analysis_types)} analysis types"
        )
        
        # Check cache first
        cache_key = f"{route_id}_{hash(tuple(analysis_types))}"
        if self._is_cache_valid(cache_key):
            self.logger.info("Returning cached analysis result")
            return self.analysis_cache[cache_key]
        
        result = MarketAnalysisResult(
            route_id=route_id,
            analysis_types=analysis_types
        )
        
        # Perform requested analyses
        for analysis_type in analysis_types:
            try:
                if analysis_type == AnalysisType.COMPETITIVE_POSITIONING:
                    result.competitive_analysis = self._analyze_competitive_positioning(
                        route_id, market, own_fare_structure
                    )
                
                elif analysis_type == AnalysisType.PRICE_INTELLIGENCE:
                    result.price_intelligence = self._analyze_price_intelligence(
                        route_id, market, own_fare_structure
                    )
                
                elif analysis_type == AnalysisType.SEASONAL_TRENDS:
                    seasonal_trends = self._analyze_seasonal_trends(route_id, market)
                    result.market_trends.extend(seasonal_trends)
                
                elif analysis_type == AnalysisType.MARKET_SHARE_ANALYSIS:
                    market_share_trends = self._analyze_market_share_trends(route_id, market)
                    result.market_trends.extend(market_share_trends)
                
                elif analysis_type == AnalysisType.CAPACITY_ANALYSIS:
                    capacity_trends = self._analyze_capacity_trends(route_id, market)
                    result.market_trends.extend(capacity_trends)
                
            except Exception as e:
                self.logger.error(f"Error in {analysis_type.value} analysis: {e}")
                continue
        
        # Calculate market characteristics
        result.market_size = self._calculate_market_size(route_id, market)
        result.market_growth_rate = self._calculate_market_growth_rate(route_id, market)
        result.market_maturity = self._assess_market_maturity(route_id, market)
        
        # Risk assessment
        result.market_risk_score = self._calculate_market_risk(route_id, market)
        result.regulatory_risk = self._assess_regulatory_risk(route_id, market)
        result.competitive_risk = self._assess_competitive_risk(route_id, market)
        result.economic_risk = self._assess_economic_risk(route_id, market)
        
        # Generate recommendations
        result.strategic_recommendations = self._generate_strategic_recommendations(result)
        result.tactical_actions = self._generate_tactical_actions(result)
        
        # Cache result
        self.analysis_cache[cache_key] = result
        
        self.logger.info(
            f"Market analysis completed for route {route_id}. "
            f"Risk score: {result.market_risk_score:.2f}, "
            f"Market size: {result.market_size:.0f} passengers"
        )
        
        return result
    
    def _analyze_competitive_positioning(
        self,
        route_id: str,
        market: Market,
        own_fare_structure: Optional[FareStructure]
    ) -> CompetitiveAnalysis:
        """Analyze competitive positioning."""
        
        # Get competitor data
        competitors = market.competitors
        total_capacity = sum(comp.capacity for comp in competitors.values())
        
        if total_capacity == 0:
            raise ValueError("No competitor capacity data available")
        
        # Calculate market shares
        market_shares = {}
        for airline, comp_data in competitors.items():
            market_shares[airline] = comp_data.capacity / total_capacity
        
        # Determine our market share (if we have capacity data)
        our_market_share = 0.0
        if own_fare_structure:
            our_capacity = own_fare_structure.calculate_total_inventory()
            our_market_share = our_capacity / (total_capacity + our_capacity)
        
        # Determine market position
        sorted_shares = sorted(market_shares.values(), reverse=True)
        if our_market_share >= sorted_shares[0] if sorted_shares else 0:
            position = CompetitivePosition.MARKET_LEADER
        elif our_market_share >= 0.20:
            position = CompetitivePosition.STRONG_COMPETITOR
        elif our_market_share >= 0.10:
            position = CompetitivePosition.AVERAGE_COMPETITOR
        elif our_market_share >= 0.05:
            position = CompetitivePosition.WEAK_COMPETITOR
        else:
            position = CompetitivePosition.NICHE_PLAYER
        
        # Calculate relative market share
        largest_competitor_share = max(market_shares.values()) if market_shares else 0
        relative_market_share = our_market_share / largest_competitor_share if largest_competitor_share > 0 else 0
        
        # Price analysis
        competitor_prices = {}
        for airline, comp_data in competitors.items():
            if comp_data.average_fare > 0:
                competitor_prices[airline] = comp_data.average_fare
        
        market_avg_price = statistics.mean(competitor_prices.values()) if competitor_prices else 0
        
        our_avg_price = 0
        if own_fare_structure:
            our_avg_price = statistics.mean(
                pp.price for pp in own_fare_structure.fare_classes.values()
            )
        
        price_index = our_avg_price / market_avg_price if market_avg_price > 0 else 1.0
        
        if price_index > 1.1:
            price_position = "premium"
        elif price_index < 0.9:
            price_position = "discount"
        else:
            price_position = "competitive"
        
        # Calculate market concentration (HHI)
        hhi = sum(share ** 2 for share in market_shares.values())
        
        # Competitive intensity
        if hhi > 0.25:
            competitive_intensity = 0.3  # Low competition (concentrated market)
        elif hhi > 0.15:
            competitive_intensity = 0.6  # Medium competition
        else:
            competitive_intensity = 0.9  # High competition
        
        # Identify key competitors (top 3 by market share)
        key_competitors = sorted(
            market_shares.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        key_competitor_names = [name for name, _ in key_competitors]
        
        # Generate competitive insights
        advantages = []
        threats = []
        opportunities = []
        
        if position in [CompetitivePosition.MARKET_LEADER, CompetitivePosition.STRONG_COMPETITOR]:
            advantages.append("Strong market position")
            advantages.append("Established customer base")
        
        if price_position == "competitive":
            advantages.append("Competitive pricing")
        elif price_position == "premium":
            advantages.append("Premium brand positioning")
            threats.append("Price sensitivity risk")
        
        if competitive_intensity > 0.7:
            threats.append("High competitive pressure")
            threats.append("Price war risk")
        
        if our_market_share < 0.10:
            opportunities.append("Market share growth potential")
            opportunities.append("Niche market development")
        
        return CompetitiveAnalysis(
            market_position=position,
            market_share=our_market_share,
            relative_market_share=relative_market_share,
            price_position=price_position,
            price_index=price_index,
            price_gap_to_leader=our_avg_price - min(competitor_prices.values()) if competitor_prices else 0,
            number_of_competitors=len(competitors),
            market_concentration=hhi,
            competitive_intensity=competitive_intensity,
            load_factor_vs_market=1.0,  # Simplified - would need load factor data
            revenue_per_passenger_vs_market=price_index,
            frequency_advantage=1.0,  # Simplified - would need frequency data
            key_competitors=key_competitor_names,
            competitive_advantages=advantages,
            competitive_threats=threats,
            market_opportunities=opportunities,
            route_id=route_id
        )
    
    def _analyze_price_intelligence(
        self,
        route_id: str,
        market: Market,
        own_fare_structure: Optional[FareStructure]
    ) -> PriceIntelligence:
        """Analyze pricing intelligence."""
        
        # Collect competitor prices
        competitor_prices = {}
        for airline, comp_data in market.competitors.items():
            if comp_data.average_fare > 0:
                competitor_prices[airline] = comp_data.average_fare
        
        if not competitor_prices:
            raise ValueError("No competitor pricing data available")
        
        # Calculate market statistics
        prices = list(competitor_prices.values())
        market_avg_price = statistics.mean(prices)
        price_range = (min(prices), max(prices))
        price_std = statistics.stdev(prices) if len(prices) > 1 else 0
        
        # Identify price leaders
        price_leader = min(competitor_prices.items(), key=lambda x: x[1])[0]
        premium_leader = max(competitor_prices.items(), key=lambda x: x[1])[0]
        
        # Analyze price trends (simplified - would use historical data)
        price_trend = "stable"  # Default
        price_volatility = price_std / market_avg_price if market_avg_price > 0 else 0
        
        # Estimate price elasticity (simplified)
        if price_volatility > 0.2:
            elasticity = -1.5  # High elasticity
            sensitivity = "high"
        elif price_volatility > 0.1:
            elasticity = -1.0  # Medium elasticity
            sensitivity = "medium"
        else:
            elasticity = -0.5  # Low elasticity
            sensitivity = "low"
        
        # Calculate optimal price range
        if sensitivity == "high":
            # Price-sensitive market - stay competitive
            optimal_min = market_avg_price * 0.95
            optimal_max = market_avg_price * 1.05
            strategy = "competitive pricing"
        elif sensitivity == "low":
            # Less price-sensitive - can charge premium
            optimal_min = market_avg_price * 1.0
            optimal_max = market_avg_price * 1.15
            strategy = "value-based pricing"
        else:
            # Medium sensitivity - balanced approach
            optimal_min = market_avg_price * 0.98
            optimal_max = market_avg_price * 1.08
            strategy = "market-based pricing"
        
        return PriceIntelligence(
            market_average_price=market_avg_price,
            market_price_range=price_range,
            price_standard_deviation=price_std,
            competitor_prices=competitor_prices,
            price_leader=price_leader,
            premium_leader=premium_leader,
            price_trend_direction=price_trend,
            price_volatility=price_volatility,
            seasonal_price_patterns={},  # Simplified
            estimated_price_elasticity=elasticity,
            demand_sensitivity=sensitivity,
            optimal_price_range=(optimal_min, optimal_max),
            pricing_strategy_recommendation=strategy,
            route_id=route_id
        )
    
    def _analyze_seasonal_trends(
        self,
        route_id: str,
        market: Market
    ) -> List[MarketTrend]:
        """Analyze seasonal market trends."""
        
        trends = []
        
        # Simplified seasonal analysis
        # In practice, this would analyze historical seasonal patterns
        
        # Summer peak trend
        summer_trend = MarketTrend(
            trend_name="Summer Peak Season",
            trend_direction="growing",
            trend_strength=0.7,
            trend_duration_months=3,
            primary_drivers=["Vacation travel", "School holidays"],
            supporting_factors=["Weather conditions", "Tourism campaigns"],
            revenue_impact=25.0,
            demand_impact=30.0,
            competitive_impact="positive",
            expected_duration_months=3,
            confidence_level=0.8
        )
        trends.append(summer_trend)
        
        # Winter low season
        winter_trend = MarketTrend(
            trend_name="Winter Low Season",
            trend_direction="declining",
            trend_strength=0.6,
            trend_duration_months=2,
            primary_drivers=["Weather conditions", "Post-holiday budget constraints"],
            supporting_factors=["Business travel reduction"],
            revenue_impact=-15.0,
            demand_impact=-20.0,
            competitive_impact="negative",
            expected_duration_months=2,
            confidence_level=0.7
        )
        trends.append(winter_trend)
        
        return trends
    
    def _analyze_market_share_trends(
        self,
        route_id: str,
        market: Market
    ) -> List[MarketTrend]:
        """Analyze market share trends."""
        
        trends = []
        
        # Market consolidation trend
        if len(market.competitors) > 5:
            consolidation_trend = MarketTrend(
                trend_name="Market Consolidation",
                trend_direction="stable",
                trend_strength=0.5,
                trend_duration_months=12,
                primary_drivers=["Industry consolidation", "Cost pressures"],
                supporting_factors=["Regulatory changes", "Economic conditions"],
                revenue_impact=5.0,
                demand_impact=0.0,
                competitive_impact="neutral",
                expected_duration_months=24,
                confidence_level=0.6
            )
            trends.append(consolidation_trend)
        
        return trends
    
    def _analyze_capacity_trends(
        self,
        route_id: str,
        market: Market
    ) -> List[MarketTrend]:
        """Analyze capacity trends."""
        
        trends = []
        
        # Calculate total market capacity
        total_capacity = sum(comp.capacity for comp in market.competitors.values())
        
        if total_capacity > 1000:  # High capacity route
            capacity_trend = MarketTrend(
                trend_name="High Capacity Market",
                trend_direction="stable",
                trend_strength=0.8,
                trend_duration_months=6,
                primary_drivers=["High demand", "Multiple competitors"],
                supporting_factors=["Airport slot availability"],
                revenue_impact=0.0,
                demand_impact=0.0,
                competitive_impact="neutral",
                expected_duration_months=12,
                confidence_level=0.7
            )
            trends.append(capacity_trend)
        
        return trends
    
    def _calculate_market_size(
        self,
        route_id: str,
        market: Market
    ) -> float:
        """Calculate annual market size in passengers."""
        
        # Simplified calculation based on competitor capacity
        total_capacity = sum(comp.capacity for comp in market.competitors.values())
        
        # Assume average load factor of 80% and daily frequency
        annual_passengers = total_capacity * 0.8 * 365
        
        return annual_passengers
    
    def _calculate_market_growth_rate(
        self,
        route_id: str,
        market: Market
    ) -> float:
        """Calculate market growth rate."""
        
        # Simplified - would use historical data in practice
        # Default to industry average growth
        return 3.5  # 3.5% annual growth
    
    def _assess_market_maturity(
        self,
        route_id: str,
        market: Market
    ) -> str:
        """Assess market maturity stage."""
        
        num_competitors = len(market.competitors)
        
        if num_competitors <= 2:
            return "emerging"
        elif num_competitors <= 4:
            return "growing"
        elif num_competitors <= 8:
            return "mature"
        else:
            return "declining"
    
    def _calculate_market_risk(
        self,
        route_id: str,
        market: Market
    ) -> float:
        """Calculate overall market risk score."""
        
        risk_factors = []
        
        # Competition risk
        num_competitors = len(market.competitors)
        if num_competitors > 5:
            risk_factors.append(0.7)  # High competition risk
        elif num_competitors > 3:
            risk_factors.append(0.5)  # Medium risk
        else:
            risk_factors.append(0.3)  # Low risk
        
        # Market concentration risk
        if market.competition_level == CompetitionLevel.HIGH:
            risk_factors.append(0.6)
        elif market.competition_level == CompetitionLevel.MEDIUM:
            risk_factors.append(0.4)
        else:
            risk_factors.append(0.2)
        
        # Calculate average risk
        return statistics.mean(risk_factors) if risk_factors else 0.5
    
    def _assess_regulatory_risk(
        self,
        route_id: str,
        market: Market
    ) -> float:
        """Assess regulatory risk."""
        
        # Simplified assessment
        if market.market_type == MarketType.INTERNATIONAL:
            return 0.4  # Higher regulatory risk for international routes
        else:
            return 0.2  # Lower risk for domestic routes
    
    def _assess_competitive_risk(
        self,
        route_id: str,
        market: Market
    ) -> float:
        """Assess competitive risk."""
        
        if market.competition_level == CompetitionLevel.HIGH:
            return 0.7
        elif market.competition_level == CompetitionLevel.MEDIUM:
            return 0.4
        else:
            return 0.2
    
    def _assess_economic_risk(
        self,
        route_id: str,
        market: Market
    ) -> float:
        """Assess economic risk."""
        
        # Simplified - would consider economic indicators in practice
        return 0.3  # Default moderate economic risk
    
    def _generate_strategic_recommendations(
        self,
        result: MarketAnalysisResult
    ) -> List[str]:
        """Generate strategic recommendations."""
        
        recommendations = []
        
        if result.competitive_analysis:
            comp_analysis = result.competitive_analysis
            
            if comp_analysis.market_position == CompetitivePosition.MARKET_LEADER:
                recommendations.append("Maintain market leadership through service excellence")
                recommendations.append("Consider premium pricing strategy")
            
            elif comp_analysis.market_position in [CompetitivePosition.WEAK_COMPETITOR, CompetitivePosition.NICHE_PLAYER]:
                recommendations.append("Focus on niche market segments")
                recommendations.append("Consider strategic partnerships")
            
            if comp_analysis.competitive_intensity > 0.7:
                recommendations.append("Implement dynamic pricing to respond to competition")
                recommendations.append("Differentiate through service quality")
        
        if result.price_intelligence:
            price_intel = result.price_intelligence
            
            if price_intel.demand_sensitivity == "high":
                recommendations.append("Maintain competitive pricing")
                recommendations.append("Focus on cost optimization")
            
            elif price_intel.demand_sensitivity == "low":
                recommendations.append("Consider value-based pricing")
                recommendations.append("Invest in premium services")
        
        if result.market_risk_score > 0.6:
            recommendations.append("Implement risk mitigation strategies")
            recommendations.append("Diversify route portfolio")
        
        return recommendations[:5]  # Top 5 recommendations
    
    def _generate_tactical_actions(
        self,
        result: MarketAnalysisResult
    ) -> List[str]:
        """Generate tactical actions."""
        
        actions = []
        
        if result.competitive_analysis:
            comp_analysis = result.competitive_analysis
            
            if comp_analysis.price_position == "premium":
                actions.append("Monitor competitor price changes daily")
                actions.append("Enhance premium service offerings")
            
            elif comp_analysis.price_position == "discount":
                actions.append("Optimize operational costs")
                actions.append("Implement aggressive marketing campaigns")
        
        if result.price_intelligence:
            price_intel = result.price_intelligence
            
            actions.append(f"Adjust prices to {price_intel.optimal_price_range[0]:.0f}-{price_intel.optimal_price_range[1]:.0f} range")
            actions.append("Implement real-time price monitoring")
        
        # Add seasonal actions based on trends
        for trend in result.market_trends:
            if "Summer" in trend.trend_name:
                actions.append("Prepare for summer capacity increase")
            elif "Winter" in trend.trend_name:
                actions.append("Plan winter schedule optimization")
        
        return actions[:5]  # Top 5 actions
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached analysis is still valid."""
        
        if cache_key not in self.analysis_cache:
            return False
        
        cached_result = self.analysis_cache[cache_key]
        age_hours = (datetime.now() - cached_result.analysis_timestamp).total_seconds() / 3600
        
        return age_hours < self.cache_expiry_hours
    
    def update_market_data(
        self,
        route_id: str,
        market_metrics: MarketMetrics
    ) -> None:
        """Update market data for analysis."""
        
        self.market_data[route_id].append(market_metrics)
        
        # Keep only recent data (last 365 days)
        if len(self.market_data[route_id]) > 365:
            self.market_data[route_id] = self.market_data[route_id][-365:]
    
    def update_competitor_data(
        self,
        route_id: str,
        airline: str,
        competitor_data: CompetitorData
    ) -> None:
        """Update competitor data."""
        
        self.competitor_data[route_id][airline].append(competitor_data)
        
        # Keep only recent data
        if len(self.competitor_data[route_id][airline]) > 100:
            self.competitor_data[route_id][airline] = self.competitor_data[route_id][airline][-100:]
    
    def update_price_data(
        self,
        route_id: str,
        price_date: date,
        competitor_prices: Dict[str, float]
    ) -> None:
        """Update competitor price data."""
        
        self.price_history[route_id].append((price_date, competitor_prices.copy()))
        
        # Keep only recent data (last 365 days)
        cutoff_date = date.today() - timedelta(days=365)
        self.price_history[route_id] = [
            (d, prices) for d, prices in self.price_history[route_id] if d >= cutoff_date
        ]
    
    def get_market_summary(
        self,
        route_id: str
    ) -> Dict:
        """Get market summary for a route."""
        
        if route_id not in self.market_data:
            return {'error': f'No market data for route {route_id}'}
        
        recent_metrics = self.market_data[route_id][-30:]  # Last 30 data points
        
        if not recent_metrics:
            return {'error': f'No recent market data for route {route_id}'}
        
        # Calculate summary statistics
        avg_demand = statistics.mean(m.total_demand for m in recent_metrics)
        avg_capacity = statistics.mean(m.total_capacity for m in recent_metrics)
        avg_load_factor = avg_demand / avg_capacity if avg_capacity > 0 else 0
        
        return {
            'route_id': route_id,
            'data_points': len(recent_metrics),
            'average_demand': avg_demand,
            'average_capacity': avg_capacity,
            'average_load_factor': avg_load_factor,
            'competitors_tracked': len(self.competitor_data.get(route_id, {})),
            'price_data_points': len(self.price_history.get(route_id, [])),
            'last_updated': recent_metrics[-1].date.isoformat() if recent_metrics else None
        }
    
    def export_analyzer_data(self) -> Dict:
        """Export market analyzer configuration and data."""
        
        return {
            'configuration': {
                'cache_expiry_hours': self.cache_expiry_hours,
                'min_data_points': self.min_data_points,
                'competitor_threshold': self.competitor_threshold,
                'price_change_threshold': self.price_change_threshold
            },
            'data_summary': {
                'routes_with_market_data': len(self.market_data),
                'routes_with_competitor_data': len(self.competitor_data),
                'routes_with_price_data': len(self.price_history),
                'cached_analyses': len(self.analysis_cache),
                'data_sources': list(self.data_sources)
            },
            'total_records': {
                'market_metrics': sum(len(data) for data in self.market_data.values()),
                'competitor_records': sum(
                    sum(len(airline_data) for airline_data in route_data.values())
                    for route_data in self.competitor_data.values()
                ),
                'price_records': sum(len(data) for data in self.price_history.values())
            }
        }