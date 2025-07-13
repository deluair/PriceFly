"""Intelligent insights and recommendations for airline pricing simulation."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class InsightType(Enum):
    """Types of insights available."""
    PRICING = "pricing"
    MARKET = "market"
    REVENUE = "revenue"
    COMPETITIVE = "competitive"
    OPERATIONAL = "operational"
    DEMAND = "demand"
    RISK = "risk"
    OPPORTUNITY = "opportunity"


class InsightSeverity(Enum):
    """Severity levels for insights."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionType(Enum):
    """Types of recommended actions."""
    PRICE_INCREASE = "price_increase"
    PRICE_DECREASE = "price_decrease"
    CAPACITY_ADJUSTMENT = "capacity_adjustment"
    ROUTE_OPTIMIZATION = "route_optimization"
    COMPETITIVE_RESPONSE = "competitive_response"
    DEMAND_STIMULATION = "demand_stimulation"
    COST_REDUCTION = "cost_reduction"
    MARKET_EXPANSION = "market_expansion"
    RISK_MITIGATION = "risk_mitigation"
    REVENUE_OPTIMIZATION = "revenue_optimization"


@dataclass
class InsightMetric:
    """Metric associated with an insight."""
    name: str
    current_value: float
    target_value: Optional[float] = None
    benchmark_value: Optional[float] = None
    trend: Optional[str] = None  # 'increasing', 'decreasing', 'stable'
    confidence: float = 0.0
    unit: str = ""


@dataclass
class RecommendedAction:
    """Recommended action based on insights."""
    action_type: ActionType
    description: str
    expected_impact: str
    priority: InsightSeverity
    estimated_revenue_impact: Optional[float] = None
    implementation_effort: str = "medium"  # low, medium, high
    timeline: str = "short-term"  # short-term, medium-term, long-term
    success_probability: float = 0.5
    dependencies: List[str] = field(default_factory=list)


@dataclass
class BaseInsight:
    """Base class for all insights."""
    insight_type: InsightType
    title: str
    description: str
    severity: InsightSeverity
    confidence: float
    metrics: List[InsightMetric] = field(default_factory=list)
    recommendations: List[RecommendedAction] = field(default_factory=list)
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)


@dataclass
class PricingInsight(BaseInsight):
    """Insights related to pricing strategies."""
    route_id: Optional[str] = None
    booking_class: Optional[str] = None
    price_elasticity: Optional[float] = None
    optimal_price_range: Optional[Tuple[float, float]] = None
    competitor_price_gap: Optional[float] = None


@dataclass
class MarketInsight(BaseInsight):
    """Insights related to market dynamics."""
    market_segment: Optional[str] = None
    market_share_change: Optional[float] = None
    competitive_intensity: Optional[float] = None
    market_growth_rate: Optional[float] = None
    customer_segment: Optional[str] = None


@dataclass
class RevenueInsight(BaseInsight):
    """Insights related to revenue optimization."""
    revenue_source: Optional[str] = None
    revenue_trend: Optional[str] = None
    profit_margin_change: Optional[float] = None
    load_factor_impact: Optional[float] = None
    ancillary_revenue_opportunity: Optional[float] = None


@dataclass
class CompetitiveInsight(BaseInsight):
    """Insights related to competitive positioning."""
    competitor_name: Optional[str] = None
    competitive_advantage: Optional[str] = None
    threat_level: Optional[str] = None
    market_position_change: Optional[str] = None
    response_urgency: Optional[str] = None


class InsightEngine:
    """Engine for generating intelligent insights from simulation data."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.insights_history: List[BaseInsight] = []
        self.thresholds = self._initialize_thresholds()
        self.benchmarks = self._initialize_benchmarks()
    
    def _initialize_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize threshold values for different metrics."""
        return {
            'pricing': {
                'price_variance_high': 0.15,  # 15% price variance
                'elasticity_sensitive': -1.5,
                'competitor_gap_significant': 0.10,  # 10% price gap
                'revenue_impact_threshold': 0.05  # 5% revenue impact
            },
            'market': {
                'market_share_decline': -0.02,  # 2% market share decline
                'competitive_intensity_high': 0.8,
                'growth_rate_low': 0.02,  # 2% growth rate
                'concentration_high': 0.7  # HHI > 0.7
            },
            'revenue': {
                'revenue_decline': -0.05,  # 5% revenue decline
                'margin_decline': -0.03,  # 3% margin decline
                'load_factor_low': 0.75,  # 75% load factor
                'yield_decline': -0.04  # 4% yield decline
            },
            'operational': {
                'cost_increase': 0.08,  # 8% cost increase
                'efficiency_decline': -0.05,  # 5% efficiency decline
                'capacity_utilization_low': 0.70  # 70% utilization
            }
        }
    
    def _initialize_benchmarks(self) -> Dict[str, float]:
        """Initialize industry benchmark values."""
        return {
            'average_load_factor': 0.82,
            'average_profit_margin': 0.08,
            'average_yield': 0.12,
            'average_rasm': 0.14,
            'competitive_price_premium': 0.05,
            'market_share_stability': 0.02,
            'demand_elasticity': -1.2,
            'booking_curve_advance': 45  # days
        }
    
    def analyze_simulation_data(
        self,
        simulation_data: Dict[str, pd.DataFrame]
    ) -> List[BaseInsight]:
        """Analyze simulation data and generate insights."""
        
        insights = []
        
        try:
            # Pricing insights
            if 'pricing' in simulation_data:
                pricing_insights = self._analyze_pricing_data(simulation_data['pricing'])
                insights.extend(pricing_insights)
            
            # Market insights
            if 'market_data' in simulation_data:
                market_insights = self._analyze_market_data(simulation_data['market_data'])
                insights.extend(market_insights)
            
            # Revenue insights
            if 'revenue' in simulation_data:
                revenue_insights = self._analyze_revenue_data(simulation_data['revenue'])
                insights.extend(revenue_insights)
            
            # Competitive insights
            if 'competitive_data' in simulation_data:
                competitive_insights = self._analyze_competitive_data(simulation_data['competitive_data'])
                insights.extend(competitive_insights)
            
            # Cross-functional insights
            cross_insights = self._analyze_cross_functional_patterns(simulation_data)
            insights.extend(cross_insights)
            
            # Store insights
            self.insights_history.extend(insights)
            
            self.logger.info(f"Generated {len(insights)} insights from simulation data")
            
        except Exception as e:
            self.logger.error(f"Error analyzing simulation data: {e}")
        
        return insights
    
    def _analyze_pricing_data(self, pricing_data: pd.DataFrame) -> List[PricingInsight]:
        """Analyze pricing data for insights."""
        
        insights = []
        
        try:
            # Price volatility analysis
            if 'price' in pricing_data.columns:
                price_volatility = pricing_data['price'].std() / pricing_data['price'].mean()
                
                if price_volatility > self.thresholds['pricing']['price_variance_high']:
                    insights.append(PricingInsight(
                        insight_type=InsightType.PRICING,
                        title="High Price Volatility Detected",
                        description=f"Price volatility of {price_volatility:.1%} exceeds recommended threshold. "
                                   f"This may indicate inconsistent pricing strategy or market instability.",
                        severity=InsightSeverity.MEDIUM,
                        confidence=0.85,
                        metrics=[
                            InsightMetric(
                                name="Price Volatility",
                                current_value=price_volatility,
                                target_value=self.thresholds['pricing']['price_variance_high'],
                                trend="increasing",
                                confidence=0.85,
                                unit="%"
                            )
                        ],
                        recommendations=[
                            RecommendedAction(
                                action_type=ActionType.REVENUE_OPTIMIZATION,
                                description="Implement more consistent pricing algorithms",
                                expected_impact="Reduce price volatility by 20-30%",
                                priority=InsightSeverity.MEDIUM,
                                estimated_revenue_impact=0.03,
                                implementation_effort="medium",
                                success_probability=0.75
                            )
                        ],
                        tags=["pricing", "volatility", "strategy"]
                    ))
            
            # Price elasticity analysis
            if 'price' in pricing_data.columns and 'demand' in pricing_data.columns:
                elasticity = self._calculate_price_elasticity(
                    pricing_data['price'], 
                    pricing_data['demand']
                )
                
                if elasticity < self.thresholds['pricing']['elasticity_sensitive']:
                    insights.append(PricingInsight(
                        insight_type=InsightType.PRICING,
                        title="High Price Sensitivity Detected",
                        description=f"Price elasticity of {elasticity:.2f} indicates high customer sensitivity. "
                                   f"Price increases may significantly reduce demand.",
                        severity=InsightSeverity.HIGH,
                        confidence=0.80,
                        price_elasticity=elasticity,
                        metrics=[
                            InsightMetric(
                                name="Price Elasticity",
                                current_value=elasticity,
                                benchmark_value=self.benchmarks['demand_elasticity'],
                                confidence=0.80,
                                unit="coefficient"
                            )
                        ],
                        recommendations=[
                            RecommendedAction(
                                action_type=ActionType.DEMAND_STIMULATION,
                                description="Focus on value-added services rather than price increases",
                                expected_impact="Maintain demand while improving margins",
                                priority=InsightSeverity.HIGH,
                                implementation_effort="high",
                                success_probability=0.65
                            )
                        ],
                        tags=["pricing", "elasticity", "demand"]
                    ))
            
            # Optimal pricing opportunities
            optimal_insights = self._identify_optimal_pricing_opportunities(pricing_data)
            insights.extend(optimal_insights)
            
        except Exception as e:
            self.logger.error(f"Error analyzing pricing data: {e}")
        
        return insights
    
    def _analyze_market_data(self, market_data: pd.DataFrame) -> List[MarketInsight]:
        """Analyze market data for insights."""
        
        insights = []
        
        try:
            # Market share analysis
            if 'market_share' in market_data.columns:
                market_share_trend = self._calculate_trend(market_data['market_share'])
                
                if market_share_trend < self.thresholds['market']['market_share_decline']:
                    insights.append(MarketInsight(
                        insight_type=InsightType.MARKET,
                        title="Market Share Decline Detected",
                        description=f"Market share declining at {market_share_trend:.1%} rate. "
                                   f"Immediate action required to prevent further erosion.",
                        severity=InsightSeverity.HIGH,
                        confidence=0.90,
                        market_share_change=market_share_trend,
                        metrics=[
                            InsightMetric(
                                name="Market Share Trend",
                                current_value=market_share_trend,
                                target_value=0.0,
                                trend="decreasing",
                                confidence=0.90,
                                unit="%"
                            )
                        ],
                        recommendations=[
                            RecommendedAction(
                                action_type=ActionType.COMPETITIVE_RESPONSE,
                                description="Implement aggressive competitive pricing strategy",
                                expected_impact="Stabilize market share within 3 months",
                                priority=InsightSeverity.HIGH,
                                estimated_revenue_impact=-0.02,
                                implementation_effort="medium",
                                timeline="short-term",
                                success_probability=0.70
                            ),
                            RecommendedAction(
                                action_type=ActionType.MARKET_EXPANSION,
                                description="Expand into underserved market segments",
                                expected_impact="Increase total addressable market",
                                priority=InsightSeverity.MEDIUM,
                                implementation_effort="high",
                                timeline="medium-term",
                                success_probability=0.60
                            )
                        ],
                        tags=["market", "share", "competitive"]
                    ))
            
            # Competitive intensity analysis
            competitive_intensity = self._calculate_competitive_intensity(market_data)
            
            if competitive_intensity > self.thresholds['market']['competitive_intensity_high']:
                insights.append(MarketInsight(
                    insight_type=InsightType.MARKET,
                    title="High Competitive Intensity",
                    description=f"Market showing high competitive intensity ({competitive_intensity:.1%}). "
                               f"Price wars and margin pressure likely.",
                    severity=InsightSeverity.MEDIUM,
                    confidence=0.75,
                    competitive_intensity=competitive_intensity,
                    recommendations=[
                        RecommendedAction(
                            action_type=ActionType.REVENUE_OPTIMIZATION,
                            description="Focus on non-price differentiation strategies",
                            expected_impact="Maintain margins despite competition",
                            priority=InsightSeverity.MEDIUM,
                            implementation_effort="high",
                            success_probability=0.55
                        )
                    ],
                    tags=["market", "competition", "intensity"]
                ))
            
            # Market growth opportunities
            growth_insights = self._identify_growth_opportunities(market_data)
            insights.extend(growth_insights)
            
        except Exception as e:
            self.logger.error(f"Error analyzing market data: {e}")
        
        return insights
    
    def _analyze_revenue_data(self, revenue_data: pd.DataFrame) -> List[RevenueInsight]:
        """Analyze revenue data for insights."""
        
        insights = []
        
        try:
            # Revenue trend analysis
            if 'total_revenue' in revenue_data.columns:
                revenue_trend = self._calculate_trend(revenue_data['total_revenue'])
                
                if revenue_trend < self.thresholds['revenue']['revenue_decline']:
                    insights.append(RevenueInsight(
                        insight_type=InsightType.REVENUE,
                        title="Revenue Decline Alert",
                        description=f"Revenue declining at {revenue_trend:.1%} rate. "
                                   f"Urgent intervention required to reverse trend.",
                        severity=InsightSeverity.CRITICAL,
                        confidence=0.95,
                        revenue_trend="decreasing",
                        metrics=[
                            InsightMetric(
                                name="Revenue Trend",
                                current_value=revenue_trend,
                                target_value=0.02,  # 2% growth target
                                trend="decreasing",
                                confidence=0.95,
                                unit="%"
                            )
                        ],
                        recommendations=[
                            RecommendedAction(
                                action_type=ActionType.REVENUE_OPTIMIZATION,
                                description="Implement dynamic pricing optimization",
                                expected_impact="Increase revenue by 5-8%",
                                priority=InsightSeverity.CRITICAL,
                                estimated_revenue_impact=0.06,
                                implementation_effort="medium",
                                timeline="short-term",
                                success_probability=0.80
                            )
                        ],
                        tags=["revenue", "decline", "critical"]
                    ))
            
            # Load factor optimization
            if 'load_factor' in revenue_data.columns:
                avg_load_factor = revenue_data['load_factor'].mean()
                
                if avg_load_factor < self.thresholds['revenue']['load_factor_low']:
                    insights.append(RevenueInsight(
                        insight_type=InsightType.REVENUE,
                        title="Low Load Factor Opportunity",
                        description=f"Average load factor of {avg_load_factor:.1%} below industry benchmark. "
                                   f"Significant revenue optimization opportunity exists.",
                        severity=InsightSeverity.MEDIUM,
                        confidence=0.85,
                        load_factor_impact=self.benchmarks['average_load_factor'] - avg_load_factor,
                        metrics=[
                            InsightMetric(
                                name="Load Factor",
                                current_value=avg_load_factor,
                                benchmark_value=self.benchmarks['average_load_factor'],
                                trend="stable",
                                confidence=0.85,
                                unit="%"
                            )
                        ],
                        recommendations=[
                            RecommendedAction(
                                action_type=ActionType.DEMAND_STIMULATION,
                                description="Implement targeted promotional pricing",
                                expected_impact="Increase load factor by 5-10 percentage points",
                                priority=InsightSeverity.MEDIUM,
                                estimated_revenue_impact=0.08,
                                implementation_effort="low",
                                success_probability=0.75
                            ),
                            RecommendedAction(
                                action_type=ActionType.CAPACITY_ADJUSTMENT,
                                description="Optimize capacity allocation across routes",
                                expected_impact="Improve overall fleet utilization",
                                priority=InsightSeverity.MEDIUM,
                                implementation_effort="medium",
                                success_probability=0.65
                            )
                        ],
                        tags=["revenue", "load_factor", "optimization"]
                    ))
            
            # Ancillary revenue opportunities
            ancillary_insights = self._identify_ancillary_opportunities(revenue_data)
            insights.extend(ancillary_insights)
            
        except Exception as e:
            self.logger.error(f"Error analyzing revenue data: {e}")
        
        return insights
    
    def _analyze_competitive_data(self, competitive_data: pd.DataFrame) -> List[CompetitiveInsight]:
        """Analyze competitive data for insights."""
        
        insights = []
        
        try:
            # Competitive positioning analysis
            if 'competitor_price' in competitive_data.columns and 'our_price' in competitive_data.columns:
                price_gaps = self._analyze_competitive_price_gaps(competitive_data)
                
                for competitor, gap in price_gaps.items():
                    if abs(gap) > self.thresholds['pricing']['competitor_gap_significant']:
                        severity = InsightSeverity.HIGH if abs(gap) > 0.15 else InsightSeverity.MEDIUM
                        
                        insights.append(CompetitiveInsight(
                            insight_type=InsightType.COMPETITIVE,
                            title=f"Significant Price Gap with {competitor}",
                            description=f"Price gap of {gap:.1%} with {competitor} may impact competitiveness.",
                            severity=severity,
                            confidence=0.80,
                            competitor_name=competitor,
                            recommendations=[
                                RecommendedAction(
                                    action_type=ActionType.COMPETITIVE_RESPONSE,
                                    description=f"Adjust pricing strategy relative to {competitor}",
                                    expected_impact="Improve competitive position",
                                    priority=severity,
                                    implementation_effort="low",
                                    success_probability=0.70
                                )
                            ],
                            tags=["competitive", "pricing", competitor.lower()]
                        ))
            
            # Market position threats
            threat_insights = self._identify_competitive_threats(competitive_data)
            insights.extend(threat_insights)
            
        except Exception as e:
            self.logger.error(f"Error analyzing competitive data: {e}")
        
        return insights
    
    def _analyze_cross_functional_patterns(self, simulation_data: Dict[str, pd.DataFrame]) -> List[BaseInsight]:
        """Analyze patterns across multiple data sources."""
        
        insights = []
        
        try:
            # Price-demand correlation analysis
            if 'pricing' in simulation_data and 'demand' in simulation_data:
                correlation_insights = self._analyze_price_demand_correlation(
                    simulation_data['pricing'], 
                    simulation_data['demand']
                )
                insights.extend(correlation_insights)
            
            # Revenue-market share relationship
            if 'revenue' in simulation_data and 'market_data' in simulation_data:
                relationship_insights = self._analyze_revenue_market_relationship(
                    simulation_data['revenue'], 
                    simulation_data['market_data']
                )
                insights.extend(relationship_insights)
            
            # Seasonal pattern analysis
            seasonal_insights = self._analyze_seasonal_patterns(simulation_data)
            insights.extend(seasonal_insights)
            
        except Exception as e:
            self.logger.error(f"Error analyzing cross-functional patterns: {e}")
        
        return insights
    
    def _calculate_price_elasticity(self, prices: pd.Series, demand: pd.Series) -> float:
        """Calculate price elasticity of demand."""
        try:
            # Calculate percentage changes
            price_pct_change = prices.pct_change().dropna()
            demand_pct_change = demand.pct_change().dropna()
            
            # Align series
            min_length = min(len(price_pct_change), len(demand_pct_change))
            price_pct_change = price_pct_change.iloc[:min_length]
            demand_pct_change = demand_pct_change.iloc[:min_length]
            
            # Calculate elasticity (slope of demand change vs price change)
            if len(price_pct_change) > 1 and price_pct_change.std() > 0:
                elasticity = np.corrcoef(price_pct_change, demand_pct_change)[0, 1]
                elasticity *= (demand_pct_change.std() / price_pct_change.std())
                return elasticity
            
            return self.benchmarks['demand_elasticity']
            
        except Exception:
            return self.benchmarks['demand_elasticity']
    
    def _calculate_trend(self, series: pd.Series) -> float:
        """Calculate trend slope for a time series."""
        try:
            if len(series) < 2:
                return 0.0
            
            x = np.arange(len(series))
            slope, _, _, _, _ = stats.linregress(x, series)
            
            # Convert to percentage change
            return slope / series.mean() if series.mean() != 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_competitive_intensity(self, market_data: pd.DataFrame) -> float:
        """Calculate competitive intensity using HHI and other metrics."""
        try:
            if 'market_share' in market_data.columns:
                # Calculate Herfindahl-Hirschman Index
                market_shares = market_data.groupby('airline')['market_share'].mean()
                hhi = (market_shares ** 2).sum()
                
                # Convert to competitive intensity (inverse of concentration)
                return 1 - hhi
            
            return 0.5  # Default moderate intensity
            
        except Exception:
            return 0.5
    
    def _identify_optimal_pricing_opportunities(self, pricing_data: pd.DataFrame) -> List[PricingInsight]:
        """Identify optimal pricing opportunities."""
        
        insights = []
        
        try:
            if 'route_id' in pricing_data.columns and 'price' in pricing_data.columns:
                # Analyze pricing by route
                route_analysis = pricing_data.groupby('route_id').agg({
                    'price': ['mean', 'std'],
                    'demand': 'mean' if 'demand' in pricing_data.columns else lambda x: 100,
                    'revenue': 'sum' if 'revenue' in pricing_data.columns else lambda x: x.iloc[0] * 100
                }).round(2)
                
                # Flatten column names
                route_analysis.columns = ['_'.join(col).strip() for col in route_analysis.columns]
                
                # Identify routes with high price variance (optimization opportunity)
                high_variance_routes = route_analysis[
                    route_analysis['price_std'] / route_analysis['price_mean'] > 0.15
                ]
                
                for route_id in high_variance_routes.index:
                    insights.append(PricingInsight(
                        insight_type=InsightType.PRICING,
                        title=f"Pricing Optimization Opportunity - Route {route_id}",
                        description=f"Route {route_id} shows high price variance, indicating optimization potential.",
                        severity=InsightSeverity.MEDIUM,
                        confidence=0.70,
                        route_id=str(route_id),
                        recommendations=[
                            RecommendedAction(
                                action_type=ActionType.REVENUE_OPTIMIZATION,
                                description=f"Implement consistent pricing strategy for route {route_id}",
                                expected_impact="Reduce price variance and improve revenue predictability",
                                priority=InsightSeverity.MEDIUM,
                                implementation_effort="low",
                                success_probability=0.75
                            )
                        ],
                        tags=["pricing", "optimization", f"route_{route_id}"]
                    ))
        
        except Exception as e:
            self.logger.error(f"Error identifying pricing opportunities: {e}")
        
        return insights
    
    def _identify_growth_opportunities(self, market_data: pd.DataFrame) -> List[MarketInsight]:
        """Identify market growth opportunities."""
        
        insights = []
        
        try:
            # Analyze market segments for growth potential
            if 'segment' in market_data.columns and 'growth_rate' in market_data.columns:
                segment_growth = market_data.groupby('segment')['growth_rate'].mean()
                
                high_growth_segments = segment_growth[segment_growth > 0.05]  # 5% growth
                
                for segment, growth_rate in high_growth_segments.items():
                    insights.append(MarketInsight(
                        insight_type=InsightType.OPPORTUNITY,
                        title=f"High Growth Opportunity in {segment} Segment",
                        description=f"{segment} segment showing {growth_rate:.1%} growth rate. "
                                   f"Consider increasing focus and capacity allocation.",
                        severity=InsightSeverity.MEDIUM,
                        confidence=0.75,
                        market_segment=segment,
                        market_growth_rate=growth_rate,
                        recommendations=[
                            RecommendedAction(
                                action_type=ActionType.MARKET_EXPANSION,
                                description=f"Increase capacity and marketing in {segment} segment",
                                expected_impact=f"Capture {growth_rate:.1%} additional market growth",
                                priority=InsightSeverity.MEDIUM,
                                estimated_revenue_impact=growth_rate * 0.5,
                                implementation_effort="medium",
                                timeline="medium-term",
                                success_probability=0.65
                            )
                        ],
                        tags=["opportunity", "growth", segment.lower()]
                    ))
        
        except Exception as e:
            self.logger.error(f"Error identifying growth opportunities: {e}")
        
        return insights
    
    def _identify_ancillary_opportunities(self, revenue_data: pd.DataFrame) -> List[RevenueInsight]:
        """Identify ancillary revenue opportunities."""
        
        insights = []
        
        try:
            # Analyze ancillary revenue potential
            if 'ancillary_revenue' in revenue_data.columns and 'total_revenue' in revenue_data.columns:
                ancillary_ratio = (revenue_data['ancillary_revenue'] / 
                                 revenue_data['total_revenue']).mean()
                
                industry_benchmark = 0.15  # 15% industry average
                
                if ancillary_ratio < industry_benchmark:
                    opportunity = (industry_benchmark - ancillary_ratio) * revenue_data['total_revenue'].sum()
                    
                    insights.append(RevenueInsight(
                        insight_type=InsightType.OPPORTUNITY,
                        title="Ancillary Revenue Opportunity",
                        description=f"Ancillary revenue at {ancillary_ratio:.1%} below industry benchmark. "
                                   f"Potential opportunity worth ${opportunity:,.0f}.",
                        severity=InsightSeverity.MEDIUM,
                        confidence=0.70,
                        ancillary_revenue_opportunity=opportunity,
                        recommendations=[
                            RecommendedAction(
                                action_type=ActionType.REVENUE_OPTIMIZATION,
                                description="Expand ancillary service offerings",
                                expected_impact=f"Increase ancillary revenue to {industry_benchmark:.1%}",
                                priority=InsightSeverity.MEDIUM,
                                estimated_revenue_impact=opportunity / revenue_data['total_revenue'].sum(),
                                implementation_effort="medium",
                                timeline="medium-term",
                                success_probability=0.60
                            )
                        ],
                        tags=["revenue", "ancillary", "opportunity"]
                    ))
        
        except Exception as e:
            self.logger.error(f"Error identifying ancillary opportunities: {e}")
        
        return insights
    
    def _analyze_competitive_price_gaps(self, competitive_data: pd.DataFrame) -> Dict[str, float]:
        """Analyze price gaps with competitors."""
        
        price_gaps = {}
        
        try:
            if 'competitor' in competitive_data.columns:
                for competitor in competitive_data['competitor'].unique():
                    competitor_data = competitive_data[competitive_data['competitor'] == competitor]
                    
                    if 'our_price' in competitor_data.columns and 'competitor_price' in competitor_data.columns:
                        avg_our_price = competitor_data['our_price'].mean()
                        avg_competitor_price = competitor_data['competitor_price'].mean()
                        
                        if avg_our_price > 0:
                            gap = (avg_competitor_price - avg_our_price) / avg_our_price
                            price_gaps[competitor] = gap
        
        except Exception as e:
            self.logger.error(f"Error analyzing competitive price gaps: {e}")
        
        return price_gaps
    
    def _identify_competitive_threats(self, competitive_data: pd.DataFrame) -> List[CompetitiveInsight]:
        """Identify competitive threats."""
        
        insights = []
        
        try:
            # Analyze competitor market share gains
            if 'competitor' in competitive_data.columns and 'market_share_change' in competitive_data.columns:
                threat_competitors = competitive_data[
                    competitive_data['market_share_change'] > 0.02  # 2% gain
                ]
                
                for _, row in threat_competitors.iterrows():
                    insights.append(CompetitiveInsight(
                        insight_type=InsightType.COMPETITIVE,
                        title=f"Competitive Threat from {row['competitor']}",
                        description=f"{row['competitor']} gaining market share at {row['market_share_change']:.1%} rate.",
                        severity=InsightSeverity.HIGH,
                        confidence=0.80,
                        competitor_name=row['competitor'],
                        threat_level="high",
                        response_urgency="immediate",
                        recommendations=[
                            RecommendedAction(
                                action_type=ActionType.COMPETITIVE_RESPONSE,
                                description=f"Develop counter-strategy against {row['competitor']}",
                                expected_impact="Prevent further market share loss",
                                priority=InsightSeverity.HIGH,
                                implementation_effort="medium",
                                timeline="short-term",
                                success_probability=0.65
                            )
                        ],
                        tags=["competitive", "threat", row['competitor'].lower()]
                    ))
        
        except Exception as e:
            self.logger.error(f"Error identifying competitive threats: {e}")
        
        return insights
    
    def _analyze_price_demand_correlation(self, pricing_data: pd.DataFrame, demand_data: pd.DataFrame) -> List[BaseInsight]:
        """Analyze correlation between price and demand."""
        
        insights = []
        
        try:
            # Merge data on common columns
            if 'date' in pricing_data.columns and 'date' in demand_data.columns:
                merged_data = pd.merge(pricing_data, demand_data, on='date', suffixes=('_price', '_demand'))
                
                if len(merged_data) > 10:  # Need sufficient data points
                    correlation = merged_data['price'].corr(merged_data['demand'])
                    
                    if abs(correlation) > 0.7:  # Strong correlation
                        insights.append(BaseInsight(
                            insight_type=InsightType.PRICING,
                            title="Strong Price-Demand Correlation Detected",
                            description=f"Strong correlation ({correlation:.2f}) between price and demand. "
                                       f"Price changes will significantly impact demand.",
                            severity=InsightSeverity.MEDIUM,
                            confidence=0.85,
                            metrics=[
                                InsightMetric(
                                    name="Price-Demand Correlation",
                                    current_value=correlation,
                                    confidence=0.85,
                                    unit="coefficient"
                                )
                            ],
                            recommendations=[
                                RecommendedAction(
                                    action_type=ActionType.REVENUE_OPTIMIZATION,
                                    description="Use correlation for predictive pricing models",
                                    expected_impact="Improve pricing accuracy by 15-20%",
                                    priority=InsightSeverity.MEDIUM,
                                    implementation_effort="medium",
                                    success_probability=0.75
                                )
                            ],
                            tags=["pricing", "demand", "correlation"]
                        ))
        
        except Exception as e:
            self.logger.error(f"Error analyzing price-demand correlation: {e}")
        
        return insights
    
    def _analyze_revenue_market_relationship(self, revenue_data: pd.DataFrame, market_data: pd.DataFrame) -> List[BaseInsight]:
        """Analyze relationship between revenue and market position."""
        
        insights = []
        
        try:
            # Analyze revenue efficiency vs market share
            if 'market_share' in market_data.columns and 'total_revenue' in revenue_data.columns:
                # Calculate revenue per market share point
                avg_market_share = market_data['market_share'].mean()
                total_revenue = revenue_data['total_revenue'].sum()
                
                if avg_market_share > 0:
                    revenue_efficiency = total_revenue / avg_market_share
                    
                    # Compare with industry benchmark (if available)
                    benchmark_efficiency = 1000000  # $1M per market share point
                    
                    if revenue_efficiency < benchmark_efficiency * 0.8:
                        insights.append(BaseInsight(
                            insight_type=InsightType.REVENUE,
                            title="Low Revenue Efficiency",
                            description=f"Revenue efficiency of ${revenue_efficiency:,.0f} per market share point "
                                       f"below industry benchmark.",
                            severity=InsightSeverity.MEDIUM,
                            confidence=0.70,
                            recommendations=[
                                RecommendedAction(
                                    action_type=ActionType.REVENUE_OPTIMIZATION,
                                    description="Optimize revenue per market share point",
                                    expected_impact="Improve revenue efficiency by 20-30%",
                                    priority=InsightSeverity.MEDIUM,
                                    implementation_effort="high",
                                    success_probability=0.60
                                )
                            ],
                            tags=["revenue", "efficiency", "market_share"]
                        ))
        
        except Exception as e:
            self.logger.error(f"Error analyzing revenue-market relationship: {e}")
        
        return insights
    
    def _analyze_seasonal_patterns(self, simulation_data: Dict[str, pd.DataFrame]) -> List[BaseInsight]:
        """Analyze seasonal patterns across datasets."""
        
        insights = []
        
        try:
            # Look for seasonal patterns in demand and pricing
            for data_type, data in simulation_data.items():
                if 'date' in data.columns and len(data) > 30:  # Need sufficient data
                    data['month'] = pd.to_datetime(data['date']).dt.month
                    
                    # Analyze monthly patterns
                    if 'demand' in data.columns:
                        monthly_demand = data.groupby('month')['demand'].mean()
                        demand_variance = monthly_demand.std() / monthly_demand.mean()
                        
                        if demand_variance > 0.2:  # 20% seasonal variance
                            peak_month = monthly_demand.idxmax()
                            low_month = monthly_demand.idxmin()
                            
                            insights.append(BaseInsight(
                                insight_type=InsightType.DEMAND,
                                title=f"Strong Seasonal Pattern in {data_type.title()}",
                                description=f"Demand varies by {demand_variance:.1%} seasonally. "
                                           f"Peak in month {peak_month}, low in month {low_month}.",
                                severity=InsightSeverity.MEDIUM,
                                confidence=0.80,
                                recommendations=[
                                    RecommendedAction(
                                        action_type=ActionType.CAPACITY_ADJUSTMENT,
                                        description="Implement seasonal capacity planning",
                                        expected_impact="Optimize capacity utilization year-round",
                                        priority=InsightSeverity.MEDIUM,
                                        implementation_effort="medium",
                                        timeline="long-term",
                                        success_probability=0.70
                                    )
                                ],
                                tags=["seasonal", "demand", data_type]
                            ))
        
        except Exception as e:
            self.logger.error(f"Error analyzing seasonal patterns: {e}")
        
        return insights
    
    def get_insights_by_type(self, insight_type: InsightType) -> List[BaseInsight]:
        """Get insights filtered by type."""
        return [insight for insight in self.insights_history if insight.insight_type == insight_type]
    
    def get_insights_by_severity(self, severity: InsightSeverity) -> List[BaseInsight]:
        """Get insights filtered by severity."""
        return [insight for insight in self.insights_history if insight.severity == severity]
    
    def get_top_insights(self, limit: int = 10) -> List[BaseInsight]:
        """Get top insights by severity and confidence."""
        
        # Sort by severity (critical first) and confidence
        severity_order = {InsightSeverity.CRITICAL: 4, InsightSeverity.HIGH: 3, 
                         InsightSeverity.MEDIUM: 2, InsightSeverity.LOW: 1}
        
        sorted_insights = sorted(
            self.insights_history,
            key=lambda x: (severity_order[x.severity], x.confidence),
            reverse=True
        )
        
        return sorted_insights[:limit]
    
    def generate_insight_summary(self) -> Dict[str, Any]:
        """Generate summary of all insights."""
        
        if not self.insights_history:
            return {"message": "No insights available"}
        
        summary = {
            'total_insights': len(self.insights_history),
            'by_type': {},
            'by_severity': {},
            'top_recommendations': [],
            'key_metrics': {},
            'last_updated': datetime.now().isoformat()
        }
        
        # Count by type
        for insight_type in InsightType:
            count = len(self.get_insights_by_type(insight_type))
            summary['by_type'][insight_type.value] = count
        
        # Count by severity
        for severity in InsightSeverity:
            count = len(self.get_insights_by_severity(severity))
            summary['by_severity'][severity.value] = count
        
        # Top recommendations
        top_insights = self.get_top_insights(5)
        summary['top_recommendations'] = [
            {
                'title': insight.title,
                'severity': insight.severity.value,
                'confidence': insight.confidence,
                'type': insight.insight_type.value
            }
            for insight in top_insights
        ]
        
        # Key metrics
        if self.insights_history:
            avg_confidence = np.mean([insight.confidence for insight in self.insights_history])
            critical_count = len(self.get_insights_by_severity(InsightSeverity.CRITICAL))
            
            summary['key_metrics'] = {
                'average_confidence': round(avg_confidence, 2),
                'critical_insights': critical_count,
                'actionable_insights': len([i for i in self.insights_history if i.recommendations])
            }
        
        return summary
    
    def export_insights(self, format: str = "json") -> str:
        """Export insights to specified format."""
        
        if format == "json":
            import json
            
            insights_data = []
            for insight in self.insights_history:
                insight_dict = {
                    'type': insight.insight_type.value,
                    'title': insight.title,
                    'description': insight.description,
                    'severity': insight.severity.value,
                    'confidence': insight.confidence,
                    'timestamp': insight.timestamp.isoformat(),
                    'metrics': [
                        {
                            'name': metric.name,
                            'current_value': metric.current_value,
                            'target_value': metric.target_value,
                            'benchmark_value': metric.benchmark_value,
                            'trend': metric.trend,
                            'confidence': metric.confidence,
                            'unit': metric.unit
                        }
                        for metric in insight.metrics
                    ],
                    'recommendations': [
                        {
                            'action_type': rec.action_type.value,
                            'description': rec.description,
                            'expected_impact': rec.expected_impact,
                            'priority': rec.priority.value,
                            'estimated_revenue_impact': rec.estimated_revenue_impact,
                            'implementation_effort': rec.implementation_effort,
                            'timeline': rec.timeline,
                            'success_probability': rec.success_probability
                        }
                        for rec in insight.recommendations
                    ],
                    'tags': insight.tags
                }
                insights_data.append(insight_dict)
            
            return json.dumps(insights_data, indent=2)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def clear_insights_history(self):
        """Clear insights history."""
        self.insights_history.clear()
        self.logger.info("Insights history cleared")