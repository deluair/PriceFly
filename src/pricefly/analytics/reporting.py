"""Reporting and dashboard generation for airline pricing simulation."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from pathlib import Path
import io
from collections import defaultdict

from .metrics import (
    PerformanceMetrics, RevenueMetrics, CompetitiveMetrics, 
    OperationalMetrics, MetricValue, MetricPeriod
)


class ReportType(Enum):
    """Types of reports."""
    EXECUTIVE_SUMMARY = "executive_summary"
    PERFORMANCE_DASHBOARD = "performance_dashboard"
    REVENUE_ANALYSIS = "revenue_analysis"
    COMPETITIVE_INTELLIGENCE = "competitive_intelligence"
    ROUTE_ANALYSIS = "route_analysis"
    OPERATIONAL_REPORT = "operational_report"
    PRICING_OPTIMIZATION = "pricing_optimization"
    MARKET_ANALYSIS = "market_analysis"
    FINANCIAL_REPORT = "financial_report"
    CUSTOM = "custom"


class ReportFormat(Enum):
    """Report output formats."""
    HTML = "html"
    PDF = "pdf"
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    DASHBOARD = "dashboard"


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    report_type: ReportType
    format: ReportFormat
    period: MetricPeriod
    start_date: datetime
    end_date: datetime
    include_charts: bool = True
    include_benchmarks: bool = True
    include_trends: bool = True
    include_forecasts: bool = False
    custom_metrics: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    output_path: Optional[str] = None
    template_path: Optional[str] = None


@dataclass
class DashboardData:
    """Data structure for dashboard components."""
    # Key performance indicators
    kpis: Dict[str, MetricValue] = field(default_factory=dict)
    
    # Chart data
    revenue_trend: List[Dict[str, Any]] = field(default_factory=list)
    load_factor_trend: List[Dict[str, Any]] = field(default_factory=list)
    competitive_position: List[Dict[str, Any]] = field(default_factory=list)
    route_performance: List[Dict[str, Any]] = field(default_factory=list)
    
    # Tables
    top_routes: pd.DataFrame = field(default_factory=pd.DataFrame)
    bottom_routes: pd.DataFrame = field(default_factory=pd.DataFrame)
    competitor_analysis: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Alerts and insights
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    insights: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    last_updated: datetime = field(default_factory=datetime.now)
    data_quality_score: float = 1.0
    coverage_percentage: float = 100.0


@dataclass
class SimulationReport:
    """Comprehensive simulation report."""
    # Report metadata
    report_id: str
    report_type: ReportType
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    
    # Executive summary
    executive_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Core metrics
    performance_metrics: Optional[PerformanceMetrics] = None
    revenue_metrics: Optional[RevenueMetrics] = None
    competitive_metrics: Optional[CompetitiveMetrics] = None
    operational_metrics: Optional[OperationalMetrics] = None
    
    # Analysis sections
    trend_analysis: Dict[str, Any] = field(default_factory=dict)
    benchmark_comparison: Dict[str, Any] = field(default_factory=dict)
    route_analysis: Dict[str, Any] = field(default_factory=dict)
    competitive_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Recommendations
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Appendices
    data_sources: List[str] = field(default_factory=list)
    methodology: str = ""
    assumptions: List[str] = field(default_factory=list)
    
    # Quality metrics
    data_quality_score: float = 1.0
    confidence_level: float = 0.95


@dataclass
class CompetitiveReport:
    """Competitive intelligence report."""
    # Market overview
    market_size: float = 0.0
    market_growth: float = 0.0
    market_concentration: float = 0.0
    
    # Competitive position
    market_share: float = 0.0
    market_rank: int = 0
    price_position: str = "neutral"
    service_position: str = "neutral"
    
    # Competitor analysis
    competitor_profiles: List[Dict[str, Any]] = field(default_factory=list)
    competitive_moves: List[Dict[str, Any]] = field(default_factory=list)
    threat_assessment: Dict[str, Any] = field(default_factory=dict)
    
    # Market dynamics
    entry_barriers: List[str] = field(default_factory=list)
    exit_barriers: List[str] = field(default_factory=list)
    market_trends: List[str] = field(default_factory=list)
    
    # Strategic recommendations
    strategic_options: List[Dict[str, Any]] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)


@dataclass
class RouteAnalysisReport:
    """Route-specific analysis report."""
    route_id: str
    route_name: str
    
    # Route characteristics
    distance: float = 0.0
    market_size: float = 0.0
    seasonality_index: float = 1.0
    
    # Performance metrics
    load_factor: float = 0.0
    average_fare: float = 0.0
    revenue: float = 0.0
    profit_margin: float = 0.0
    
    # Competitive landscape
    competitors_count: int = 0
    market_share: float = 0.0
    price_position: str = "neutral"
    
    # Demand analysis
    demand_elasticity: float = 0.0
    price_sensitivity: float = 0.0
    booking_curve: List[Dict[str, Any]] = field(default_factory=list)
    
    # Optimization opportunities
    revenue_opportunity: float = 0.0
    capacity_optimization: Dict[str, Any] = field(default_factory=dict)
    pricing_recommendations: List[str] = field(default_factory=list)


class ReportGenerator:
    """Generates various types of reports and dashboards."""
    
    def __init__(self, output_dir: str = "reports"):
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Report templates
        self.templates = self._load_templates()
        
        # Chart configurations
        self.chart_configs = self._initialize_chart_configs()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load report templates."""
        return {
            'executive_summary': self._get_executive_template(),
            'performance_dashboard': self._get_dashboard_template(),
            'revenue_analysis': self._get_revenue_template(),
            'competitive_intelligence': self._get_competitive_template()
        }
    
    def _initialize_chart_configs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize chart configurations."""
        return {
            'revenue_trend': {
                'type': 'line',
                'title': 'Revenue Trend',
                'x_axis': 'Date',
                'y_axis': 'Revenue ($)',
                'color': '#1f77b4'
            },
            'load_factor_trend': {
                'type': 'line',
                'title': 'Load Factor Trend',
                'x_axis': 'Date',
                'y_axis': 'Load Factor (%)',
                'color': '#ff7f0e'
            },
            'market_share': {
                'type': 'pie',
                'title': 'Market Share by Airline',
                'colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            },
            'route_performance': {
                'type': 'scatter',
                'title': 'Route Performance (Load Factor vs Revenue)',
                'x_axis': 'Load Factor (%)',
                'y_axis': 'Revenue ($)',
                'color': '#2ca02c'
            }
        }
    
    def generate_simulation_report(
        self,
        config: ReportConfig,
        performance_metrics: PerformanceMetrics,
        revenue_metrics: RevenueMetrics,
        competitive_metrics: CompetitiveMetrics,
        operational_metrics: OperationalMetrics,
        simulation_data: Dict[str, pd.DataFrame]
    ) -> SimulationReport:
        """Generate comprehensive simulation report."""
        
        report = SimulationReport(
            report_id=f"sim_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            report_type=config.report_type,
            generated_at=datetime.now(),
            period_start=config.start_date,
            period_end=config.end_date,
            performance_metrics=performance_metrics,
            revenue_metrics=revenue_metrics,
            competitive_metrics=competitive_metrics,
            operational_metrics=operational_metrics
        )
        
        # Generate executive summary
        report.executive_summary = self._generate_executive_summary(
            performance_metrics, revenue_metrics, competitive_metrics, operational_metrics
        )
        
        # Generate trend analysis
        if config.include_trends:
            report.trend_analysis = self._generate_trend_analysis(simulation_data)
        
        # Generate benchmark comparison
        if config.include_benchmarks:
            report.benchmark_comparison = self._generate_benchmark_comparison(
                performance_metrics, revenue_metrics, competitive_metrics, operational_metrics
            )
        
        # Generate route analysis
        if 'routes' in simulation_data:
            report.route_analysis = self._generate_route_analysis(simulation_data['routes'])
        
        # Generate competitive analysis
        if 'competitors' in simulation_data:
            report.competitive_analysis = self._generate_competitive_analysis(
                simulation_data['competitors'], competitive_metrics
            )
        
        # Generate recommendations
        report.recommendations = self._generate_recommendations(
            performance_metrics, revenue_metrics, competitive_metrics, operational_metrics
        )
        
        # Calculate data quality score
        report.data_quality_score = self._calculate_data_quality_score(simulation_data)
        
        return report
    
    def generate_dashboard_data(
        self,
        performance_metrics: PerformanceMetrics,
        revenue_metrics: RevenueMetrics,
        competitive_metrics: CompetitiveMetrics,
        operational_metrics: OperationalMetrics,
        time_series_data: Dict[str, pd.DataFrame]
    ) -> DashboardData:
        """Generate data for interactive dashboard."""
        
        dashboard = DashboardData()
        
        # Key Performance Indicators
        dashboard.kpis = {
            'total_revenue': MetricValue(
                name='Total Revenue',
                value=performance_metrics.total_revenue,
                unit='$',
                period=MetricPeriod.MONTHLY,
                timestamp=datetime.now()
            ),
            'load_factor': MetricValue(
                name='Load Factor',
                value=performance_metrics.load_factor * 100,
                unit='%',
                period=MetricPeriod.MONTHLY,
                timestamp=datetime.now()
            ),
            'market_share': MetricValue(
                name='Market Share',
                value=competitive_metrics.market_share_passengers * 100,
                unit='%',
                period=MetricPeriod.MONTHLY,
                timestamp=datetime.now()
            ),
            'on_time_performance': MetricValue(
                name='On-Time Performance',
                value=operational_metrics.on_time_arrival * 100,
                unit='%',
                period=MetricPeriod.MONTHLY,
                timestamp=datetime.now()
            )
        }
        
        # Generate chart data
        if 'revenue_daily' in time_series_data:
            dashboard.revenue_trend = self._prepare_chart_data(
                time_series_data['revenue_daily'], 'date', 'revenue'
            )
        
        if 'load_factor_daily' in time_series_data:
            dashboard.load_factor_trend = self._prepare_chart_data(
                time_series_data['load_factor_daily'], 'date', 'load_factor'
            )
        
        # Generate alerts
        dashboard.alerts = self._generate_alerts(
            performance_metrics, revenue_metrics, competitive_metrics, operational_metrics
        )
        
        # Generate insights
        dashboard.insights = self._generate_insights(
            performance_metrics, revenue_metrics, competitive_metrics, operational_metrics
        )
        
        return dashboard
    
    def generate_competitive_report(
        self,
        competitive_metrics: CompetitiveMetrics,
        market_data: pd.DataFrame,
        competitor_data: pd.DataFrame
    ) -> CompetitiveReport:
        """Generate competitive intelligence report."""
        
        report = CompetitiveReport()
        
        # Market overview
        if 'market_size' in market_data.columns:
            report.market_size = market_data['market_size'].sum()
        
        report.market_share = competitive_metrics.market_share_passengers
        report.market_concentration = competitive_metrics.market_concentration_hhi
        report.price_position = competitive_metrics.price_position
        
        # Competitor profiles
        if not competitor_data.empty:
            report.competitor_profiles = self._generate_competitor_profiles(competitor_data)
        
        # Threat assessment
        report.threat_assessment = self._assess_competitive_threats(
            competitive_metrics, competitor_data
        )
        
        # Strategic recommendations
        report.strategic_options = self._generate_strategic_options(
            competitive_metrics, market_data
        )
        
        return report
    
    def generate_route_analysis_report(
        self,
        route_id: str,
        route_data: pd.DataFrame,
        booking_data: pd.DataFrame,
        competitive_data: pd.DataFrame
    ) -> RouteAnalysisReport:
        """Generate route-specific analysis report."""
        
        route_info = route_data[route_data['route_id'] == route_id].iloc[0]
        route_bookings = booking_data[booking_data['route_id'] == route_id]
        route_competition = competitive_data[competitive_data['route_id'] == route_id]
        
        report = RouteAnalysisReport(
            route_id=route_id,
            route_name=route_info.get('route_name', f"Route {route_id}")
        )
        
        # Route characteristics
        report.distance = route_info.get('distance', 0)
        report.market_size = route_bookings['passengers'].sum() if 'passengers' in route_bookings.columns else len(route_bookings)
        
        # Performance metrics
        if not route_bookings.empty:
            report.revenue = route_bookings['fare'].sum()
            report.average_fare = route_bookings['fare'].mean()
            
            # Calculate load factor if capacity data available
            if 'capacity' in route_info:
                report.load_factor = len(route_bookings) / route_info['capacity']
        
        # Competitive analysis
        if not route_competition.empty:
            report.competitors_count = len(route_competition['airline'].unique())
            
            # Market share calculation
            total_passengers = route_competition['passengers'].sum()
            our_passengers = route_competition[
                route_competition['airline'] == 'our_airline'
            ]['passengers'].sum()
            
            if total_passengers > 0:
                report.market_share = our_passengers / total_passengers
        
        # Generate optimization recommendations
        report.pricing_recommendations = self._generate_route_pricing_recommendations(
            route_bookings, route_competition
        )
        
        return report
    
    def export_report(
        self,
        report: SimulationReport,
        format: ReportFormat,
        filename: Optional[str] = None
    ) -> str:
        """Export report in specified format."""
        
        if filename is None:
            filename = f"{report.report_id}.{format.value}"
        
        output_path = self.output_dir / filename
        
        if format == ReportFormat.JSON:
            self._export_json(report, output_path)
        elif format == ReportFormat.HTML:
            self._export_html(report, output_path)
        elif format == ReportFormat.CSV:
            self._export_csv(report, output_path)
        elif format == ReportFormat.EXCEL:
            self._export_excel(report, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Report exported to {output_path}")
        return str(output_path)
    
    def _generate_executive_summary(
        self,
        performance_metrics: PerformanceMetrics,
        revenue_metrics: RevenueMetrics,
        competitive_metrics: CompetitiveMetrics,
        operational_metrics: OperationalMetrics
    ) -> Dict[str, Any]:
        """Generate executive summary."""
        
        return {
            'key_highlights': [
                f"Total revenue: ${performance_metrics.total_revenue:,.0f}",
                f"Load factor: {performance_metrics.load_factor:.1%}",
                f"Market share: {competitive_metrics.market_share_passengers:.1%}",
                f"On-time performance: {operational_metrics.on_time_arrival:.1%}"
            ],
            'performance_summary': {
                'revenue_growth': 'TBD',  # Would need historical data
                'profitability': performance_metrics.operating_margin,
                'efficiency': performance_metrics.load_factor,
                'competitiveness': competitive_metrics.market_share_passengers
            },
            'key_challenges': self._identify_key_challenges(
                performance_metrics, competitive_metrics, operational_metrics
            ),
            'opportunities': self._identify_opportunities(
                performance_metrics, revenue_metrics, competitive_metrics
            )
        }
    
    def _generate_trend_analysis(self, simulation_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate trend analysis."""
        trends = {}
        
        # Revenue trends
        if 'revenue_daily' in simulation_data:
            revenue_data = simulation_data['revenue_daily']
            if len(revenue_data) > 1:
                trends['revenue'] = {
                    'direction': 'up' if revenue_data['revenue'].iloc[-1] > revenue_data['revenue'].iloc[0] else 'down',
                    'volatility': revenue_data['revenue'].std() / revenue_data['revenue'].mean(),
                    'growth_rate': (revenue_data['revenue'].iloc[-1] / revenue_data['revenue'].iloc[0] - 1) * 100
                }
        
        # Load factor trends
        if 'load_factor_daily' in simulation_data:
            lf_data = simulation_data['load_factor_daily']
            if len(lf_data) > 1:
                trends['load_factor'] = {
                    'direction': 'up' if lf_data['load_factor'].iloc[-1] > lf_data['load_factor'].iloc[0] else 'down',
                    'volatility': lf_data['load_factor'].std(),
                    'average': lf_data['load_factor'].mean()
                }
        
        return trends
    
    def _generate_benchmark_comparison(
        self,
        performance_metrics: PerformanceMetrics,
        revenue_metrics: RevenueMetrics,
        competitive_metrics: CompetitiveMetrics,
        operational_metrics: OperationalMetrics
    ) -> Dict[str, Any]:
        """Generate benchmark comparison."""
        
        # Industry benchmarks
        benchmarks = {
            'load_factor': 0.82,
            'on_time_performance': 0.85,
            'operating_margin': 0.08,
            'market_concentration': 0.60
        }
        
        comparison = {}
        
        # Load factor comparison
        comparison['load_factor'] = {
            'actual': performance_metrics.load_factor,
            'benchmark': benchmarks['load_factor'],
            'variance': performance_metrics.load_factor - benchmarks['load_factor'],
            'performance': 'above' if performance_metrics.load_factor > benchmarks['load_factor'] else 'below'
        }
        
        # On-time performance comparison
        comparison['on_time_performance'] = {
            'actual': operational_metrics.on_time_arrival,
            'benchmark': benchmarks['on_time_performance'],
            'variance': operational_metrics.on_time_arrival - benchmarks['on_time_performance'],
            'performance': 'above' if operational_metrics.on_time_arrival > benchmarks['on_time_performance'] else 'below'
        }
        
        return comparison
    
    def _generate_route_analysis(self, routes_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate route analysis."""
        
        analysis = {
            'total_routes': len(routes_data),
            'top_performing_routes': [],
            'underperforming_routes': [],
            'route_concentration': 0.0
        }
        
        if 'revenue' in routes_data.columns:
            # Top performing routes
            top_routes = routes_data.nlargest(5, 'revenue')
            analysis['top_performing_routes'] = top_routes[['route_id', 'revenue']].to_dict('records')
            
            # Underperforming routes
            bottom_routes = routes_data.nsmallest(5, 'revenue')
            analysis['underperforming_routes'] = bottom_routes[['route_id', 'revenue']].to_dict('records')
            
            # Route concentration (HHI)
            total_revenue = routes_data['revenue'].sum()
            if total_revenue > 0:
                route_shares = routes_data['revenue'] / total_revenue
                analysis['route_concentration'] = (route_shares ** 2).sum()
        
        return analysis
    
    def _generate_competitive_analysis(
        self,
        competitors_data: pd.DataFrame,
        competitive_metrics: CompetitiveMetrics
    ) -> Dict[str, Any]:
        """Generate competitive analysis."""
        
        analysis = {
            'market_position': competitive_metrics.price_position,
            'market_share_rank': 0,  # Would need to calculate from data
            'competitive_threats': [],
            'competitive_advantages': []
        }
        
        # Identify competitive threats
        if 'market_share' in competitors_data.columns:
            growing_competitors = competitors_data[
                competitors_data['market_share_growth'] > 0.05
            ] if 'market_share_growth' in competitors_data.columns else pd.DataFrame()
            
            analysis['competitive_threats'] = growing_competitors['airline'].tolist()
        
        return analysis
    
    def _generate_recommendations(
        self,
        performance_metrics: PerformanceMetrics,
        revenue_metrics: RevenueMetrics,
        competitive_metrics: CompetitiveMetrics,
        operational_metrics: OperationalMetrics
    ) -> List[Dict[str, Any]]:
        """Generate strategic recommendations."""
        
        recommendations = []
        
        # Load factor recommendations
        if performance_metrics.load_factor < 0.75:
            recommendations.append({
                'category': 'Revenue Management',
                'priority': 'High',
                'recommendation': 'Implement dynamic pricing to improve load factor',
                'expected_impact': 'Increase load factor by 5-10%',
                'implementation_effort': 'Medium'
            })
        
        # Competitive position recommendations
        if competitive_metrics.market_share_passengers < 0.15:
            recommendations.append({
                'category': 'Market Strategy',
                'priority': 'High',
                'recommendation': 'Develop market share growth strategy',
                'expected_impact': 'Increase market share by 2-3%',
                'implementation_effort': 'High'
            })
        
        # Operational efficiency recommendations
        if operational_metrics.on_time_arrival < 0.80:
            recommendations.append({
                'category': 'Operations',
                'priority': 'Medium',
                'recommendation': 'Improve operational efficiency and on-time performance',
                'expected_impact': 'Reduce customer complaints by 15%',
                'implementation_effort': 'Medium'
            })
        
        return recommendations
    
    def _prepare_chart_data(
        self,
        data: pd.DataFrame,
        x_column: str,
        y_column: str
    ) -> List[Dict[str, Any]]:
        """Prepare data for chart visualization."""
        
        chart_data = []
        
        for _, row in data.iterrows():
            chart_data.append({
                'x': row[x_column].isoformat() if hasattr(row[x_column], 'isoformat') else row[x_column],
                'y': float(row[y_column]) if pd.notna(row[y_column]) else 0.0
            })
        
        return chart_data
    
    def _generate_alerts(
        self,
        performance_metrics: PerformanceMetrics,
        revenue_metrics: RevenueMetrics,
        competitive_metrics: CompetitiveMetrics,
        operational_metrics: OperationalMetrics
    ) -> List[Dict[str, Any]]:
        """Generate alerts for dashboard."""
        
        alerts = []
        
        # Performance alerts
        if performance_metrics.load_factor < 0.70:
            alerts.append({
                'type': 'warning',
                'category': 'Performance',
                'message': f'Load factor below target: {performance_metrics.load_factor:.1%}',
                'severity': 'medium'
            })
        
        if operational_metrics.on_time_arrival < 0.75:
            alerts.append({
                'type': 'warning',
                'category': 'Operations',
                'message': f'On-time performance below target: {operational_metrics.on_time_arrival:.1%}',
                'severity': 'high'
            })
        
        # Competitive alerts
        if competitive_metrics.market_share_passengers < 0.10:
            alerts.append({
                'type': 'info',
                'category': 'Market',
                'message': f'Market share below 10%: {competitive_metrics.market_share_passengers:.1%}',
                'severity': 'medium'
            })
        
        return alerts
    
    def _generate_insights(
        self,
        performance_metrics: PerformanceMetrics,
        revenue_metrics: RevenueMetrics,
        competitive_metrics: CompetitiveMetrics,
        operational_metrics: OperationalMetrics
    ) -> List[Dict[str, Any]]:
        """Generate insights for dashboard."""
        
        insights = []
        
        # Revenue insights
        if revenue_metrics.ancillary_revenue > 0:
            ancillary_percentage = (revenue_metrics.ancillary_revenue / 
                                   revenue_metrics.passenger_revenue) * 100
            insights.append({
                'type': 'revenue',
                'title': 'Ancillary Revenue Performance',
                'description': f'Ancillary revenue represents {ancillary_percentage:.1f}% of passenger revenue',
                'impact': 'positive' if ancillary_percentage > 15 else 'neutral'
            })
        
        # Operational insights
        if operational_metrics.aircraft_utilization_rate > 0:
            insights.append({
                'type': 'operational',
                'title': 'Aircraft Utilization',
                'description': f'Aircraft utilization rate: {operational_metrics.aircraft_utilization_rate:.1%}',
                'impact': 'positive' if operational_metrics.aircraft_utilization_rate > 0.75 else 'negative'
            })
        
        return insights
    
    def _calculate_data_quality_score(self, simulation_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate overall data quality score."""
        
        quality_scores = []
        
        for dataset_name, dataset in simulation_data.items():
            if dataset.empty:
                quality_scores.append(0.0)
                continue
            
            # Calculate completeness
            completeness = 1.0 - (dataset.isnull().sum().sum() / (len(dataset) * len(dataset.columns)))
            
            # Calculate consistency (simplified)
            consistency = 1.0  # Would implement actual consistency checks
            
            # Calculate accuracy (simplified)
            accuracy = 1.0  # Would implement actual accuracy checks
            
            dataset_quality = (completeness + consistency + accuracy) / 3
            quality_scores.append(dataset_quality)
        
        return np.mean(quality_scores) if quality_scores else 1.0
    
    def _export_json(self, report: SimulationReport, output_path: Path) -> None:
        """Export report as JSON."""
        
        # Convert report to dictionary
        report_dict = {
            'report_id': report.report_id,
            'report_type': report.report_type.value,
            'generated_at': report.generated_at.isoformat(),
            'period_start': report.period_start.isoformat(),
            'period_end': report.period_end.isoformat(),
            'executive_summary': report.executive_summary,
            'trend_analysis': report.trend_analysis,
            'benchmark_comparison': report.benchmark_comparison,
            'recommendations': report.recommendations,
            'data_quality_score': report.data_quality_score
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
    
    def _export_html(self, report: SimulationReport, output_path: Path) -> None:
        """Export report as HTML."""
        
        html_template = self.templates.get('executive_summary', self._get_default_html_template())
        
        # Replace placeholders with actual data
        html_content = html_template.format(
            report_title=f"{report.report_type.value.replace('_', ' ').title()} Report",
            generated_at=report.generated_at.strftime('%Y-%m-%d %H:%M:%S'),
            executive_summary=self._format_executive_summary_html(report.executive_summary),
            recommendations=self._format_recommendations_html(report.recommendations)
        )
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _export_csv(self, report: SimulationReport, output_path: Path) -> None:
        """Export report as CSV."""
        
        # Create a summary DataFrame
        summary_data = []
        
        if report.performance_metrics:
            summary_data.append(['Total Revenue', report.performance_metrics.total_revenue, '$'])
            summary_data.append(['Load Factor', report.performance_metrics.load_factor, '%'])
            summary_data.append(['Average Fare', report.performance_metrics.average_fare, '$'])
        
        if report.competitive_metrics:
            summary_data.append(['Market Share', report.competitive_metrics.market_share_passengers, '%'])
            summary_data.append(['Price Position', report.competitive_metrics.price_position, 'text'])
        
        df = pd.DataFrame(summary_data, columns=['Metric', 'Value', 'Unit'])
        df.to_csv(output_path, index=False)
    
    def _export_excel(self, report: SimulationReport, output_path: Path) -> None:
        """Export report as Excel with multiple sheets."""
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = self._create_summary_dataframe(report)
            summary_data.to_excel(writer, sheet_name='Summary', index=False)
            
            # Recommendations sheet
            if report.recommendations:
                rec_df = pd.DataFrame(report.recommendations)
                rec_df.to_excel(writer, sheet_name='Recommendations', index=False)
    
    def _create_summary_dataframe(self, report: SimulationReport) -> pd.DataFrame:
        """Create summary DataFrame for export."""
        
        data = []
        
        if report.performance_metrics:
            pm = report.performance_metrics
            data.extend([
                ['Performance', 'Total Revenue', pm.total_revenue, '$'],
                ['Performance', 'Load Factor', pm.load_factor, '%'],
                ['Performance', 'Average Fare', pm.average_fare, '$'],
                ['Performance', 'Operating Margin', pm.operating_margin, '%']
            ])
        
        if report.competitive_metrics:
            cm = report.competitive_metrics
            data.extend([
                ['Competitive', 'Market Share (Passengers)', cm.market_share_passengers, '%'],
                ['Competitive', 'Market Share (Revenue)', cm.market_share_revenue, '%'],
                ['Competitive', 'Price Position', cm.price_position, 'text']
            ])
        
        return pd.DataFrame(data, columns=['Category', 'Metric', 'Value', 'Unit'])
    
    # Template methods
    def _get_executive_template(self) -> str:
        return """
        <h1>Executive Summary</h1>
        <p>Generated: {generated_at}</p>
        <div>{executive_summary}</div>
        <div>{recommendations}</div>
        """
    
    def _get_dashboard_template(self) -> str:
        return """
        <div class="dashboard">
            <div class="kpis">{kpis}</div>
            <div class="charts">{charts}</div>
            <div class="alerts">{alerts}</div>
        </div>
        """
    
    def _get_revenue_template(self) -> str:
        return """
        <h1>Revenue Analysis</h1>
        <div class="revenue-metrics">{revenue_metrics}</div>
        <div class="revenue-trends">{revenue_trends}</div>
        """
    
    def _get_competitive_template(self) -> str:
        return """
        <h1>Competitive Intelligence</h1>
        <div class="market-position">{market_position}</div>
        <div class="competitor-analysis">{competitor_analysis}</div>
        """
    
    def _get_default_html_template(self) -> str:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report_title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 10px; }}
                .content {{ margin: 20px 0; }}
                .metric {{ margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{report_title}</h1>
                <p>Generated: {generated_at}</p>
            </div>
            <div class="content">
                {executive_summary}
                {recommendations}
            </div>
        </body>
        </html>
        """
    
    def _format_executive_summary_html(self, summary: Dict[str, Any]) -> str:
        """Format executive summary for HTML."""
        html = "<h2>Executive Summary</h2>"
        
        if 'key_highlights' in summary:
            html += "<h3>Key Highlights</h3><ul>"
            for highlight in summary['key_highlights']:
                html += f"<li>{highlight}</li>"
            html += "</ul>"
        
        return html
    
    def _format_recommendations_html(self, recommendations: List[Dict[str, Any]]) -> str:
        """Format recommendations for HTML."""
        html = "<h2>Recommendations</h2>"
        
        for rec in recommendations:
            html += f"""
            <div class="recommendation">
                <h4>{rec.get('category', 'General')} - {rec.get('priority', 'Medium')} Priority</h4>
                <p>{rec.get('recommendation', '')}</p>
                <p><strong>Expected Impact:</strong> {rec.get('expected_impact', 'TBD')}</p>
            </div>
            """
        
        return html
    
    # Helper methods for competitive analysis
    def _generate_competitor_profiles(self, competitor_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate competitor profiles."""
        profiles = []
        
        for _, competitor in competitor_data.iterrows():
            profile = {
                'airline': competitor.get('airline', 'Unknown'),
                'market_share': competitor.get('market_share', 0),
                'average_price': competitor.get('average_price', 0),
                'routes_served': competitor.get('routes_count', 0),
                'strengths': [],  # Would analyze from data
                'weaknesses': []  # Would analyze from data
            }
            profiles.append(profile)
        
        return profiles
    
    def _assess_competitive_threats(
        self,
        competitive_metrics: CompetitiveMetrics,
        competitor_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Assess competitive threats."""
        
        threats = {
            'high_threats': [],
            'medium_threats': [],
            'low_threats': [],
            'overall_threat_level': 'medium'
        }
        
        # Analyze based on market concentration
        if competitive_metrics.market_concentration_hhi > 0.25:
            threats['overall_threat_level'] = 'high'
        elif competitive_metrics.market_concentration_hhi < 0.15:
            threats['overall_threat_level'] = 'low'
        
        return threats
    
    def _generate_strategic_options(
        self,
        competitive_metrics: CompetitiveMetrics,
        market_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Generate strategic options."""
        
        options = []
        
        # Market share growth options
        if competitive_metrics.market_share_passengers < 0.20:
            options.append({
                'strategy': 'Market Share Growth',
                'description': 'Aggressive pricing and capacity expansion',
                'risk_level': 'high',
                'expected_outcome': 'Increase market share by 3-5%'
            })
        
        # Premium positioning options
        if competitive_metrics.price_position == 'discount':
            options.append({
                'strategy': 'Premium Positioning',
                'description': 'Focus on service quality and premium pricing',
                'risk_level': 'medium',
                'expected_outcome': 'Improve margins by 2-3%'
            })
        
        return options
    
    def _generate_route_pricing_recommendations(
        self,
        route_bookings: pd.DataFrame,
        route_competition: pd.DataFrame
    ) -> List[str]:
        """Generate route-specific pricing recommendations."""
        
        recommendations = []
        
        if not route_bookings.empty:
            avg_fare = route_bookings['fare'].mean()
            fare_variance = route_bookings['fare'].var()
            
            if fare_variance > avg_fare * 0.5:  # High variance
                recommendations.append("Consider implementing more consistent pricing strategy")
            
            # Compare with competition
            if not route_competition.empty and 'average_price' in route_competition.columns:
                competitor_avg = route_competition['average_price'].mean()
                
                if avg_fare > competitor_avg * 1.1:
                    recommendations.append("Consider reducing prices to match competition")
                elif avg_fare < competitor_avg * 0.9:
                    recommendations.append("Opportunity to increase prices")
        
        return recommendations
    
    def _identify_key_challenges(
        self,
        performance_metrics: PerformanceMetrics,
        competitive_metrics: CompetitiveMetrics,
        operational_metrics: OperationalMetrics
    ) -> List[str]:
        """Identify key business challenges."""
        
        challenges = []
        
        if performance_metrics.load_factor < 0.75:
            challenges.append("Low load factor affecting profitability")
        
        if competitive_metrics.market_share_passengers < 0.15:
            challenges.append("Limited market share in competitive environment")
        
        if operational_metrics.on_time_arrival < 0.80:
            challenges.append("Operational reliability issues")
        
        if performance_metrics.operating_margin < 0.05:
            challenges.append("Thin operating margins")
        
        return challenges
    
    def _identify_opportunities(
        self,
        performance_metrics: PerformanceMetrics,
        revenue_metrics: RevenueMetrics,
        competitive_metrics: CompetitiveMetrics
    ) -> List[str]:
        """Identify business opportunities."""
        
        opportunities = []
        
        if revenue_metrics.ancillary_revenue / revenue_metrics.passenger_revenue < 0.15:
            opportunities.append("Increase ancillary revenue through enhanced services")
        
        if competitive_metrics.price_position == "discount":
            opportunities.append("Potential for premium positioning and higher margins")
        
        if performance_metrics.load_factor > 0.85:
            opportunities.append("High demand suggests capacity expansion opportunities")
        
        return opportunities