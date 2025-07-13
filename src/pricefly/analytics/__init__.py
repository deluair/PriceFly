"""Analytics module for PriceFly simulation platform.

This module provides comprehensive analytics and reporting capabilities
for airline pricing simulation results, including performance metrics,
revenue analysis, competitive intelligence, and market insights.
"""

from .metrics import (
    PerformanceMetrics,
    RevenueMetrics,
    CompetitiveMetrics,
    OperationalMetrics,
    MetricsCalculator
)

from .reporting import (
    ReportGenerator,
    DashboardData,
    SimulationReport,
    CompetitiveReport,
    RouteAnalysisReport
)

from .visualization import (
    PricingVisualizer,
    DemandVisualizer,
    CompetitiveVisualizer,
    RevenueVisualizer
)

from .insights import (
    InsightEngine,
    PricingInsight,
    MarketInsight,
    RevenueInsight,
    CompetitiveInsight
)

__version__ = "1.0.0"
__author__ = "PriceFly Development Team"

__all__ = [
    # Metrics
    "PerformanceMetrics",
    "RevenueMetrics",
    "CompetitiveMetrics",
    "OperationalMetrics",
    "MetricsCalculator",
    
    # Reporting
    "ReportGenerator",
    "DashboardData",
    "SimulationReport",
    "CompetitiveReport",
    "RouteAnalysisReport",
    
    # Visualization
    "PricingVisualizer",
    "DemandVisualizer",
    "CompetitiveVisualizer",
    "RevenueVisualizer",
    
    # Insights
    "InsightEngine",
    "PricingInsight",
    "MarketInsight",
    "RevenueInsight",
    "CompetitiveInsight"
]