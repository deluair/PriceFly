"""PriceFly: Comprehensive Airline Pricing Simulation Platform

A sophisticated simulation platform that models airline pricing strategies,
market dynamics, competitive intelligence, and revenue optimization.
"""

__version__ = "1.0.0"
__author__ = "PriceFly Development Team"
__email__ = "contact@pricefly.ai"

from .core import *
from .models import *
from .simulation import *
from .analytics import *

__all__ = [
    "DynamicPricingEngine",
    "RevenueManager",
    "MarketSimulator",
    "CompetitiveIntelligence",
    "DataGenerator",
    "Analytics"
]