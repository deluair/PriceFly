"""Simulation framework for airline pricing dynamics."""

from .engine import SimulationEngine, SimulationConfig
from .market import MarketSimulator, CompetitorAgent
from .demand import DemandSimulator, DemandScenario
from .events import EventSimulator, MarketEvent
from .scenarios import ScenarioManager, SimulationScenario

__all__ = [
    'SimulationEngine',
    'SimulationConfig',
    'MarketSimulator',
    'CompetitorAgent',
    'DemandSimulator',
    'DemandScenario',
    'EventSimulator',
    'MarketEvent',
    'ScenarioManager',
    'SimulationScenario'
]

# Version info
__version__ = '1.0.0'
__author__ = 'PriceFly Development Team'
__description__ = 'Advanced airline pricing simulation framework'