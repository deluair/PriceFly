"""Event simulation for external factors affecting airline pricing."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict
import random


class EventType(Enum):
    """Types of external events."""
    ECONOMIC = "economic"  # Economic indicators, recessions, etc.
    GEOPOLITICAL = "geopolitical"  # Wars, trade disputes, etc.
    NATURAL_DISASTER = "natural_disaster"  # Hurricanes, earthquakes, etc.
    PANDEMIC = "pandemic"  # Health crises
    FUEL_PRICE = "fuel_price"  # Oil price shocks
    REGULATORY = "regulatory"  # New regulations, policy changes
    TECHNOLOGY = "technology"  # New tech adoption, disruptions
    SEASONAL = "seasonal"  # Holidays, conferences, sports events
    COMPETITIVE = "competitive"  # New entrants, mergers, etc.
    OPERATIONAL = "operational"  # Strikes, system failures, etc.


class EventSeverity(Enum):
    """Severity levels for events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EventImpactType(Enum):
    """Types of impact an event can have."""
    DEMAND_INCREASE = "demand_increase"
    DEMAND_DECREASE = "demand_decrease"
    COST_INCREASE = "cost_increase"
    COST_DECREASE = "cost_decrease"
    CAPACITY_REDUCTION = "capacity_reduction"
    CAPACITY_INCREASE = "capacity_increase"
    PRICE_PRESSURE_UP = "price_pressure_up"
    PRICE_PRESSURE_DOWN = "price_pressure_down"
    VOLATILITY_INCREASE = "volatility_increase"
    REGULATORY_CONSTRAINT = "regulatory_constraint"


@dataclass
class EventImpact:
    """Represents the impact of an event."""
    impact_type: EventImpactType
    magnitude: float  # Percentage change or multiplier
    affected_routes: List[str] = field(default_factory=list)
    affected_segments: List[str] = field(default_factory=list)  # business, leisure, etc.
    affected_metrics: List[str] = field(default_factory=list)  # demand, cost, price, etc.
    duration_days: int = 7
    decay_rate: float = 0.1  # How quickly impact diminishes
    confidence: float = 0.8  # Confidence in impact estimate


@dataclass
class ExternalEvent:
    """Represents an external event affecting the airline industry."""
    event_id: str
    event_type: EventType
    severity: EventSeverity
    name: str
    description: str
    start_date: datetime
    end_date: Optional[datetime] = None
    
    # Geographic scope
    affected_countries: List[str] = field(default_factory=list)
    affected_cities: List[str] = field(default_factory=list)
    global_impact: bool = False
    
    # Impact details
    impacts: List[EventImpact] = field(default_factory=list)
    
    # Probability and uncertainty
    probability: float = 1.0  # For future events
    uncertainty_factor: float = 0.1
    
    # Related events
    triggers: List[str] = field(default_factory=list)  # Events that trigger this one
    consequences: List[str] = field(default_factory=list)  # Events this triggers
    
    # Metadata
    source: str = "simulation"
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)


# Alias for backward compatibility and specific market event usage
MarketEvent = ExternalEvent


class EventSimulator:
    """Simulates external events and their impacts on airline operations."""
    
    def __init__(self, simulation_start_date: datetime, simulation_end_date: datetime):
        self.simulation_start_date = simulation_start_date
        self.simulation_end_date = simulation_end_date
        
        # Event storage
        self.active_events: Dict[str, ExternalEvent] = {}
        self.historical_events: List[ExternalEvent] = []
        self.scheduled_events: List[ExternalEvent] = []
        
        # Event templates for random generation
        self.event_templates = self._initialize_event_templates()
        
        # Impact tracking
        self.current_impacts: Dict[str, List[EventImpact]] = defaultdict(list)
        
        # Event probabilities
        self.event_probabilities = self._initialize_event_probabilities()
        
        # Economic indicators affected by events
        self.economic_state = {
            'gdp_growth': 0.025,
            'inflation_rate': 0.02,
            'unemployment_rate': 0.05,
            'consumer_confidence': 0.8,
            'fuel_price_index': 1.0,
            'exchange_rate_volatility': 0.1,
            'interest_rates': 0.03
        }
        
        self.logger = logging.getLogger(__name__)
    
    def _initialize_event_templates(self) -> Dict[EventType, List[Dict[str, Any]]]:
        """Initialize templates for different types of events."""
        templates = {
            EventType.ECONOMIC: [
                {
                    'name': 'Economic Recession',
                    'description': 'Economic downturn affecting travel demand',
                    'severity': EventSeverity.HIGH,
                    'duration_range': (90, 365),
                    'impacts': [
                        EventImpact(EventImpactType.DEMAND_DECREASE, -0.25, duration_days=180),
                        EventImpact(EventImpactType.PRICE_PRESSURE_DOWN, -0.15, duration_days=180)
                    ]
                },
                {
                    'name': 'Economic Boom',
                    'description': 'Strong economic growth boosting travel',
                    'severity': EventSeverity.MEDIUM,
                    'duration_range': (180, 720),
                    'impacts': [
                        EventImpact(EventImpactType.DEMAND_INCREASE, 0.20, duration_days=360),
                        EventImpact(EventImpactType.PRICE_PRESSURE_UP, 0.10, duration_days=360)
                    ]
                },
                {
                    'name': 'Currency Crisis',
                    'description': 'Major currency devaluation affecting international travel',
                    'severity': EventSeverity.HIGH,
                    'duration_range': (30, 180),
                    'impacts': [
                        EventImpact(EventImpactType.DEMAND_DECREASE, -0.30, affected_segments=['leisure'], duration_days=120),
                        EventImpact(EventImpactType.COST_INCREASE, 0.15, duration_days=120)
                    ]
                }
            ],
            
            EventType.GEOPOLITICAL: [
                {
                    'name': 'Trade War',
                    'description': 'Trade tensions affecting business travel',
                    'severity': EventSeverity.MEDIUM,
                    'duration_range': (180, 1095),
                    'impacts': [
                        EventImpact(EventImpactType.DEMAND_DECREASE, -0.15, affected_segments=['business'], duration_days=365),
                        EventImpact(EventImpactType.VOLATILITY_INCREASE, 0.25, duration_days=365)
                    ]
                },
                {
                    'name': 'Regional Conflict',
                    'description': 'Military conflict affecting regional travel',
                    'severity': EventSeverity.CRITICAL,
                    'duration_range': (30, 365),
                    'impacts': [
                        EventImpact(EventImpactType.DEMAND_DECREASE, -0.80, duration_days=180),
                        EventImpact(EventImpactType.CAPACITY_REDUCTION, -0.50, duration_days=180)
                    ]
                },
                {
                    'name': 'Diplomatic Relations Improvement',
                    'description': 'Improved relations opening new routes',
                    'severity': EventSeverity.LOW,
                    'duration_range': (365, 1825),
                    'impacts': [
                        EventImpact(EventImpactType.DEMAND_INCREASE, 0.30, duration_days=730),
                        EventImpact(EventImpactType.CAPACITY_INCREASE, 0.20, duration_days=730)
                    ]
                }
            ],
            
            EventType.NATURAL_DISASTER: [
                {
                    'name': 'Major Hurricane',
                    'description': 'Hurricane disrupting air travel',
                    'severity': EventSeverity.HIGH,
                    'duration_range': (3, 14),
                    'impacts': [
                        EventImpact(EventImpactType.CAPACITY_REDUCTION, -0.70, duration_days=7),
                        EventImpact(EventImpactType.COST_INCREASE, 0.20, duration_days=14)
                    ]
                },
                {
                    'name': 'Earthquake',
                    'description': 'Major earthquake affecting airport operations',
                    'severity': EventSeverity.HIGH,
                    'duration_range': (7, 60),
                    'impacts': [
                        EventImpact(EventImpactType.CAPACITY_REDUCTION, -0.60, duration_days=30),
                        EventImpact(EventImpactType.DEMAND_DECREASE, -0.40, duration_days=60)
                    ]
                },
                {
                    'name': 'Volcanic Eruption',
                    'description': 'Volcanic ash disrupting flights',
                    'severity': EventSeverity.CRITICAL,
                    'duration_range': (3, 21),
                    'impacts': [
                        EventImpact(EventImpactType.CAPACITY_REDUCTION, -0.90, duration_days=14),
                        EventImpact(EventImpactType.VOLATILITY_INCREASE, 0.50, duration_days=21)
                    ]
                }
            ],
            
            EventType.PANDEMIC: [
                {
                    'name': 'Global Pandemic',
                    'description': 'Worldwide health crisis restricting travel',
                    'severity': EventSeverity.CRITICAL,
                    'duration_range': (365, 1095),
                    'impacts': [
                        EventImpact(EventImpactType.DEMAND_DECREASE, -0.70, duration_days=730),
                        EventImpact(EventImpactType.CAPACITY_REDUCTION, -0.60, duration_days=365),
                        EventImpact(EventImpactType.REGULATORY_CONSTRAINT, 0.80, duration_days=730)
                    ]
                },
                {
                    'name': 'Regional Health Crisis',
                    'description': 'Regional disease outbreak affecting travel',
                    'severity': EventSeverity.HIGH,
                    'duration_range': (30, 180),
                    'impacts': [
                        EventImpact(EventImpactType.DEMAND_DECREASE, -0.40, duration_days=120),
                        EventImpact(EventImpactType.VOLATILITY_INCREASE, 0.30, duration_days=180)
                    ]
                }
            ],
            
            EventType.FUEL_PRICE: [
                {
                    'name': 'Oil Price Spike',
                    'description': 'Sharp increase in fuel costs',
                    'severity': EventSeverity.HIGH,
                    'duration_range': (30, 365),
                    'impacts': [
                        EventImpact(EventImpactType.COST_INCREASE, 0.40, duration_days=180),
                        EventImpact(EventImpactType.PRICE_PRESSURE_UP, 0.15, duration_days=180)
                    ]
                },
                {
                    'name': 'Oil Price Collapse',
                    'description': 'Sharp decrease in fuel costs',
                    'severity': EventSeverity.MEDIUM,
                    'duration_range': (90, 365),
                    'impacts': [
                        EventImpact(EventImpactType.COST_DECREASE, -0.30, duration_days=270),
                        EventImpact(EventImpactType.PRICE_PRESSURE_DOWN, -0.10, duration_days=270)
                    ]
                }
            ],
            
            EventType.REGULATORY: [
                {
                    'name': 'New Environmental Regulations',
                    'description': 'Stricter emissions standards increasing costs',
                    'severity': EventSeverity.MEDIUM,
                    'duration_range': (365, 1825),
                    'impacts': [
                        EventImpact(EventImpactType.COST_INCREASE, 0.08, duration_days=1095),
                        EventImpact(EventImpactType.REGULATORY_CONSTRAINT, 0.20, duration_days=1095)
                    ]
                },
                {
                    'name': 'Deregulation',
                    'description': 'Market deregulation increasing competition',
                    'severity': EventSeverity.MEDIUM,
                    'duration_range': (730, 3650),
                    'impacts': [
                        EventImpact(EventImpactType.PRICE_PRESSURE_DOWN, -0.20, duration_days=1825),
                        EventImpact(EventImpactType.CAPACITY_INCREASE, 0.25, duration_days=1825)
                    ]
                },
                {
                    'name': 'Security Regulations Tightening',
                    'description': 'Enhanced security measures increasing costs',
                    'severity': EventSeverity.MEDIUM,
                    'duration_range': (180, 1095),
                    'impacts': [
                        EventImpact(EventImpactType.COST_INCREASE, 0.05, duration_days=730),
                        EventImpact(EventImpactType.DEMAND_DECREASE, -0.05, duration_days=365)
                    ]
                }
            ],
            
            EventType.TECHNOLOGY: [
                {
                    'name': 'Virtual Meeting Technology Adoption',
                    'description': 'Widespread adoption of virtual meetings reducing business travel',
                    'severity': EventSeverity.HIGH,
                    'duration_range': (365, 1825),
                    'impacts': [
                        EventImpact(EventImpactType.DEMAND_DECREASE, -0.25, affected_segments=['business'], duration_days=1095)
                    ]
                },
                {
                    'name': 'New Aircraft Technology',
                    'description': 'More efficient aircraft reducing operating costs',
                    'severity': EventSeverity.MEDIUM,
                    'duration_range': (1095, 3650),
                    'impacts': [
                        EventImpact(EventImpactType.COST_DECREASE, -0.15, duration_days=2190),
                        EventImpact(EventImpactType.CAPACITY_INCREASE, 0.10, duration_days=2190)
                    ]
                }
            ],
            
            EventType.SEASONAL: [
                {
                    'name': 'Major Sporting Event',
                    'description': 'Olympics, World Cup, or similar major event',
                    'severity': EventSeverity.MEDIUM,
                    'duration_range': (14, 30),
                    'impacts': [
                        EventImpact(EventImpactType.DEMAND_INCREASE, 0.80, affected_segments=['leisure'], duration_days=21),
                        EventImpact(EventImpactType.PRICE_PRESSURE_UP, 0.30, duration_days=21)
                    ]
                },
                {
                    'name': 'Business Conference Season',
                    'description': 'Peak conference and convention period',
                    'severity': EventSeverity.LOW,
                    'duration_range': (30, 90),
                    'impacts': [
                        EventImpact(EventImpactType.DEMAND_INCREASE, 0.20, affected_segments=['business'], duration_days=60)
                    ]
                },
                {
                    'name': 'Holiday Travel Peak',
                    'description': 'Major holiday period with high leisure travel',
                    'severity': EventSeverity.MEDIUM,
                    'duration_range': (7, 21),
                    'impacts': [
                        EventImpact(EventImpactType.DEMAND_INCREASE, 0.50, affected_segments=['leisure', 'vfr'], duration_days=14),
                        EventImpact(EventImpactType.PRICE_PRESSURE_UP, 0.20, duration_days=14)
                    ]
                }
            ],
            
            EventType.COMPETITIVE: [
                {
                    'name': 'New Low-Cost Carrier Entry',
                    'description': 'New budget airline entering market',
                    'severity': EventSeverity.MEDIUM,
                    'duration_range': (180, 1095),
                    'impacts': [
                        EventImpact(EventImpactType.PRICE_PRESSURE_DOWN, -0.15, duration_days=730),
                        EventImpact(EventImpactType.CAPACITY_INCREASE, 0.20, duration_days=730)
                    ]
                },
                {
                    'name': 'Airline Merger',
                    'description': 'Major airline consolidation',
                    'severity': EventSeverity.HIGH,
                    'duration_range': (365, 1095),
                    'impacts': [
                        EventImpact(EventImpactType.CAPACITY_REDUCTION, -0.10, duration_days=730),
                        EventImpact(EventImpactType.PRICE_PRESSURE_UP, 0.08, duration_days=730)
                    ]
                },
                {
                    'name': 'Airline Bankruptcy',
                    'description': 'Major carrier filing for bankruptcy',
                    'severity': EventSeverity.HIGH,
                    'duration_range': (90, 365),
                    'impacts': [
                        EventImpact(EventImpactType.CAPACITY_REDUCTION, -0.15, duration_days=180),
                        EventImpact(EventImpactType.VOLATILITY_INCREASE, 0.30, duration_days=365)
                    ]
                }
            ],
            
            EventType.OPERATIONAL: [
                {
                    'name': 'Pilot Strike',
                    'description': 'Airline pilot labor strike',
                    'severity': EventSeverity.HIGH,
                    'duration_range': (3, 30),
                    'impacts': [
                        EventImpact(EventImpactType.CAPACITY_REDUCTION, -0.60, duration_days=14),
                        EventImpact(EventImpactType.COST_INCREASE, 0.25, duration_days=30)
                    ]
                },
                {
                    'name': 'Air Traffic Control System Failure',
                    'description': 'Major ATC system outage',
                    'severity': EventSeverity.CRITICAL,
                    'duration_range': (1, 7),
                    'impacts': [
                        EventImpact(EventImpactType.CAPACITY_REDUCTION, -0.80, duration_days=3),
                        EventImpact(EventImpactType.VOLATILITY_INCREASE, 0.60, duration_days=7)
                    ]
                },
                {
                    'name': 'Cybersecurity Incident',
                    'description': 'Major cyber attack on airline systems',
                    'severity': EventSeverity.HIGH,
                    'duration_range': (1, 14),
                    'impacts': [
                        EventImpact(EventImpactType.CAPACITY_REDUCTION, -0.40, duration_days=7),
                        EventImpact(EventImpactType.COST_INCREASE, 0.15, duration_days=14)
                    ]
                }
            ]
        }
        
        return templates
    
    def _initialize_event_probabilities(self) -> Dict[EventType, float]:
        """Initialize daily probabilities for different event types."""
        return {
            EventType.ECONOMIC: 0.001,  # 0.1% daily chance
            EventType.GEOPOLITICAL: 0.0005,
            EventType.NATURAL_DISASTER: 0.002,
            EventType.PANDEMIC: 0.0001,
            EventType.FUEL_PRICE: 0.003,
            EventType.REGULATORY: 0.0008,
            EventType.TECHNOLOGY: 0.0003,
            EventType.SEASONAL: 0.01,  # More frequent
            EventType.COMPETITIVE: 0.002,
            EventType.OPERATIONAL: 0.005
        }
    
    def simulate_daily_events(self, current_date: datetime) -> List[ExternalEvent]:
        """Simulate events that might occur on a given day."""
        new_events = []
        
        # Check for scheduled events
        scheduled_today = [
            event for event in self.scheduled_events
            if event.start_date.date() == current_date.date()
        ]
        
        for event in scheduled_today:
            self.activate_event(event)
            new_events.append(event)
            self.scheduled_events.remove(event)
        
        # Generate random events based on probabilities
        for event_type, probability in self.event_probabilities.items():
            if random.random() < probability:
                event = self._generate_random_event(event_type, current_date)
                if event:
                    self.activate_event(event)
                    new_events.append(event)
        
        # Update active events
        self._update_active_events(current_date)
        
        return new_events
    
    def _generate_random_event(self, event_type: EventType, start_date: datetime) -> Optional[ExternalEvent]:
        """Generate a random event of the specified type."""
        templates = self.event_templates.get(event_type, [])
        if not templates:
            return None
        
        template = random.choice(templates)
        
        # Generate duration
        min_duration, max_duration = template['duration_range']
        duration = random.randint(min_duration, max_duration)
        end_date = start_date + timedelta(days=duration)
        
        # Generate event ID
        event_id = f"{event_type.value}_{start_date.strftime('%Y%m%d')}_{random.randint(1000, 9999)}"
        
        # Create impacts with some randomization
        impacts = []
        for impact_template in template['impacts']:
            # Add some randomness to magnitude
            magnitude = impact_template.magnitude * random.uniform(0.7, 1.3)
            
            impact = EventImpact(
                impact_type=impact_template.impact_type,
                magnitude=magnitude,
                affected_routes=impact_template.affected_routes.copy(),
                affected_segments=impact_template.affected_segments.copy(),
                duration_days=impact_template.duration_days,
                decay_rate=impact_template.decay_rate,
                confidence=random.uniform(0.6, 0.9)
            )
            impacts.append(impact)
        
        # Determine geographic scope
        affected_countries, affected_cities, global_impact = self._determine_geographic_scope(event_type)
        
        event = ExternalEvent(
            event_id=event_id,
            event_type=event_type,
            severity=template['severity'],
            name=template['name'],
            description=template['description'],
            start_date=start_date,
            end_date=end_date,
            affected_countries=affected_countries,
            affected_cities=affected_cities,
            global_impact=global_impact,
            impacts=impacts,
            probability=1.0,
            uncertainty_factor=random.uniform(0.05, 0.25),
            source="random_generation"
        )
        
        return event
    
    def _determine_geographic_scope(self, event_type: EventType) -> Tuple[List[str], List[str], bool]:
        """Determine geographic scope of an event."""
        # Sample countries and cities (in a real implementation, this would be more comprehensive)
        all_countries = ['US', 'UK', 'DE', 'FR', 'JP', 'CN', 'CA', 'AU', 'BR', 'IN']
        all_cities = ['NYC', 'LON', 'PAR', 'TKY', 'BER', 'SYD', 'TOR', 'SAO', 'DEL', 'PEK']
        
        if event_type in [EventType.PANDEMIC, EventType.ECONOMIC, EventType.FUEL_PRICE]:
            # Global events
            return all_countries.copy(), all_cities.copy(), True
        
        elif event_type in [EventType.GEOPOLITICAL, EventType.REGULATORY]:
            # Regional events
            num_countries = random.randint(2, 5)
            affected_countries = random.sample(all_countries, num_countries)
            affected_cities = [city for city in all_cities if random.random() < 0.3]
            return affected_countries, affected_cities, False
        
        elif event_type in [EventType.NATURAL_DISASTER, EventType.OPERATIONAL]:
            # Local events
            num_countries = random.randint(1, 2)
            affected_countries = random.sample(all_countries, num_countries)
            affected_cities = random.sample(all_cities, random.randint(1, 3))
            return affected_countries, affected_cities, False
        
        else:
            # Default to regional
            num_countries = random.randint(1, 3)
            affected_countries = random.sample(all_countries, num_countries)
            affected_cities = random.sample(all_cities, random.randint(1, 4))
            return affected_countries, affected_cities, False
    
    def activate_event(self, event: ExternalEvent):
        """Activate an event and apply its impacts."""
        self.active_events[event.event_id] = event
        
        # Apply impacts to current state
        for impact in event.impacts:
            self.current_impacts[event.event_id].append(impact)
        
        # Update economic indicators
        self._update_economic_indicators(event)
        
        self.logger.info(
            f"Activated event: {event.name} ({event.event_type.value}) "
            f"from {event.start_date.date()} to {event.end_date.date() if event.end_date else 'ongoing'}"
        )
    
    def _update_active_events(self, current_date: datetime):
        """Update active events and remove expired ones."""
        expired_events = []
        
        for event_id, event in self.active_events.items():
            if event.end_date and current_date > event.end_date:
                expired_events.append(event_id)
        
        for event_id in expired_events:
            event = self.active_events.pop(event_id)
            self.historical_events.append(event)
            
            # Remove impacts
            if event_id in self.current_impacts:
                del self.current_impacts[event_id]
            
            self.logger.info(f"Event expired: {event.name} ({event.event_id})")
    
    def _update_economic_indicators(self, event: ExternalEvent):
        """Update economic indicators based on event impacts."""
        for impact in event.impacts:
            if impact.impact_type == EventImpactType.COST_INCREASE:
                if event.event_type == EventType.FUEL_PRICE:
                    self.economic_state['fuel_price_index'] *= (1 + impact.magnitude)
                else:
                    # General cost increase affects inflation
                    self.economic_state['inflation_rate'] *= (1 + impact.magnitude * 0.1)
            
            elif impact.impact_type == EventImpactType.DEMAND_DECREASE:
                if event.severity in [EventSeverity.HIGH, EventSeverity.CRITICAL]:
                    self.economic_state['consumer_confidence'] *= (1 + impact.magnitude * 0.5)
                    self.economic_state['unemployment_rate'] *= (1 - impact.magnitude * 0.2)
            
            elif impact.impact_type == EventImpactType.VOLATILITY_INCREASE:
                self.economic_state['exchange_rate_volatility'] *= (1 + impact.magnitude)
    
    def get_current_impacts(
        self,
        route: Optional[str] = None,
        segment: Optional[str] = None,
        impact_type: Optional[EventImpactType] = None
    ) -> List[Tuple[ExternalEvent, EventImpact]]:
        """Get current impacts affecting the simulation."""
        current_impacts = []
        
        for event_id, impacts in self.current_impacts.items():
            event = self.active_events[event_id]
            
            for impact in impacts:
                # Filter by criteria
                if route and impact.affected_routes and route not in impact.affected_routes:
                    continue
                
                if segment and impact.affected_segments and segment not in impact.affected_segments:
                    continue
                
                if impact_type and impact.impact_type != impact_type:
                    continue
                
                current_impacts.append((event, impact))
        
        return current_impacts
    
    def calculate_demand_adjustment(
        self,
        base_demand: float,
        route: str,
        segment: str = 'all'
    ) -> Tuple[float, List[str]]:
        """Calculate demand adjustment based on current events."""
        adjusted_demand = base_demand
        affecting_events = []
        
        # Get relevant impacts
        demand_impacts = self.get_current_impacts(
            route=route,
            segment=segment if segment != 'all' else None
        )
        
        for event, impact in demand_impacts:
            if impact.impact_type == EventImpactType.DEMAND_INCREASE:
                adjusted_demand *= (1 + impact.magnitude)
                affecting_events.append(f"{event.name}: +{impact.magnitude:.1%}")
            
            elif impact.impact_type == EventImpactType.DEMAND_DECREASE:
                adjusted_demand *= (1 + impact.magnitude)  # magnitude is negative
                affecting_events.append(f"{event.name}: {impact.magnitude:.1%}")
        
        return adjusted_demand, affecting_events
    
    def calculate_cost_adjustment(
        self,
        base_cost: float,
        route: str
    ) -> Tuple[float, List[str]]:
        """Calculate cost adjustment based on current events."""
        adjusted_cost = base_cost
        affecting_events = []
        
        # Get relevant impacts
        cost_impacts = self.get_current_impacts(route=route)
        
        for event, impact in cost_impacts:
            if impact.impact_type == EventImpactType.COST_INCREASE:
                adjusted_cost *= (1 + impact.magnitude)
                affecting_events.append(f"{event.name}: +{impact.magnitude:.1%}")
            
            elif impact.impact_type == EventImpactType.COST_DECREASE:
                adjusted_cost *= (1 + impact.magnitude)  # magnitude is negative
                affecting_events.append(f"{event.name}: {impact.magnitude:.1%}")
        
        return adjusted_cost, affecting_events
    
    def calculate_capacity_adjustment(
        self,
        base_capacity: int,
        route: str
    ) -> Tuple[int, List[str]]:
        """Calculate capacity adjustment based on current events."""
        adjusted_capacity = base_capacity
        affecting_events = []
        
        # Get relevant impacts
        capacity_impacts = self.get_current_impacts(route=route)
        
        for event, impact in capacity_impacts:
            if impact.impact_type == EventImpactType.CAPACITY_REDUCTION:
                adjusted_capacity = int(adjusted_capacity * (1 + impact.magnitude))  # magnitude is negative
                affecting_events.append(f"{event.name}: {impact.magnitude:.1%}")
            
            elif impact.impact_type == EventImpactType.CAPACITY_INCREASE:
                adjusted_capacity = int(adjusted_capacity * (1 + impact.magnitude))
                affecting_events.append(f"{event.name}: +{impact.magnitude:.1%}")
        
        return max(0, adjusted_capacity), affecting_events
    
    def calculate_price_pressure(
        self,
        route: str
    ) -> Tuple[float, List[str]]:
        """Calculate price pressure from current events."""
        price_pressure = 0.0
        affecting_events = []
        
        # Get relevant impacts
        price_impacts = self.get_current_impacts(route=route)
        
        for event, impact in price_impacts:
            if impact.impact_type == EventImpactType.PRICE_PRESSURE_UP:
                price_pressure += impact.magnitude
                affecting_events.append(f"{event.name}: +{impact.magnitude:.1%}")
            
            elif impact.impact_type == EventImpactType.PRICE_PRESSURE_DOWN:
                price_pressure += impact.magnitude  # magnitude is negative
                affecting_events.append(f"{event.name}: {impact.magnitude:.1%}")
        
        return price_pressure, affecting_events
    
    def schedule_event(self, event: ExternalEvent):
        """Schedule an event for future activation."""
        self.scheduled_events.append(event)
        self.scheduled_events.sort(key=lambda x: x.start_date)
        
        self.logger.info(
            f"Scheduled event: {event.name} for {event.start_date.date()}"
        )
    
    def create_custom_event(
        self,
        name: str,
        description: str,
        event_type: EventType,
        severity: EventSeverity,
        start_date: datetime,
        duration_days: int,
        impacts: List[EventImpact],
        affected_countries: Optional[List[str]] = None,
        affected_cities: Optional[List[str]] = None
    ) -> ExternalEvent:
        """Create a custom event."""
        event_id = f"custom_{event_type.value}_{start_date.strftime('%Y%m%d')}_{random.randint(1000, 9999)}"
        end_date = start_date + timedelta(days=duration_days)
        
        event = ExternalEvent(
            event_id=event_id,
            event_type=event_type,
            severity=severity,
            name=name,
            description=description,
            start_date=start_date,
            end_date=end_date,
            affected_countries=affected_countries or [],
            affected_cities=affected_cities or [],
            global_impact=len(affected_countries or []) > 5,
            impacts=impacts,
            source="custom"
        )
        
        return event
    
    def get_event_forecast(
        self,
        forecast_days: int = 30
    ) -> Dict[str, Any]:
        """Get forecast of potential events in the coming period."""
        forecast_end = datetime.now() + timedelta(days=forecast_days)
        
        # Scheduled events
        scheduled_in_period = [
            event for event in self.scheduled_events
            if event.start_date <= forecast_end
        ]
        
        # Estimate probability of random events
        expected_events = {}
        for event_type, daily_prob in self.event_probabilities.items():
            expected_count = daily_prob * forecast_days
            expected_events[event_type.value] = {
                'expected_count': expected_count,
                'probability_at_least_one': 1 - (1 - daily_prob) ** forecast_days
            }
        
        return {
            'forecast_period_days': forecast_days,
            'scheduled_events': [
                {
                    'name': event.name,
                    'type': event.event_type.value,
                    'severity': event.severity.value,
                    'start_date': event.start_date.isoformat(),
                    'duration_days': (event.end_date - event.start_date).days if event.end_date else None
                }
                for event in scheduled_in_period
            ],
            'expected_random_events': expected_events,
            'current_active_events': len(self.active_events),
            'risk_assessment': self._assess_event_risk(forecast_days)
        }
    
    def _assess_event_risk(self, forecast_days: int) -> Dict[str, str]:
        """Assess overall event risk for the forecast period."""
        # Calculate total expected disruptive events
        high_impact_types = [EventType.PANDEMIC, EventType.GEOPOLITICAL, EventType.NATURAL_DISASTER]
        total_high_impact_prob = sum(
            1 - (1 - self.event_probabilities[event_type]) ** forecast_days
            for event_type in high_impact_types
        )
        
        if total_high_impact_prob > 0.5:
            risk_level = "HIGH"
        elif total_high_impact_prob > 0.2:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            'overall_risk': risk_level,
            'high_impact_probability': f"{total_high_impact_prob:.1%}",
            'recommendation': self._get_risk_recommendation(risk_level)
        }
    
    def _get_risk_recommendation(self, risk_level: str) -> str:
        """Get risk management recommendation."""
        if risk_level == "HIGH":
            return "Consider conservative pricing strategies and maintain higher cash reserves"
        elif risk_level == "MEDIUM":
            return "Monitor market conditions closely and prepare contingency plans"
        else:
            return "Normal operations with standard risk management practices"
    
    def get_economic_state(self) -> Dict[str, float]:
        """Get current economic state affected by events."""
        return self.economic_state.copy()
    
    def get_active_events_summary(self) -> List[Dict[str, Any]]:
        """Get summary of currently active events."""
        summaries = []
        
        for event in self.active_events.values():
            days_remaining = (event.end_date - datetime.now()).days if event.end_date else None
            
            summary = {
                'event_id': event.event_id,
                'name': event.name,
                'type': event.event_type.value,
                'severity': event.severity.value,
                'start_date': event.start_date.isoformat(),
                'days_remaining': days_remaining,
                'affected_countries': event.affected_countries,
                'affected_cities': event.affected_cities,
                'global_impact': event.global_impact,
                'impact_count': len(event.impacts),
                'description': event.description
            }
            
            summaries.append(summary)
        
        return summaries
    
    def export_event_history(self, days_back: int = 365) -> pd.DataFrame:
        """Export event history as a DataFrame."""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        events_data = []
        for event in self.historical_events:
            if event.start_date >= cutoff_date:
                events_data.append({
                    'event_id': event.event_id,
                    'name': event.name,
                    'type': event.event_type.value,
                    'severity': event.severity.value,
                    'start_date': event.start_date,
                    'end_date': event.end_date,
                    'duration_days': (event.end_date - event.start_date).days if event.end_date else None,
                    'global_impact': event.global_impact,
                    'affected_countries_count': len(event.affected_countries),
                    'affected_cities_count': len(event.affected_cities),
                    'impact_count': len(event.impacts),
                    'source': event.source
                })
        
        return pd.DataFrame(events_data)