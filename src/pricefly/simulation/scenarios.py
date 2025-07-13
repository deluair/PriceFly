"""Scenario management for airline pricing simulations."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from pathlib import Path

from .events import ExternalEvent, EventType, EventSeverity, EventImpact, EventImpactType


class ScenarioType(Enum):
    """Types of simulation scenarios."""
    BASELINE = "baseline"  # Normal market conditions
    STRESS_TEST = "stress_test"  # Adverse conditions
    GROWTH = "growth"  # Favorable conditions
    CRISIS = "crisis"  # Major disruption
    COMPETITIVE = "competitive"  # High competition
    HIGH_COMPETITION = "high_competition"  # Intense competition scenario
    REGULATORY = "regulatory"  # Regulatory changes
    TECHNOLOGY = "technology"  # Technology disruption
    CUSTOM = "custom"  # User-defined scenario


class MarketCondition(Enum):
    """Overall market conditions."""
    RECESSION = "recession"
    SLOW_GROWTH = "slow_growth"
    NORMAL = "normal"
    STRONG_GROWTH = "strong_growth"
    BOOM = "boom"


@dataclass
class EconomicParameters:
    """Economic parameters for scenarios."""
    gdp_growth_rate: float = 0.025  # Annual GDP growth
    inflation_rate: float = 0.02  # Annual inflation
    unemployment_rate: float = 0.05  # Unemployment rate
    interest_rate: float = 0.03  # Interest rate
    consumer_confidence: float = 0.8  # Consumer confidence index (0-1)
    fuel_price_volatility: float = 0.15  # Fuel price volatility
    exchange_rate_volatility: float = 0.10  # Exchange rate volatility
    
    # Seasonal factors
    seasonal_amplitude: float = 0.20  # Seasonal demand variation
    business_travel_factor: float = 1.0  # Business travel multiplier
    leisure_travel_factor: float = 1.0  # Leisure travel multiplier


@dataclass
class CompetitiveParameters:
    """Competitive environment parameters."""
    market_concentration: float = 0.6  # HHI-like measure (0-1)
    price_competition_intensity: float = 0.5  # Price competition level (0-1)
    capacity_competition_intensity: float = 0.3  # Capacity competition level (0-1)
    new_entrant_probability: float = 0.1  # Probability of new entrants
    merger_probability: float = 0.05  # Probability of mergers
    
    # Competitive response parameters
    price_response_speed: float = 0.7  # How quickly competitors respond to price changes
    capacity_response_speed: float = 0.3  # How quickly competitors adjust capacity
    competitive_intelligence_quality: float = 0.8  # Quality of competitive intelligence


@dataclass
class OperationalParameters:
    """Operational environment parameters."""
    fuel_cost_base: float = 0.30  # Base fuel cost as % of operating cost
    labor_cost_base: float = 0.31  # Base labor cost as % of operating cost
    airport_fee_inflation: float = 0.03  # Annual airport fee inflation
    maintenance_cost_inflation: float = 0.025  # Annual maintenance cost inflation
    
    # Operational efficiency
    load_factor_target: float = 0.82  # Target load factor
    on_time_performance: float = 0.85  # On-time performance rate
    cancellation_rate: float = 0.02  # Flight cancellation rate
    
    # Technology adoption
    dynamic_pricing_adoption: float = 0.8  # Dynamic pricing adoption rate
    ai_optimization_level: float = 0.6  # AI optimization sophistication


@dataclass
class RegulatoryParameters:
    """Regulatory environment parameters."""
    environmental_regulations_strictness: float = 0.5  # Environmental regulation level (0-1)
    consumer_protection_level: float = 0.7  # Consumer protection strength (0-1)
    market_liberalization: float = 0.8  # Market openness (0-1)
    
    # Specific regulations
    carbon_tax_rate: float = 0.0  # Carbon tax per ton CO2
    passenger_rights_compensation: float = 600.0  # Max compensation for delays/cancellations
    price_transparency_requirements: bool = True  # Price transparency requirements
    
    # International regulations
    bilateral_agreement_openness: float = 0.7  # Openness of bilateral agreements
    visa_restrictions_level: float = 0.3  # Level of visa restrictions


@dataclass
class TechnologyParameters:
    """Technology environment parameters."""
    virtual_meeting_adoption: float = 0.6  # Virtual meeting technology adoption
    mobile_booking_penetration: float = 0.85  # Mobile booking penetration
    ai_personalization_level: float = 0.5  # AI personalization sophistication
    
    # Distribution technology
    ndc_adoption_rate: float = 0.4  # NDC adoption rate
    direct_booking_preference: float = 0.6  # Direct booking preference
    ota_market_share: float = 0.35  # Online travel agency market share
    
    # Operational technology
    predictive_maintenance_adoption: float = 0.7  # Predictive maintenance adoption
    automated_operations_level: float = 0.5  # Automation level in operations


@dataclass
class ScenarioConfiguration:
    """Complete scenario configuration."""
    scenario_id: str
    name: str
    description: str
    scenario_type: ScenarioType
    
    # Time parameters
    start_date: datetime
    end_date: datetime
    
    # Environment parameters
    economic_params: EconomicParameters = field(default_factory=EconomicParameters)
    competitive_params: CompetitiveParameters = field(default_factory=CompetitiveParameters)
    operational_params: OperationalParameters = field(default_factory=OperationalParameters)
    regulatory_params: RegulatoryParameters = field(default_factory=RegulatoryParameters)
    technology_params: TechnologyParameters = field(default_factory=TechnologyParameters)
    
    # Events
    scheduled_events: List[ExternalEvent] = field(default_factory=list)
    event_probability_multipliers: Dict[EventType, float] = field(default_factory=dict)
    
    # Market conditions
    market_condition: MarketCondition = MarketCondition.NORMAL
    
    # Simulation parameters
    random_seed: Optional[int] = None
    monte_carlo_runs: int = 1
    
    # Custom parameters
    custom_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"
    tags: List[str] = field(default_factory=list)


# Alias for backward compatibility
SimulationScenario = ScenarioConfiguration


class ScenarioManager:
    """Manages simulation scenarios and configurations."""
    
    def __init__(self, scenarios_dir: Optional[Path] = None):
        self.scenarios_dir = scenarios_dir or Path("scenarios")
        self.scenarios_dir.mkdir(exist_ok=True)
        
        # Built-in scenarios
        self.built_in_scenarios: Dict[str, ScenarioConfiguration] = {}
        self._initialize_built_in_scenarios()
        
        # Custom scenarios
        self.custom_scenarios: Dict[str, ScenarioConfiguration] = {}
        self._load_custom_scenarios()
        
        self.logger = logging.getLogger(__name__)
    
    def _initialize_built_in_scenarios(self):
        """Initialize built-in scenario templates."""
        base_start = datetime(2024, 1, 1)
        base_end = datetime(2024, 12, 31)
        
        # Baseline scenario
        self.built_in_scenarios["baseline"] = ScenarioConfiguration(
            scenario_id="baseline",
            name="Baseline Scenario",
            description="Normal market conditions with typical seasonal patterns",
            scenario_type=ScenarioType.BASELINE,
            start_date=base_start,
            end_date=base_end,
            market_condition=MarketCondition.NORMAL
        )
        
        # Economic recession scenario
        recession_economic = EconomicParameters(
            gdp_growth_rate=-0.02,
            inflation_rate=0.01,
            unemployment_rate=0.08,
            consumer_confidence=0.4,
            business_travel_factor=0.7,
            leisure_travel_factor=0.6
        )
        
        recession_events = [
            self._create_recession_event(base_start + timedelta(days=30))
        ]
        
        self.built_in_scenarios["recession"] = ScenarioConfiguration(
            scenario_id="recession",
            name="Economic Recession",
            description="Economic downturn with reduced travel demand",
            scenario_type=ScenarioType.STRESS_TEST,
            start_date=base_start,
            end_date=base_end,
            economic_params=recession_economic,
            scheduled_events=recession_events,
            market_condition=MarketCondition.RECESSION,
            event_probability_multipliers={
                EventType.ECONOMIC: 2.0,
                EventType.COMPETITIVE: 1.5
            }
        )
        
        # High growth scenario
        growth_economic = EconomicParameters(
            gdp_growth_rate=0.06,
            inflation_rate=0.025,
            unemployment_rate=0.03,
            consumer_confidence=0.9,
            business_travel_factor=1.3,
            leisure_travel_factor=1.4
        )
        
        self.built_in_scenarios["high_growth"] = ScenarioConfiguration(
            scenario_id="high_growth",
            name="High Growth Economy",
            description="Strong economic growth driving increased travel demand",
            scenario_type=ScenarioType.GROWTH,
            start_date=base_start,
            end_date=base_end,
            economic_params=growth_economic,
            market_condition=MarketCondition.STRONG_GROWTH
        )
        
        # Pandemic crisis scenario
        pandemic_economic = EconomicParameters(
            gdp_growth_rate=-0.05,
            unemployment_rate=0.12,
            consumer_confidence=0.3,
            business_travel_factor=0.2,
            leisure_travel_factor=0.1
        )
        
        pandemic_regulatory = RegulatoryParameters(
            consumer_protection_level=0.9,
            market_liberalization=0.5,
            visa_restrictions_level=0.8
        )
        
        pandemic_events = [
            self._create_pandemic_event(base_start + timedelta(days=60))
        ]
        
        self.built_in_scenarios["pandemic"] = ScenarioConfiguration(
            scenario_id="pandemic",
            name="Global Pandemic",
            description="Global health crisis severely impacting travel",
            scenario_type=ScenarioType.CRISIS,
            start_date=base_start,
            end_date=base_end,
            economic_params=pandemic_economic,
            regulatory_params=pandemic_regulatory,
            scheduled_events=pandemic_events,
            market_condition=MarketCondition.RECESSION,
            event_probability_multipliers={
                EventType.PANDEMIC: 5.0,
                EventType.REGULATORY: 3.0,
                EventType.OPERATIONAL: 2.0
            }
        )
        
        # High competition scenario
        competitive_params = CompetitiveParameters(
            market_concentration=0.4,
            price_competition_intensity=0.9,
            capacity_competition_intensity=0.7,
            new_entrant_probability=0.3,
            price_response_speed=0.9
        )
        
        competitive_events = [
            self._create_new_entrant_event(base_start + timedelta(days=90)),
            self._create_price_war_event(base_start + timedelta(days=180))
        ]
        
        self.built_in_scenarios["high_competition"] = ScenarioConfiguration(
            scenario_id="high_competition",
            name="Intense Competition",
            description="Highly competitive market with new entrants and price wars",
            scenario_type=ScenarioType.HIGH_COMPETITION,
            start_date=base_start,
            end_date=base_end,
            competitive_params=competitive_params,
            scheduled_events=competitive_events,
            event_probability_multipliers={
                EventType.COMPETITIVE: 3.0
            }
        )
        
        # Fuel price volatility scenario
        fuel_economic = EconomicParameters(
            fuel_price_volatility=0.40,
            exchange_rate_volatility=0.20
        )
        
        fuel_operational = OperationalParameters(
            fuel_cost_base=0.45
        )
        
        fuel_events = [
            self._create_oil_shock_event(base_start + timedelta(days=120)),
            self._create_oil_shock_event(base_start + timedelta(days=240))
        ]
        
        self.built_in_scenarios["fuel_volatility"] = ScenarioConfiguration(
            scenario_id="fuel_volatility",
            name="Fuel Price Volatility",
            description="High fuel price volatility affecting operating costs",
            scenario_type=ScenarioType.STRESS_TEST,
            start_date=base_start,
            end_date=base_end,
            economic_params=fuel_economic,
            operational_params=fuel_operational,
            scheduled_events=fuel_events,
            event_probability_multipliers={
                EventType.FUEL_PRICE: 4.0
            }
        )
        
        # Technology disruption scenario
        tech_params = TechnologyParameters(
            virtual_meeting_adoption=0.9,
            ai_personalization_level=0.9,
            ndc_adoption_rate=0.8,
            automated_operations_level=0.8
        )
        
        tech_economic = EconomicParameters(
            business_travel_factor=0.6  # Reduced business travel due to virtual meetings
        )
        
        tech_events = [
            self._create_tech_disruption_event(base_start + timedelta(days=150))
        ]
        
        self.built_in_scenarios["tech_disruption"] = ScenarioConfiguration(
            scenario_id="tech_disruption",
            name="Technology Disruption",
            description="Rapid technology adoption changing travel patterns",
            scenario_type=ScenarioType.TECHNOLOGY,
            start_date=base_start,
            end_date=base_end,
            economic_params=tech_economic,
            technology_params=tech_params,
            scheduled_events=tech_events,
            event_probability_multipliers={
                EventType.TECHNOLOGY: 3.0
            }
        )
        
        # Environmental regulations scenario
        env_regulatory = RegulatoryParameters(
            environmental_regulations_strictness=0.9,
            carbon_tax_rate=50.0,  # $50 per ton CO2
            consumer_protection_level=0.8
        )
        
        env_operational = OperationalParameters(
            fuel_cost_base=0.35  # Higher effective fuel costs due to carbon tax
        )
        
        env_events = [
            self._create_environmental_regulation_event(base_start + timedelta(days=100))
        ]
        
        self.built_in_scenarios["environmental_regulations"] = ScenarioConfiguration(
            scenario_id="environmental_regulations",
            name="Strict Environmental Regulations",
            description="New environmental regulations increasing operational costs",
            scenario_type=ScenarioType.REGULATORY,
            start_date=base_start,
            end_date=base_end,
            regulatory_params=env_regulatory,
            operational_params=env_operational,
            scheduled_events=env_events,
            event_probability_multipliers={
                EventType.REGULATORY: 2.0
            }
        )
    
    def _create_recession_event(self, start_date: datetime) -> ExternalEvent:
        """Create a recession event."""
        impacts = [
            EventImpact(EventImpactType.DEMAND_DECREASE, -0.25, duration_days=365),
            EventImpact(EventImpactType.PRICE_PRESSURE_DOWN, -0.15, duration_days=365),
            EventImpact(EventImpactType.VOLATILITY_INCREASE, 0.30, duration_days=365)
        ]
        
        return ExternalEvent(
            event_id=f"recession_{start_date.strftime('%Y%m%d')}",
            event_type=EventType.ECONOMIC,
            severity=EventSeverity.HIGH,
            name="Economic Recession",
            description="Major economic downturn affecting travel demand",
            start_date=start_date,
            end_date=start_date + timedelta(days=365),
            global_impact=True,
            impacts=impacts,
            source="scenario"
        )
    
    def _create_pandemic_event(self, start_date: datetime) -> ExternalEvent:
        """Create a pandemic event."""
        impacts = [
            EventImpact(EventImpactType.DEMAND_DECREASE, -0.70, duration_days=730),
            EventImpact(EventImpactType.CAPACITY_REDUCTION, -0.60, duration_days=365),
            EventImpact(EventImpactType.REGULATORY_CONSTRAINT, 0.80, duration_days=730),
            EventImpact(EventImpactType.VOLATILITY_INCREASE, 0.50, duration_days=730)
        ]
        
        return ExternalEvent(
            event_id=f"pandemic_{start_date.strftime('%Y%m%d')}",
            event_type=EventType.PANDEMIC,
            severity=EventSeverity.CRITICAL,
            name="Global Pandemic",
            description="Worldwide health crisis severely restricting travel",
            start_date=start_date,
            end_date=start_date + timedelta(days=730),
            global_impact=True,
            impacts=impacts,
            source="scenario"
        )
    
    def _create_new_entrant_event(self, start_date: datetime) -> ExternalEvent:
        """Create a new market entrant event."""
        impacts = [
            EventImpact(EventImpactType.PRICE_PRESSURE_DOWN, -0.15, duration_days=365),
            EventImpact(EventImpactType.CAPACITY_INCREASE, 0.20, duration_days=365),
            EventImpact(EventImpactType.VOLATILITY_INCREASE, 0.25, duration_days=180)
        ]
        
        return ExternalEvent(
            event_id=f"new_entrant_{start_date.strftime('%Y%m%d')}",
            event_type=EventType.COMPETITIVE,
            severity=EventSeverity.MEDIUM,
            name="New Low-Cost Carrier Entry",
            description="New budget airline entering major markets",
            start_date=start_date,
            end_date=start_date + timedelta(days=365),
            impacts=impacts,
            source="scenario"
        )
    
    def _create_price_war_event(self, start_date: datetime) -> ExternalEvent:
        """Create a price war event."""
        impacts = [
            EventImpact(EventImpactType.PRICE_PRESSURE_DOWN, -0.25, duration_days=120),
            EventImpact(EventImpactType.VOLATILITY_INCREASE, 0.40, duration_days=120)
        ]
        
        return ExternalEvent(
            event_id=f"price_war_{start_date.strftime('%Y%m%d')}",
            event_type=EventType.COMPETITIVE,
            severity=EventSeverity.HIGH,
            name="Industry Price War",
            description="Aggressive price competition across major routes",
            start_date=start_date,
            end_date=start_date + timedelta(days=120),
            impacts=impacts,
            source="scenario"
        )
    
    def _create_oil_shock_event(self, start_date: datetime) -> ExternalEvent:
        """Create an oil price shock event."""
        impacts = [
            EventImpact(EventImpactType.COST_INCREASE, 0.40, duration_days=180),
            EventImpact(EventImpactType.PRICE_PRESSURE_UP, 0.15, duration_days=180),
            EventImpact(EventImpactType.VOLATILITY_INCREASE, 0.30, duration_days=180)
        ]
        
        return ExternalEvent(
            event_id=f"oil_shock_{start_date.strftime('%Y%m%d')}",
            event_type=EventType.FUEL_PRICE,
            severity=EventSeverity.HIGH,
            name="Oil Price Shock",
            description="Sharp increase in crude oil and jet fuel prices",
            start_date=start_date,
            end_date=start_date + timedelta(days=180),
            global_impact=True,
            impacts=impacts,
            source="scenario"
        )
    
    def _create_tech_disruption_event(self, start_date: datetime) -> ExternalEvent:
        """Create a technology disruption event."""
        impacts = [
            EventImpact(
                EventImpactType.DEMAND_DECREASE, 
                -0.25, 
                affected_segments=['business'], 
                duration_days=1095
            ),
            EventImpact(EventImpactType.COST_DECREASE, -0.10, duration_days=1095)
        ]
        
        return ExternalEvent(
            event_id=f"tech_disruption_{start_date.strftime('%Y%m%d')}",
            event_type=EventType.TECHNOLOGY,
            severity=EventSeverity.MEDIUM,
            name="Virtual Meeting Technology Adoption",
            description="Widespread adoption of virtual meeting technology reducing business travel",
            start_date=start_date,
            end_date=start_date + timedelta(days=1095),
            global_impact=True,
            impacts=impacts,
            source="scenario"
        )
    
    def _create_environmental_regulation_event(self, start_date: datetime) -> ExternalEvent:
        """Create an environmental regulation event."""
        impacts = [
            EventImpact(EventImpactType.COST_INCREASE, 0.08, duration_days=1095),
            EventImpact(EventImpactType.REGULATORY_CONSTRAINT, 0.20, duration_days=1095)
        ]
        
        return ExternalEvent(
            event_id=f"env_regulation_{start_date.strftime('%Y%m%d')}",
            event_type=EventType.REGULATORY,
            severity=EventSeverity.MEDIUM,
            name="New Environmental Regulations",
            description="Stricter emissions standards and carbon pricing",
            start_date=start_date,
            end_date=start_date + timedelta(days=1095),
            global_impact=True,
            impacts=impacts,
            source="scenario"
        )
    
    def _load_custom_scenarios(self):
        """Load custom scenarios from files."""
        for scenario_file in self.scenarios_dir.glob("*.json"):
            try:
                with open(scenario_file, 'r') as f:
                    scenario_data = json.load(f)
                
                scenario = self._deserialize_scenario(scenario_data)
                self.custom_scenarios[scenario.scenario_id] = scenario
                
                self.logger.info(f"Loaded custom scenario: {scenario.name}")
                
            except Exception as e:
                self.logger.error(f"Failed to load scenario from {scenario_file}: {e}")
    
    def get_scenario(self, scenario_id: str) -> Optional[ScenarioConfiguration]:
        """Get a scenario by ID."""
        if scenario_id in self.built_in_scenarios:
            return self.built_in_scenarios[scenario_id]
        elif scenario_id in self.custom_scenarios:
            return self.custom_scenarios[scenario_id]
        else:
            return None
    
    def list_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """List all available scenarios."""
        scenarios = {}
        
        # Built-in scenarios
        for scenario_id, scenario in self.built_in_scenarios.items():
            scenarios[scenario_id] = {
                'name': scenario.name,
                'description': scenario.description,
                'type': scenario.scenario_type.value,
                'market_condition': scenario.market_condition.value,
                'source': 'built-in',
                'duration_days': (scenario.end_date - scenario.start_date).days,
                'scheduled_events': len(scenario.scheduled_events)
            }
        
        # Custom scenarios
        for scenario_id, scenario in self.custom_scenarios.items():
            scenarios[scenario_id] = {
                'name': scenario.name,
                'description': scenario.description,
                'type': scenario.scenario_type.value,
                'market_condition': scenario.market_condition.value,
                'source': 'custom',
                'duration_days': (scenario.end_date - scenario.start_date).days,
                'scheduled_events': len(scenario.scheduled_events),
                'created_at': scenario.created_at.isoformat(),
                'created_by': scenario.created_by
            }
        
        return scenarios
    
    def create_custom_scenario(
        self,
        scenario_id: str,
        name: str,
        description: str,
        scenario_type: ScenarioType,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> ScenarioConfiguration:
        """Create a custom scenario."""
        scenario = ScenarioConfiguration(
            scenario_id=scenario_id,
            name=name,
            description=description,
            scenario_type=scenario_type,
            start_date=start_date,
            end_date=end_date,
            **kwargs
        )
        
        self.custom_scenarios[scenario_id] = scenario
        return scenario
    
    def save_scenario(self, scenario: ScenarioConfiguration):
        """Save a scenario to file."""
        scenario_file = self.scenarios_dir / f"{scenario.scenario_id}.json"
        
        scenario_data = self._serialize_scenario(scenario)
        
        with open(scenario_file, 'w') as f:
            json.dump(scenario_data, f, indent=2, default=str)
        
        self.logger.info(f"Saved scenario: {scenario.name} to {scenario_file}")
    
    def _serialize_scenario(self, scenario: ScenarioConfiguration) -> Dict[str, Any]:
        """Serialize scenario to dictionary."""
        return {
            'scenario_id': scenario.scenario_id,
            'name': scenario.name,
            'description': scenario.description,
            'scenario_type': scenario.scenario_type.value,
            'start_date': scenario.start_date.isoformat(),
            'end_date': scenario.end_date.isoformat(),
            'market_condition': scenario.market_condition.value,
            'economic_params': {
                'gdp_growth_rate': scenario.economic_params.gdp_growth_rate,
                'inflation_rate': scenario.economic_params.inflation_rate,
                'unemployment_rate': scenario.economic_params.unemployment_rate,
                'interest_rate': scenario.economic_params.interest_rate,
                'consumer_confidence': scenario.economic_params.consumer_confidence,
                'fuel_price_volatility': scenario.economic_params.fuel_price_volatility,
                'exchange_rate_volatility': scenario.economic_params.exchange_rate_volatility,
                'seasonal_amplitude': scenario.economic_params.seasonal_amplitude,
                'business_travel_factor': scenario.economic_params.business_travel_factor,
                'leisure_travel_factor': scenario.economic_params.leisure_travel_factor
            },
            'competitive_params': {
                'market_concentration': scenario.competitive_params.market_concentration,
                'price_competition_intensity': scenario.competitive_params.price_competition_intensity,
                'capacity_competition_intensity': scenario.competitive_params.capacity_competition_intensity,
                'new_entrant_probability': scenario.competitive_params.new_entrant_probability,
                'merger_probability': scenario.competitive_params.merger_probability,
                'price_response_speed': scenario.competitive_params.price_response_speed,
                'capacity_response_speed': scenario.competitive_params.capacity_response_speed,
                'competitive_intelligence_quality': scenario.competitive_params.competitive_intelligence_quality
            },
            'operational_params': {
                'fuel_cost_base': scenario.operational_params.fuel_cost_base,
                'labor_cost_base': scenario.operational_params.labor_cost_base,
                'airport_fee_inflation': scenario.operational_params.airport_fee_inflation,
                'maintenance_cost_inflation': scenario.operational_params.maintenance_cost_inflation,
                'load_factor_target': scenario.operational_params.load_factor_target,
                'on_time_performance': scenario.operational_params.on_time_performance,
                'cancellation_rate': scenario.operational_params.cancellation_rate,
                'dynamic_pricing_adoption': scenario.operational_params.dynamic_pricing_adoption,
                'ai_optimization_level': scenario.operational_params.ai_optimization_level
            },
            'regulatory_params': {
                'environmental_regulations_strictness': scenario.regulatory_params.environmental_regulations_strictness,
                'consumer_protection_level': scenario.regulatory_params.consumer_protection_level,
                'market_liberalization': scenario.regulatory_params.market_liberalization,
                'carbon_tax_rate': scenario.regulatory_params.carbon_tax_rate,
                'passenger_rights_compensation': scenario.regulatory_params.passenger_rights_compensation,
                'price_transparency_requirements': scenario.regulatory_params.price_transparency_requirements,
                'bilateral_agreement_openness': scenario.regulatory_params.bilateral_agreement_openness,
                'visa_restrictions_level': scenario.regulatory_params.visa_restrictions_level
            },
            'technology_params': {
                'virtual_meeting_adoption': scenario.technology_params.virtual_meeting_adoption,
                'mobile_booking_penetration': scenario.technology_params.mobile_booking_penetration,
                'ai_personalization_level': scenario.technology_params.ai_personalization_level,
                'ndc_adoption_rate': scenario.technology_params.ndc_adoption_rate,
                'direct_booking_preference': scenario.technology_params.direct_booking_preference,
                'ota_market_share': scenario.technology_params.ota_market_share,
                'predictive_maintenance_adoption': scenario.technology_params.predictive_maintenance_adoption,
                'automated_operations_level': scenario.technology_params.automated_operations_level
            },
            'scheduled_events': [
                self._serialize_event(event) for event in scenario.scheduled_events
            ],
            'event_probability_multipliers': {
                event_type.value: multiplier 
                for event_type, multiplier in scenario.event_probability_multipliers.items()
            },
            'random_seed': scenario.random_seed,
            'monte_carlo_runs': scenario.monte_carlo_runs,
            'custom_parameters': scenario.custom_parameters,
            'created_at': scenario.created_at.isoformat(),
            'created_by': scenario.created_by,
            'tags': scenario.tags
        }
    
    def _serialize_event(self, event: ExternalEvent) -> Dict[str, Any]:
        """Serialize an event to dictionary."""
        return {
            'event_id': event.event_id,
            'event_type': event.event_type.value,
            'severity': event.severity.value,
            'name': event.name,
            'description': event.description,
            'start_date': event.start_date.isoformat(),
            'end_date': event.end_date.isoformat() if event.end_date else None,
            'affected_countries': event.affected_countries,
            'affected_cities': event.affected_cities,
            'global_impact': event.global_impact,
            'impacts': [
                {
                    'impact_type': impact.impact_type.value,
                    'magnitude': impact.magnitude,
                    'affected_routes': impact.affected_routes,
                    'affected_segments': impact.affected_segments,
                    'affected_metrics': impact.affected_metrics,
                    'duration_days': impact.duration_days,
                    'decay_rate': impact.decay_rate,
                    'confidence': impact.confidence
                }
                for impact in event.impacts
            ],
            'probability': event.probability,
            'uncertainty_factor': event.uncertainty_factor,
            'triggers': event.triggers,
            'consequences': event.consequences,
            'source': event.source,
            'tags': event.tags
        }
    
    def _deserialize_scenario(self, data: Dict[str, Any]) -> ScenarioConfiguration:
        """Deserialize scenario from dictionary."""
        # Parse dates
        start_date = datetime.fromisoformat(data['start_date'])
        end_date = datetime.fromisoformat(data['end_date'])
        created_at = datetime.fromisoformat(data.get('created_at', datetime.now().isoformat()))
        
        # Parse parameters
        economic_params = EconomicParameters(**data.get('economic_params', {}))
        competitive_params = CompetitiveParameters(**data.get('competitive_params', {}))
        operational_params = OperationalParameters(**data.get('operational_params', {}))
        regulatory_params = RegulatoryParameters(**data.get('regulatory_params', {}))
        technology_params = TechnologyParameters(**data.get('technology_params', {}))
        
        # Parse events
        scheduled_events = [
            self._deserialize_event(event_data)
            for event_data in data.get('scheduled_events', [])
        ]
        
        # Parse event probability multipliers
        event_probability_multipliers = {
            EventType(event_type): multiplier
            for event_type, multiplier in data.get('event_probability_multipliers', {}).items()
        }
        
        return ScenarioConfiguration(
            scenario_id=data['scenario_id'],
            name=data['name'],
            description=data['description'],
            scenario_type=ScenarioType(data['scenario_type']),
            start_date=start_date,
            end_date=end_date,
            economic_params=economic_params,
            competitive_params=competitive_params,
            operational_params=operational_params,
            regulatory_params=regulatory_params,
            technology_params=technology_params,
            scheduled_events=scheduled_events,
            event_probability_multipliers=event_probability_multipliers,
            market_condition=MarketCondition(data.get('market_condition', 'normal')),
            random_seed=data.get('random_seed'),
            monte_carlo_runs=data.get('monte_carlo_runs', 1),
            custom_parameters=data.get('custom_parameters', {}),
            created_at=created_at,
            created_by=data.get('created_by', 'unknown'),
            tags=data.get('tags', [])
        )
    
    def _deserialize_event(self, data: Dict[str, Any]) -> ExternalEvent:
        """Deserialize an event from dictionary."""
        start_date = datetime.fromisoformat(data['start_date'])
        end_date = datetime.fromisoformat(data['end_date']) if data.get('end_date') else None
        
        impacts = [
            EventImpact(
                impact_type=EventImpactType(impact_data['impact_type']),
                magnitude=impact_data['magnitude'],
                affected_routes=impact_data.get('affected_routes', []),
                affected_segments=impact_data.get('affected_segments', []),
                affected_metrics=impact_data.get('affected_metrics', []),
                duration_days=impact_data.get('duration_days', 7),
                decay_rate=impact_data.get('decay_rate', 0.1),
                confidence=impact_data.get('confidence', 0.8)
            )
            for impact_data in data.get('impacts', [])
        ]
        
        return ExternalEvent(
            event_id=data['event_id'],
            event_type=EventType(data['event_type']),
            severity=EventSeverity(data['severity']),
            name=data['name'],
            description=data['description'],
            start_date=start_date,
            end_date=end_date,
            affected_countries=data.get('affected_countries', []),
            affected_cities=data.get('affected_cities', []),
            global_impact=data.get('global_impact', False),
            impacts=impacts,
            probability=data.get('probability', 1.0),
            uncertainty_factor=data.get('uncertainty_factor', 0.1),
            triggers=data.get('triggers', []),
            consequences=data.get('consequences', []),
            source=data.get('source', 'custom'),
            tags=data.get('tags', [])
        )
    
    def create_scenario_comparison(
        self,
        scenario_ids: List[str],
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """Create a comparison of multiple scenarios."""
        if metrics is None:
            metrics = [
                'gdp_growth_rate', 'consumer_confidence', 'market_concentration',
                'price_competition_intensity', 'fuel_cost_base', 'scheduled_events_count'
            ]
        
        comparison = {
            'scenarios': {},
            'metrics_comparison': {},
            'risk_assessment': {}
        }
        
        scenarios = []
        for scenario_id in scenario_ids:
            scenario = self.get_scenario(scenario_id)
            if scenario:
                scenarios.append(scenario)
                comparison['scenarios'][scenario_id] = {
                    'name': scenario.name,
                    'type': scenario.scenario_type.value,
                    'market_condition': scenario.market_condition.value
                }
        
        # Compare metrics
        for metric in metrics:
            comparison['metrics_comparison'][metric] = {}
            
            for scenario in scenarios:
                value = self._extract_metric_value(scenario, metric)
                comparison['metrics_comparison'][metric][scenario.scenario_id] = value
        
        # Risk assessment
        for scenario in scenarios:
            risk_score = self._calculate_scenario_risk(scenario)
            comparison['risk_assessment'][scenario.scenario_id] = {
                'risk_score': risk_score,
                'risk_level': self._categorize_risk(risk_score)
            }
        
        return comparison
    
    def _extract_metric_value(self, scenario: ScenarioConfiguration, metric: str) -> Any:
        """Extract a specific metric value from a scenario."""
        if metric == 'scheduled_events_count':
            return len(scenario.scheduled_events)
        
        # Economic parameters
        if hasattr(scenario.economic_params, metric):
            return getattr(scenario.economic_params, metric)
        
        # Competitive parameters
        if hasattr(scenario.competitive_params, metric):
            return getattr(scenario.competitive_params, metric)
        
        # Operational parameters
        if hasattr(scenario.operational_params, metric):
            return getattr(scenario.operational_params, metric)
        
        # Regulatory parameters
        if hasattr(scenario.regulatory_params, metric):
            return getattr(scenario.regulatory_params, metric)
        
        # Technology parameters
        if hasattr(scenario.technology_params, metric):
            return getattr(scenario.technology_params, metric)
        
        return None
    
    def _calculate_scenario_risk(self, scenario: ScenarioConfiguration) -> float:
        """Calculate overall risk score for a scenario."""
        risk_score = 0.0
        
        # Economic risk factors
        if scenario.economic_params.gdp_growth_rate < 0:
            risk_score += 0.3
        if scenario.economic_params.consumer_confidence < 0.5:
            risk_score += 0.2
        if scenario.economic_params.fuel_price_volatility > 0.3:
            risk_score += 0.15
        
        # Competitive risk factors
        if scenario.competitive_params.price_competition_intensity > 0.7:
            risk_score += 0.2
        if scenario.competitive_params.new_entrant_probability > 0.2:
            risk_score += 0.1
        
        # Event risk factors
        high_severity_events = sum(
            1 for event in scenario.scheduled_events
            if event.severity in [EventSeverity.HIGH, EventSeverity.CRITICAL]
        )
        risk_score += min(high_severity_events * 0.1, 0.3)
        
        # Event probability multipliers
        high_risk_multipliers = sum(
            max(0, multiplier - 1.0) for multiplier in scenario.event_probability_multipliers.values()
        )
        risk_score += min(high_risk_multipliers * 0.05, 0.2)
        
        return min(risk_score, 1.0)
    
    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk score into levels."""
        if risk_score >= 0.7:
            return "HIGH"
        elif risk_score >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def generate_scenario_report(self, scenario_id: str) -> Dict[str, Any]:
        """Generate a comprehensive report for a scenario."""
        scenario = self.get_scenario(scenario_id)
        if not scenario:
            raise ValueError(f"Scenario {scenario_id} not found")
        
        report = {
            'scenario_info': {
                'id': scenario.scenario_id,
                'name': scenario.name,
                'description': scenario.description,
                'type': scenario.scenario_type.value,
                'market_condition': scenario.market_condition.value,
                'duration_days': (scenario.end_date - scenario.start_date).days,
                'created_at': scenario.created_at.isoformat(),
                'created_by': scenario.created_by
            },
            'economic_environment': {
                'gdp_growth': f"{scenario.economic_params.gdp_growth_rate:.1%}",
                'inflation': f"{scenario.economic_params.inflation_rate:.1%}",
                'unemployment': f"{scenario.economic_params.unemployment_rate:.1%}",
                'consumer_confidence': f"{scenario.economic_params.consumer_confidence:.1%}",
                'fuel_volatility': f"{scenario.economic_params.fuel_price_volatility:.1%}",
                'business_travel_factor': f"{scenario.economic_params.business_travel_factor:.1%}",
                'leisure_travel_factor': f"{scenario.economic_params.leisure_travel_factor:.1%}"
            },
            'competitive_environment': {
                'market_concentration': f"{scenario.competitive_params.market_concentration:.1%}",
                'price_competition': f"{scenario.competitive_params.price_competition_intensity:.1%}",
                'capacity_competition': f"{scenario.competitive_params.capacity_competition_intensity:.1%}",
                'new_entrant_probability': f"{scenario.competitive_params.new_entrant_probability:.1%}",
                'price_response_speed': f"{scenario.competitive_params.price_response_speed:.1%}"
            },
            'operational_environment': {
                'fuel_cost_share': f"{scenario.operational_params.fuel_cost_base:.1%}",
                'labor_cost_share': f"{scenario.operational_params.labor_cost_base:.1%}",
                'target_load_factor': f"{scenario.operational_params.load_factor_target:.1%}",
                'on_time_performance': f"{scenario.operational_params.on_time_performance:.1%}",
                'dynamic_pricing_adoption': f"{scenario.operational_params.dynamic_pricing_adoption:.1%}"
            },
            'regulatory_environment': {
                'environmental_strictness': f"{scenario.regulatory_params.environmental_regulations_strictness:.1%}",
                'consumer_protection': f"{scenario.regulatory_params.consumer_protection_level:.1%}",
                'market_liberalization': f"{scenario.regulatory_params.market_liberalization:.1%}",
                'carbon_tax': f"${scenario.regulatory_params.carbon_tax_rate}/ton CO2",
                'visa_restrictions': f"{scenario.regulatory_params.visa_restrictions_level:.1%}"
            },
            'technology_environment': {
                'virtual_meeting_adoption': f"{scenario.technology_params.virtual_meeting_adoption:.1%}",
                'mobile_booking': f"{scenario.technology_params.mobile_booking_penetration:.1%}",
                'ai_personalization': f"{scenario.technology_params.ai_personalization_level:.1%}",
                'ndc_adoption': f"{scenario.technology_params.ndc_adoption_rate:.1%}",
                'automation_level': f"{scenario.technology_params.automated_operations_level:.1%}"
            },
            'scheduled_events': [
                {
                    'name': event.name,
                    'type': event.event_type.value,
                    'severity': event.severity.value,
                    'start_date': event.start_date.strftime('%Y-%m-%d'),
                    'duration_days': (event.end_date - event.start_date).days if event.end_date else None,
                    'global_impact': event.global_impact,
                    'impact_count': len(event.impacts)
                }
                for event in scenario.scheduled_events
            ],
            'risk_assessment': {
                'overall_risk_score': self._calculate_scenario_risk(scenario),
                'risk_level': self._categorize_risk(self._calculate_scenario_risk(scenario)),
                'key_risk_factors': self._identify_key_risks(scenario)
            },
            'simulation_parameters': {
                'random_seed': scenario.random_seed,
                'monte_carlo_runs': scenario.monte_carlo_runs,
                'event_probability_multipliers': {
                    event_type.value: multiplier
                    for event_type, multiplier in scenario.event_probability_multipliers.items()
                }
            }
        }
        
        return report
    
    def _identify_key_risks(self, scenario: ScenarioConfiguration) -> List[str]:
        """Identify key risk factors in a scenario."""
        risks = []
        
        if scenario.economic_params.gdp_growth_rate < 0:
            risks.append("Economic recession reducing travel demand")
        
        if scenario.economic_params.consumer_confidence < 0.5:
            risks.append("Low consumer confidence affecting discretionary travel")
        
        if scenario.competitive_params.price_competition_intensity > 0.7:
            risks.append("Intense price competition pressuring margins")
        
        if scenario.competitive_params.new_entrant_probability > 0.2:
            risks.append("High probability of new market entrants")
        
        if scenario.economic_params.fuel_price_volatility > 0.3:
            risks.append("High fuel price volatility increasing cost uncertainty")
        
        critical_events = [
            event for event in scenario.scheduled_events
            if event.severity == EventSeverity.CRITICAL
        ]
        if critical_events:
            risks.append(f"{len(critical_events)} critical events scheduled")
        
        if scenario.technology_params.virtual_meeting_adoption > 0.8:
            risks.append("High virtual meeting adoption reducing business travel")
        
        return risks