"""Passenger and Customer Segment models for demand simulation."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta
import uuid
import random


class TravelPurpose(Enum):
    """Purpose of travel classification."""
    BUSINESS = "business"
    LEISURE = "leisure"
    VFR = "visiting_friends_relatives"  # Visiting Friends and Relatives
    EMERGENCY = "emergency"
    OTHER = "other"


class BookingChannel(Enum):
    """Channel through which booking was made."""
    AIRLINE_WEBSITE = "airline_website"
    TRAVEL_AGENT = "travel_agent"
    OTA = "online_travel_agency"  # Online Travel Agency
    MOBILE_APP = "mobile_app"
    PHONE = "phone"
    CORPORATE = "corporate_booking"


class LoyaltyTier(Enum):
    """Airline loyalty program tiers."""
    NONE = "none"
    BASIC = "basic"
    SILVER = "silver"
    GOLD = "gold"
    PLATINUM = "platinum"
    DIAMOND = "diamond"


@dataclass
class CustomerSegment:
    """Represents a customer segment with behavioral characteristics."""
    
    # Segment Identification
    segment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    segment_name: str = ""
    description: str = ""
    
    # Demographics
    age_range: Tuple[int, int] = (25, 65)
    income_range: Tuple[float, float] = (30000, 100000)  # USD
    gender_distribution: Dict[str, float] = field(default_factory=lambda: {
        "male": 0.5, "female": 0.5
    })
    
    # Travel Behavior
    primary_travel_purpose: TravelPurpose = TravelPurpose.LEISURE
    booking_lead_time_days: Tuple[int, int] = (7, 90)  # Min, Max days in advance
    preferred_booking_channels: Dict[BookingChannel, float] = field(default_factory=lambda: {
        BookingChannel.AIRLINE_WEBSITE: 0.4,
        BookingChannel.OTA: 0.3,
        BookingChannel.MOBILE_APP: 0.2,
        BookingChannel.TRAVEL_AGENT: 0.1
    })
    
    # Price Sensitivity
    price_elasticity: float = -1.2  # How responsive to price changes
    willingness_to_pay_multiplier: float = 1.0  # Relative to base fare
    price_sensitivity_score: float = 0.7  # 0-1, higher = more sensitive
    
    # Service Preferences
    cabin_class_preference: Dict[str, float] = field(default_factory=lambda: {
        "economy": 0.8,
        "premium_economy": 0.15,
        "business": 0.05
    })
    
    # Flexibility and Convenience
    schedule_flexibility: float = 0.5  # 0-1, higher = more flexible
    change_fee_sensitivity: float = 0.8  # 0-1, higher = more sensitive
    cancellation_probability: float = 0.05  # Probability of cancellation
    no_show_probability: float = 0.02  # Probability of no-show
    
    # Loyalty and Retention
    loyalty_tier_distribution: Dict[LoyaltyTier, float] = field(default_factory=lambda: {
        LoyaltyTier.NONE: 0.6,
        LoyaltyTier.BASIC: 0.25,
        LoyaltyTier.SILVER: 0.1,
        LoyaltyTier.GOLD: 0.04,
        LoyaltyTier.PLATINUM: 0.01
    })
    
    brand_loyalty_score: float = 0.3  # 0-1, higher = more loyal
    repeat_purchase_probability: float = 0.4
    
    # Ancillary Services
    baggage_fee_acceptance: float = 0.6  # Willingness to pay baggage fees
    seat_upgrade_propensity: float = 0.2  # Likelihood to upgrade seats
    meal_purchase_propensity: float = 0.3  # Likelihood to buy meals
    
    # Digital Behavior
    mobile_usage_rate: float = 0.7  # Preference for mobile booking
    social_media_influence: float = 0.4  # Influence of social media on decisions
    review_sensitivity: float = 0.6  # How much reviews affect decisions
    
    # Market Share
    segment_size_percentage: float = 0.1  # Percentage of total market
    
    def generate_booking_lead_time(self) -> int:
        """Generate a random booking lead time for this segment."""
        min_days, max_days = self.booking_lead_time_days
        return random.randint(min_days, max_days)
    
    def calculate_price_sensitivity_factor(self, base_price: float, 
                                         offered_price: float) -> float:
        """Calculate demand adjustment based on price sensitivity."""
        price_change_ratio = (offered_price - base_price) / base_price
        return 1 + (self.price_elasticity * price_change_ratio)
    
    def get_preferred_cabin_class(self) -> str:
        """Get randomly selected preferred cabin class based on preferences."""
        rand_val = random.random()
        cumulative = 0
        for cabin_class, probability in self.cabin_class_preference.items():
            cumulative += probability
            if rand_val <= cumulative:
                return cabin_class
        return "economy"  # Default fallback
    
    def calculate_willingness_to_pay(self, base_fare: float, 
                                   route_characteristics: Dict) -> float:
        """Calculate willingness to pay based on segment characteristics."""
        wtp = base_fare * self.willingness_to_pay_multiplier
        
        # Adjust based on travel purpose
        if self.primary_travel_purpose == TravelPurpose.BUSINESS:
            wtp *= 1.5  # Business travelers pay more
        elif self.primary_travel_purpose == TravelPurpose.EMERGENCY:
            wtp *= 2.0  # Emergency travelers less price sensitive
        elif self.primary_travel_purpose == TravelPurpose.LEISURE:
            wtp *= 0.8  # Leisure travelers more price sensitive
        
        # Adjust based on route characteristics
        if route_characteristics.get("competition_level", "moderate") == "low":
            wtp *= 1.2  # Less competition, higher WTP
        elif route_characteristics.get("competition_level", "moderate") == "high":
            wtp *= 0.9  # High competition, lower WTP
        
        return wtp


@dataclass
class Passenger:
    """Represents an individual passenger with booking characteristics."""
    
    # Passenger Identification
    passenger_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Demographics
    age: int = 35
    gender: str = "unknown"
    income_usd: float = 50000.0
    home_country: str = "US"
    home_city: str = ""
    
    # Segment Assignment
    customer_segment: Optional[CustomerSegment] = None
    segment_id: str = ""
    
    # Travel Characteristics
    travel_purpose: TravelPurpose = TravelPurpose.LEISURE
    loyalty_tier: LoyaltyTier = LoyaltyTier.NONE
    loyalty_program_number: Optional[str] = None
    
    # Booking Behavior
    booking_lead_time_preference: int = 30  # Days in advance
    preferred_booking_channel: BookingChannel = BookingChannel.AIRLINE_WEBSITE
    price_sensitivity: float = 0.7  # 0-1, higher = more sensitive
    
    # Service Preferences
    preferred_cabin_class: str = "economy"
    seat_preference: str = "window"  # window, aisle, middle, no_preference
    meal_preference: str = "standard"  # standard, vegetarian, kosher, etc.
    
    # Flexibility
    schedule_flexibility_hours: int = 4  # Hours of flexibility
    willing_to_connect: bool = True
    max_connection_time_hours: int = 4
    
    # Historical Data
    total_flights_taken: int = 0
    flights_with_airline: int = 0
    last_flight_date: Optional[datetime] = None
    average_annual_spend: float = 0.0
    
    # Current Trip
    party_size: int = 1
    has_checked_baggage: bool = False
    special_assistance_needed: bool = False
    
    def __post_init__(self):
        """Initialize derived fields after creation."""
        if self.customer_segment:
            self.segment_id = self.customer_segment.segment_id
            # Inherit some characteristics from segment
            self.price_sensitivity = self.customer_segment.price_sensitivity_score
    
    @property
    def is_frequent_flyer(self) -> bool:
        """Check if passenger is a frequent flyer."""
        return self.loyalty_tier not in [LoyaltyTier.NONE, LoyaltyTier.BASIC]
    
    @property
    def customer_lifetime_value_estimate(self) -> float:
        """Estimate customer lifetime value."""
        if self.total_flights_taken == 0:
            return self.average_annual_spend * 5  # 5-year estimate for new customer
        
        annual_flight_frequency = max(1, self.total_flights_taken / 5)  # Assume 5-year history
        return self.average_annual_spend * annual_flight_frequency * 3  # 3-year forward estimate
    
    def calculate_booking_probability(self, offered_price: float, 
                                    base_price: float) -> float:
        """Calculate probability of booking at offered price."""
        if self.customer_segment is None:
            # Simple price sensitivity model
            price_ratio = offered_price / base_price
            return max(0, 1 - (price_ratio - 1) * self.price_sensitivity)
        
        # Use segment-based model
        sensitivity_factor = self.customer_segment.calculate_price_sensitivity_factor(
            base_price, offered_price
        )
        
        base_probability = 0.7  # Base booking probability
        adjusted_probability = base_probability * sensitivity_factor
        
        # Adjust for loyalty
        if self.is_frequent_flyer:
            adjusted_probability *= 1.2  # Loyal customers more likely to book
        
        return max(0, min(1, adjusted_probability))
    
    def generate_ancillary_purchases(self) -> Dict[str, bool]:
        """Generate ancillary purchase decisions."""
        purchases = {
            "checked_baggage": False,
            "seat_upgrade": False,
            "priority_boarding": False,
            "meal": False,
            "wifi": False,
            "travel_insurance": False
        }
        
        if self.customer_segment:
            # Use segment propensities
            purchases["checked_baggage"] = (
                random.random() < self.customer_segment.baggage_fee_acceptance
            )
            purchases["seat_upgrade"] = (
                random.random() < self.customer_segment.seat_upgrade_propensity
            )
            purchases["meal"] = (
                random.random() < self.customer_segment.meal_purchase_propensity
            )
        else:
            # Default probabilities
            purchases["checked_baggage"] = random.random() < 0.6
            purchases["seat_upgrade"] = random.random() < 0.2
            purchases["meal"] = random.random() < 0.3
        
        # Business travelers more likely to purchase services
        if self.travel_purpose == TravelPurpose.BUSINESS:
            for service in purchases:
                if random.random() < 0.3:  # 30% boost for business travelers
                    purchases[service] = True
        
        return purchases
    
    def update_travel_history(self, flight_cost: float):
        """Update passenger's travel history after a flight."""
        self.total_flights_taken += 1
        self.flights_with_airline += 1
        self.last_flight_date = datetime.now()
        
        # Update average annual spend (simple moving average)
        if self.average_annual_spend == 0:
            self.average_annual_spend = flight_cost
        else:
            self.average_annual_spend = (
                self.average_annual_spend * 0.9 + flight_cost * 0.1
            )
    
    def get_preferred_flight_times(self) -> List[Tuple[int, int]]:
        """Get preferred departure time ranges (hour ranges)."""
        if self.travel_purpose == TravelPurpose.BUSINESS:
            # Business travelers prefer morning and evening flights
            return [(6, 10), (17, 20)]
        elif self.travel_purpose == TravelPurpose.LEISURE:
            # Leisure travelers more flexible, avoid very early flights
            return [(8, 12), (13, 18)]
        else:
            # Default: flexible timing
            return [(6, 22)]
    
    def calculate_schedule_value(self, departure_hour: int) -> float:
        """Calculate value/utility of a specific departure time."""
        preferred_times = self.get_preferred_flight_times()
        
        max_value = 1.0
        for start_hour, end_hour in preferred_times:
            if start_hour <= departure_hour <= end_hour:
                return max_value
        
        # Calculate penalty for non-preferred times
        min_distance = min(
            min(abs(departure_hour - start), abs(departure_hour - end))
            for start, end in preferred_times
        )
        
        # Linear decay with distance from preferred time
        penalty = min_distance * 0.1
        return max(0.1, max_value - penalty)