"""Data validation utilities for PriceFly simulation data."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import re

from ..models.aircraft import Aircraft, AircraftType
from ..models.airport import Airport, AirportType
from ..models.airline import Airline, AirlineType
from ..models.passenger import CustomerSegment


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a data validation issue."""
    severity: ValidationSeverity
    category: str
    message: str
    affected_records: Optional[List[str]] = None
    suggested_action: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    dataset_name: str
    validation_timestamp: datetime
    total_records: int
    issues: List[ValidationIssue]
    summary_stats: Dict[str, Any]
    
    @property
    def error_count(self) -> int:
        return len([i for i in self.issues if i.severity == ValidationSeverity.ERROR])
    
    @property
    def warning_count(self) -> int:
        return len([i for i in self.issues if i.severity == ValidationSeverity.WARNING])
    
    @property
    def critical_count(self) -> int:
        return len([i for i in self.issues if i.severity == ValidationSeverity.CRITICAL])
    
    @property
    def is_valid(self) -> bool:
        return self.critical_count == 0 and self.error_count == 0


class DataValidator:
    """Comprehensive data validator for airline simulation data."""
    
    def __init__(self):
        self.validation_rules = {
            'airports': self._get_airport_validation_rules(),
            'airlines': self._get_airline_validation_rules(),
            'aircraft': self._get_aircraft_validation_rules(),
            'routes': self._get_route_validation_rules(),
            'bookings': self._get_booking_validation_rules(),
            'pricing': self._get_pricing_validation_rules(),
            'operational': self._get_operational_validation_rules()
        }
    
    def validate_airports(self, airports: List[Airport]) -> ValidationReport:
        """Validate airport data."""
        issues = []
        
        # Check for required fields and basic constraints
        for airport in airports:
            # IATA code validation
            if not re.match(r'^[A-Z]{3}$', airport.iata_code):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="format",
                    message=f"Invalid IATA code format: {airport.iata_code}",
                    affected_records=[airport.iata_code]
                ))
            
            # Coordinate validation
            if not (-90 <= airport.latitude <= 90):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="coordinates",
                    message=f"Invalid latitude: {airport.latitude} for {airport.iata_code}",
                    affected_records=[airport.iata_code]
                ))
            
            if not (-180 <= airport.longitude <= 180):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="coordinates",
                    message=f"Invalid longitude: {airport.longitude} for {airport.iata_code}",
                    affected_records=[airport.iata_code]
                ))
            
            # Passenger volume validation
            if airport.annual_passengers < 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="business_logic",
                    message=f"Negative passenger volume: {airport.annual_passengers} for {airport.iata_code}",
                    affected_records=[airport.iata_code]
                ))
            
            # Runway count validation
            if airport.runway_count <= 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="business_logic",
                    message=f"Airport {airport.iata_code} has {airport.runway_count} runways",
                    affected_records=[airport.iata_code]
                ))
            
            # Fee validation
            if airport.landing_fee_base < 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="business_logic",
                    message=f"Negative landing fee: {airport.landing_fee_base} for {airport.iata_code}",
                    affected_records=[airport.iata_code]
                ))
        
        # Check for duplicates
        iata_codes = [a.iata_code for a in airports]
        duplicates = [code for code in set(iata_codes) if iata_codes.count(code) > 1]
        if duplicates:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="duplicates",
                message=f"Duplicate IATA codes found: {duplicates}",
                affected_records=duplicates
            ))
        
        # Statistical validation
        passenger_volumes = [a.annual_passengers for a in airports]
        if passenger_volumes:
            avg_passengers = np.mean(passenger_volumes)
            outliers = [
                a.iata_code for a in airports 
                if a.annual_passengers > avg_passengers * 10
            ]
            if outliers:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="outliers",
                    message=f"Airports with unusually high passenger volumes: {outliers}",
                    affected_records=outliers
                ))
        
        return ValidationReport(
            dataset_name="airports",
            validation_timestamp=datetime.now(),
            total_records=len(airports),
            issues=issues,
            summary_stats={
                "total_airports": len(airports),
                "unique_countries": len(set(a.country for a in airports)),
                "avg_annual_passengers": np.mean([a.annual_passengers for a in airports]),
                "airport_types": {t.value: len([a for a in airports if a.airport_type == t]) 
                                for t in AirportType}
            }
        )
    
    def validate_airlines(self, airlines: List[Airline]) -> ValidationReport:
        """Validate airline data."""
        issues = []
        
        for airline in airlines:
            # Airline code validation
            if not re.match(r'^[A-Z0-9]{2,3}$', airline.airline_code):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="format",
                    message=f"Invalid airline code format: {airline.airline_code}",
                    affected_records=[airline.airline_code]
                ))
            
            # Financial validation
            if airline.annual_revenue_usd < 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="business_logic",
                    message=f"Negative revenue: {airline.annual_revenue_usd} for {airline.airline_code}",
                    affected_records=[airline.airline_code]
                ))
            
            # Load factor validation
            if not (0 <= airline.load_factor <= 1):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="business_logic",
                    message=f"Invalid load factor: {airline.load_factor} for {airline.airline_code}",
                    affected_records=[airline.airline_code]
                ))
            
            # On-time performance validation
            if not (0 <= airline.on_time_performance <= 1):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="business_logic",
                    message=f"Invalid OTP: {airline.on_time_performance} for {airline.airline_code}",
                    affected_records=[airline.airline_code]
                ))
            
            # Fleet validation
            if len(airline.fleet.aircraft) == 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="business_logic",
                    message=f"Airline {airline.airline_code} has no aircraft",
                    affected_records=[airline.airline_code]
                ))
            
            # Hub validation (would need airport data for cross-validation)
            if not airline.hub_airports:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="business_logic",
                    message=f"Airline {airline.airline_code} has no hub airports",
                    affected_records=[airline.airline_code]
                ))
        
        # Check for duplicates
        airline_codes = [a.airline_code for a in airlines]
        duplicates = [code for code in set(airline_codes) if airline_codes.count(code) > 1]
        if duplicates:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="duplicates",
                message=f"Duplicate airline codes found: {duplicates}",
                affected_records=duplicates
            ))
        
        return ValidationReport(
            dataset_name="airlines",
            validation_timestamp=datetime.now(),
            total_records=len(airlines),
            issues=issues,
            summary_stats={
                "total_airlines": len(airlines),
                "total_aircraft": sum(len(a.fleet.aircraft) for a in airlines),
                "avg_load_factor": np.mean([a.load_factor for a in airlines]),
                "airline_types": {t.value: len([a for a in airlines if a.airline_type == t]) 
                                for t in AirlineType}
            }
        )
    
    def validate_bookings_dataframe(self, bookings_df: pd.DataFrame) -> ValidationReport:
        """Validate booking transaction data."""
        issues = []
        
        required_columns = [
            'booking_id', 'booking_date', 'travel_date', 'origin', 'destination',
            'total_fare', 'cabin_class', 'passenger_segment'
        ]
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in bookings_df.columns]
        if missing_columns:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="schema",
                message=f"Missing required columns: {missing_columns}"
            ))
            return ValidationReport(
                dataset_name="bookings",
                validation_timestamp=datetime.now(),
                total_records=len(bookings_df),
                issues=issues,
                summary_stats={}
            )
        
        # Check for null values in critical fields
        for col in required_columns:
            null_count = bookings_df[col].isnull().sum()
            if null_count > 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="completeness",
                    message=f"Column {col} has {null_count} null values"
                ))
        
        # Date validation
        if 'booking_date' in bookings_df.columns and 'travel_date' in bookings_df.columns:
            try:
                booking_dates = pd.to_datetime(bookings_df['booking_date'])
                travel_dates = pd.to_datetime(bookings_df['travel_date'])
                
                # Check for travel dates before booking dates
                invalid_dates = (travel_dates < booking_dates).sum()
                if invalid_dates > 0:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category="business_logic",
                        message=f"{invalid_dates} bookings have travel date before booking date"
                    ))
                
                # Check for unrealistic lead times
                lead_times = (travel_dates - booking_dates).dt.days
                long_lead_times = (lead_times > 365).sum()
                if long_lead_times > 0:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="business_logic",
                        message=f"{long_lead_times} bookings have lead time > 365 days"
                    ))
                
            except Exception as e:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="format",
                    message=f"Date parsing error: {str(e)}"
                ))
        
        # Fare validation
        if 'total_fare' in bookings_df.columns:
            negative_fares = (bookings_df['total_fare'] <= 0).sum()
            if negative_fares > 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="business_logic",
                    message=f"{negative_fares} bookings have non-positive fares"
                ))
            
            # Check for outlier fares
            fare_q99 = bookings_df['total_fare'].quantile(0.99)
            outlier_fares = (bookings_df['total_fare'] > fare_q99 * 5).sum()
            if outlier_fares > 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="outliers",
                    message=f"{outlier_fares} bookings have extremely high fares"
                ))
        
        # Route validation
        if 'origin' in bookings_df.columns and 'destination' in bookings_df.columns:
            same_origin_dest = (bookings_df['origin'] == bookings_df['destination']).sum()
            if same_origin_dest > 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="business_logic",
                    message=f"{same_origin_dest} bookings have same origin and destination"
                ))
        
        # Duplicate booking ID check
        if 'booking_id' in bookings_df.columns:
            duplicate_bookings = bookings_df['booking_id'].duplicated().sum()
            if duplicate_bookings > 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="duplicates",
                    message=f"{duplicate_bookings} duplicate booking IDs found"
                ))
        
        return ValidationReport(
            dataset_name="bookings",
            validation_timestamp=datetime.now(),
            total_records=len(bookings_df),
            issues=issues,
            summary_stats={
                "total_bookings": len(bookings_df),
                "date_range": {
                    "start": bookings_df['booking_date'].min() if 'booking_date' in bookings_df.columns else None,
                    "end": bookings_df['booking_date'].max() if 'booking_date' in bookings_df.columns else None
                },
                "avg_fare": bookings_df['total_fare'].mean() if 'total_fare' in bookings_df.columns else None,
                "unique_routes": len(bookings_df[['origin', 'destination']].drop_duplicates()) if all(col in bookings_df.columns for col in ['origin', 'destination']) else None
            }
        )
    
    def validate_pricing_dataframe(self, pricing_df: pd.DataFrame) -> ValidationReport:
        """Validate pricing history data."""
        issues = []
        
        required_columns = [
            'timestamp', 'route_code', 'cabin_class', 'booking_class', 'price', 'availability'
        ]
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in pricing_df.columns]
        if missing_columns:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="schema",
                message=f"Missing required columns: {missing_columns}"
            ))
            return ValidationReport(
                dataset_name="pricing",
                validation_timestamp=datetime.now(),
                total_records=len(pricing_df),
                issues=issues,
                summary_stats={}
            )
        
        # Price validation
        if 'price' in pricing_df.columns:
            negative_prices = (pricing_df['price'] <= 0).sum()
            if negative_prices > 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="business_logic",
                    message=f"{negative_prices} pricing records have non-positive prices"
                ))
        
        # Availability validation
        if 'availability' in pricing_df.columns:
            negative_availability = (pricing_df['availability'] < 0).sum()
            if negative_availability > 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="business_logic",
                    message=f"{negative_availability} pricing records have negative availability"
                ))
        
        # Booking class validation
        if 'booking_class' in pricing_df.columns:
            valid_classes = ['Y', 'B', 'M', 'H', 'Q', 'V', 'W', 'S', 'T', 'L', 'K']
            invalid_classes = ~pricing_df['booking_class'].isin(valid_classes)
            if invalid_classes.sum() > 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="format",
                    message=f"{invalid_classes.sum()} records have non-standard booking classes"
                ))
        
        return ValidationReport(
            dataset_name="pricing",
            validation_timestamp=datetime.now(),
            total_records=len(pricing_df),
            issues=issues,
            summary_stats={
                "total_observations": len(pricing_df),
                "unique_routes": pricing_df['route_code'].nunique() if 'route_code' in pricing_df.columns else None,
                "avg_price": pricing_df['price'].mean() if 'price' in pricing_df.columns else None,
                "price_range": {
                    "min": pricing_df['price'].min() if 'price' in pricing_df.columns else None,
                    "max": pricing_df['price'].max() if 'price' in pricing_df.columns else None
                }
            }
        )
    
    def validate_cross_dataset_consistency(
        self,
        airports: List[Airport],
        airlines: List[Airline],
        bookings_df: Optional[pd.DataFrame] = None
    ) -> ValidationReport:
        """Validate consistency across multiple datasets."""
        issues = []
        
        # Create lookup sets
        airport_codes = {a.iata_code for a in airports}
        airline_codes = {a.airline_code for a in airlines}
        
        # Validate airline hub airports
        for airline in airlines:
            missing_hubs = set(airline.hub_airports) - airport_codes
            if missing_hubs:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="referential_integrity",
                    message=f"Airline {airline.airline_code} references non-existent hub airports: {missing_hubs}",
                    affected_records=[airline.airline_code]
                ))
        
        # Validate booking references
        if bookings_df is not None:
            if 'origin' in bookings_df.columns:
                missing_origins = set(bookings_df['origin'].unique()) - airport_codes
                if missing_origins:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category="referential_integrity",
                        message=f"Bookings reference non-existent origin airports: {missing_origins}"
                    ))
            
            if 'destination' in bookings_df.columns:
                missing_destinations = set(bookings_df['destination'].unique()) - airport_codes
                if missing_destinations:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category="referential_integrity",
                        message=f"Bookings reference non-existent destination airports: {missing_destinations}"
                    ))
        
        return ValidationReport(
            dataset_name="cross_dataset",
            validation_timestamp=datetime.now(),
            total_records=len(airports) + len(airlines) + (len(bookings_df) if bookings_df is not None else 0),
            issues=issues,
            summary_stats={
                "airports_count": len(airports),
                "airlines_count": len(airlines),
                "bookings_count": len(bookings_df) if bookings_df is not None else 0
            }
        )
    
    def _get_airport_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for airports."""
        return {
            "required_fields": ["iata_code", "name", "city", "country", "latitude", "longitude"],
            "format_rules": {
                "iata_code": r"^[A-Z]{3}$",
                "icao_code": r"^[A-Z]{4}$"
            },
            "range_rules": {
                "latitude": (-90, 90),
                "longitude": (-180, 180),
                "annual_passengers": (0, float('inf')),
                "runway_count": (1, 10)
            }
        }
    
    def _get_airline_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for airlines."""
        return {
            "required_fields": ["airline_code", "airline_name", "country", "airline_type"],
            "format_rules": {
                "airline_code": r"^[A-Z0-9]{2,3}$",
                "icao_code": r"^[A-Z]{3}$"
            },
            "range_rules": {
                "load_factor": (0, 1),
                "on_time_performance": (0, 1),
                "annual_revenue_usd": (0, float('inf'))
            }
        }
    
    def _get_aircraft_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for aircraft."""
        return {
            "required_fields": ["model", "manufacturer", "aircraft_type", "total_seats"],
            "range_rules": {
                "total_seats": (1, 1000),
                "max_range_km": (100, 20000),
                "cruise_speed_kmh": (200, 1200),
                "fuel_burn_per_hour": (100, 15000)
            }
        }
    
    def _get_route_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for routes."""
        return {
            "required_fields": ["origin", "destination", "route_type"],
            "range_rules": {
                "annual_demand": (0, float('inf')),
                "frequency_per_day": (0, 50),
                "average_fare_economy": (0, float('inf'))
            }
        }
    
    def _get_booking_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for bookings."""
        return {
            "required_fields": ["booking_id", "booking_date", "travel_date", "origin", "destination", "total_fare"],
            "range_rules": {
                "total_fare": (0, float('inf')),
                "party_size": (1, 20),
                "lead_time_days": (0, 365)
            }
        }
    
    def _get_pricing_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for pricing."""
        return {
            "required_fields": ["timestamp", "route_code", "cabin_class", "price"],
            "range_rules": {
                "price": (0, float('inf')),
                "availability": (0, 1000)
            }
        }
    
    def _get_operational_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for operational data."""
        return {
            "required_fields": ["flight_id", "date", "airline_code", "origin", "destination"],
            "range_rules": {
                "load_factor": (0, 1),
                "passengers_boarded": (0, 1000),
                "fuel_consumed_liters": (0, 50000)
            }
        }


class DataQualityMonitor:
    """Monitor data quality over time and detect anomalies."""
    
    def __init__(self):
        self.quality_history: List[ValidationReport] = []
        self.thresholds = {
            "max_error_rate": 0.05,  # 5% error rate threshold
            "max_missing_rate": 0.10,  # 10% missing data threshold
            "min_completeness": 0.95   # 95% completeness threshold
        }
    
    def add_validation_report(self, report: ValidationReport) -> None:
        """Add a validation report to the quality history."""
        self.quality_history.append(report)
    
    def get_quality_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get data quality trends over the specified period."""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_reports = [
            r for r in self.quality_history 
            if r.validation_timestamp >= cutoff_date
        ]
        
        if not recent_reports:
            return {"message": "No recent validation reports available"}
        
        trends = {
            "period_days": days,
            "total_reports": len(recent_reports),
            "error_trend": [r.error_count for r in recent_reports],
            "warning_trend": [r.warning_count for r in recent_reports],
            "critical_trend": [r.critical_count for r in recent_reports],
            "avg_error_rate": np.mean([r.error_count / max(r.total_records, 1) for r in recent_reports]),
            "datasets_monitored": list(set(r.dataset_name for r in recent_reports))
        }
        
        return trends
    
    def detect_quality_anomalies(self) -> List[ValidationIssue]:
        """Detect anomalies in data quality patterns."""
        anomalies = []
        
        if len(self.quality_history) < 2:
            return anomalies
        
        recent_reports = self.quality_history[-10:]  # Last 10 reports
        
        # Check for sudden increase in errors
        error_rates = [r.error_count / max(r.total_records, 1) for r in recent_reports]
        if len(error_rates) >= 2:
            latest_rate = error_rates[-1]
            avg_rate = np.mean(error_rates[:-1])
            
            if latest_rate > avg_rate * 2 and latest_rate > self.thresholds["max_error_rate"]:
                anomalies.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="quality_anomaly",
                    message=f"Sudden increase in error rate: {latest_rate:.3f} vs avg {avg_rate:.3f}"
                ))
        
        return anomalies
    
    def generate_quality_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for a quality monitoring dashboard."""
        if not self.quality_history:
            return {"message": "No validation history available"}
        
        latest_report = self.quality_history[-1]
        
        dashboard_data = {
            "last_validation": latest_report.validation_timestamp.isoformat(),
            "overall_status": "healthy" if latest_report.is_valid else "issues_detected",
            "current_issues": {
                "critical": latest_report.critical_count,
                "errors": latest_report.error_count,
                "warnings": latest_report.warning_count
            },
            "trends": self.get_quality_trends(30),
            "anomalies": self.detect_quality_anomalies(),
            "recommendations": self._generate_recommendations(latest_report)
        }
        
        return dashboard_data
    
    def _generate_recommendations(self, report: ValidationReport) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if report.critical_count > 0:
            recommendations.append("Address critical issues immediately before proceeding with analysis")
        
        if report.error_count > report.total_records * 0.05:
            recommendations.append("High error rate detected - review data collection processes")
        
        # Category-specific recommendations
        error_categories = [issue.category for issue in report.issues if issue.severity == ValidationSeverity.ERROR]
        
        if "duplicates" in error_categories:
            recommendations.append("Implement deduplication process in data pipeline")
        
        if "referential_integrity" in error_categories:
            recommendations.append("Review data relationships and foreign key constraints")
        
        if "business_logic" in error_categories:
            recommendations.append("Validate business rules in data generation process")
        
        return recommendations