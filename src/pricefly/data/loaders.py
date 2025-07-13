"""Data loading and processing utilities for PriceFly simulation."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..models.aircraft import Aircraft, Fleet, AircraftType
from ..models.airport import Airport, Route, AirportType, RouteType
from ..models.airline import Airline, AirlineType, AllianceType
from ..models.passenger import CustomerSegment, TravelPurpose, LoyaltyTier


@dataclass
class DataLoadConfig:
    """Configuration for data loading."""
    data_directory: str
    cache_enabled: bool = True
    validate_data: bool = True
    load_time_series: bool = True
    time_series_start_date: Optional[datetime] = None
    time_series_end_date: Optional[datetime] = None


class DataLoader:
    """Comprehensive data loader for airline simulation data."""
    
    def __init__(self, config: DataLoadConfig):
        self.config = config
        self.data_path = Path(config.data_directory)
        
        # Cached data
        self._airports_cache: Optional[List[Airport]] = None
        self._airlines_cache: Optional[List[Airline]] = None
        self._routes_cache: Optional[List[Route]] = None
        self._customer_segments_cache: Optional[List[CustomerSegment]] = None
        
        # Time series cache
        self._fuel_prices_cache: Optional[pd.DataFrame] = None
        self._economic_indicators_cache: Optional[pd.DataFrame] = None
        self._demand_patterns_cache: Optional[pd.DataFrame] = None
        
        # Validation results
        self._validation_results: Dict[str, Any] = {}
    
    def load_all_data(self) -> Dict[str, Any]:
        """Load all simulation data."""
        print("Loading simulation data...")
        
        data = {
            'airports': self.load_airports(),
            'airlines': self.load_airlines(),
            'routes': self.load_routes(),
            'customer_segments': self.load_customer_segments()
        }
        
        if self.config.load_time_series:
            data.update({
                'fuel_prices': self.load_fuel_prices(),
                'economic_indicators': self.load_economic_indicators(),
                'demand_patterns': self.load_demand_patterns()
            })
        
        if self.config.validate_data:
            self._validation_results = self.validate_data(data)
            data['validation_results'] = self._validation_results
        
        print("Data loading completed.")
        return data
    
    def load_airports(self) -> List[Airport]:
        """Load airport data."""
        if self.config.cache_enabled and self._airports_cache is not None:
            return self._airports_cache
        
        print("Loading airports...")
        
        airports_file = self.data_path / 'airports.json'
        if not airports_file.exists():
            raise FileNotFoundError(f"Airports data file not found: {airports_file}")
        
        with open(airports_file, 'r') as f:
            airports_data = json.load(f)
        
        airports = []
        for airport_dict in airports_data:
            # Convert string enums back to enum types
            airport_dict['airport_type'] = AirportType(airport_dict['airport_type'])
            
            airport = Airport(**airport_dict)
            airports.append(airport)
        
        if self.config.cache_enabled:
            self._airports_cache = airports
        
        print(f"Loaded {len(airports)} airports")
        return airports
    
    def load_airlines(self) -> List[Airline]:
        """Load airline data with fleet and routes."""
        if self.config.cache_enabled and self._airlines_cache is not None:
            return self._airlines_cache
        
        print("Loading airlines...")
        
        airlines_file = self.data_path / 'airlines.json'
        if not airlines_file.exists():
            raise FileNotFoundError(f"Airlines data file not found: {airlines_file}")
        
        with open(airlines_file, 'r') as f:
            airlines_data = json.load(f)
        
        airlines = []
        for airline_dict in airlines_data:
            # Extract fleet and routes data
            fleet_data = airline_dict.pop('fleet', [])
            routes_data = airline_dict.pop('routes_operated', [])
            
            # Convert enum fields
            airline_dict['airline_type'] = AirlineType(airline_dict['airline_type'])
            airline_dict['alliance'] = AllianceType(airline_dict['alliance'])
            
            # Create airline
            airline = Airline(**airline_dict)
            
            # Load fleet
            for aircraft_dict in fleet_data:
                aircraft_dict['aircraft_type'] = AircraftType(aircraft_dict['aircraft_type'])
                aircraft = Aircraft(**aircraft_dict)
                airline.fleet.add_aircraft(aircraft)
            
            # Load routes (simplified - would need full route reconstruction)
            # This is a placeholder - in practice, routes would be loaded separately
            # and linked to airlines
            
            airlines.append(airline)
        
        if self.config.cache_enabled:
            self._airlines_cache = airlines
        
        print(f"Loaded {len(airlines)} airlines")
        return airlines
    
    def load_routes(self) -> List[Route]:
        """Load route data."""
        if self.config.cache_enabled and self._routes_cache is not None:
            return self._routes_cache
        
        print("Loading routes...")
        
        # Routes are typically embedded in airline data or need to be reconstructed
        # This is a simplified implementation
        routes = []
        
        # Load from airlines if available
        airlines = self.load_airlines()
        for airline in airlines:
            routes.extend(airline.routes_operated)
        
        if self.config.cache_enabled:
            self._routes_cache = routes
        
        print(f"Loaded {len(routes)} routes")
        return routes
    
    def load_customer_segments(self) -> List[CustomerSegment]:
        """Load customer segment data."""
        if self.config.cache_enabled and self._customer_segments_cache is not None:
            return self._customer_segments_cache
        
        print("Loading customer segments...")
        
        # Customer segments are typically generated programmatically
        # This would load from a dedicated file if available
        segments_file = self.data_path / 'customer_segments.json'
        
        if segments_file.exists():
            with open(segments_file, 'r') as f:
                segments_data = json.load(f)
            
            segments = []
            for segment_dict in segments_data:
                # Convert enum fields
                segment_dict['primary_travel_purpose'] = TravelPurpose(
                    segment_dict['primary_travel_purpose']
                )
                
                # Convert loyalty tier distribution
                loyalty_dist = segment_dict['loyalty_tier_distribution']
                converted_dist = {
                    LoyaltyTier(k): v for k, v in loyalty_dist.items()
                }
                segment_dict['loyalty_tier_distribution'] = converted_dist
                
                segment = CustomerSegment(**segment_dict)
                segments.append(segment)
        else:
            # Generate default segments if file doesn't exist
            from .synthetic_data import SyntheticDataEngine
            engine = SyntheticDataEngine()
            segments = engine.generate_customer_segments()
        
        if self.config.cache_enabled:
            self._customer_segments_cache = segments
        
        print(f"Loaded {len(segments)} customer segments")
        return segments
    
    def load_fuel_prices(self) -> pd.DataFrame:
        """Load fuel price time series."""
        if self.config.cache_enabled and self._fuel_prices_cache is not None:
            return self._fuel_prices_cache
        
        print("Loading fuel prices...")
        
        fuel_file = self.data_path / 'fuel_prices.csv'
        if not fuel_file.exists():
            raise FileNotFoundError(f"Fuel prices file not found: {fuel_file}")
        
        fuel_df = pd.read_csv(fuel_file)
        fuel_df['date'] = pd.to_datetime(fuel_df['date'])
        
        # Filter by date range if specified
        if self.config.time_series_start_date:
            fuel_df = fuel_df[fuel_df['date'] >= self.config.time_series_start_date]
        
        if self.config.time_series_end_date:
            fuel_df = fuel_df[fuel_df['date'] <= self.config.time_series_end_date]
        
        if self.config.cache_enabled:
            self._fuel_prices_cache = fuel_df
        
        print(f"Loaded fuel prices for {len(fuel_df)} days")
        return fuel_df
    
    def load_economic_indicators(self) -> pd.DataFrame:
        """Load economic indicators time series."""
        if self.config.cache_enabled and self._economic_indicators_cache is not None:
            return self._economic_indicators_cache
        
        print("Loading economic indicators...")
        
        econ_file = self.data_path / 'economic_indicators.csv'
        if not econ_file.exists():
            raise FileNotFoundError(f"Economic indicators file not found: {econ_file}")
        
        econ_df = pd.read_csv(econ_file)
        econ_df['date'] = pd.to_datetime(econ_df['date'])
        
        # Filter by date range if specified
        if self.config.time_series_start_date:
            econ_df = econ_df[econ_df['date'] >= self.config.time_series_start_date]
        
        if self.config.time_series_end_date:
            econ_df = econ_df[econ_df['date'] <= self.config.time_series_end_date]
        
        if self.config.cache_enabled:
            self._economic_indicators_cache = econ_df
        
        print(f"Loaded economic indicators for {len(econ_df)} days")
        return econ_df
    
    def load_demand_patterns(self) -> pd.DataFrame:
        """Load demand pattern data."""
        if self.config.cache_enabled and self._demand_patterns_cache is not None:
            return self._demand_patterns_cache
        
        print("Loading demand patterns...")
        
        demand_file = self.data_path / 'demand_patterns.csv'
        if not demand_file.exists():
            raise FileNotFoundError(f"Demand patterns file not found: {demand_file}")
        
        demand_df = pd.read_csv(demand_file)
        
        if self.config.cache_enabled:
            self._demand_patterns_cache = demand_df
        
        print(f"Loaded demand patterns for {len(demand_df)} route-months")
        return demand_df
    
    def load_booking_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Load booking transaction data for date range."""
        print(f"Loading booking data from {start_date} to {end_date}...")
        
        booking_file = self.data_path / 'bookings.csv'
        if not booking_file.exists():
            raise FileNotFoundError(f"Booking data file not found: {booking_file}")
        
        # Load with date filtering
        bookings_df = pd.read_csv(booking_file)
        bookings_df['booking_date'] = pd.to_datetime(bookings_df['booking_date'])
        bookings_df['travel_date'] = pd.to_datetime(bookings_df['travel_date'])
        
        # Filter by booking date range
        mask = (
            (bookings_df['booking_date'] >= start_date) &
            (bookings_df['booking_date'] <= end_date)
        )
        filtered_df = bookings_df[mask]
        
        print(f"Loaded {len(filtered_df)} booking records")
        return filtered_df
    
    def load_pricing_history(
        self, 
        route_codes: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Load pricing history data."""
        print("Loading pricing history...")
        
        pricing_file = self.data_path / 'pricing_history.csv'
        if not pricing_file.exists():
            raise FileNotFoundError(f"Pricing history file not found: {pricing_file}")
        
        pricing_df = pd.read_csv(pricing_file)
        pricing_df['timestamp'] = pd.to_datetime(pricing_df['timestamp'])
        
        # Apply filters
        if route_codes:
            pricing_df = pricing_df[pricing_df['route_code'].isin(route_codes)]
        
        if start_date:
            pricing_df = pricing_df[pricing_df['timestamp'] >= start_date]
        
        if end_date:
            pricing_df = pricing_df[pricing_df['timestamp'] <= end_date]
        
        print(f"Loaded {len(pricing_df)} pricing observations")
        return pricing_df
    
    def load_operational_data(
        self, 
        airline_codes: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Load operational performance data."""
        print("Loading operational data...")
        
        ops_file = self.data_path / 'flight_operations.csv'
        if not ops_file.exists():
            raise FileNotFoundError(f"Operational data file not found: {ops_file}")
        
        ops_df = pd.read_csv(ops_file)
        ops_df['date'] = pd.to_datetime(ops_df['date'])
        ops_df['scheduled_departure'] = pd.to_datetime(ops_df['scheduled_departure'])
        ops_df['actual_departure'] = pd.to_datetime(ops_df['actual_departure'])
        
        # Apply filters
        if airline_codes:
            ops_df = ops_df[ops_df['airline_code'].isin(airline_codes)]
        
        if start_date:
            ops_df = ops_df[ops_df['date'] >= start_date]
        
        if end_date:
            ops_df = ops_df[ops_df['date'] <= end_date]
        
        print(f"Loaded {len(ops_df)} flight operations")
        return ops_df
    
    def validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate loaded data for consistency and completeness."""
        print("Validating data...")
        
        validation_results = {
            'validation_timestamp': datetime.now().isoformat(),
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Validate airports
        airports = data.get('airports', [])
        if airports:
            validation_results['statistics']['airports_count'] = len(airports)
            
            # Check for duplicate IATA codes
            iata_codes = [a.iata_code for a in airports]
            if len(iata_codes) != len(set(iata_codes)):
                validation_results['errors'].append("Duplicate airport IATA codes found")
            
            # Check coordinate validity
            invalid_coords = [
                a.iata_code for a in airports 
                if not (-90 <= a.latitude <= 90) or not (-180 <= a.longitude <= 180)
            ]
            if invalid_coords:
                validation_results['errors'].append(
                    f"Invalid coordinates for airports: {invalid_coords}"
                )
        
        # Validate airlines
        airlines = data.get('airlines', [])
        if airlines:
            validation_results['statistics']['airlines_count'] = len(airlines)
            
            # Check for duplicate airline codes
            airline_codes = [a.airline_code for a in airlines]
            if len(airline_codes) != len(set(airline_codes)):
                validation_results['errors'].append("Duplicate airline codes found")
            
            # Check fleet sizes
            empty_fleets = [a.airline_code for a in airlines if len(a.fleet.aircraft) == 0]
            if empty_fleets:
                validation_results['warnings'].append(
                    f"Airlines with empty fleets: {empty_fleets}"
                )
        
        # Validate routes
        routes = data.get('routes', [])
        if routes:
            validation_results['statistics']['routes_count'] = len(routes)
            
            # Check for routes with zero demand
            zero_demand_routes = [r.route_code for r in routes if r.annual_demand <= 0]
            if zero_demand_routes:
                validation_results['warnings'].append(
                    f"Routes with zero demand: {len(zero_demand_routes)}"
                )
        
        # Validate time series data
        if 'fuel_prices' in data:
            fuel_df = data['fuel_prices']
            validation_results['statistics']['fuel_price_days'] = len(fuel_df)
            
            # Check for missing dates
            date_range = pd.date_range(
                start=fuel_df['date'].min(),
                end=fuel_df['date'].max(),
                freq='D'
            )
            missing_dates = set(date_range) - set(fuel_df['date'])
            if missing_dates:
                validation_results['warnings'].append(
                    f"Missing fuel price dates: {len(missing_dates)}"
                )
        
        # Cross-validation between datasets
        if airports and airlines:
            # Check if airline hubs exist in airport data
            airport_codes = {a.iata_code for a in airports}
            for airline in airlines:
                missing_hubs = set(airline.hub_airports) - airport_codes
                if missing_hubs:
                    validation_results['errors'].append(
                        f"Airline {airline.airline_code} has non-existent hub airports: {missing_hubs}"
                    )
        
        error_count = len(validation_results['errors'])
        warning_count = len(validation_results['warnings'])
        
        print(f"Validation completed: {error_count} errors, {warning_count} warnings")
        
        return validation_results
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics of loaded data."""
        summary = {
            'data_directory': str(self.data_path),
            'cache_status': {
                'airports_cached': self._airports_cache is not None,
                'airlines_cached': self._airlines_cache is not None,
                'routes_cached': self._routes_cache is not None,
                'customer_segments_cached': self._customer_segments_cache is not None
            }
        }
        
        # Add counts if data is cached
        if self._airports_cache:
            summary['airports_count'] = len(self._airports_cache)
        
        if self._airlines_cache:
            summary['airlines_count'] = len(self._airlines_cache)
            summary['total_aircraft'] = sum(len(a.fleet.aircraft) for a in self._airlines_cache)
        
        if self._routes_cache:
            summary['routes_count'] = len(self._routes_cache)
        
        if self._customer_segments_cache:
            summary['customer_segments_count'] = len(self._customer_segments_cache)
        
        return summary
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        print("Clearing data cache...")
        
        self._airports_cache = None
        self._airlines_cache = None
        self._routes_cache = None
        self._customer_segments_cache = None
        self._fuel_prices_cache = None
        self._economic_indicators_cache = None
        self._demand_patterns_cache = None
        
        print("Cache cleared.")


class DataProcessor:
    """Data processing utilities for analysis and modeling."""
    
    @staticmethod
    def aggregate_bookings_by_route(
        bookings_df: pd.DataFrame,
        time_period: str = 'D'  # D=daily, W=weekly, M=monthly
    ) -> pd.DataFrame:
        """Aggregate booking data by route and time period."""
        
        # Set booking_date as index for resampling
        bookings_df = bookings_df.set_index('booking_date')
        
        # Group by route and resample
        aggregated = bookings_df.groupby('route_code').resample(time_period).agg({
            'total_fare': ['sum', 'mean', 'count'],
            'lead_time_days': 'mean',
            'party_size': 'sum'
        }).reset_index()
        
        # Flatten column names
        aggregated.columns = [
            'route_code', 'period', 'total_revenue', 'average_fare', 
            'booking_count', 'avg_lead_time', 'total_passengers'
        ]
        
        return aggregated
    
    @staticmethod
    def calculate_route_performance(
        operational_df: pd.DataFrame,
        bookings_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate route performance metrics."""
        
        # Operational metrics by route
        ops_metrics = operational_df.groupby('route_code').agg({
            'load_factor': 'mean',
            'passengers_boarded': 'sum',
            'fuel_consumed_liters': 'sum',
            'fuel_cost': 'sum',
            'crew_cost': 'sum',
            'airport_fees': 'sum'
        }).reset_index()
        
        # Revenue metrics by route
        revenue_metrics = bookings_df.groupby('route_code').agg({
            'total_fare': 'sum',
            'booking_id': 'count'
        }).reset_index()
        revenue_metrics.columns = ['route_code', 'total_revenue', 'total_bookings']
        
        # Merge metrics
        performance = ops_metrics.merge(revenue_metrics, on='route_code', how='outer')
        
        # Calculate derived metrics
        performance['total_costs'] = (
            performance['fuel_cost'] + 
            performance['crew_cost'] + 
            performance['airport_fees']
        )
        
        performance['profit'] = performance['total_revenue'] - performance['total_costs']
        performance['profit_margin'] = (
            performance['profit'] / performance['total_revenue']
        ).fillna(0)
        
        return performance
    
    @staticmethod
    def analyze_price_elasticity(
        pricing_df: pd.DataFrame,
        bookings_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Analyze price elasticity by route and segment."""
        
        # This is a simplified elasticity analysis
        # In practice, would use more sophisticated econometric methods
        
        elasticity_results = []
        
        for route_code in pricing_df['route_code'].unique():
            route_pricing = pricing_df[pricing_df['route_code'] == route_code]
            route_bookings = bookings_df[bookings_df['route_code'] == route_code]
            
            if len(route_pricing) > 10 and len(route_bookings) > 10:
                # Simple correlation analysis
                # Group by date and calculate daily averages
                daily_prices = route_pricing.groupby(
                    route_pricing['timestamp'].dt.date
                )['price'].mean()
                
                daily_bookings = route_bookings.groupby(
                    route_bookings['booking_date'].dt.date
                )['booking_id'].count()
                
                # Align dates
                common_dates = set(daily_prices.index) & set(daily_bookings.index)
                if len(common_dates) > 5:
                    aligned_prices = daily_prices.loc[list(common_dates)]
                    aligned_bookings = daily_bookings.loc[list(common_dates)]
                    
                    # Calculate correlation (proxy for elasticity)
                    correlation = np.corrcoef(aligned_prices, aligned_bookings)[0, 1]
                    
                    elasticity_results.append({
                        'route_code': route_code,
                        'price_demand_correlation': correlation,
                        'avg_price': aligned_prices.mean(),
                        'avg_daily_bookings': aligned_bookings.mean(),
                        'observations': len(common_dates)
                    })
        
        return pd.DataFrame(elasticity_results)