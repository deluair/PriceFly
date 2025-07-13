"""Demand forecasting engine for airline revenue management.

This module provides advanced demand forecasting capabilities using various
statistical and machine learning models to predict passenger demand patterns.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, date, timedelta
from enum import Enum
import numpy as np
import logging
from collections import defaultdict
import math

from ..models.demand import DemandPattern, BookingCurve, DemandType, SeasonalityType
from ..models.route import Route
from ..models.market import Market


class ForecastModel(Enum):
    """Demand forecasting models."""
    SIMPLE_MOVING_AVERAGE = "simple_moving_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    LINEAR_REGRESSION = "linear_regression"
    SEASONAL_DECOMPOSITION = "seasonal_decomposition"
    ARIMA = "arima"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"


class ForecastHorizon(Enum):
    """Forecast time horizons."""
    SHORT_TERM = "short_term"  # 1-7 days
    MEDIUM_TERM = "medium_term"  # 1-4 weeks
    LONG_TERM = "long_term"  # 1-6 months
    STRATEGIC = "strategic"  # 6+ months


class ForecastAccuracy(Enum):
    """Forecast accuracy levels."""
    LOW = "low"  # ±25%
    MEDIUM = "medium"  # ±15%
    HIGH = "high"  # ±10%
    VERY_HIGH = "very_high"  # ±5%


@dataclass
class ForecastInput:
    """Input data for demand forecasting."""
    # Historical data
    historical_demand: List[float]
    historical_dates: List[date]
    historical_prices: List[float]
    
    # External factors
    economic_indicators: Dict[str, float] = field(default_factory=dict)
    seasonal_factors: Dict[str, float] = field(default_factory=dict)
    competitor_data: Dict[str, float] = field(default_factory=dict)
    
    # Route characteristics
    route_id: str = ""
    market_segment: str = "leisure"
    
    # Forecast parameters
    forecast_horizon_days: int = 30
    confidence_level: float = 0.95
    
    def validate(self) -> bool:
        """Validate input data quality."""
        if not self.historical_demand or not self.historical_dates:
            return False
        
        if len(self.historical_demand) != len(self.historical_dates):
            return False
        
        if len(self.historical_demand) < 7:  # Minimum data points
            return False
        
        return True
    
    def get_data_quality_score(self) -> float:
        """Calculate data quality score."""
        score = 1.0
        
        # Penalize for missing data
        missing_points = sum(1 for x in self.historical_demand if x is None or x < 0)
        if missing_points > 0:
            score *= (1 - missing_points / len(self.historical_demand))
        
        # Penalize for insufficient data
        if len(self.historical_demand) < 30:
            score *= 0.8
        elif len(self.historical_demand) < 90:
            score *= 0.9
        
        # Bonus for additional data sources
        if self.economic_indicators:
            score *= 1.05
        if self.competitor_data:
            score *= 1.05
        
        return min(1.0, score)


@dataclass
class ForecastResult:
    """Result of demand forecasting."""
    # Forecast values
    forecast_values: List[float]
    forecast_dates: List[date]
    
    # Confidence intervals
    upper_bound: List[float]
    lower_bound: List[float]
    confidence_level: float
    
    # Model performance
    model_used: ForecastModel
    accuracy_score: float
    mean_absolute_error: float
    root_mean_square_error: float
    
    # Forecast metadata
    forecast_horizon: ForecastHorizon
    data_quality_score: float
    model_confidence: float
    
    # Trend analysis
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    trend_strength: float  # 0-1 scale
    seasonality_detected: bool
    
    # Risk assessment
    forecast_risk: float  # 0-1 scale
    volatility_measure: float
    
    # Metadata
    forecast_timestamp: datetime = field(default_factory=datetime.now)
    route_id: str = ""
    
    def get_forecast_summary(self) -> Dict:
        """Get summary of forecast results."""
        return {
            'total_forecast_demand': sum(self.forecast_values),
            'average_daily_demand': np.mean(self.forecast_values),
            'peak_demand': max(self.forecast_values),
            'min_demand': min(self.forecast_values),
            'forecast_horizon_days': len(self.forecast_values),
            'model_used': self.model_used.value,
            'accuracy_score': self.accuracy_score,
            'confidence_level': self.confidence_level,
            'trend_direction': self.trend_direction,
            'seasonality_detected': self.seasonality_detected,
            'forecast_risk': self.forecast_risk,
            'data_quality': self.data_quality_score
        }
    
    def get_demand_for_date(self, target_date: date) -> Optional[Tuple[float, float, float]]:
        """Get forecast demand for specific date.
        
        Returns:
            Tuple of (forecast, lower_bound, upper_bound) or None if date not in forecast
        """
        try:
            index = self.forecast_dates.index(target_date)
            return (
                self.forecast_values[index],
                self.lower_bound[index],
                self.upper_bound[index]
            )
        except ValueError:
            return None


@dataclass
class SeasonalPattern:
    """Seasonal demand pattern."""
    pattern_name: str
    seasonal_factors: Dict[int, float]  # Month -> factor
    weekly_factors: Dict[int, float]  # Day of week -> factor
    holiday_factors: Dict[str, float]  # Holiday name -> factor
    
    # Pattern characteristics
    peak_months: List[int]
    low_months: List[int]
    volatility: float
    
    def get_seasonal_factor(self, date_obj: date) -> float:
        """Get seasonal factor for a specific date."""
        month_factor = self.seasonal_factors.get(date_obj.month, 1.0)
        weekday_factor = self.weekly_factors.get(date_obj.weekday(), 1.0)
        
        # Simple combination - in practice, this would be more sophisticated
        combined_factor = (month_factor + weekday_factor) / 2
        
        return combined_factor


class DemandForecaster:
    """Advanced demand forecasting engine for airline revenue management."""
    
    def __init__(
        self,
        default_model: ForecastModel = ForecastModel.EXPONENTIAL_SMOOTHING,
        default_horizon: ForecastHorizon = ForecastHorizon.MEDIUM_TERM
    ):
        self.default_model = default_model
        self.default_horizon = default_horizon
        
        # Historical data storage
        self.demand_history: Dict[str, List[Tuple[date, float]]] = defaultdict(list)
        self.price_history: Dict[str, List[Tuple[date, float]]] = defaultdict(list)
        self.forecast_history: Dict[str, List[ForecastResult]] = defaultdict(list)
        
        # Seasonal patterns
        self.seasonal_patterns: Dict[str, SeasonalPattern] = {}
        
        # Model parameters
        self.smoothing_alpha = 0.3  # Exponential smoothing parameter
        self.trend_beta = 0.1  # Trend smoothing parameter
        self.seasonal_gamma = 0.1  # Seasonal smoothing parameter
        
        # Forecast validation
        self.min_data_points = 14  # Minimum historical data points
        self.max_forecast_days = 180  # Maximum forecast horizon
        
        self.logger = logging.getLogger(__name__)
    
    def forecast_demand(
        self,
        forecast_input: ForecastInput,
        model: Optional[ForecastModel] = None,
        horizon: Optional[ForecastHorizon] = None
    ) -> ForecastResult:
        """Generate demand forecast."""
        
        if not forecast_input.validate():
            raise ValueError("Invalid forecast input data")
        
        model = model or self.default_model
        horizon = horizon or self.default_horizon
        
        self.logger.info(
            f"Generating demand forecast for route {forecast_input.route_id} "
            f"using {model.value} model with {horizon.value} horizon"
        )
        
        # Determine forecast horizon in days
        horizon_days = self._get_horizon_days(horizon, forecast_input.forecast_horizon_days)
        
        # Select and apply forecasting model
        if model == ForecastModel.SIMPLE_MOVING_AVERAGE:
            result = self._forecast_moving_average(forecast_input, horizon_days)
        elif model == ForecastModel.EXPONENTIAL_SMOOTHING:
            result = self._forecast_exponential_smoothing(forecast_input, horizon_days)
        elif model == ForecastModel.LINEAR_REGRESSION:
            result = self._forecast_linear_regression(forecast_input, horizon_days)
        elif model == ForecastModel.SEASONAL_DECOMPOSITION:
            result = self._forecast_seasonal_decomposition(forecast_input, horizon_days)
        elif model == ForecastModel.ENSEMBLE:
            result = self._forecast_ensemble(forecast_input, horizon_days)
        else:
            # Default to exponential smoothing
            result = self._forecast_exponential_smoothing(forecast_input, horizon_days)
        
        # Store forecast for validation
        self.forecast_history[forecast_input.route_id].append(result)
        
        # Keep only recent forecasts
        if len(self.forecast_history[forecast_input.route_id]) > 100:
            self.forecast_history[forecast_input.route_id] = \
                self.forecast_history[forecast_input.route_id][-100:]
        
        self.logger.info(
            f"Forecast completed. Average daily demand: {np.mean(result.forecast_values):.1f}, "
            f"Accuracy score: {result.accuracy_score:.3f}"
        )
        
        return result
    
    def _get_horizon_days(self, horizon: ForecastHorizon, requested_days: int) -> int:
        """Convert horizon enum to days."""
        horizon_mapping = {
            ForecastHorizon.SHORT_TERM: min(7, requested_days),
            ForecastHorizon.MEDIUM_TERM: min(30, requested_days),
            ForecastHorizon.LONG_TERM: min(180, requested_days),
            ForecastHorizon.STRATEGIC: min(365, requested_days)
        }
        
        return horizon_mapping.get(horizon, requested_days)
    
    def _forecast_moving_average(
        self,
        forecast_input: ForecastInput,
        horizon_days: int
    ) -> ForecastResult:
        """Simple moving average forecast."""
        
        historical_data = np.array(forecast_input.historical_demand)
        
        # Use last N points for moving average
        window_size = min(7, len(historical_data))
        moving_avg = np.mean(historical_data[-window_size:])
        
        # Generate forecast dates
        start_date = forecast_input.historical_dates[-1] + timedelta(days=1)
        forecast_dates = [start_date + timedelta(days=i) for i in range(horizon_days)]
        
        # Simple forecast - constant value
        forecast_values = [moving_avg] * horizon_days
        
        # Calculate confidence intervals (simplified)
        std_dev = np.std(historical_data[-window_size:])
        margin = 1.96 * std_dev  # 95% confidence interval
        
        upper_bound = [f + margin for f in forecast_values]
        lower_bound = [max(0, f - margin) for f in forecast_values]
        
        # Calculate accuracy metrics
        accuracy_score = self._calculate_accuracy_score(historical_data, window_size)
        mae = std_dev  # Simplified
        rmse = std_dev * 1.2  # Simplified
        
        return ForecastResult(
            forecast_values=forecast_values,
            forecast_dates=forecast_dates,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            confidence_level=0.95,
            model_used=ForecastModel.SIMPLE_MOVING_AVERAGE,
            accuracy_score=accuracy_score,
            mean_absolute_error=mae,
            root_mean_square_error=rmse,
            forecast_horizon=self._classify_horizon(horizon_days),
            data_quality_score=forecast_input.get_data_quality_score(),
            model_confidence=0.7,
            trend_direction=self._detect_trend(historical_data),
            trend_strength=self._calculate_trend_strength(historical_data),
            seasonality_detected=False,
            forecast_risk=0.3,
            volatility_measure=std_dev / np.mean(historical_data) if np.mean(historical_data) > 0 else 0,
            route_id=forecast_input.route_id
        )
    
    def _forecast_exponential_smoothing(
        self,
        forecast_input: ForecastInput,
        horizon_days: int
    ) -> ForecastResult:
        """Exponential smoothing forecast with trend and seasonality."""
        
        historical_data = np.array(forecast_input.historical_demand)
        
        # Initialize components
        level = historical_data[0]
        trend = 0
        seasonal = np.ones(7)  # Weekly seasonality
        
        # Fit exponential smoothing model
        smoothed_values = []
        
        for i, value in enumerate(historical_data):
            # Update level
            season_index = i % 7
            level_new = self.smoothing_alpha * (value / seasonal[season_index]) + \
                       (1 - self.smoothing_alpha) * (level + trend)
            
            # Update trend
            trend_new = self.trend_beta * (level_new - level) + \
                       (1 - self.trend_beta) * trend
            
            # Update seasonal
            seasonal[season_index] = self.seasonal_gamma * (value / level_new) + \
                                   (1 - self.seasonal_gamma) * seasonal[season_index]
            
            level = level_new
            trend = trend_new
            
            smoothed_values.append(level * seasonal[season_index])
        
        # Generate forecast
        start_date = forecast_input.historical_dates[-1] + timedelta(days=1)
        forecast_dates = [start_date + timedelta(days=i) for i in range(horizon_days)]
        forecast_values = []
        
        for i in range(horizon_days):
            season_index = (len(historical_data) + i) % 7
            forecast_value = (level + trend * (i + 1)) * seasonal[season_index]
            forecast_values.append(max(0, forecast_value))
        
        # Calculate confidence intervals
        residuals = historical_data - np.array(smoothed_values)
        std_dev = np.std(residuals)
        
        # Expanding confidence intervals for longer horizons
        upper_bound = []
        lower_bound = []
        
        for i, f in enumerate(forecast_values):
            margin = 1.96 * std_dev * math.sqrt(1 + i * 0.1)  # Expanding uncertainty
            upper_bound.append(f + margin)
            lower_bound.append(max(0, f - margin))
        
        # Calculate accuracy metrics
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals ** 2))
        accuracy_score = max(0, 1 - mae / np.mean(historical_data)) if np.mean(historical_data) > 0 else 0
        
        return ForecastResult(
            forecast_values=forecast_values,
            forecast_dates=forecast_dates,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            confidence_level=0.95,
            model_used=ForecastModel.EXPONENTIAL_SMOOTHING,
            accuracy_score=accuracy_score,
            mean_absolute_error=mae,
            root_mean_square_error=rmse,
            forecast_horizon=self._classify_horizon(horizon_days),
            data_quality_score=forecast_input.get_data_quality_score(),
            model_confidence=0.85,
            trend_direction=self._detect_trend(historical_data),
            trend_strength=self._calculate_trend_strength(historical_data),
            seasonality_detected=True,
            forecast_risk=0.2,
            volatility_measure=std_dev / np.mean(historical_data) if np.mean(historical_data) > 0 else 0,
            route_id=forecast_input.route_id
        )
    
    def _forecast_linear_regression(
        self,
        forecast_input: ForecastInput,
        horizon_days: int
    ) -> ForecastResult:
        """Linear regression forecast."""
        
        historical_data = np.array(forecast_input.historical_demand)
        x = np.arange(len(historical_data))
        
        # Fit linear regression
        coeffs = np.polyfit(x, historical_data, 1)
        slope, intercept = coeffs
        
        # Generate forecast
        start_date = forecast_input.historical_dates[-1] + timedelta(days=1)
        forecast_dates = [start_date + timedelta(days=i) for i in range(horizon_days)]
        
        forecast_values = []
        for i in range(horizon_days):
            forecast_value = slope * (len(historical_data) + i) + intercept
            forecast_values.append(max(0, forecast_value))
        
        # Calculate residuals and confidence intervals
        fitted_values = slope * x + intercept
        residuals = historical_data - fitted_values
        std_dev = np.std(residuals)
        
        upper_bound = [f + 1.96 * std_dev for f in forecast_values]
        lower_bound = [max(0, f - 1.96 * std_dev) for f in forecast_values]
        
        # Calculate accuracy metrics
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals ** 2))
        r_squared = 1 - np.sum(residuals ** 2) / np.sum((historical_data - np.mean(historical_data)) ** 2)
        accuracy_score = max(0, r_squared)
        
        return ForecastResult(
            forecast_values=forecast_values,
            forecast_dates=forecast_dates,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            confidence_level=0.95,
            model_used=ForecastModel.LINEAR_REGRESSION,
            accuracy_score=accuracy_score,
            mean_absolute_error=mae,
            root_mean_square_error=rmse,
            forecast_horizon=self._classify_horizon(horizon_days),
            data_quality_score=forecast_input.get_data_quality_score(),
            model_confidence=0.75,
            trend_direction=self._detect_trend(historical_data),
            trend_strength=abs(slope) / np.mean(historical_data) if np.mean(historical_data) > 0 else 0,
            seasonality_detected=False,
            forecast_risk=0.25,
            volatility_measure=std_dev / np.mean(historical_data) if np.mean(historical_data) > 0 else 0,
            route_id=forecast_input.route_id
        )
    
    def _forecast_seasonal_decomposition(
        self,
        forecast_input: ForecastInput,
        horizon_days: int
    ) -> ForecastResult:
        """Seasonal decomposition forecast."""
        
        historical_data = np.array(forecast_input.historical_demand)
        
        # Simple seasonal decomposition (weekly pattern)
        if len(historical_data) < 14:  # Need at least 2 weeks
            return self._forecast_exponential_smoothing(forecast_input, horizon_days)
        
        # Calculate weekly averages
        weekly_pattern = np.zeros(7)
        weekly_counts = np.zeros(7)
        
        for i, value in enumerate(historical_data):
            day_of_week = i % 7
            weekly_pattern[day_of_week] += value
            weekly_counts[day_of_week] += 1
        
        # Normalize weekly pattern
        for i in range(7):
            if weekly_counts[i] > 0:
                weekly_pattern[i] /= weekly_counts[i]
        
        # Calculate trend
        x = np.arange(len(historical_data))
        trend_coeffs = np.polyfit(x, historical_data, 1)
        trend_slope, trend_intercept = trend_coeffs
        
        # Generate forecast
        start_date = forecast_input.historical_dates[-1] + timedelta(days=1)
        forecast_dates = [start_date + timedelta(days=i) for i in range(horizon_days)]
        forecast_values = []
        
        base_level = np.mean(historical_data)
        
        for i in range(horizon_days):
            # Trend component
            trend_value = trend_slope * (len(historical_data) + i) + trend_intercept
            
            # Seasonal component
            day_of_week = (len(historical_data) + i) % 7
            seasonal_factor = weekly_pattern[day_of_week] / base_level if base_level > 0 else 1
            
            forecast_value = trend_value * seasonal_factor
            forecast_values.append(max(0, forecast_value))
        
        # Calculate confidence intervals
        detrended = historical_data - (trend_slope * x + trend_intercept)
        std_dev = np.std(detrended)
        
        upper_bound = [f + 1.96 * std_dev for f in forecast_values]
        lower_bound = [max(0, f - 1.96 * std_dev) for f in forecast_values]
        
        # Calculate accuracy metrics
        fitted_values = []
        for i in range(len(historical_data)):
            trend_val = trend_slope * i + trend_intercept
            seasonal_factor = weekly_pattern[i % 7] / base_level if base_level > 0 else 1
            fitted_values.append(trend_val * seasonal_factor)
        
        residuals = historical_data - np.array(fitted_values)
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals ** 2))
        accuracy_score = max(0, 1 - mae / np.mean(historical_data)) if np.mean(historical_data) > 0 else 0
        
        return ForecastResult(
            forecast_values=forecast_values,
            forecast_dates=forecast_dates,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            confidence_level=0.95,
            model_used=ForecastModel.SEASONAL_DECOMPOSITION,
            accuracy_score=accuracy_score,
            mean_absolute_error=mae,
            root_mean_square_error=rmse,
            forecast_horizon=self._classify_horizon(horizon_days),
            data_quality_score=forecast_input.get_data_quality_score(),
            model_confidence=0.8,
            trend_direction=self._detect_trend(historical_data),
            trend_strength=abs(trend_slope) / np.mean(historical_data) if np.mean(historical_data) > 0 else 0,
            seasonality_detected=True,
            forecast_risk=0.2,
            volatility_measure=std_dev / np.mean(historical_data) if np.mean(historical_data) > 0 else 0,
            route_id=forecast_input.route_id
        )
    
    def _forecast_ensemble(
        self,
        forecast_input: ForecastInput,
        horizon_days: int
    ) -> ForecastResult:
        """Ensemble forecast combining multiple models."""
        
        # Generate forecasts from multiple models
        models = [
            ForecastModel.EXPONENTIAL_SMOOTHING,
            ForecastModel.LINEAR_REGRESSION,
            ForecastModel.SEASONAL_DECOMPOSITION
        ]
        
        forecasts = []
        weights = [0.4, 0.3, 0.3]  # Weights for ensemble
        
        for model in models:
            try:
                if model == ForecastModel.EXPONENTIAL_SMOOTHING:
                    forecast = self._forecast_exponential_smoothing(forecast_input, horizon_days)
                elif model == ForecastModel.LINEAR_REGRESSION:
                    forecast = self._forecast_linear_regression(forecast_input, horizon_days)
                elif model == ForecastModel.SEASONAL_DECOMPOSITION:
                    forecast = self._forecast_seasonal_decomposition(forecast_input, horizon_days)
                
                forecasts.append(forecast)
            except Exception as e:
                self.logger.warning(f"Model {model.value} failed: {e}")
                continue
        
        if not forecasts:
            # Fallback to simple moving average
            return self._forecast_moving_average(forecast_input, horizon_days)
        
        # Combine forecasts
        ensemble_values = []
        ensemble_upper = []
        ensemble_lower = []
        
        for i in range(horizon_days):
            weighted_value = sum(
                w * f.forecast_values[i] for w, f in zip(weights[:len(forecasts)], forecasts)
            ) / sum(weights[:len(forecasts)])
            
            weighted_upper = sum(
                w * f.upper_bound[i] for w, f in zip(weights[:len(forecasts)], forecasts)
            ) / sum(weights[:len(forecasts)])
            
            weighted_lower = sum(
                w * f.lower_bound[i] for w, f in zip(weights[:len(forecasts)], forecasts)
            ) / sum(weights[:len(forecasts)])
            
            ensemble_values.append(weighted_value)
            ensemble_upper.append(weighted_upper)
            ensemble_lower.append(weighted_lower)
        
        # Calculate ensemble accuracy
        ensemble_accuracy = sum(
            w * f.accuracy_score for w, f in zip(weights[:len(forecasts)], forecasts)
        ) / sum(weights[:len(forecasts)])
        
        ensemble_mae = sum(
            w * f.mean_absolute_error for w, f in zip(weights[:len(forecasts)], forecasts)
        ) / sum(weights[:len(forecasts)])
        
        ensemble_rmse = sum(
            w * f.root_mean_square_error for w, f in zip(weights[:len(forecasts)], forecasts)
        ) / sum(weights[:len(forecasts)])
        
        return ForecastResult(
            forecast_values=ensemble_values,
            forecast_dates=forecasts[0].forecast_dates,
            upper_bound=ensemble_upper,
            lower_bound=ensemble_lower,
            confidence_level=0.95,
            model_used=ForecastModel.ENSEMBLE,
            accuracy_score=ensemble_accuracy,
            mean_absolute_error=ensemble_mae,
            root_mean_square_error=ensemble_rmse,
            forecast_horizon=self._classify_horizon(horizon_days),
            data_quality_score=forecast_input.get_data_quality_score(),
            model_confidence=0.9,
            trend_direction=forecasts[0].trend_direction,
            trend_strength=forecasts[0].trend_strength,
            seasonality_detected=any(f.seasonality_detected for f in forecasts),
            forecast_risk=0.15,
            volatility_measure=np.mean([f.volatility_measure for f in forecasts]),
            route_id=forecast_input.route_id
        )
    
    def _detect_trend(self, data: np.ndarray) -> str:
        """Detect trend direction in historical data."""
        if len(data) < 3:
            return 'stable'
        
        # Simple trend detection using linear regression
        x = np.arange(len(data))
        slope = np.polyfit(x, data, 1)[0]
        
        # Threshold for trend detection
        mean_value = np.mean(data)
        threshold = mean_value * 0.01  # 1% of mean value
        
        if slope > threshold:
            return 'increasing'
        elif slope < -threshold:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_trend_strength(self, data: np.ndarray) -> float:
        """Calculate strength of trend (0-1 scale)."""
        if len(data) < 3:
            return 0.0
        
        x = np.arange(len(data))
        slope, intercept = np.polyfit(x, data, 1)
        
        # Calculate R-squared for trend line
        fitted_values = slope * x + intercept
        ss_res = np.sum((data - fitted_values) ** 2)
        ss_tot = np.sum((data - np.mean(data)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        r_squared = 1 - (ss_res / ss_tot)
        return max(0.0, min(1.0, r_squared))
    
    def _calculate_accuracy_score(self, data: np.ndarray, window_size: int) -> float:
        """Calculate accuracy score based on historical performance."""
        if len(data) < window_size * 2:
            return 0.7  # Default moderate accuracy
        
        # Calculate coefficient of variation
        cv = np.std(data) / np.mean(data) if np.mean(data) > 0 else 1.0
        
        # Lower CV means higher predictability
        accuracy = max(0.3, 1.0 - cv)
        return min(1.0, accuracy)
    
    def _classify_horizon(self, days: int) -> ForecastHorizon:
        """Classify forecast horizon based on number of days."""
        if days <= 7:
            return ForecastHorizon.SHORT_TERM
        elif days <= 30:
            return ForecastHorizon.MEDIUM_TERM
        elif days <= 180:
            return ForecastHorizon.LONG_TERM
        else:
            return ForecastHorizon.STRATEGIC
    
    def validate_forecast(
        self,
        route_id: str,
        actual_demand: List[float],
        actual_dates: List[date]
    ) -> Dict:
        """Validate previous forecasts against actual demand."""
        
        if route_id not in self.forecast_history:
            return {'error': 'No forecast history for route'}
        
        recent_forecasts = self.forecast_history[route_id][-5:]  # Last 5 forecasts
        validation_results = []
        
        for forecast in recent_forecasts:
            # Find overlapping dates
            overlapping_results = []
            
            for i, forecast_date in enumerate(forecast.forecast_dates):
                if forecast_date in actual_dates:
                    actual_index = actual_dates.index(forecast_date)
                    actual_value = actual_demand[actual_index]
                    forecast_value = forecast.forecast_values[i]
                    
                    error = abs(actual_value - forecast_value)
                    percentage_error = (error / actual_value * 100) if actual_value > 0 else 0
                    
                    overlapping_results.append({
                        'date': forecast_date.isoformat(),
                        'actual': actual_value,
                        'forecast': forecast_value,
                        'error': error,
                        'percentage_error': percentage_error
                    })
            
            if overlapping_results:
                avg_error = np.mean([r['error'] for r in overlapping_results])
                avg_percentage_error = np.mean([r['percentage_error'] for r in overlapping_results])
                
                validation_results.append({
                    'forecast_timestamp': forecast.forecast_timestamp.isoformat(),
                    'model_used': forecast.model_used.value,
                    'overlapping_points': len(overlapping_results),
                    'average_error': avg_error,
                    'average_percentage_error': avg_percentage_error,
                    'predicted_accuracy': forecast.accuracy_score,
                    'details': overlapping_results
                })
        
        return {
            'route_id': route_id,
            'validation_results': validation_results,
            'total_forecasts_validated': len(validation_results)
        }
    
    def update_demand_history(
        self,
        route_id: str,
        demand_data: List[Tuple[date, float]]
    ) -> None:
        """Update historical demand data."""
        
        self.demand_history[route_id].extend(demand_data)
        
        # Sort by date and remove duplicates
        self.demand_history[route_id] = sorted(
            list(set(self.demand_history[route_id])),
            key=lambda x: x[0]
        )
        
        # Keep only recent history (last 2 years)
        cutoff_date = date.today() - timedelta(days=730)
        self.demand_history[route_id] = [
            (d, v) for d, v in self.demand_history[route_id] if d >= cutoff_date
        ]
    
    def get_forecast_performance(
        self,
        route_id: str,
        days: int = 30
    ) -> Dict:
        """Get forecast performance summary."""
        
        if route_id not in self.forecast_history:
            return {'error': f'No forecast history for route {route_id}'}
        
        recent_forecasts = self.forecast_history[route_id][-days:]
        
        if not recent_forecasts:
            return {'error': f'No recent forecasts for route {route_id}'}
        
        # Calculate performance metrics
        avg_accuracy = np.mean([f.accuracy_score for f in recent_forecasts])
        avg_confidence = np.mean([f.model_confidence for f in recent_forecasts])
        avg_risk = np.mean([f.forecast_risk for f in recent_forecasts])
        
        model_usage = defaultdict(int)
        for forecast in recent_forecasts:
            model_usage[forecast.model_used.value] += 1
        
        return {
            'route_id': route_id,
            'period_days': days,
            'total_forecasts': len(recent_forecasts),
            'average_accuracy': avg_accuracy,
            'average_confidence': avg_confidence,
            'average_risk': avg_risk,
            'model_usage': dict(model_usage),
            'most_used_model': max(model_usage.items(), key=lambda x: x[1])[0] if model_usage else None
        }
    
    def create_seasonal_pattern(
        self,
        pattern_name: str,
        route_id: str
    ) -> SeasonalPattern:
        """Create seasonal pattern from historical data."""
        
        if route_id not in self.demand_history:
            raise ValueError(f"No historical data for route {route_id}")
        
        demand_data = self.demand_history[route_id]
        
        if len(demand_data) < 30:  # Need at least 30 data points
            raise ValueError("Insufficient data to create seasonal pattern")
        
        # Calculate monthly factors
        monthly_totals = defaultdict(list)
        weekly_totals = defaultdict(list)
        
        for date_obj, demand in demand_data:
            monthly_totals[date_obj.month].append(demand)
            weekly_totals[date_obj.weekday()].append(demand)
        
        # Calculate average factors
        overall_avg = np.mean([d for _, d in demand_data])
        
        seasonal_factors = {}
        for month in range(1, 13):
            if month in monthly_totals:
                month_avg = np.mean(monthly_totals[month])
                seasonal_factors[month] = month_avg / overall_avg if overall_avg > 0 else 1.0
            else:
                seasonal_factors[month] = 1.0
        
        weekly_factors = {}
        for day in range(7):
            if day in weekly_totals:
                day_avg = np.mean(weekly_totals[day])
                weekly_factors[day] = day_avg / overall_avg if overall_avg > 0 else 1.0
            else:
                weekly_factors[day] = 1.0
        
        # Identify peak and low months
        peak_months = [m for m, f in seasonal_factors.items() if f > 1.2]
        low_months = [m for m, f in seasonal_factors.items() if f < 0.8]
        
        # Calculate volatility
        volatility = np.std([d for _, d in demand_data]) / overall_avg if overall_avg > 0 else 0
        
        pattern = SeasonalPattern(
            pattern_name=pattern_name,
            seasonal_factors=seasonal_factors,
            weekly_factors=weekly_factors,
            holiday_factors={},  # Simplified - no holiday data
            peak_months=peak_months,
            low_months=low_months,
            volatility=volatility
        )
        
        self.seasonal_patterns[pattern_name] = pattern
        return pattern
    
    def export_forecaster_data(self) -> Dict:
        """Export demand forecaster configuration and data."""
        
        return {
            'configuration': {
                'default_model': self.default_model.value,
                'default_horizon': self.default_horizon.value,
                'smoothing_alpha': self.smoothing_alpha,
                'trend_beta': self.trend_beta,
                'seasonal_gamma': self.seasonal_gamma,
                'min_data_points': self.min_data_points,
                'max_forecast_days': self.max_forecast_days
            },
            'data_summary': {
                'routes_with_demand_history': len(self.demand_history),
                'routes_with_forecast_history': len(self.forecast_history),
                'seasonal_patterns': len(self.seasonal_patterns),
                'total_demand_records': sum(len(history) for history in self.demand_history.values()),
                'total_forecast_records': sum(len(history) for history in self.forecast_history.values())
            },
            'seasonal_patterns': list(self.seasonal_patterns.keys())
        }