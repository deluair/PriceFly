"""Visualization components for airline pricing simulation analytics."""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import base64
import io


class ChartType(Enum):
    """Types of charts available."""
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    PIE = "pie"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"
    BOX = "box"
    AREA = "area"
    CANDLESTICK = "candlestick"
    GAUGE = "gauge"
    TREEMAP = "treemap"
    SUNBURST = "sunburst"


class ChartLibrary(Enum):
    """Chart libraries available."""
    PLOTLY = "plotly"
    MATPLOTLIB = "matplotlib"
    SEABORN = "seaborn"


@dataclass
class ChartConfig:
    """Configuration for chart generation."""
    chart_type: ChartType
    title: str
    x_axis_title: str = ""
    y_axis_title: str = ""
    width: int = 800
    height: int = 600
    color_scheme: str = "viridis"
    show_legend: bool = True
    show_grid: bool = True
    library: ChartLibrary = ChartLibrary.PLOTLY
    custom_colors: Optional[List[str]] = None
    annotations: List[Dict[str, Any]] = None
    theme: str = "plotly_white"


@dataclass
class ChartData:
    """Data structure for chart visualization."""
    data: pd.DataFrame
    x_column: str
    y_column: str
    color_column: Optional[str] = None
    size_column: Optional[str] = None
    hover_columns: List[str] = None
    group_column: Optional[str] = None


class PricingVisualizer:
    """Visualizations for pricing analysis."""
    
    def __init__(self, output_dir: str = "charts"):
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set default styling
        self._setup_styling()
    
    def _setup_styling(self):
        """Setup default styling for charts."""
        # Matplotlib styling
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Custom color schemes
        self.color_schemes = {
            'airline_blue': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            'revenue': ['#2E8B57', '#32CD32', '#90EE90', '#98FB98', '#F0FFF0'],
            'competitive': ['#B22222', '#DC143C', '#FF6347', '#FFA07A', '#FFE4E1'],
            'operational': ['#4169E1', '#6495ED', '#87CEEB', '#B0E0E6', '#F0F8FF']
        }
    
    def create_price_trend_chart(
        self,
        pricing_data: pd.DataFrame,
        config: Optional[ChartConfig] = None
    ) -> go.Figure:
        """Create price trend visualization."""
        
        if config is None:
            config = ChartConfig(
                chart_type=ChartType.LINE,
                title="Price Trends Over Time",
                x_axis_title="Date",
                y_axis_title="Price ($)"
            )
        
        fig = go.Figure()
        
        # Group by route or airline if available
        if 'route_id' in pricing_data.columns:
            for route in pricing_data['route_id'].unique():
                route_data = pricing_data[pricing_data['route_id'] == route]
                fig.add_trace(go.Scatter(
                    x=route_data['date'],
                    y=route_data['price'],
                    mode='lines+markers',
                    name=f'Route {route}',
                    line=dict(width=2)
                ))
        else:
            fig.add_trace(go.Scatter(
                x=pricing_data['date'],
                y=pricing_data['price'],
                mode='lines+markers',
                name='Average Price',
                line=dict(width=3, color='#1f77b4')
            ))
        
        fig.update_layout(
            title=config.title,
            xaxis_title=config.x_axis_title,
            yaxis_title=config.y_axis_title,
            width=config.width,
            height=config.height,
            template=config.theme,
            showlegend=config.show_legend
        )
        
        return fig
    
    def create_price_distribution_chart(
        self,
        pricing_data: pd.DataFrame,
        config: Optional[ChartConfig] = None
    ) -> go.Figure:
        """Create price distribution histogram."""
        
        if config is None:
            config = ChartConfig(
                chart_type=ChartType.HISTOGRAM,
                title="Price Distribution",
                x_axis_title="Price ($)",
                y_axis_title="Frequency"
            )
        
        fig = go.Figure()
        
        # Create histogram
        fig.add_trace(go.Histogram(
            x=pricing_data['price'],
            nbinsx=30,
            name='Price Distribution',
            marker_color='#1f77b4',
            opacity=0.7
        ))
        
        # Add statistical lines
        mean_price = pricing_data['price'].mean()
        median_price = pricing_data['price'].median()
        
        fig.add_vline(
            x=mean_price,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: ${mean_price:.2f}"
        )
        
        fig.add_vline(
            x=median_price,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Median: ${median_price:.2f}"
        )
        
        fig.update_layout(
            title=config.title,
            xaxis_title=config.x_axis_title,
            yaxis_title=config.y_axis_title,
            width=config.width,
            height=config.height,
            template=config.theme
        )
        
        return fig
    
    def create_price_elasticity_chart(
        self,
        elasticity_data: pd.DataFrame,
        config: Optional[ChartConfig] = None
    ) -> go.Figure:
        """Create price elasticity visualization."""
        
        if config is None:
            config = ChartConfig(
                chart_type=ChartType.SCATTER,
                title="Price Elasticity Analysis",
                x_axis_title="Price Change (%)",
                y_axis_title="Demand Change (%)"
            )
        
        fig = go.Figure()
        
        # Scatter plot of price vs demand changes
        fig.add_trace(go.Scatter(
            x=elasticity_data['price_change_pct'],
            y=elasticity_data['demand_change_pct'],
            mode='markers',
            marker=dict(
                size=8,
                color=elasticity_data.get('route_revenue', elasticity_data.index),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Revenue")
            ),
            text=elasticity_data.get('route_name', elasticity_data.index),
            hovertemplate='<b>%{text}</b><br>' +
                         'Price Change: %{x:.1f}%<br>' +
                         'Demand Change: %{y:.1f}%<br>' +
                         '<extra></extra>'
        ))
        
        # Add elasticity reference lines
        x_range = [-50, 50]
        
        # Elastic demand (elasticity > 1)
        fig.add_trace(go.Scatter(
            x=x_range,
            y=[-x * 1.5 for x in x_range],  # Elasticity = -1.5
            mode='lines',
            line=dict(dash='dash', color='red'),
            name='Elastic (E = -1.5)',
            showlegend=True
        ))
        
        # Unit elastic (elasticity = 1)
        fig.add_trace(go.Scatter(
            x=x_range,
            y=[-x for x in x_range],  # Elasticity = -1.0
            mode='lines',
            line=dict(dash='dash', color='orange'),
            name='Unit Elastic (E = -1.0)',
            showlegend=True
        ))
        
        # Inelastic demand (elasticity < 1)
        fig.add_trace(go.Scatter(
            x=x_range,
            y=[-x * 0.5 for x in x_range],  # Elasticity = -0.5
            mode='lines',
            line=dict(dash='dash', color='green'),
            name='Inelastic (E = -0.5)',
            showlegend=True
        ))
        
        fig.update_layout(
            title=config.title,
            xaxis_title=config.x_axis_title,
            yaxis_title=config.y_axis_title,
            width=config.width,
            height=config.height,
            template=config.theme
        )
        
        return fig
    
    def create_booking_class_mix_chart(
        self,
        booking_data: pd.DataFrame,
        config: Optional[ChartConfig] = None
    ) -> go.Figure:
        """Create booking class mix visualization."""
        
        if config is None:
            config = ChartConfig(
                chart_type=ChartType.PIE,
                title="Booking Class Mix",
                width=600,
                height=600
            )
        
        # Calculate booking class distribution
        class_counts = booking_data['booking_class'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=class_counts.index,
            values=class_counts.values,
            hole=0.3,
            textinfo='label+percent',
            textposition='outside',
            marker_colors=self.color_schemes['airline_blue']
        )])
        
        fig.update_layout(
            title=config.title,
            width=config.width,
            height=config.height,
            template=config.theme,
            annotations=[dict(text='Booking<br>Classes', x=0.5, y=0.5, font_size=16, showarrow=False)]
        )
        
        return fig


class DemandVisualizer:
    """Visualizations for demand analysis."""
    
    def __init__(self, output_dir: str = "charts"):
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def create_demand_forecast_chart(
        self,
        demand_data: pd.DataFrame,
        forecast_data: pd.DataFrame,
        config: Optional[ChartConfig] = None
    ) -> go.Figure:
        """Create demand forecast visualization."""
        
        if config is None:
            config = ChartConfig(
                chart_type=ChartType.LINE,
                title="Demand Forecast",
                x_axis_title="Date",
                y_axis_title="Demand (Passengers)"
            )
        
        fig = go.Figure()
        
        # Historical demand
        fig.add_trace(go.Scatter(
            x=demand_data['date'],
            y=demand_data['demand'],
            mode='lines+markers',
            name='Historical Demand',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Forecasted demand
        fig.add_trace(go.Scatter(
            x=forecast_data['date'],
            y=forecast_data['forecast'],
            mode='lines',
            name='Forecast',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
        
        # Confidence intervals
        if 'upper_bound' in forecast_data.columns and 'lower_bound' in forecast_data.columns:
            fig.add_trace(go.Scatter(
                x=forecast_data['date'],
                y=forecast_data['upper_bound'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_data['date'],
                y=forecast_data['lower_bound'],
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(255, 127, 14, 0.2)',
                fill='tonexty',
                name='Confidence Interval',
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title=config.title,
            xaxis_title=config.x_axis_title,
            yaxis_title=config.y_axis_title,
            width=config.width,
            height=config.height,
            template=config.theme
        )
        
        return fig
    
    def create_booking_curve_chart(
        self,
        booking_data: pd.DataFrame,
        config: Optional[ChartConfig] = None
    ) -> go.Figure:
        """Create booking curve visualization."""
        
        if config is None:
            config = ChartConfig(
                chart_type=ChartType.AREA,
                title="Booking Curve Analysis",
                x_axis_title="Days Before Departure",
                y_axis_title="Cumulative Bookings (%)"
            )
        
        fig = go.Figure()
        
        # Calculate cumulative booking percentage
        booking_data = booking_data.sort_values('days_before_departure', ascending=False)
        booking_data['cumulative_pct'] = (booking_data['bookings'].cumsum() / 
                                         booking_data['bookings'].sum() * 100)
        
        # Group by customer segment if available
        if 'segment' in booking_data.columns:
            for segment in booking_data['segment'].unique():
                segment_data = booking_data[booking_data['segment'] == segment]
                segment_data = segment_data.sort_values('days_before_departure', ascending=False)
                segment_data['cumulative_pct'] = (segment_data['bookings'].cumsum() / 
                                                segment_data['bookings'].sum() * 100)
                
                fig.add_trace(go.Scatter(
                    x=segment_data['days_before_departure'],
                    y=segment_data['cumulative_pct'],
                    mode='lines',
                    name=f'{segment.title()} Travelers',
                    fill='tonexty' if segment != booking_data['segment'].unique()[0] else None,
                    stackgroup='one'
                ))
        else:
            fig.add_trace(go.Scatter(
                x=booking_data['days_before_departure'],
                y=booking_data['cumulative_pct'],
                mode='lines+markers',
                name='All Bookings',
                fill='tozeroy',
                line=dict(color='#1f77b4', width=2)
            ))
        
        fig.update_layout(
            title=config.title,
            xaxis_title=config.x_axis_title,
            yaxis_title=config.y_axis_title,
            width=config.width,
            height=config.height,
            template=config.theme,
            xaxis=dict(autorange='reversed')  # Reverse x-axis (closer to departure on right)
        )
        
        return fig
    
    def create_seasonality_heatmap(
        self,
        demand_data: pd.DataFrame,
        config: Optional[ChartConfig] = None
    ) -> go.Figure:
        """Create seasonality heatmap."""
        
        if config is None:
            config = ChartConfig(
                chart_type=ChartType.HEATMAP,
                title="Demand Seasonality Heatmap",
                width=800,
                height=500
            )
        
        # Prepare data for heatmap
        demand_data['month'] = demand_data['date'].dt.month
        demand_data['day_of_week'] = demand_data['date'].dt.day_name()
        
        # Create pivot table
        heatmap_data = demand_data.pivot_table(
            values='demand',
            index='day_of_week',
            columns='month',
            aggfunc='mean'
        )
        
        # Reorder days of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data = heatmap_data.reindex(day_order)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=[f'Month {i}' for i in heatmap_data.columns],
            y=heatmap_data.index,
            colorscale='Viridis',
            hoverongaps=False,
            colorbar=dict(title="Average Demand")
        ))
        
        fig.update_layout(
            title=config.title,
            width=config.width,
            height=config.height,
            template=config.theme
        )
        
        return fig


class CompetitiveVisualizer:
    """Visualizations for competitive analysis."""
    
    def __init__(self, output_dir: str = "charts"):
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def create_market_share_chart(
        self,
        market_data: pd.DataFrame,
        config: Optional[ChartConfig] = None
    ) -> go.Figure:
        """Create market share visualization."""
        
        if config is None:
            config = ChartConfig(
                chart_type=ChartType.PIE,
                title="Market Share by Airline",
                width=700,
                height=700
            )
        
        # Calculate market share
        market_share = market_data.groupby('airline')['passengers'].sum()
        market_share_pct = (market_share / market_share.sum() * 100).round(1)
        
        fig = go.Figure(data=[go.Pie(
            labels=market_share.index,
            values=market_share.values,
            textinfo='label+percent',
            textposition='outside',
            marker_colors=px.colors.qualitative.Set3,
            hole=0.4
        )])
        
        fig.update_layout(
            title=config.title,
            width=config.width,
            height=config.height,
            template=config.theme,
            annotations=[dict(
                text='Market<br>Share',
                x=0.5, y=0.5,
                font_size=16,
                showarrow=False
            )]
        )
        
        return fig
    
    def create_competitive_positioning_chart(
        self,
        competitive_data: pd.DataFrame,
        config: Optional[ChartConfig] = None
    ) -> go.Figure:
        """Create competitive positioning scatter plot."""
        
        if config is None:
            config = ChartConfig(
                chart_type=ChartType.SCATTER,
                title="Competitive Positioning: Price vs Service Quality",
                x_axis_title="Average Price ($)",
                y_axis_title="Service Quality Index"
            )
        
        fig = go.Figure()
        
        # Create scatter plot
        fig.add_trace(go.Scatter(
            x=competitive_data['average_price'],
            y=competitive_data['service_quality'],
            mode='markers+text',
            text=competitive_data['airline'],
            textposition='top center',
            marker=dict(
                size=competitive_data.get('market_share', 10) * 100,  # Size by market share
                color=competitive_data.get('profitability', 0),
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Profitability"),
                line=dict(width=2, color='black')
            ),
            hovertemplate='<b>%{text}</b><br>' +
                         'Price: $%{x:.0f}<br>' +
                         'Service Quality: %{y:.2f}<br>' +
                         'Market Share: %{marker.size:.1f}%<br>' +
                         '<extra></extra>'
        ))
        
        # Add quadrant lines
        avg_price = competitive_data['average_price'].mean()
        avg_quality = competitive_data['service_quality'].mean()
        
        fig.add_hline(y=avg_quality, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=avg_price, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Add quadrant labels
        fig.add_annotation(
            x=avg_price * 1.2, y=avg_quality * 1.1,
            text="Premium", showarrow=False,
            font=dict(size=14, color="blue")
        )
        fig.add_annotation(
            x=avg_price * 0.8, y=avg_quality * 1.1,
            text="Value", showarrow=False,
            font=dict(size=14, color="green")
        )
        fig.add_annotation(
            x=avg_price * 0.8, y=avg_quality * 0.9,
            text="Budget", showarrow=False,
            font=dict(size=14, color="orange")
        )
        fig.add_annotation(
            x=avg_price * 1.2, y=avg_quality * 0.9,
            text="Overpriced", showarrow=False,
            font=dict(size=14, color="red")
        )
        
        fig.update_layout(
            title=config.title,
            xaxis_title=config.x_axis_title,
            yaxis_title=config.y_axis_title,
            width=config.width,
            height=config.height,
            template=config.theme
        )
        
        return fig
    
    def create_price_comparison_chart(
        self,
        price_data: pd.DataFrame,
        config: Optional[ChartConfig] = None
    ) -> go.Figure:
        """Create price comparison chart across airlines."""
        
        if config is None:
            config = ChartConfig(
                chart_type=ChartType.BOX,
                title="Price Distribution by Airline",
                x_axis_title="Airline",
                y_axis_title="Price ($)"
            )
        
        fig = go.Figure()
        
        # Create box plots for each airline
        for airline in price_data['airline'].unique():
            airline_prices = price_data[price_data['airline'] == airline]['price']
            
            fig.add_trace(go.Box(
                y=airline_prices,
                name=airline,
                boxpoints='outliers',
                marker_color=px.colors.qualitative.Set1[hash(airline) % len(px.colors.qualitative.Set1)]
            ))
        
        fig.update_layout(
            title=config.title,
            xaxis_title=config.x_axis_title,
            yaxis_title=config.y_axis_title,
            width=config.width,
            height=config.height,
            template=config.theme
        )
        
        return fig


class RevenueVisualizer:
    """Visualizations for revenue analysis."""
    
    def __init__(self, output_dir: str = "charts"):
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def create_revenue_waterfall_chart(
        self,
        revenue_components: Dict[str, float],
        config: Optional[ChartConfig] = None
    ) -> go.Figure:
        """Create revenue waterfall chart."""
        
        if config is None:
            config = ChartConfig(
                chart_type=ChartType.BAR,
                title="Revenue Waterfall Analysis",
                x_axis_title="Components",
                y_axis_title="Revenue ($)"
            )
        
        # Prepare waterfall data
        categories = list(revenue_components.keys())
        values = list(revenue_components.values())
        
        # Calculate cumulative values
        cumulative = [0]
        for i, value in enumerate(values[:-1]):
            cumulative.append(cumulative[-1] + value)
        
        fig = go.Figure(go.Waterfall(
            name="Revenue",
            orientation="v",
            measure=["relative"] * (len(categories) - 1) + ["total"],
            x=categories,
            textposition="outside",
            text=["+$" + str(v) if v > 0 else "$" + str(v) for v in values],
            y=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
            title=config.title,
            width=config.width,
            height=config.height,
            template=config.theme,
            showlegend=False
        )
        
        return fig
    
    def create_revenue_by_route_chart(
        self,
        route_revenue_data: pd.DataFrame,
        config: Optional[ChartConfig] = None
    ) -> go.Figure:
        """Create revenue by route visualization."""
        
        if config is None:
            config = ChartConfig(
                chart_type=ChartType.TREEMAP,
                title="Revenue by Route",
                width=800,
                height=600
            )
        
        # Sort by revenue and take top routes
        top_routes = route_revenue_data.nlargest(20, 'revenue')
        
        fig = go.Figure(go.Treemap(
            labels=top_routes['route_name'],
            values=top_routes['revenue'],
            parents=[""] * len(top_routes),
            textinfo="label+value+percent parent",
            hovertemplate='<b>%{label}</b><br>' +
                         'Revenue: $%{value:,.0f}<br>' +
                         'Percentage: %{percentParent}<br>' +
                         '<extra></extra>',
            maxdepth=2
        ))
        
        fig.update_layout(
            title=config.title,
            width=config.width,
            height=config.height,
            template=config.theme
        )
        
        return fig
    
    def create_revenue_trend_chart(
        self,
        revenue_data: pd.DataFrame,
        config: Optional[ChartConfig] = None
    ) -> go.Figure:
        """Create revenue trend with multiple metrics."""
        
        if config is None:
            config = ChartConfig(
                chart_type=ChartType.LINE,
                title="Revenue Trends and Metrics",
                x_axis_title="Date",
                y_axis_title="Revenue ($)"
            )
        
        # Create subplot with secondary y-axis
        fig = make_subplots(
            specs=[[{"secondary_y": True}]],
            subplot_titles=[config.title]
        )
        
        # Revenue trend
        fig.add_trace(
            go.Scatter(
                x=revenue_data['date'],
                y=revenue_data['total_revenue'],
                mode='lines+markers',
                name='Total Revenue',
                line=dict(color='#1f77b4', width=3)
            ),
            secondary_y=False
        )
        
        # Load factor on secondary axis
        if 'load_factor' in revenue_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=revenue_data['date'],
                    y=revenue_data['load_factor'] * 100,
                    mode='lines',
                    name='Load Factor (%)',
                    line=dict(color='#ff7f0e', width=2, dash='dash')
                ),
                secondary_y=True
            )
        
        # Average fare on secondary axis
        if 'average_fare' in revenue_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=revenue_data['date'],
                    y=revenue_data['average_fare'],
                    mode='lines',
                    name='Average Fare ($)',
                    line=dict(color='#2ca02c', width=2)
                ),
                secondary_y=True
            )
        
        # Update layout
        fig.update_xaxes(title_text=config.x_axis_title)
        fig.update_yaxes(title_text="Revenue ($)", secondary_y=False)
        fig.update_yaxes(title_text="Load Factor (%) / Fare ($)", secondary_y=True)
        
        fig.update_layout(
            width=config.width,
            height=config.height,
            template=config.theme,
            hovermode='x unified'
        )
        
        return fig
    
    def create_profitability_analysis_chart(
        self,
        profitability_data: pd.DataFrame,
        config: Optional[ChartConfig] = None
    ) -> go.Figure:
        """Create profitability analysis chart."""
        
        if config is None:
            config = ChartConfig(
                chart_type=ChartType.SCATTER,
                title="Route Profitability Analysis",
                x_axis_title="Revenue ($)",
                y_axis_title="Profit Margin (%)"
            )
        
        fig = go.Figure()
        
        # Create bubble chart
        fig.add_trace(go.Scatter(
            x=profitability_data['revenue'],
            y=profitability_data['profit_margin'] * 100,
            mode='markers',
            marker=dict(
                size=profitability_data.get('passengers', 100) / 10,  # Size by passenger volume
                color=profitability_data.get('load_factor', 0.8),
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Load Factor"),
                line=dict(width=1, color='black'),
                sizemode='diameter',
                sizeref=2. * max(profitability_data.get('passengers', [100])) / (40.**2),
                sizemin=4
            ),
            text=profitability_data.get('route_name', profitability_data.index),
            hovertemplate='<b>%{text}</b><br>' +
                         'Revenue: $%{x:,.0f}<br>' +
                         'Profit Margin: %{y:.1f}%<br>' +
                         'Load Factor: %{marker.color:.1%}<br>' +
                         '<extra></extra>'
        ))
        
        # Add reference lines
        fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
        fig.add_hline(y=10, line_dash="dash", line_color="green", opacity=0.5)
        
        fig.update_layout(
            title=config.title,
            xaxis_title=config.x_axis_title,
            yaxis_title=config.y_axis_title,
            width=config.width,
            height=config.height,
            template=config.theme
        )
        
        return fig


class VisualizationManager:
    """Manages all visualization components."""
    
    def __init__(self, output_dir: str = "visualizations"):
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize visualizers
        self.pricing_viz = PricingVisualizer(str(self.output_dir / "pricing"))
        self.demand_viz = DemandVisualizer(str(self.output_dir / "demand"))
        self.competitive_viz = CompetitiveVisualizer(str(self.output_dir / "competitive"))
        self.revenue_viz = RevenueVisualizer(str(self.output_dir / "revenue"))
        
        # Chart registry
        self.chart_registry: Dict[str, go.Figure] = {}
    
    def create_dashboard_charts(
        self,
        simulation_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, go.Figure]:
        """Create all charts for dashboard."""
        
        charts = {}
        
        try:
            # Pricing charts
            if 'pricing' in simulation_data:
                charts['price_trends'] = self.pricing_viz.create_price_trend_chart(
                    simulation_data['pricing']
                )
                charts['price_distribution'] = self.pricing_viz.create_price_distribution_chart(
                    simulation_data['pricing']
                )
            
            # Demand charts
            if 'demand' in simulation_data:
                if 'demand_forecast' in simulation_data:
                    charts['demand_forecast'] = self.demand_viz.create_demand_forecast_chart(
                        simulation_data['demand'],
                        simulation_data['demand_forecast']
                    )
                
                charts['seasonality'] = self.demand_viz.create_seasonality_heatmap(
                    simulation_data['demand']
                )
            
            # Competitive charts
            if 'market_data' in simulation_data:
                charts['market_share'] = self.competitive_viz.create_market_share_chart(
                    simulation_data['market_data']
                )
            
            if 'competitive_data' in simulation_data:
                charts['competitive_positioning'] = self.competitive_viz.create_competitive_positioning_chart(
                    simulation_data['competitive_data']
                )
            
            # Revenue charts
            if 'revenue' in simulation_data:
                charts['revenue_trends'] = self.revenue_viz.create_revenue_trend_chart(
                    simulation_data['revenue']
                )
            
            if 'route_revenue' in simulation_data:
                charts['revenue_by_route'] = self.revenue_viz.create_revenue_by_route_chart(
                    simulation_data['route_revenue']
                )
        
        except Exception as e:
            self.logger.error(f"Error creating dashboard charts: {e}")
        
        # Store in registry
        self.chart_registry.update(charts)
        
        return charts
    
    def save_chart(
        self,
        chart: go.Figure,
        filename: str,
        format: str = "html"
    ) -> str:
        """Save chart to file."""
        
        output_path = self.output_dir / f"{filename}.{format}"
        
        if format == "html":
            chart.write_html(str(output_path))
        elif format == "png":
            chart.write_image(str(output_path))
        elif format == "pdf":
            chart.write_image(str(output_path))
        elif format == "svg":
            chart.write_image(str(output_path))
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Chart saved to {output_path}")
        return str(output_path)
    
    def export_charts_to_html(
        self,
        charts: Dict[str, go.Figure],
        filename: str = "dashboard.html"
    ) -> str:
        """Export all charts to a single HTML file."""
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>PriceFly Analytics Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .chart-container { margin: 20px 0; padding: 20px; border: 1px solid #ddd; }
                .chart-title { font-size: 18px; font-weight: bold; margin-bottom: 10px; }
            </style>
        </head>
        <body>
            <h1>PriceFly Analytics Dashboard</h1>
            <p>Generated: {timestamp}</p>
        """.format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # Add each chart
        for chart_name, chart in charts.items():
            chart_html = chart.to_html(include_plotlyjs=False, div_id=chart_name)
            html_content += f"""
            <div class="chart-container">
                <div class="chart-title">{chart_name.replace('_', ' ').title()}</div>
                {chart_html}
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Dashboard exported to {output_path}")
        return str(output_path)
    
    def get_chart_as_base64(
        self,
        chart: go.Figure,
        format: str = "png",
        width: int = 800,
        height: int = 600
    ) -> str:
        """Convert chart to base64 string for embedding."""
        
        img_bytes = chart.to_image(
            format=format,
            width=width,
            height=height
        )
        
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return f"data:image/{format};base64,{img_base64}"
    
    def create_custom_chart(
        self,
        data: pd.DataFrame,
        chart_config: ChartConfig,
        chart_data: ChartData
    ) -> go.Figure:
        """Create custom chart based on configuration."""
        
        fig = go.Figure()
        
        if chart_config.chart_type == ChartType.LINE:
            fig.add_trace(go.Scatter(
                x=data[chart_data.x_column],
                y=data[chart_data.y_column],
                mode='lines+markers',
                name=chart_data.y_column
            ))
        
        elif chart_config.chart_type == ChartType.BAR:
            fig.add_trace(go.Bar(
                x=data[chart_data.x_column],
                y=data[chart_data.y_column],
                name=chart_data.y_column
            ))
        
        elif chart_config.chart_type == ChartType.SCATTER:
            marker_config = dict(size=8)
            
            if chart_data.color_column:
                marker_config['color'] = data[chart_data.color_column]
                marker_config['colorscale'] = chart_config.color_scheme
                marker_config['showscale'] = True
            
            if chart_data.size_column:
                marker_config['size'] = data[chart_data.size_column]
            
            fig.add_trace(go.Scatter(
                x=data[chart_data.x_column],
                y=data[chart_data.y_column],
                mode='markers',
                marker=marker_config,
                name=chart_data.y_column
            ))
        
        elif chart_config.chart_type == ChartType.HEATMAP:
            # Assume data is already in matrix format or pivot
            fig.add_trace(go.Heatmap(
                z=data.values,
                x=data.columns,
                y=data.index,
                colorscale=chart_config.color_scheme
            ))
        
        # Update layout
        fig.update_layout(
            title=chart_config.title,
            xaxis_title=chart_config.x_axis_title,
            yaxis_title=chart_config.y_axis_title,
            width=chart_config.width,
            height=chart_config.height,
            template=chart_config.theme,
            showlegend=chart_config.show_legend
        )
        
        return fig
    
    def clear_registry(self):
        """Clear the chart registry."""
        self.chart_registry.clear()
        self.logger.info("Chart registry cleared")
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """Get summary of charts in registry."""
        return {
            'total_charts': len(self.chart_registry),
            'chart_names': list(self.chart_registry.keys()),
            'last_updated': datetime.now().isoformat()
        }