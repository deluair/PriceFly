#!/usr/bin/env python3
"""Basic simulation example for PriceFly airline pricing platform.

This example demonstrates how to:
1. Generate synthetic airline data
2. Run a pricing simulation
3. Analyze results and generate reports
4. Visualize key metrics
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json
from enum import Enum
import pandas as pd

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import PriceFly modules
from pricefly.data.synthetic_data import SyntheticDataEngine, DataGenerationConfig
from pricefly.simulation.engine import SimulationEngine, SimulationConfig
from pricefly.simulation.scenarios import ScenarioManager, ScenarioType
from pricefly.analytics.metrics import MetricsCalculator, MetricPeriod
from pricefly.analytics.reporting import ReportGenerator, ReportConfig, ReportType, ReportFormat
from pricefly.analytics.visualization import VisualizationManager
from pricefly.analytics.insights import InsightEngine


def setup_directories(base_dir: str) -> dict:
    """Create necessary directories for the simulation."""
    
    base_path = Path(base_dir)
    directories = {
        'base': base_path,
        'data': base_path / 'data',
        'results': base_path / 'results',
        'reports': base_path / 'reports',
        'visualizations': base_path / 'visualizations'
    }
    
    # Create directories
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return {k: str(v) for k, v in directories.items()}


def generate_sample_data(output_dir: str) -> dict:
    """Generate synthetic airline data for the simulation."""
    
    print("🔄 Generating synthetic airline data...")
    
    # Configure data generation
    config = DataGenerationConfig(
        num_airports=50,
        num_airlines=5,
        num_routes_per_airline=20,
        num_aircraft_per_airline=25,
        num_passengers_per_day=5000,
        simulation_start_date=datetime.now(),
        simulation_duration_days=90
    )
    
    # Generate data
    data_engine = SyntheticDataEngine(config)
    datasets = data_engine.generate_all_data()
    
    # Save datasets
    data_engine.save_data(output_dir)
    
    print(f"✅ Generated {len(datasets)} datasets:")
    for dataset_name in datasets.keys():
        print(f"   - {dataset_name}")
    
    return datasets


def run_pricing_simulation(datasets: dict, output_dir: str, scenario: ScenarioType) -> dict:
    """Run the airline pricing simulation."""
    
    print(f"🚀 Running pricing simulation with {scenario.value} scenario...")
    
    # Get scenario configuration
    scenario_manager = ScenarioManager()
    scenario_config = scenario_manager.get_scenario(scenario)
    
    # Create simulation configuration
    sim_config = SimulationConfig(
        start_date=datetime.now(),
        end_date=(datetime.now() + timedelta(days=90)),
        parallel_execution=True,
        save_intermediate_results=True
    )
    
    # Initialize and run simulation
    engine = SimulationEngine(
        config=sim_config,
        airlines=datasets['airlines'][:3],  # Use first 3 airlines
        airports=datasets['airports'],
        routes=datasets['routes'][:20],     # Use first 20 routes
        customer_segments=datasets['customer_segments']
    )
    
    print("   - Initializing pricing engines...")
    print("   - Setting up market dynamics...")
    print("   - Running simulation...")
    
    results = engine.run()
    
    print(f"✅ Simulation completed successfully")
    print(f"   - Simulated {len(datasets['airlines'][:3])} airlines")
    print(f"   - Analyzed {len(datasets['routes'][:20])} routes")
    print(f"   - Generated {len(results.get('metrics', {}))} metric categories")
    
    return results


def analyze_results(simulation_results: dict, output_dir: str) -> dict:
    """Analyze simulation results and generate insights."""
    
    print("📊 Analyzing simulation results...")
    
    analysis_results = {}
    
    # Extract simulation data
    simulation_data = simulation_results.get('data', {})
    
    if simulation_data:
        # Calculate comprehensive metrics
        print("   - Calculating performance metrics...")
        metrics_calculator = MetricsCalculator()
        
        performance_metrics = metrics_calculator.calculate_performance_metrics(simulation_data)
        revenue_metrics = metrics_calculator.calculate_revenue_metrics(simulation_data)
        competitive_metrics = metrics_calculator.calculate_competitive_metrics(simulation_data)
        
        analysis_results['metrics'] = {
            'performance': performance_metrics,
            'revenue': revenue_metrics,
            'competitive': competitive_metrics
        }
        
        # Generate insights
        print("   - Generating AI-driven insights...")
        insight_engine = InsightEngine()
        insights = insight_engine.analyze_simulation_data(simulation_data)
        
        # Get insights summary
        insights_summary = insight_engine.generate_insight_summary()
        analysis_results['insights'] = {
            'total_insights': len(insights),
            'summary': insights_summary
        }
        
        print(f"   - Generated {len(insights)} insights")
        print(f"   - Critical insights: {insights_summary.get('key_metrics', {}).get('critical_insights', 0)}")
    
    return analysis_results


def generate_reports(simulation_results: dict, output_dir: str) -> dict:
    """Generate comprehensive reports from simulation results."""
    
    print("📋 Generating reports...")
    
    reports = {}
    
    # Import required classes
    from pricefly.analytics.reporting import ReportGenerator, ReportConfig
    from pricefly.analytics.metrics import MetricsCalculator
    
    # Create a comprehensive summary report
    print("   - Creating simulation summary...")
    summary_path = Path(output_dir) / "simulation_summary.json"
    
    summary_data = {
        'simulation_completed': True,
        'timestamp': datetime.now().isoformat(),
        'results_summary': {
            'total_metrics': len(simulation_results.get('metrics', {})),
            'simulation_steps': simulation_results.get('simulation_config', {}).get('total_steps', 0),
            'airlines_simulated': len(simulation_results.get('airline_metrics', {})),
        },
        'airline_metrics': simulation_results.get('airline_metrics', {}),
        'market_summary': simulation_results.get('market_summary', {}),
        'event_summary': simulation_results.get('event_summary', {})
    }
    
    def convert_for_json(obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            # Convert enum keys to strings
            new_dict = {}
            for k, v in obj.items():
                if isinstance(k, Enum):
                    new_key = k.value if hasattr(k, 'value') else str(k)
                else:
                    new_key = k
                new_dict[new_key] = convert_for_json(v)
            return new_dict
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        elif isinstance(obj, Enum):
            return obj.value
        elif hasattr(obj, 'isoformat'):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return convert_for_json(obj.__dict__)
        else:
            return obj
    
    # Convert the data before JSON serialization
    json_safe_data = convert_for_json(summary_data)
    
    with open(summary_path, 'w') as f:
        json.dump(json_safe_data, f, indent=2, default=str)
    
    reports['simulation_summary'] = str(summary_path)
    
    # Generate comprehensive HTML report if metrics are available
    if simulation_results.get('airline_metrics'):
        try:
            print("   - Creating comprehensive HTML report...")
            
            # Create report configuration
            report_config = ReportConfig(
                report_type=ReportType.EXECUTIVE_SUMMARY,
                format=ReportFormat.HTML,
                period=MetricPeriod.MONTHLY,
                start_date=datetime.now() - timedelta(days=30),
                end_date=datetime.now(),
                include_charts=True
            )
            
            # Generate report
            report_generator = ReportGenerator()
            
            # Get first airline's metrics for report generation
            first_airline_metrics = list(simulation_results['airline_metrics'].values())[0]
            
            report_data = report_generator.generate_simulation_report(
                config=report_config,
                performance_metrics=first_airline_metrics.get('performance'),
                revenue_metrics=first_airline_metrics.get('revenue'),
                competitive_metrics=first_airline_metrics.get('competitive'),
                operational_metrics=first_airline_metrics.get('operational'),
                simulation_data=simulation_results
            )
            
            # Save HTML report
            html_report_path = Path(output_dir) / "comprehensive_report.html"
            with open(html_report_path, 'w') as f:
                f.write(report_data.get('html_content', '<html><body><h1>Simulation Report Generated</h1></body></html>'))
            
            reports['comprehensive_report'] = str(html_report_path)
            
        except Exception as e:
            print(f"   - Warning: Could not generate comprehensive report: {e}")
    
    print(f"✅ Generated {len(reports)} reports")
    
    return reports


def create_visualizations(simulation_results: dict, output_dir: str) -> dict:
    """Create visualizations and dashboards."""
    
    print("📈 Creating visualizations...")
    
    visualizations = {}
    
    # Import required classes
    from pricefly.analytics.visualization import VisualizationManager, ChartConfig, ChartType, ChartData
    
    # Create visualization manager
    viz_manager = VisualizationManager()
    charts = {}
    
    # Generate charts if airline metrics are available
    if simulation_results.get('airline_metrics'):
        try:
            print("   - Creating revenue and performance charts...")
            
            # Prepare data for charts
            airline_data = []
            for airline_code, metrics in simulation_results['airline_metrics'].items():
                performance = metrics.get('performance')
                if performance:
                    airline_data.append({
                        'airline': airline_code,
                        'revenue': getattr(performance, 'total_revenue', 0),
                        'passengers': getattr(performance, 'passengers_carried', 0),
                        'load_factor': getattr(performance, 'load_factor', 0)
                    })
            
            if airline_data:
                # Revenue by airline chart
                revenue_chart_config = ChartConfig(
                    chart_type=ChartType.BAR,
                    title="Revenue by Airline",
                    x_axis_title="Airline",
                    y_axis_title="Revenue ($)"
                )
                
                import pandas as pd
                revenue_chart_df = pd.DataFrame({
                    'airline': [item['airline'] for item in airline_data],
                    'revenue': [item['revenue'] for item in airline_data]
                })
                revenue_chart_data = ChartData(
                    data=revenue_chart_df,
                    x_column="airline",
                    y_column="revenue"
                )
                
                revenue_chart = viz_manager.create_custom_chart(
                    data=revenue_chart_df,
                    chart_config=revenue_chart_config,
                    chart_data=revenue_chart_data
                )
                charts['revenue_by_airline'] = revenue_chart
                
                # Load factor chart
                load_factor_config = ChartConfig(
                    chart_type=ChartType.BAR,
                    title="Load Factor by Airline",
                    x_axis_title="Airline",
                    y_axis_title="Load Factor"
                )
                
                load_factor_df = pd.DataFrame({
                    'airline': [item['airline'] for item in airline_data],
                    'load_factor': [item['load_factor'] for item in airline_data]
                })
                load_factor_data = ChartData(
                    data=load_factor_df,
                    x_column="airline",
                    y_column="load_factor"
                )
                
                load_factor_chart = viz_manager.create_custom_chart(
                    data=load_factor_df,
                    chart_config=load_factor_config,
                    chart_data=load_factor_data
                )
                charts['load_factor_by_airline'] = load_factor_chart
                
                # Save individual charts
                print("   - Saving individual charts...")
                chart_files = {}
                for chart_name, chart in charts.items():
                    chart_filename = f"{chart_name}.html"
                    chart_path = Path(output_dir) / chart_filename
                    chart.write_html(str(chart_path))
                    chart_files[chart_name] = str(chart_path)
                    print(f"     • Saved {chart_name} to {chart_path}")
                
                visualizations['chart_files'] = chart_files
            
            # Export charts to HTML dashboard
            if charts:
                print("   - Creating HTML dashboard...")
                html_dashboard = viz_manager.export_charts_to_html(
                    charts
                )
                
                dashboard_path = Path(output_dir) / "dashboard.html"
                with open(dashboard_path, 'w') as f:
                    f.write(html_dashboard)
                
                visualizations['dashboard'] = str(dashboard_path)
                visualizations['charts'] = list(charts.keys())
                print(f"     • Saved dashboard to {dashboard_path}")
                
        except Exception as e:
            print(f"   - Warning: Could not create charts: {e}")
    
    # Create visualization summary
    print("   - Creating visualization summary...")
    viz_summary_path = Path(output_dir) / "visualization_summary.json"
    
    viz_data = {
        'visualization_completed': True,
        'timestamp': datetime.now().isoformat(),
        'available_data': list(simulation_results.keys()),
        'metrics_available': len(simulation_results.get('airline_metrics', {})) > 0,
        'charts_generated': len(charts),
        'dashboard_available': 'dashboard' in visualizations
    }
    
    def json_serializer(obj):
        """Custom JSON serializer for complex objects."""
        if isinstance(obj, Enum):
            return obj.value
        elif hasattr(obj, 'isoformat'):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)
    
    with open(viz_summary_path, 'w') as f:
        json.dump(viz_data, f, indent=2, default=json_serializer)
    
    visualizations['summary'] = str(viz_summary_path)
    
    print(f"   - Created {len(charts)} charts and dashboard")
    
    return visualizations


def print_summary(results: dict, directories: dict):
    """Print a summary of the simulation results."""
    
    print("\n" + "="*60)
    print("🎯 PRICEFLY SIMULATION SUMMARY")
    print("="*60)
    
    # Simulation overview
    simulation_results = results.get('simulation', {})
    metrics = simulation_results.get('metrics', {})
    
    print(f"\n📊 Simulation Overview:")
    print(f"   - Scenario: {results.get('scenario', 'Unknown')}")
    print(f"   - Duration: {results.get('duration_days', 'Unknown')} days")
    print(f"   - Airlines: {len(results.get('airlines', []))}")
    print(f"   - Routes: {len(results.get('routes', []))}")
    
    # Key metrics
    analysis = results.get('analysis', {})
    if 'insights' in analysis:
        insights_summary = analysis['insights']['summary']
        print(f"\n🔍 Key Insights:")
        print(f"   - Total insights generated: {analysis['insights']['total_insights']}")
        
        key_metrics = insights_summary.get('key_metrics', {})
        if key_metrics:
            print(f"   - Critical insights: {key_metrics.get('critical_insights', 0)}")
            print(f"   - Revenue opportunities: {key_metrics.get('revenue_opportunities', 0)}")
            print(f"   - Risk alerts: {key_metrics.get('risk_alerts', 0)}")
    
    # Generated outputs
    print(f"\n📁 Generated Outputs:")
    print(f"   - Data directory: {directories['data']}")
    print(f"   - Results directory: {directories['results']}")
    
    reports = results.get('reports', {})
    if reports:
        print(f"   - Reports:")
        for report_name, report_path in reports.items():
            print(f"     • {report_name}: {report_path}")
    
    visualizations = results.get('visualizations', {})
    if visualizations:
        print(f"   - Visualizations:")
        print(f"     • Dashboard: {visualizations.get('dashboard', 'Not generated')}")
        print(f"     • Charts: {len(visualizations.get('charts', []))} types")
    
    print("\n" + "="*60)
    print("✅ Simulation completed successfully!")
    print("="*60)


def main():
    """Main function to run the basic simulation example."""
    
    print("🚀 PriceFly Basic Simulation Example")
    print("====================================\n")
    
    # Setup
    base_dir = "./pricefly_example_output"
    directories = setup_directories(base_dir)
    
    # Choose scenario
    scenario = ScenarioType.HIGH_COMPETITION
    
    try:
        # Step 1: Generate synthetic data
        datasets = generate_sample_data(directories['data'])
        
        # Step 2: Run simulation
        simulation_results = run_pricing_simulation(
            datasets, 
            directories['results'], 
            scenario
        )
        
        # Step 3: Calculate comprehensive metrics
        print("\n📊 Calculating comprehensive metrics...")
        
        # Import required classes
        from pricefly.analytics.metrics import MetricsCalculator, MetricPeriod
        from pricefly.analytics.metrics import CompetitiveMetrics, OperationalMetrics
        
        # Create metrics calculator
        metrics_calculator = MetricsCalculator()
        
        # Convert datasets to DataFrames for metrics calculation
        # Handle missing datasets gracefully
        import random
        
        bookings_data = datasets.get('bookings', [])
        flights_data = datasets.get('flights', [])
        routes_data = datasets.get('routes', [])
        airlines_data = datasets.get('airlines', [])
        
        print(f"   - Available datasets: {list(datasets.keys())}")
        print(f"   - Bookings: {len(bookings_data)} records")
        print(f"   - Flights: {len(flights_data)} records")
        print(f"   - Routes: {len(routes_data)} records")
        print(f"   - Airlines: {len(airlines_data)} records")
        
        # Helper function to get attribute from object or dict
        def get_attr(obj, attr, default=None):
            if hasattr(obj, attr):
                return getattr(obj, attr)
            elif isinstance(obj, dict):
                return obj.get(attr, default)
            else:
                return default
        
        # Generate sample data if needed
        if not bookings_data and airlines_data:
            print("   - Generating sample booking data...")
            bookings_data = []
            for i in range(1000):
                airline = airlines_data[i % len(airlines_data)]
                airline_code = get_attr(airline, 'airline_code') or get_attr(airline, 'code', 'AA')
                # Generate a flight_id that matches with flight data
                flight_id = f'FL{random.randint(0, 499):06d}'
                booking = {
                    'booking_id': f'BK{i:06d}',
                    'airline_code': airline_code,
                    'route_id': f'RT{random.randint(1, 100):03d}',
                    'flight_id': flight_id,  # Add flight_id to link with flight data
                    'passenger_count': random.randint(1, 4),
                    'fare_paid': random.uniform(100, 1500),
                    'booking_class': random.choice(['economy', 'business', 'first']),
                    'advance_booking_days': random.randint(1, 365),
                    'booking_date': datetime.now() + timedelta(days=random.randint(0, 90))
                }
                bookings_data.append(booking)
            print(f"   - Generated {len(bookings_data)} sample bookings")
        
        if not flights_data and airlines_data:
            print("   - Generating sample flight data...")
            flights_data = []
            for i in range(500):
                airline = airlines_data[i % len(airlines_data)]
                airline_code = get_attr(airline, 'airline_code') or get_attr(airline, 'code', 'AA')
                flight = {
                    'flight_id': f'FL{i:06d}',
                    'airline_code': airline_code,
                    'route_id': f'RT{random.randint(1, 100):03d}',
                    'aircraft_id': f'AC{i:06d}',
                    'departure_time': '2024-01-01T08:00:00',
                    'arrival_time': '2024-01-01T12:00:00',
                    'seats_sold': random.randint(50, 180),
                    'total_seats': random.randint(150, 200),
                    'distance': random.uniform(500, 5000),
                    'fuel_cost': random.uniform(5000, 15000),
                    'crew_cost': random.uniform(2000, 5000),
                    'maintenance_cost': random.uniform(1000, 3000)
                }
                flights_data.append(flight)
            print(f"   - Generated {len(flights_data)} sample flights")
        
        if not routes_data:
            print("   - Generating sample route data...")
            routes_data = []
            for i in range(100):
                route = {
                    'route_id': f'RT{i:03d}',
                    'origin': random.choice(['ATL', 'LAX', 'ORD', 'DFW', 'JFK']),
                    'destination': random.choice(['ATL', 'LAX', 'ORD', 'DFW', 'JFK']),
                    'distance_km': random.uniform(500, 5000),
                    'route_type': random.choice(['domestic', 'international']),
                    'domestic': random.choice([True, False])
                }
                routes_data.append(route)
            print(f"   - Generated {len(routes_data)} sample routes")
        
        # Convert to DataFrames
        bookings_df = pd.DataFrame(bookings_data)
        flights_df = pd.DataFrame(flights_data)
        routes_df = pd.DataFrame(routes_data)
        airlines_df = pd.DataFrame(airlines_data)
        
        # Add required columns if missing
        if not routes_df.empty:
            if 'route_id' not in routes_df.columns:
                routes_df['route_id'] = [f'RT{i:03d}' for i in range(len(routes_df))]
            if 'domestic' not in routes_df.columns:
                routes_df['domestic'] = [random.choice([True, False]) for _ in range(len(routes_df))]
            if 'route_type' not in routes_df.columns:
                routes_df['route_type'] = ['domestic' if domestic else 'international' for domestic in routes_df['domestic']]
        
        # Create cost data (simplified)
        cost_data = pd.DataFrame({
            'flight_id': flights_df['flight_id'] if not flights_df.empty else [],
            'total_cost': [50000] * len(flights_df) if not flights_df.empty else []
        })
        
        # Ensure consistent column names
        if not bookings_df.empty:
            if 'fare' not in bookings_df.columns and 'fare_paid' in bookings_df.columns:
                bookings_df['fare'] = bookings_df['fare_paid']
            if 'airline_code' not in bookings_df.columns and 'airline' in bookings_df.columns:
                bookings_df['airline_code'] = bookings_df['airline']
            
            # Debug: Print sample booking data
            print(f"   - Sample booking data columns: {list(bookings_df.columns)}")
            print(f"   - Sample fare values: {bookings_df['fare'].head().tolist() if 'fare' in bookings_df.columns else 'No fare column'}")
            print(f"   - Total revenue in bookings: ${bookings_df['fare'].sum() if 'fare' in bookings_df.columns else 0:,.2f}")
        
        if not flights_df.empty:
            if 'capacity' not in flights_df.columns and 'total_seats' in flights_df.columns:
                flights_df['capacity'] = flights_df['total_seats']
            if 'distance' not in flights_df.columns and 'distance_km' in flights_df.columns:
                flights_df['distance'] = flights_df['distance_km']
        
        # Calculate metrics for each airline
        airline_metrics = {}
        # Handle both object and dictionary formats for airlines
        airlines_list = airlines_data if airlines_data else []
        for airline in airlines_list:
            airline_code = get_attr(airline, 'airline_code') or get_attr(airline, 'code', 'AA')
            
            # Filter data for this airline
            airline_flights = flights_df[flights_df['airline_code'] == airline_code] if not flights_df.empty and 'airline_code' in flights_df.columns else pd.DataFrame()
            airline_bookings = bookings_df[bookings_df['airline_code'] == airline_code] if not bookings_df.empty and 'airline_code' in bookings_df.columns else pd.DataFrame()
            airline_costs = cost_data[cost_data['flight_id'].isin(airline_flights['flight_id'])] if not airline_flights.empty and not cost_data.empty else pd.DataFrame()
            
            # Calculate performance metrics with proper data handling
            try:
                performance_metrics = metrics_calculator.calculate_performance_metrics(
                    airline_bookings if not airline_bookings.empty else pd.DataFrame({'fare': []}),
                    airline_flights if not airline_flights.empty else pd.DataFrame({'capacity': [], 'distance': []}),
                    airline_costs if not airline_costs.empty else pd.DataFrame({'total_cost': []})
                )
            except Exception as e:
                print(f"   - Warning: Could not calculate performance metrics for {airline_code}: {e}")
                from pricefly.analytics.metrics import PerformanceMetrics
                performance_metrics = PerformanceMetrics()
            
            # Calculate revenue metrics with proper data handling
            try:
                revenue_metrics = metrics_calculator.calculate_revenue_metrics(
                    airline_bookings if not airline_bookings.empty else pd.DataFrame({'fare': []}),
                    routes_df if not routes_df.empty else pd.DataFrame()
                )
            except Exception as e:
                print(f"   - Warning: Could not calculate revenue metrics for {airline_code}: {e}")
                from pricefly.analytics.metrics import RevenueMetrics
                revenue_metrics = RevenueMetrics()
            
            # Create simplified competitive and operational metrics
            competitive_metrics = CompetitiveMetrics()
            operational_metrics = OperationalMetrics()
            
            airline_metrics[airline_code] = {
                'performance': performance_metrics,
                'revenue': revenue_metrics,
                'competitive': competitive_metrics,
                'operational': operational_metrics
            }
        
        # Add airline metrics to simulation results
        simulation_results['airline_metrics'] = airline_metrics
        
        print(f"✅ Calculated metrics for {len(airline_metrics)} airlines")
        
        # Step 4: Analyze results
        analysis_results = analyze_results(
            simulation_results, 
            directories['results']
        )
        
        # Step 5: Generate reports
        reports = generate_reports(
            simulation_results, 
            directories['reports']
        )
        
        # Step 6: Create visualizations
        visualizations = create_visualizations(
            simulation_results, 
            directories['visualizations']
        )
        
        # Compile final results
        final_results = {
            'scenario': scenario.value,
            'duration_days': 90,
            'airlines': datasets['airlines'][:3],
            'routes': datasets['routes'][:20],
            'simulation': simulation_results,
            'analysis': analysis_results,
            'reports': reports,
            'visualizations': visualizations
        }
        
        # Save complete results
        results_file = Path(directories['base']) / "complete_results.json"
        
        def convert_for_json(obj):
            """Convert objects to JSON-serializable format."""
            if isinstance(obj, dict):
                # Convert enum keys to strings
                new_dict = {}
                for k, v in obj.items():
                    if isinstance(k, Enum):
                        new_key = k.value if hasattr(k, 'value') else str(k)
                    else:
                        new_key = k
                    new_dict[new_key] = convert_for_json(v)
                return new_dict
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, Enum):
                return obj.value
            elif hasattr(obj, 'isoformat'):
                return obj.isoformat()
            elif hasattr(obj, '__dict__'):
                return convert_for_json(obj.__dict__)
            else:
                return obj
        
        # Convert the data before JSON serialization
        json_safe_results = convert_for_json(final_results)
        
        with open(results_file, 'w') as f:
            json.dump(json_safe_results, f, indent=2, default=str)
        
        # Print summary
        print_summary(final_results, directories)
        
    except Exception as e:
        print(f"❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)