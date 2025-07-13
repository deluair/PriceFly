"""Main entry point for the PriceFly airline pricing simulation platform."""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import json
import pandas as pd

# Import PriceFly modules
from pricefly.simulation.engine import SimulationEngine, SimulationConfig
from pricefly.simulation.scenarios import ScenarioManager, ScenarioType
from pricefly.data.synthetic_data import SyntheticDataEngine, DataGenerationConfig
from pricefly.data.loader import DataLoader
from pricefly.analytics.metrics import MetricsCalculator
from pricefly.analytics.reporting import ReportGenerator, ReportConfig, ReportType, ReportFormat
from pricefly.analytics.visualization import VisualizationManager
from pricefly.analytics.insights import InsightEngine


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup logging configuration."""
    
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )
    
    # Reduce noise from external libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('plotly').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def generate_synthetic_data(config: DataGenerationConfig, output_dir: str) -> Dict[str, str]:
    """Generate synthetic data for simulation."""
    
    logger = logging.getLogger(__name__)
    logger.info("Starting synthetic data generation...")
    
    # Initialize data engine
    data_engine = SyntheticDataEngine(config)
    
    # Generate all datasets
    datasets = data_engine.generate_complete_dataset()
    
    # Save datasets
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    file_paths = data_engine.save_datasets(datasets, str(output_path))
    
    logger.info(f"Synthetic data generated and saved to {output_path}")
    logger.info(f"Generated {len(file_paths)} dataset files")
    
    return file_paths


def run_simulation(
    scenario_type: ScenarioType,
    data_dir: str,
    output_dir: str,
    simulation_days: int = 365,
    custom_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Run the airline pricing simulation."""
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting simulation with scenario: {scenario_type.value}")
    
    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading simulation data...")
    data_loader = DataLoader(data_dir)
    
    # Load required datasets
    airports = data_loader.load_airports()
    airlines = data_loader.load_airlines()
    routes = data_loader.load_routes()
    customer_segments = data_loader.load_customer_segments()
    
    logger.info(f"Loaded {len(airports)} airports, {len(airlines)} airlines, {len(routes)} routes")
    
    # Initialize scenario manager
    scenario_manager = ScenarioManager()
    scenario_config = scenario_manager.get_scenario(scenario_type)
    
    # Override with custom config if provided
    if custom_config:
        for key, value in custom_config.items():
            if hasattr(scenario_config, key):
                setattr(scenario_config, key, value)
    
    # Create simulation configuration
    sim_config = SimulationConfig(
        start_date=datetime.now().date(),
        end_date=(datetime.now() + timedelta(days=simulation_days)).date(),
        airlines=list(airlines.keys()),
        routes=list(routes.keys()),
        scenario=scenario_config,
        output_dir=str(output_path),
        parallel_execution=True,
        save_intermediate_results=True
    )
    
    # Initialize and run simulation
    logger.info("Initializing simulation engine...")
    simulation_engine = SimulationEngine(sim_config)
    
    # Load data into simulation
    simulation_engine.load_data({
        'airports': airports,
        'airlines': airlines,
        'routes': routes,
        'customer_segments': customer_segments
    })
    
    logger.info(f"Running simulation for {simulation_days} days...")
    results = simulation_engine.run_simulation()
    
    logger.info("Simulation completed successfully")
    
    return results


def generate_analytics(
    simulation_results: Dict[str, Any],
    output_dir: str,
    generate_reports: bool = True,
    generate_visualizations: bool = True,
    generate_insights: bool = True
) -> Dict[str, Any]:
    """Generate analytics from simulation results."""
    
    logger = logging.getLogger(__name__)
    logger.info("Generating analytics from simulation results...")
    
    output_path = Path(output_dir)
    analytics_dir = output_path / "analytics"
    analytics_dir.mkdir(parents=True, exist_ok=True)
    
    analytics_results = {}
    
    # Extract simulation data
    simulation_data = simulation_results.get('data', {})
    metrics = simulation_results.get('metrics', {})
    
    # Calculate additional metrics
    if simulation_data:
        logger.info("Calculating performance metrics...")
        metrics_calculator = MetricsCalculator()
        
        # Calculate comprehensive metrics
        performance_metrics = metrics_calculator.calculate_performance_metrics(simulation_data)
        revenue_metrics = metrics_calculator.calculate_revenue_metrics(simulation_data)
        competitive_metrics = metrics_calculator.calculate_competitive_metrics(simulation_data)
        
        analytics_results['metrics'] = {
            'performance': performance_metrics,
            'revenue': revenue_metrics,
            'competitive': competitive_metrics
        }
    
    # Generate reports
    if generate_reports:
        logger.info("Generating reports...")
        report_generator = ReportGenerator()
        
        # Generate different types of reports
        reports = {}
        
        # Executive summary report
        exec_config = ReportConfig(
            report_type=ReportType.EXECUTIVE_SUMMARY,
            format=ReportFormat.HTML,
            include_charts=True,
            include_recommendations=True
        )
        
        exec_report = report_generator.generate_simulation_report(
            simulation_results, exec_config
        )
        
        exec_report_path = analytics_dir / "executive_summary.html"
        report_generator.export_report(exec_report, str(exec_report_path), ReportFormat.HTML)
        reports['executive_summary'] = str(exec_report_path)
        
        # Competitive analysis report
        comp_config = ReportConfig(
            report_type=ReportType.COMPETITIVE_ANALYSIS,
            format=ReportFormat.HTML,
            include_charts=True
        )
        
        comp_report = report_generator.generate_competitive_report(
            simulation_data, comp_config
        )
        
        comp_report_path = analytics_dir / "competitive_analysis.html"
        report_generator.export_report(comp_report, str(comp_report_path), ReportFormat.HTML)
        reports['competitive_analysis'] = str(comp_report_path)
        
        analytics_results['reports'] = reports
    
    # Generate visualizations
    if generate_visualizations:
        logger.info("Generating visualizations...")
        viz_manager = VisualizationManager(str(analytics_dir / "visualizations"))
        
        # Create dashboard charts
        charts = viz_manager.create_dashboard_charts(simulation_data)
        
        # Export dashboard
        dashboard_path = viz_manager.export_charts_to_html(
            charts, "simulation_dashboard.html"
        )
        
        analytics_results['visualizations'] = {
            'dashboard': dashboard_path,
            'charts': list(charts.keys())
        }
    
    # Generate insights
    if generate_insights:
        logger.info("Generating intelligent insights...")
        insight_engine = InsightEngine()
        
        # Analyze simulation data
        insights = insight_engine.analyze_simulation_data(simulation_data)
        
        # Export insights
        insights_json = insight_engine.export_insights("json")
        insights_path = analytics_dir / "insights.json"
        
        with open(insights_path, 'w') as f:
            f.write(insights_json)
        
        # Generate insights summary
        insights_summary = insight_engine.generate_insight_summary()
        summary_path = analytics_dir / "insights_summary.json"
        
        with open(summary_path, 'w') as f:
            json.dump(insights_summary, f, indent=2)
        
        analytics_results['insights'] = {
            'total_insights': len(insights),
            'insights_file': str(insights_path),
            'summary_file': str(summary_path),
            'summary': insights_summary
        }
    
    logger.info(f"Analytics generated and saved to {analytics_dir}")
    
    return analytics_results


def main():
    """Main entry point for the PriceFly simulation platform."""
    
    parser = argparse.ArgumentParser(
        description="PriceFly - Airline Pricing Simulation Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate synthetic data
  python -m pricefly.main --mode generate-data --output-dir ./data
  
  # Run baseline simulation
  python -m pricefly.main --mode simulate --scenario baseline --data-dir ./data --output-dir ./results
  
  # Run full pipeline (data generation + simulation + analytics)
  python -m pricefly.main --mode full-pipeline --scenario recession --output-dir ./output
  
  # Run simulation with custom parameters
  python -m pricefly.main --mode simulate --scenario custom --config config.json --data-dir ./data
        """
    )
    
    # Main arguments
    parser.add_argument(
        "--mode",
        choices=["generate-data", "simulate", "analytics", "full-pipeline"],
        required=True,
        help="Operation mode"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path (optional)"
    )
    
    # Data generation arguments
    parser.add_argument(
        "--num-airports",
        type=int,
        default=100,
        help="Number of airports to generate"
    )
    
    parser.add_argument(
        "--num-airlines",
        type=int,
        default=10,
        help="Number of airlines to generate"
    )
    
    parser.add_argument(
        "--num-routes",
        type=int,
        default=500,
        help="Number of routes to generate"
    )
    
    # Simulation arguments
    parser.add_argument(
        "--scenario",
        choices=["baseline", "recession", "high_growth", "pandemic", "high_competition", 
                "fuel_volatility", "tech_disruption", "environmental_regulations", "custom"],
        default="baseline",
        help="Simulation scenario"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Directory containing simulation data"
    )
    
    parser.add_argument(
        "--simulation-days",
        type=int,
        default=365,
        help="Number of days to simulate"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom configuration JSON file"
    )
    
    # Analytics arguments
    parser.add_argument(
        "--results-file",
        type=str,
        help="Path to simulation results file for analytics"
    )
    
    parser.add_argument(
        "--no-reports",
        action="store_true",
        help="Skip report generation"
    )
    
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Skip visualization generation"
    )
    
    parser.add_argument(
        "--no-insights",
        action="store_true",
        help="Skip insights generation"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting PriceFly simulation platform")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        if args.mode == "generate-data":
            # Generate synthetic data
            config = DataGenerationConfig(
                num_airports=args.num_airports,
                num_airlines=args.num_airlines,
                num_routes=args.num_routes,
                num_aircraft=50,
                num_passengers=10000,
                simulation_start_date=datetime.now().date(),
                simulation_end_date=(datetime.now() + timedelta(days=args.simulation_days)).date()
            )
            
            file_paths = generate_synthetic_data(config, args.output_dir)
            
            logger.info("Data generation completed successfully")
            logger.info(f"Generated files: {list(file_paths.keys())}")
            
        elif args.mode == "simulate":
            # Run simulation
            if not args.data_dir:
                logger.error("--data-dir is required for simulation mode")
                sys.exit(1)
            
            # Load custom config if provided
            custom_config = None
            if args.config:
                with open(args.config, 'r') as f:
                    custom_config = json.load(f)
            
            # Map scenario string to enum
            scenario_map = {
                "baseline": ScenarioType.BASELINE,
                "recession": ScenarioType.RECESSION,
                "high_growth": ScenarioType.HIGH_GROWTH,
                "pandemic": ScenarioType.PANDEMIC,
                "high_competition": ScenarioType.HIGH_COMPETITION,
                "fuel_volatility": ScenarioType.FUEL_VOLATILITY,
                "tech_disruption": ScenarioType.TECHNOLOGY_DISRUPTION,
                "environmental_regulations": ScenarioType.ENVIRONMENTAL_REGULATIONS,
                "custom": ScenarioType.BASELINE  # Default for custom
            }
            
            scenario_type = scenario_map[args.scenario]
            
            results = run_simulation(
                scenario_type,
                args.data_dir,
                args.output_dir,
                args.simulation_days,
                custom_config
            )
            
            # Save results
            results_path = Path(args.output_dir) / "simulation_results.json"
            with open(results_path, 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                json_results = json.loads(json.dumps(results, default=str))
                json.dump(json_results, f, indent=2)
            
            logger.info(f"Simulation completed successfully")
            logger.info(f"Results saved to {results_path}")
            
        elif args.mode == "analytics":
            # Generate analytics from existing results
            if not args.results_file:
                logger.error("--results-file is required for analytics mode")
                sys.exit(1)
            
            with open(args.results_file, 'r') as f:
                simulation_results = json.load(f)
            
            analytics_results = generate_analytics(
                simulation_results,
                args.output_dir,
                generate_reports=not args.no_reports,
                generate_visualizations=not args.no_visualizations,
                generate_insights=not args.no_insights
            )
            
            logger.info("Analytics generation completed successfully")
            
        elif args.mode == "full-pipeline":
            # Run complete pipeline
            logger.info("Running full pipeline: data generation + simulation + analytics")
            
            # Step 1: Generate data
            data_dir = Path(args.output_dir) / "data"
            config = DataGenerationConfig(
                num_airports=args.num_airports,
                num_airlines=args.num_airlines,
                num_routes=args.num_routes,
                num_aircraft=50,
                num_passengers=10000,
                simulation_start_date=datetime.now().date(),
                simulation_end_date=(datetime.now() + timedelta(days=args.simulation_days)).date()
            )
            
            generate_synthetic_data(config, str(data_dir))
            
            # Step 2: Run simulation
            scenario_map = {
                "baseline": ScenarioType.BASELINE,
                "recession": ScenarioType.RECESSION,
                "high_growth": ScenarioType.HIGH_GROWTH,
                "pandemic": ScenarioType.PANDEMIC,
                "high_competition": ScenarioType.HIGH_COMPETITION,
                "fuel_volatility": ScenarioType.FUEL_VOLATILITY,
                "tech_disruption": ScenarioType.TECHNOLOGY_DISRUPTION,
                "environmental_regulations": ScenarioType.ENVIRONMENTAL_REGULATIONS,
                "custom": ScenarioType.BASELINE
            }
            
            scenario_type = scenario_map[args.scenario]
            
            custom_config = None
            if args.config:
                with open(args.config, 'r') as f:
                    custom_config = json.load(f)
            
            simulation_results = run_simulation(
                scenario_type,
                str(data_dir),
                args.output_dir,
                args.simulation_days,
                custom_config
            )
            
            # Step 3: Generate analytics
            analytics_results = generate_analytics(
                simulation_results,
                args.output_dir,
                generate_reports=not args.no_reports,
                generate_visualizations=not args.no_visualizations,
                generate_insights=not args.no_insights
            )
            
            logger.info("Full pipeline completed successfully")
            logger.info(f"All outputs saved to {args.output_dir}")
            
            # Print summary
            print("\n" + "="*60)
            print("PRICEFLY SIMULATION COMPLETED")
            print("="*60)
            print(f"Scenario: {args.scenario}")
            print(f"Simulation Days: {args.simulation_days}")
            print(f"Output Directory: {args.output_dir}")
            
            if 'insights' in analytics_results:
                insights_summary = analytics_results['insights']['summary']
                print(f"\nInsights Generated: {insights_summary.get('total_insights', 0)}")
                print(f"Critical Insights: {insights_summary.get('key_metrics', {}).get('critical_insights', 0)}")
            
            if 'visualizations' in analytics_results:
                print(f"\nDashboard: {analytics_results['visualizations']['dashboard']}")
            
            if 'reports' in analytics_results:
                print(f"\nReports Generated:")
                for report_name, report_path in analytics_results['reports'].items():
                    print(f"  - {report_name}: {report_path}")
            
            print("\n" + "="*60)
    
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    logger.info("PriceFly execution completed successfully")


if __name__ == "__main__":
    main()