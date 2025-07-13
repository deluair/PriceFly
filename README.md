# PriceFly - Airline Pricing Simulation Platform

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

PriceFly is a comprehensive simulation platform that models the complex ecosystem of airline pricing strategies, incorporating real-time market dynamics, competitive intelligence, and advanced revenue optimization algorithms. This project simulates how airlines set prices across different market segments, routes, and booking horizons while accounting for operational costs, regulatory constraints, and macroeconomic factors.

## 🚀 Key Features

### Dynamic Pricing Engine
- **Willingness-to-Pay (WTP) Modeling**: AI-driven algorithms that segment passengers into behavioral cohorts
- **Real-time Price Optimization**: Continuous pricing adjustments based on demand forecasting and competitive positioning
- **Booking Class Management**: Traditional 26-character booking class systems transitioning to flexible continuous pricing
- **Demand Elasticity Calculations**: Price sensitivity modeling across different customer segments

### Multi-Variable Cost Structure
- **Fuel Price Volatility**: Integration of jet fuel price fluctuations with hedging strategies
- **Exchange Rate Impact**: Currency fluctuation effects on international routes
- **Operational Cost Modeling**: Labor costs, airport fees, maintenance, and regulatory compliance
- **Route-Specific Analysis**: Hub-and-spoke vs. point-to-point operational models

### Competitive Intelligence Framework
- **Market Positioning Analysis**: Real-time competitor pricing strategy monitoring
- **Capacity Management**: Seat inventory allocation across booking classes and timeframes
- **Route Competition Dynamics**: Market concentration effects and competitive responses
- **Alliance Impact**: Revenue-sharing mechanisms and joint pricing strategies

### Advanced Analytics
- **Revenue Management Optimization**: Historical booking pattern analysis and forecasting
- **Seasonal Adjustments**: Pricing modifications based on holidays and business cycles
- **Ancillary Revenue Integration**: Comprehensive passenger value modeling
- **Risk Management**: Overbooking strategies and compensation cost modeling

## 📁 Project Structure

```
PriceFly/
├── src/pricefly/
│   ├── __init__.py
│   ├── main.py                    # Main entry point
│   ├── core/
│   │   ├── __init__.py
│   │   └── pricing_engine.py      # Core pricing algorithms
│   ├── data/
│   │   ├── __init__.py
│   │   ├── synthetic_data.py       # Data generation engine
│   │   ├── generators.py          # Specific data generators
│   │   ├── loader.py              # Data loading utilities
│   │   └── validator.py           # Data validation
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── engine.py              # Simulation orchestration
│   │   ├── market.py              # Market dynamics
│   │   ├── demand.py              # Demand modeling
│   │   ├── events.py              # External events
│   │   └── scenarios.py           # Scenario management
│   └── analytics/
│       ├── __init__.py
│       ├── metrics.py             # Performance metrics
│       ├── reporting.py           # Report generation
│       ├── visualization.py       # Charts and dashboards
│       └── insights.py            # AI-driven insights
├── tests/
├── docs/
├── examples/
├── requirements.txt
├── setup.py
└── README.md
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Install

```bash
# Clone the repository
git clone https://github.com/deluair/PriceFly.git
cd PriceFly

# Install in development mode
pip install -e .

# Or install from PyPI (when available)
pip install pricefly
```

### Development Installation

```bash
# Install with development dependencies
pip install -e ".[dev,docs,jupyter]"

# Install pre-commit hooks
pre-commit install
```

## 🚀 Quick Start

### 1. Generate Synthetic Data

```bash
# Generate synthetic airline data
pricefly --mode generate-data --output-dir ./data --num-airports 100 --num-airlines 10
```

### 2. Run a Basic Simulation

```bash
# Run baseline scenario simulation
pricefly --mode simulate --scenario baseline --data-dir ./data --output-dir ./results
```

### 3. Full Pipeline Execution

```bash
# Run complete pipeline: data generation + simulation + analytics
pricefly --mode full-pipeline --scenario recession --output-dir ./output --simulation-days 365
```

### 4. Python API Usage

```python
# Run the basic simulation example
python3 examples/basic_simulation.py
```

This will generate:
- Synthetic airline data (airports, airlines, routes, bookings, flights)
- Performance metrics and revenue analysis
- Interactive HTML charts showing revenue and load factor by airline
- Comprehensive simulation reports

**Recent Updates:**
- ✅ Fixed blank chart generation issue
- ✅ Improved booking data linkage with flight data
- ✅ Enhanced revenue calculation accuracy
- ✅ Added debug output for data validation

## 📊 Available Scenarios

- **Baseline**: Normal market conditions with standard competition
- **Recession**: Economic downturn with reduced demand and price sensitivity
- **High Growth**: Expanding market with increased demand and capacity
- **Pandemic**: Crisis scenario with travel restrictions and demand shocks
- **High Competition**: Intense price competition with new market entrants
- **Fuel Volatility**: Fluctuating fuel costs affecting operational expenses
- **Technology Disruption**: New distribution channels and pricing technologies
- **Environmental Regulations**: Carbon pricing and sustainability requirements

## 📈 Analytics and Reporting

### Generated Reports
- **Executive Summary**: High-level performance metrics and KPIs
- **Competitive Analysis**: Market positioning and competitor benchmarking
- **Route Performance**: Route-specific profitability and optimization opportunities
- **Revenue Analysis**: Detailed revenue breakdown and trend analysis

### Visualizations
- **Price Trend Charts**: Historical and forecasted pricing patterns
- **Demand Heatmaps**: Geographic and temporal demand visualization
- **Market Share Analysis**: Competitive positioning over time
- **Revenue Waterfalls**: Revenue component analysis

### AI-Driven Insights
- **Pricing Recommendations**: Optimal pricing strategies based on market conditions
- **Market Opportunities**: Underserved routes and segments
- **Risk Assessments**: Potential threats and mitigation strategies
- **Performance Benchmarks**: Industry comparison and best practices

## 🔧 Configuration

### Custom Scenario Configuration

```json
{
  "economic_parameters": {
    "gdp_growth_rate": 0.025,
    "inflation_rate": 0.03,
    "fuel_price_volatility": 0.15
  },
  "competitive_parameters": {
    "market_concentration": 0.7,
    "price_sensitivity": 0.8,
    "capacity_utilization": 0.85
  },
  "operational_parameters": {
    "cost_efficiency": 0.9,
    "schedule_reliability": 0.95,
    "fleet_utilization": 0.88
  }
}
```

### Environment Variables

```bash
# Optional configuration
export PRICEFLY_LOG_LEVEL=INFO
export PRICEFLY_PARALLEL_WORKERS=4
export PRICEFLY_CACHE_DIR=./cache
export PRICEFLY_OUTPUT_FORMAT=json
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pricefly --cov-report=html

# Run specific test module
pytest tests/test_pricing_engine.py
```

## 📚 Documentation

- [API Reference](docs/api/)
- [User Guide](docs/user_guide/)
- [Developer Guide](docs/developer_guide/)
- [Examples](examples/)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Format code (`black src/ tests/`)
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Airline industry experts for domain knowledge
- Open source community for foundational libraries
- Academic researchers in revenue management and pricing

## 📞 Support


- **Issues**: [GitHub Issues](https://github.com/deluair/PriceFly/issues)
- **Discussions**: [GitHub Discussions](https://github.com/deluair/PriceFly/discussions)


## 🔮 Roadmap

- [ ] Real-time data integration APIs
- [ ] Machine learning model marketplace
- [ ] Cloud deployment templates
- [ ] Interactive web dashboard
- [ ] Mobile analytics app
- [ ] Integration with airline reservation systems

---

**PriceFly** - Transforming airline pricing through intelligent simulation and analytics.