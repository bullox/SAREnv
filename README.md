# SARenv: UAV Search and Rescue Dataset and Evaluation Framework

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.1.7-green.svg)](https://github.com/your-repo/sarenv)

SARenv is an open-access dataset and evaluation framework designed to support research in UAV-based search and rescue (SAR) algorithms. This toolkit addresses the critical need for standardized datasets and benchmarks in wilderness SAR operations, enabling systematic evaluation and comparison of algorithmic approaches including coverage path planning, probabilistic search, and information-theoretic exploration.

## 🎯 Project Goals

Unmanned Aerial Vehicles (UAVs) play an increasingly vital role in wilderness search and rescue operations by enhancing situational awareness and extending the reach of human teams. However, the absence of standardized datasets and benchmarks has hindered systematic evaluation and comparison of UAV-based SAR algorithms. SARenv bridges this gap by providing:

- **Realistic geospatial scenarios** across diverse terrain types
- **Synthetic victim locations** derived from statistical models of lost person behavior
- **Comprehensive evaluation metrics** for search trajectory assessment
- **Baseline planners** for reproducible algorithm comparisons
- **Extensible framework** for custom algorithm development

## 🌟 Key Features

### 📊 Dataset Generation

- **Multi-scale environments**: Small, medium, large, and extra-large search areas
- **Diverse terrain types**: Flat and mountainous environments
- **Climate variations**: Temperate and dry climate conditions
- **Realistic geospatial features**: Roads, water bodies, vegetation, structures, and terrain features extracted from OpenStreetMap

### 🎯 Lost Person Modeling

- Statistical models based on established lost person behavior research
- Probability heatmaps incorporating environmental factors
- Configurable victim location generation with terrain-aware distributions

### 🚁 Path Planning Algorithms

- **Spiral Coverage**: Efficient outward spiral search patterns
- **Concentric Circles**: Systematic circular search patterns
- **Pizza Zigzag**: Sector-based zigzag coverage
- **Greedy Search**: Probability-driven adaptive search
- **Extensible framework** for custom algorithm integration

### 📈 Evaluation Metrics

- **Coverage metrics**: Area coverage and search efficiency
- **Likelihood scores**: Probability-weighted path evaluation
- **Time-discounted scoring**: Temporal effectiveness assessment
- **Victim detection rates**: Success probability and timeliness analysis
- **Multi-drone coordination**: Support for collaborative search strategies

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/sarenv.git
cd sarenv

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

#### 1. Generate Dataset

```python
import sarenv

# Initialize data generator
generator = sarenv.DataGenerator()

# Generate dataset for different locations and sizes
generator.export_dataset()
```

#### 2. Load and Visualize Data

```python
import sarenv

# Load a dataset
loader = sarenv.DatasetLoader("sarenv_dataset")
item = loader.load_environment("large")

# Visualize the environment
from examples.02_load_and_visualize import visualize_heatmap, visualize_features
visualize_heatmap(item)
visualize_features(item)
```

#### 3. Generate Lost Person Locations

```python
import sarenv

# Load environment
loader = sarenv.DatasetLoader("sarenv_dataset")
item = loader.load_environment("medium")

# Generate victim locations
victim_generator = sarenv.LostPersonLocationGenerator(item)
locations = victim_generator.generate_locations(num_locations=100)
```

#### 4. Evaluate Search Algorithms

```python
import sarenv

# Initialize comparative evaluator
evaluator = sarenv.ComparativeEvaluator(
    dataset_directory="sarenv_dataset",
    evaluation_sizes=["medium", "large"],
    num_drones=5,
    num_lost_persons=50
)

# Run baseline evaluations
results = evaluator.run_baseline_evaluations()

# Plot comparative results
evaluator.plot_results(results)
```

## 📁 Repository Structure

```text
sarenv/
├── sarenv/                     # Main package
│   ├── analytics/              # Path planning and evaluation
│   │   ├── paths.py           # Coverage path algorithms
│   │   ├── metrics.py         # Evaluation metrics
│   │   └── evaluator.py       # Comparative evaluation framework
│   ├── core/                   # Core functionality
│   │   ├── generation.py      # Dataset generation
│   │   ├── loading.py         # Dataset loading utilities
│   │   └── lost_person.py     # Lost person modeling
│   └── utils/                  # Utility functions
│       ├── geo.py             # Geospatial utilities
│       ├── plot.py            # Visualization tools
│       └── lost_person_behavior.py  # Behavioral models
├── examples/                   # Usage examples
│   ├── 02_load_and_visualize.py
│   ├── 03_generate_survivors.py
│   └── 04_evaluate_coverage_paths.py
├── data/                       # Data processing scripts
└── sarenv_dataset/            # Generated datasets (created after running)
```

## 🔬 Research Applications

### Supported Algorithm Types

- **Coverage Path Planning**: Systematic area coverage strategies
- **Probabilistic Search**: Bayesian and heuristic search methods
- **Information-Theoretic Exploration**: Entropy-based search optimization
- **Multi-Agent Coordination**: Collaborative UAV search strategies

### Evaluation Dimensions

- **Spatial Coverage**: Area covered vs. time efficiency
- **Probability Optimization**: Likelihood-weighted search performance
- **Temporal Dynamics**: Time-sensitive victim detection modeling
- **Resource Utilization**: Multi-drone coordination effectiveness

## 📊 Dataset Specifications

### Environment Scales

| Size    | Radius (km) | Area (km²) | Use Case |
|---------|-------------|------------|----------|
| Small   | 1.0         | ~3.14      | Algorithm development |
| Medium  | 2.0         | ~12.57     | Comparative testing |
| Large   | 4.0         | ~50.27     | Realistic scenarios |
| XLarge  | 8.0         | ~201.06    | Challenging benchmarks |

### Terrain Types

- **Flat Environments**: Plains, fields, minimal elevation variation
- **Mountainous Environments**: Hills, valleys, significant elevation changes

### Climate Conditions

- **Temperate**: Moderate conditions, mixed vegetation
- **Dry**: Arid conditions, sparse vegetation patterns

## 🛠️ Custom Algorithm Integration

Add your own path planning algorithm:

```python
def custom_search_algorithm(center_x, center_y, max_radius, num_drones, **kwargs):
    """
    Custom search algorithm implementation.
    
    Args:
        center_x, center_y: Search center coordinates
        max_radius: Maximum search radius in meters
        num_drones: Number of UAVs
        **kwargs: Additional parameters (fov_deg, altitude, etc.)
    
    Returns:
        list[LineString]: Path for each drone
    """
    # Your algorithm implementation
    paths = []
    # ... algorithm logic ...
    return paths

# Register with evaluator
evaluator = sarenv.ComparativeEvaluator()
evaluator.path_generators['custom'] = custom_search_algorithm
```

## 📈 Performance Metrics

The framework provides comprehensive metrics for algorithm evaluation:

- **Total Likelihood Score**: Probability-weighted coverage assessment
- **Time-Discounted Score**: Temporal efficiency with decay factors
- **Victim Detection Rate**: Percentage of victims found
- **Average Detection Distance**: Mean travel distance to victim discovery
- **Coverage Efficiency**: Area covered per unit time/distance

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- Adding new path planning algorithms
- Extending evaluation metrics
- Improving dataset generation
- Documentation and examples

## 📝 Citation

If you use SARenv in your research, please cite:

```bibtex
@article{sarenv2024,
  title={An Open-Access Dataset and Evaluation Framework for UAV-Based Search and Rescue Algorithms},
  author={[Author Names]},
  journal={[Journal Name]},
  year={2024},
  publisher={[Publisher]}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Lost person behavior models based on research by Koester (2008)
- Geospatial data provided by OpenStreetMap contributors
- Built with Python geospatial libraries: GeoPandas, Shapely, Rasterio

## 📞 Support

- **Documentation**: [Link to documentation]
- **Issues**: Please report bugs and feature requests via GitHub Issues
- **Discussions**: Join our community discussions for questions and ideas
- **Contact**: [Contact information]

---

**Note**: This framework is designed for research purposes. For real-world SAR operations, please consult with professional search and rescue organizations and follow established protocols.
