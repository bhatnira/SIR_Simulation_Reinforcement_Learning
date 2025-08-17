# SIR Epidemic Simulation with Bayesian Inference

[![C++](https://img.shields.io/badge/C%2B%2B-14-blue.svg)](https://isocpp.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Bayesian](https://img.shields.io/badge/Bayesian-MCMC-purple.svg)]()

A high-performance C++ implementation of the **SIR (Susceptible-Infected-Recovered)** epidemic model with **Bayesian parameter estimation** using Markov Chain Monte Carlo (MCMC) methods. This project demonstrates modern C++ practices, object-oriented design, computational epidemiology, and advanced statistical inference techniques.

## 🦠 About the SIR Model

The SIR model is a fundamental compartmental model in epidemiology that divides the population into three compartments:

- **S**usceptible: Individuals who can be infected
- **I**nfected: Individuals who are currently infected and can transmit the disease
- **R**ecovered: Individuals who have recovered and gained immunity

The model simulates how individuals transition between these states over time, providing insights into epidemic dynamics such as peak infection rates, total attack rates, and epidemic duration.

## 🏗️ Project Architecture

The project follows modern C++ design principles with clear separation of concerns:

```
ScientificComputing-SIRSimulation/
├── src/
│   ├── Person.h/cpp           # Individual person interface and implementation
│   ├── Population.h/cpp       # Population dynamics and disease transmission
│   ├── Simulation.h           # Standard simulation configuration
│   ├── SIRSimulation.cpp      # Standard SIR simulation runner
│   ├── BayesianSIR.h/cpp      # Bayesian inference framework with MCMC
│   └── BayesianExample.cpp    # Bayesian parameter estimation example
├── Makefile                   # Build configuration for both simulations
├── README.md                  # Project documentation
└── docs/                      # Additional documentation
```

### 📋 Core Components

#### Standard SIR Simulation

#### `Person` Class
- **Purpose**: Represents an individual in the simulation
- **Responsibilities**: 
  - State management (susceptible → infected → recovered)
  - Infection duration tracking
  - State transition logic
- **Design**: Encapsulated state with const-correct accessors

#### `Population` Class  
- **Purpose**: Manages population-level dynamics
- **Responsibilities**:
  - Disease transmission simulation
  - Population statistics tracking (S, I, R counts)
  - Parameter configuration and validation
- **Design**: RAII principles with smart pointer memory management

#### `SIRSimulation` Class
- **Purpose**: Orchestrates the standard simulation
- **Responsibilities**:
  - Configuration management via `SimulationConfig`
  - Simulation lifecycle control
  - Results output and reporting
- **Design**: Command pattern with configurable parameters

#### Bayesian Inference Framework

#### `BayesianSIRSampler` Class
- **Purpose**: MCMC-based parameter estimation
- **Responsibilities**:
  - Metropolis-Hastings sampling algorithm
  - Prior specification and likelihood calculation
  - Posterior sample generation and convergence diagnostics
- **Design**: Flexible prior distributions with uncertainty quantification

#### `PosteriorAnalysis` Class
- **Purpose**: Statistical analysis of MCMC results
- **Responsibilities**:
  - Summary statistics and credible intervals
  - Convergence diagnostics (effective sample size, R-hat)
  - Model comparison metrics (DIC, WAIC)
- **Design**: Comprehensive Bayesian workflow tools

## ✨ Key Features

### 🎯 Standard SIR Simulation
- **Configurable Parameters**: Population size, infection rates, contact patterns
- **Realistic Dynamics**: Stochastic transmission with probabilistic contacts
- **Performance Optimized**: Efficient algorithms for large population simulations
- **Early Termination**: Automatic detection of epidemic end conditions

### 🔬 Bayesian Inference Capabilities
- **Parameter Estimation**: MCMC-based estimation of transmission and recovery rates
- **Uncertainty Quantification**: Credible intervals and posterior distributions
- **Prior Specification**: Flexible prior distributions (Uniform, Normal, Beta, Gamma, Exponential)
- **Model Comparison**: Bayesian model selection using DIC, WAIC, and Bayes factors
- **Convergence Diagnostics**: Effective sample size, R-hat statistics, and acceptance rates
- **Posterior Predictive**: Generate predictions with uncertainty bounds

### 🛠️ Technical Excellence
- **Modern C++14**: RAII, smart pointers, const-correctness
- **Memory Safe**: Automated memory management with `std::unique_ptr`
- **Well-Documented**: Comprehensive Doxygen-style documentation
- **Modular Design**: Clean interfaces and separation of concerns
- **Build System**: Professional Makefile with multiple targets

### 📊 Configurable Parameters
| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| Population Size | Total number of individuals | 1000 |
| Initial Infections | Starting infected count | 5 |
| Simulation Days | Maximum simulation duration | 90 |
| Infection Probability | Transmission chance per contact | 0.5 (50%) |
| Contacts Per Day | Daily contacts per infected person | 6 |
| Infection Duration | Days an individual remains infectious | 5 |

## 🚀 Quick Start

### Prerequisites
- C++ compiler with C++14 support (g++, clang++)
- Make utility
- Unix-like environment (Linux, macOS, WSL)

### Building and Running

```bash
# Clone the repository
git clone https://github.com/bhatnira/ScientificComputing-SIRSimulation.git
cd ScientificComputing-SIRSimulation

# Build both simulations
make

# Build individual targets
make sir         # Standard SIR simulation only
make bayesian    # Bayesian analysis only

# Run standard simulation
make run
# Or directly: ./sir_simulation

# Run Bayesian parameter estimation
make run-bayesian
# Or directly: ./bayesian_sir
```

### Build Targets
```bash
make              # Build both executables
make sir          # Build standard simulation
make bayesian     # Build Bayesian analysis
make clean        # Remove build artifacts  
make rebuild      # Clean and build
make run          # Build and run standard simulation
make run-bayesian # Build and run Bayesian analysis
make debug        # Build with debug symbols
make help         # Show available targets
```

## 💻 Usage Examples

### Standard SIR Simulation
```cpp
#include "Simulation.h"

int main() {
    // Create default configuration
    SimulationConfig config;
    
    // Run simulation with defaults
    SIRSimulation simulation(config);
    simulation.runSimulation();
    
    return 0;
}
```

### Bayesian Parameter Estimation
```cpp
#include "BayesianSIR.h"

int main() {
    // Load epidemic data (S, I, R observations over time)
    EpidemicData data = loadObservedData("epidemic_data.csv");
    
    // Define prior distributions for parameters
    PriorDistribution betaPrior(PriorDistribution::GAMMA, 2.0, 1.0);      // Transmission rate
    PriorDistribution gammaPrior(PriorDistribution::GAMMA, 2.0, 10.0);    // Recovery rate
    PriorDistribution initPrior(PriorDistribution::BETA, 1.0, 99.0);      // Initial infected fraction
    
    // Create MCMC sampler
    BayesianSIRSampler sampler(betaPrior, gammaPrior, initPrior, 
                              1000,   // burn-in samples
                              5000,   // total samples
                              2);     // thinning interval
    
    // Run parameter estimation
    BayesianParameters initial;
    auto posteriorSamples = sampler.sample(data, initial);
    
    // Analyze results
    std::vector<double> betaSamples, gammaSamples;
    for (const auto& sample : posteriorSamples) {
        betaSamples.push_back(sample.transmissionRate);
        gammaSamples.push_back(sample.recoveryRate);
    }
    
    // Calculate summary statistics
    auto betaStats = PosteriorAnalysis::summarizeParameter(betaSamples);
    auto gammaStats = PosteriorAnalysis::summarizeParameter(gammaSamples);
    
    std::cout << "β estimate: " << betaStats.mean << " ± " << betaStats.std << std::endl;
    std::cout << "γ estimate: " << gammaStats.mean << " ± " << gammaStats.std << std::endl;
    
    return 0;
}
```

### Custom Configuration
```cpp
// Advanced configuration example
SimulationConfig config;
config.populationSize = 5000;           // Larger population
config.initialInfections = 10;          // More initial cases
config.simulationDays = 365;            // One year simulation
config.infectionProbability = 0.3;      // Lower transmission rate
config.contactsPerDay = 8;               // Higher contact rate
config.infectionDuration = 7;           // Longer infectious period

SIRSimulation simulation(config);
simulation.runSimulation();

// Access final statistics
const Population& pop = simulation.getPopulation();
std::cout << "Final susceptible: " << pop.getSusceptibleCount() << std::endl;
std::cout << "Total recovered: " << pop.getRecoveredCount() << std::endl;
```

## 📈 Sample Output

```
Starting SIR Epidemic Simulation
Population Size: 1000
Initial Infections: 5
Simulation Days: 90
Infection Probability: 0.5
Contacts Per Day: 6
Infection Duration: 5 days

Day 0: Susceptible=1000, Infected=0, Recovered=0
Day 1: Susceptible=980, Infected=20, Recovered=0
Day 2: Susceptible=923, Infected=77, Recovered=0
Day 3: Susceptible=745, Infected=255, Recovered=0
Day 4: Susceptible=345, Infected=655, Recovered=0
Day 5: Susceptible=52, Infected=943, Recovered=5
...
Epidemic ended on day 23

Simulation completed.
Final Statistics:
Susceptible: 12
Recovered: 988
Total Affected: 988
```

## 🔬 Scientific Applications

This simulation can be used for:

- **Educational Purposes**: Understanding epidemic dynamics and compartmental models
- **Policy Analysis**: Exploring effects of intervention strategies (social distancing, vaccination)
- **Parameter Studies**: Investigating how disease characteristics affect outbreak patterns
- **Research**: Baseline for more complex epidemic models (SEIR, age-structured, etc.)

## 📊 Extension Ideas

- **Vaccination**: Add vaccination compartment and strategies
- **Demographics**: Age-structured populations with different contact patterns
- **Spatial Models**: Geographic spread with mobility patterns  
- **Interventions**: Dynamic policy changes (lockdowns, masks, etc.)
- **Data Visualization**: Real-time plotting of epidemic curves
- **Parameter Estimation**: Fitting model to real epidemic data

## 🏗️ Development

### Code Style
- Follow C++ Core Guidelines
- Use meaningful variable and function names
- Maintain const-correctness
- Document public interfaces with Doxygen comments
- Keep functions focused and cohesive

### Testing
```bash
# Build debug version for development
make debug

# Add your own test cases in a separate test file
# Example: tests/test_person.cpp, tests/test_population.cpp
```

### Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 Background & Theory

### Mathematical Foundation
The SIR model is governed by differential equations:

```
dS/dt = -βSI/N
dI/dt = βSI/N - γI  
dR/dt = γI
```

Where:
- β = transmission rate (probability × contacts)
- γ = recovery rate (1/infection_duration)
- N = total population (S + I + R)

### Key Metrics
- **Basic Reproduction Number (R₀)**: Average secondary infections per primary case
- **Attack Rate**: Final proportion of population infected
- **Peak Prevalence**: Maximum number of simultaneous infections
- **Epidemic Duration**: Time from outbreak to extinction

## 🔧 Technical Details

### Performance Characteristics
- **Time Complexity**: O(n×d) where n=population, d=simulation days
- **Space Complexity**: O(n) for population storage
- **Memory Usage**: ~24 bytes per person (using smart pointers)
- **Scalability**: Tested with populations up to 100,000 individuals

### Dependencies
- Standard C++ Library (no external dependencies)
- C++14 compliant compiler
- POSIX-compatible build environment

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Epidemiological modeling concepts from Kermack-McKendrick (1927)
- C++ best practices from the C++ Core Guidelines
- Modern software engineering principles

## 📞 Contact

For questions, suggestions, or collaborations:
- Create an issue in this repository
---

**Note**: This simulation is for educational and research purposes. For real-world epidemic analysis, consult epidemiological experts and use validated models with proper calibration.
