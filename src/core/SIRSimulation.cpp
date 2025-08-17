/**
 * @file SIRSimulation.cpp
 * @brief Implementation of SIR epidemic simulation orchestrator
 * @author Scientific Computing Team
 * @date 2025
 */

#include "Simulation.h"
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <iomanip>

// SimulationConfig implementation
SimulationConfig::SimulationConfig(int popSize, int initInfections, int simDays, 
                                  float infProb, int contacts, int duration)
    : populationSize(popSize), initialInfections(initInfections), 
      simulationDays(simDays), infectionProbability(infProb),
      contactsPerDay(contacts), infectionDuration(duration) {
    
    if (!isValid()) {
        throw std::invalid_argument("Invalid simulation configuration parameters");
    }
}

bool SimulationConfig::isValid() const {
    return populationSize > 0 && 
           initialInfections > 0 && 
           initialInfections <= populationSize &&
           simulationDays > 0 &&
           infectionProbability >= 0.0f && 
           infectionProbability <= 1.0f &&
           contactsPerDay >= 0 &&
           infectionDuration > 0;
}

std::string SimulationConfig::toString() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "Population: " << populationSize << ", "
        << "Initial Infections: " << initialInfections << ", "
        << "Days: " << simulationDays << ", "
        << "Infection Prob: " << infectionProbability << ", "
        << "Contacts/Day: " << contactsPerDay << ", "
        << "Duration: " << infectionDuration << " days";
    return oss.str();
}

// SIRSimulation implementation
SIRSimulation::SIRSimulation(const SimulationConfig& simConfig) 
    : config(simConfig), population(simConfig.populationSize) {
    
    // Validate configuration
    if (!config.isValid()) {
        throw std::invalid_argument("Invalid simulation configuration: " + config.toString());
    }
    
    // Configure population parameters
    population.setInfectionProbability(config.infectionProbability);
    population.setContactsPerDay(config.contactsPerDay);
    population.setInfectionDuration(config.infectionDuration);
}

void SIRSimulation::initializeSimulation() {
    // Introduce initial infections
    for (int i = 0; i < config.initialInfections; i++) {
        population.infectRandomPerson();
    }
}

void SIRSimulation::outputDailyStats(int day) const {
    std::cout << "Day " << std::setw(3) << day << ": "
              << "S=" << std::setw(4) << population.getSusceptibleCount() << ", "
              << "I=" << std::setw(4) << population.getInfectedCount() << ", "
              << "R=" << std::setw(4) << population.getRecoveredCount() << std::endl;
}

void SIRSimulation::runSimulation() {
    std::cout << "=== SIR Epidemic Simulation ===" << std::endl;
    std::cout << config.toString() << std::endl;
    std::cout << std::endl;
    
    // Initialize with initial infections
    initializeSimulation();
    
    // Output initial state (day 0)
    outputDailyStats(0);
    
    // Run simulation for specified number of days
    for (int day = 1; day <= config.simulationDays; day++) {
        population.simulateOneDay();
        outputDailyStats(day);
        
        // Early termination if no more infected individuals
        if (population.getInfectedCount() == 0) {
            std::cout << std::endl;
            std::cout << "*** Epidemic ended on day " << day << " ***" << std::endl;
            break;
        }
    }
    
    // Final summary
    std::cout << std::endl;
    std::cout << "=== Final Statistics ===" << std::endl;
    std::cout << "Susceptible: " << population.getSusceptibleCount() 
              << " (" << std::fixed << std::setprecision(1) 
              << (100.0 * population.getSusceptibleCount() / config.populationSize) << "%)" << std::endl;
    std::cout << "Recovered: " << population.getRecoveredCount() 
              << " (" << std::fixed << std::setprecision(1) 
              << (100.0 * population.getRecoveredCount() / config.populationSize) << "%)" << std::endl;
    std::cout << "Total Affected: " << (config.populationSize - population.getSusceptibleCount())
              << " (" << std::fixed << std::setprecision(1) 
              << (100.0 * (config.populationSize - population.getSusceptibleCount()) / config.populationSize) << "%)" << std::endl;
    std::cout << "Attack Rate: " << std::fixed << std::setprecision(1) 
              << (100.0 * (config.populationSize - population.getSusceptibleCount()) / config.populationSize) << "%" << std::endl;
}

int main() {
    try {
        // Create simulation configuration with default parameters
        SimulationConfig config;
        
        // Optionally customize parameters for different scenarios
        // Example: config.populationSize = 5000;
        // Example: config.infectionProbability = 0.3f;
        
        std::cout << "Initializing simulation with configuration:" << std::endl;
        std::cout << config.toString() << std::endl;
        std::cout << std::endl;
        
        // Create and run simulation
        SIRSimulation simulation(config);
        simulation.runSimulation();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
