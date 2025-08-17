/**
 * @file Simulation.h
 * @brief Main simulation orchestrator for SIR epidemic model
 * @author Scientific Computing Team
 * @date 2025
 * 
 * This file defines the simulation configuration and orchestrator classes
 * for running SIR epidemic simulations with configurable parameters.
 */

#ifndef SIMULATION_H
#define SIMULATION_H

#include "Population.h"

/**
 * @brief Configuration structure for SIR epidemic simulation parameters
 * 
 * This structure encapsulates all configurable parameters for the simulation,
 * providing default values that represent a typical respiratory disease outbreak.
 * All parameters can be customized for different disease characteristics or
 * population behaviors.
 * 
 * @note Parameter validation is performed in the SIRSimulation constructor
 */
struct SimulationConfig {
    int populationSize;         ///< Total population size (must be > 0)
    int initialInfections;      ///< Number of initially infected individuals (0 < initialInfections <= populationSize)
    int simulationDays;         ///< Number of days to simulate (must be > 0)
    float infectionProbability; ///< Probability of infection upon contact (0.0 <= p <= 1.0)
    int contactsPerDay;         ///< Number of contacts per infected person per day (must be >= 0)
    int infectionDuration;      ///< Duration of infection in days (must be > 0)
    
    /**
     * @brief Default constructor with epidemiologically reasonable default values
     * 
     * Default parameters represent a moderate respiratory disease outbreak:
     * - Population: 1000 individuals
     * - Initial infections: 5 cases (0.5% attack rate)
     * - Duration: 90 days (approximately 3 months)
     * - Transmission probability: 50% per contact
     * - Contact rate: 6 contacts per day per infected individual
     * - Infectious period: 5 days
     */
    SimulationConfig() 
        : populationSize(1000), initialInfections(5), simulationDays(90),
          infectionProbability(0.5f), contactsPerDay(6), infectionDuration(5) {}
    
    /**
     * @brief Parameterized constructor with validation
     * 
     * @param popSize Total population size (must be > 0)
     * @param initInfections Initial infected count (must be > 0 and <= popSize)
     * @param simDays Simulation duration in days (must be > 0)
     * @param infProb Infection probability per contact (must be 0.0-1.0)
     * @param contacts Daily contacts per infected person (must be >= 0)
     * @param duration Infection duration in days (must be > 0)
     * @throws std::invalid_argument if any parameter is invalid
     */
    SimulationConfig(int popSize, int initInfections, int simDays, 
                    float infProb, int contacts, int duration);
    
    /**
     * @brief Validates all configuration parameters
     * 
     * @return true if all parameters are valid, false otherwise
     * @note This method can be used for runtime validation without exceptions
     */
    bool isValid() const;
    
    /**
     * @brief Gets a string description of the configuration
     * 
     * @return Formatted string with all parameter values
     */
    std::string toString() const;
};

/**
 * @brief Main simulation runner class
 * 
 * Orchestrates the SIR epidemic simulation with configurable parameters
 */
class SIRSimulation {
private:
    SimulationConfig config;   ///< Simulation configuration
    Population population;     ///< Population being simulated
    
    /**
     * @brief Initializes the simulation with initial infections
     */
    void initializeSimulation();
    
    /**
     * @brief Outputs the current state of the simulation
     * 
     * @param day Current simulation day
     */
    void outputDailyStats(int day) const;

public:
    /**
     * @brief Constructor with configuration
     * 
     * @param simConfig Simulation configuration parameters
     */
    explicit SIRSimulation(const SimulationConfig& simConfig);
    
    /**
     * @brief Runs the complete simulation
     * 
     * Initializes the population, introduces initial infections,
     * and runs the simulation for the specified number of days
     */
    void runSimulation();
    
    /**
     * @brief Gets the current population statistics
     * 
     * @return Reference to the population object
     */
    const Population& getPopulation() const { return population; }
};

#endif // SIMULATION_H
