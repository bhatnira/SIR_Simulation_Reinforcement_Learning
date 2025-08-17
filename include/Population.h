/**
 * @file Population.h
 * @brief Population management for SIR epidemic simulation
 * @author Scientific Computing Team
 * @date 2025
 * 
 * This file defines the Population class which manages a collection of
 * individuals and handles the epidemiological dynamics of disease transmission.
 */

#ifndef POPULATION_H
#define POPULATION_H

#include "Person.h"
#include <vector>
#include <memory>

/**
 * @brief Manages a population of individuals in the SIR epidemic model
 * 
 * The Population class is responsible for:
 * - Managing a collection of Person objects
 * - Simulating disease transmission dynamics  
 * - Tracking epidemiological statistics (S, I, R counts)
 * - Configuring simulation parameters
 * - Advancing the simulation state over time
 * 
 * The class uses modern C++ practices including RAII for memory management
 * and const-correctness for data access methods.
 * 
 * @note This class is designed for computational efficiency with large populations
 *       while maintaining clear interfaces for parameter configuration.
 */
class Population {
private:
    int size;                               ///< Total population size
    int day;                               ///< Current simulation day
    
    // Compartment counts
    int countInfected;                     ///< Number of currently infected individuals
    int countSusceptible;                  ///< Number of susceptible individuals
    int countRecovered;                    ///< Number of recovered individuals
    
    // Simulation parameters
    float infectionProbability;            ///< Probability of infection upon contact
    int contactsPerDay;                    ///< Number of contacts per infected person per day
    int infectionDuration;                 ///< Duration of infection in days
    
    std::vector<std::unique_ptr<Person>> population;  ///< Container for all individuals

    /**
     * @brief Helper function to infect a specific person
     * 
     * @param person Pointer to the person to infect
     */
    void infectPerson(Person* person);

    /**
     * @brief Simulates disease transmission from infected individuals
     * 
     * @param newlyInfected Vector to store newly infected individuals
     */
    void simulateTransmission(std::vector<Person*>& newlyInfected);

    /**
     * @brief Updates the compartment counts based on current population state
     */
    void updateCounts();

public:
    /**
     * @brief Constructor to create a population of specified size
     * 
     * @param populationSize Number of individuals in the population
     */
    explicit Population(int populationSize);

    /**
     * @brief Destructor
     */
    ~Population() = default;

    /**
     * @brief Randomly infects one person in the population
     */
    void infectRandomPerson();

    /**
     * @brief Advances the simulation by one day
     * 
     * Updates all individual states, handles disease transmission,
     * and updates population statistics
     */
    void simulateOneDay();

    /**
     * @brief Sets the probability of infection upon contact
     * 
     * @param probability Value between 0.0 and 1.0
     */
    void setInfectionProbability(float probability);

    /**
     * @brief Sets the number of contacts per infected person per day
     * 
     * @param contacts Number of contacts
     */
    void setContactsPerDay(int contacts);

    /**
     * @brief Sets the duration of infection in days
     * 
     * @param days Duration in days
     */
    void setInfectionDuration(int days);

    // Getters for population statistics
    int getCurrentDay() const { return day; }
    int getPopulationSize() const { return size; }
    int getSusceptibleCount() const { return countSusceptible; }
    int getInfectedCount() const { return countInfected; }
    int getRecoveredCount() const { return countRecovered; }
    
    float getInfectionProbability() const { return infectionProbability; }
    int getContactsPerDay() const { return contactsPerDay; }
    int getInfectionDuration() const { return infectionDuration; }
};

#endif // POPULATION_H
