/**
 * @file Person.h
 * @brief Individual person representation in the SIR epidemic model
 * @author Scientific Computing Team
 * @date 2025
 * 
 * This file defines the Person class which represents an individual
 * in the SIR (Susceptible-Infected-Recovered) epidemic simulation.
 */

#ifndef PERSON_H
#define PERSON_H

#include <string>

/**
 * @brief Represents an individual in the SIR (Susceptible-Infected-Recovered) model
 * 
 * The Person class manages the state transitions of an individual through the
 * epidemic states: susceptible -> infected -> recovered. Each person maintains
 * their current health status and tracks the remaining duration of infection.
 * 
 * State Transitions:
 * - Susceptible: Can be infected when exposed
 * - Infected (Sick): Can transmit disease, has limited duration
 * - Recovered: Immune to further infection
 * 
 * @note This class uses the RAII principle and is designed to be lightweight
 *       for use in large population simulations.
 */
class Person {
private:
    int infectionDays;      ///< Number of days remaining to be infectious (0 if not infected)
    std::string current;    ///< Current health state: "susceptible", "sick", or "recovered"

public:
    /**
     * @brief Default constructor - creates a susceptible person
     * 
     * Initializes a person in the susceptible state with zero infection days.
     * This represents a healthy individual who can potentially be infected.
     */
    Person();

    /**
     * @brief Copy constructor
     */
    Person(const Person& other) = default;

    /**
     * @brief Assignment operator
     */
    Person& operator=(const Person& other) = default;

    /**
     * @brief Destructor
     */
    ~Person() = default;

    /**
     * @brief Updates the current state of a person based on infection progression
     * 
     * This method should be called once per simulation day to advance the
     * person's health state. If the person is infected, it decrements the
     * remaining infection days and transitions to recovered when the infection
     * period ends.
     * 
     * @note Only affects infected individuals; susceptible and recovered 
     *       individuals remain unchanged.
     */
    void updateState();

    /**
     * @brief Infects a susceptible person for a specified duration
     * 
     * Attempts to infect the person if they are currently susceptible.
     * Sets the infection duration and changes state to "sick". Has no effect
     * if the person is already infected or recovered.
     * 
     * @param duration Number of days the person will remain infectious (must be > 0)
     * @pre duration > 0
     * @post If successful, person state becomes "sick" and infectionDays == duration
     */
    void infect(int duration);

    // State query methods (const-correct accessors)
    
    /**
     * @brief Checks if the person has recovered and gained immunity
     * 
     * @return true if the person is in the recovered state, false otherwise
     * @note Recovered individuals are immune to reinfection in this model
     */
    bool isRecovered() const;

    /**
     * @brief Checks if the person is currently infected and infectious
     * 
     * @return true if the person is in the sick state, false otherwise
     * @note Only infected individuals can transmit the disease
     */
    bool isInfected() const;

    /**
     * @brief Checks if the person is susceptible to infection
     * 
     * @return true if the person is in the susceptible state, false otherwise
     * @note Only susceptible individuals can become infected
     */
    bool isSusceptible() const;

    /**
     * @brief Gets the current health status of the person
     * 
     * @return Current state as a string: "susceptible", "sick", or "recovered"
     * @note This method is primarily for debugging and output purposes
     */
    std::string getStatus() const;

    /**
     * @brief Gets the remaining infection days
     * 
     * @return Number of days remaining in infection period (0 if not infected)
     * @note Useful for detailed epidemiological analysis
     */
    int getRemainingInfectionDays() const { return infectionDays; }
};

#endif // PERSON_H
