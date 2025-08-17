#include "Population.h"
#include <random>
#include <algorithm>
#include <iostream>

Population::Population(int populationSize) 
    : size(populationSize), day(0), countInfected(0), 
      countSusceptible(populationSize), countRecovered(0),
      infectionProbability(0.0), contactsPerDay(0), infectionDuration(0) {
    
    // Initialize population with susceptible individuals
    population.reserve(populationSize);
    for (int i = 0; i < populationSize; i++) {
        population.push_back(std::make_unique<Person>());
    }
}

void Population::infectRandomPerson() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, size - 1);
    
    int index = dis(gen);
    Person* chosenPerson = population[index].get();
    infectPerson(chosenPerson);
}

void Population::simulateOneDay() {
    std::vector<Person*> newlyInfected;
    
    // Process each person in the population
    for (auto& person : population) {
        if (person->isInfected()) {
            // Infected people can transmit disease
            simulateTransmission(newlyInfected);
        }
        // Update each person's state (progression of disease)
        person->updateState();
    }
    
    // Infect newly infected people
    for (Person* person : newlyInfected) {
        infectPerson(person);
    }
    
    // Update day counter and population counts
    day++;
    updateCounts();
}

void Population::setInfectionProbability(float probability) {
    this->infectionProbability = probability;
}

void Population::setContactsPerDay(int contacts) {
    this->contactsPerDay = contacts;
}

void Population::setInfectionDuration(int days) {
    this->infectionDuration = days;
}

void Population::infectPerson(Person* person) {
    person->infect(infectionDuration);
}

void Population::simulateTransmission(std::vector<Person*>& newlyInfected) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> personDis(0, population.size() - 1);
    std::uniform_real_distribution<float> probDis(0.0, 1.0);
    
    // Each infected person makes 'contactsPerDay' contacts
    for (int i = 0; i < contactsPerDay && i < static_cast<int>(population.size()) - 1; ++i) {
        // Random contact
        int contactIndex = personDis(gen);
        Person* contactPerson = population[contactIndex].get();
        
        // Check if contact is susceptible and transmission occurs
        if (contactPerson->isSusceptible() && probDis(gen) <= infectionProbability) {
            newlyInfected.push_back(contactPerson);
        }
    }
}

void Population::updateCounts() {
    countInfected = 0;
    countSusceptible = 0;
    countRecovered = 0;
    
    for (const auto& person : population) {
        if (person->isInfected()) {
            countInfected++;
        } else if (person->isSusceptible()) {
            countSusceptible++;
        } else if (person->isRecovered()) {
            countRecovered++;
        }
    }
}
