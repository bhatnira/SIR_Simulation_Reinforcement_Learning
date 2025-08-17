/**
 * @file Person.cpp
 * @brief Implementation of the Person class for SIR epidemic simulation
 * @author Scientific Computing Team
 * @date 2025
 */

#include "Person.h"
#include <stdexcept>

Person::Person() {
    current = "susceptible";
    infectionDays = 0;
}

void Person::updateState() {
    if (current == "sick") {
        infectionDays--;
        if (infectionDays <= 0) {
            current = "recovered";
            infectionDays = 0;  // Ensure non-negative
        }
    }
}

void Person::infect(int duration) {
    if (duration <= 0) {
        throw std::invalid_argument("Infection duration must be positive");
    }
    
    if (current == "susceptible") {
        infectionDays = duration;
        current = "sick";
    }
    // Silently ignore attempts to infect non-susceptible individuals
}

bool Person::isRecovered() const {
    return current == "recovered";
}

bool Person::isInfected() const {
    return current == "sick";
}

bool Person::isSusceptible() const {
    return current == "susceptible";
}

std::string Person::getStatus() const {
    return current;
}
