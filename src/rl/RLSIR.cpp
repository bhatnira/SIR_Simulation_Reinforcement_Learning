/**
 * @file RLSIR.cpp
 * @brief Implementation of Reinforcement Learning framework for epidemic control
 * @author Scientific Computing Team
 * @date 2025
 */

#include "RLSIR.h"
#include "Population.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <tuple>
#include <random>

// EpidemicState implementation
std::vector<double> EpidemicState::toVector() const {
    return {
        static_cast<double>(susceptible),
        static_cast<double>(exposed),
        static_cast<double>(infected),
        static_cast<double>(recovered),
        static_cast<double>(hospitalized),
        static_cast<double>(icu),
        static_cast<double>(deaths),
        static_cast<double>(vaccinated),
        effective_reproduction_number,
        hospital_capacity_utilization,
        icu_capacity_utilization,
        economic_cost,
        social_mobility_index
    };
}

void EpidemicState::fromVector(const std::vector<double>& vec) {
    if (vec.size() != 13) {
        throw std::invalid_argument("State vector must have 13 elements");
    }
    
    susceptible = static_cast<int>(vec[0]);
    exposed = static_cast<int>(vec[1]);
    infected = static_cast<int>(vec[2]);
    recovered = static_cast<int>(vec[3]);
    hospitalized = static_cast<int>(vec[4]);
    icu = static_cast<int>(vec[5]);
    deaths = static_cast<int>(vec[6]);
    vaccinated = static_cast<int>(vec[7]);
    effective_reproduction_number = vec[8];
    hospital_capacity_utilization = vec[9];
    icu_capacity_utilization = vec[10];
    economic_cost = vec[11];
    social_mobility_index = vec[12];
}

// EpidemicRewardFunction implementation
EpidemicRewardFunction::EpidemicRewardFunction(double health_w, double economic_w, 
                                             double social_w, double consistency_w)
    : health_weight(health_w), economic_weight(economic_w), 
      social_weight(social_w), policy_consistency_weight(consistency_w) {}

double EpidemicRewardFunction::calculateReward(const EpidemicState& state, 
                                             const EpidemicAction& action,
                                             const EpidemicState& next_state) const {
    double health_reward = calculateHealthReward(state, next_state);
    double economic_reward = calculateEconomicReward(action, next_state);
    double social_reward = calculateSocialReward(action, next_state);
    
    return health_weight * health_reward + 
           economic_weight * economic_reward + 
           social_weight * social_reward;
}

double EpidemicRewardFunction::calculateHealthReward(const EpidemicState& state,
                                                   const EpidemicState& next_state) const {
    // Negative reward for increasing infection and hospitalization rates
    double infection_penalty = -100.0 * (next_state.infected_fraction - state.infected_fraction);
    double hospitalization_penalty = -50.0 * (next_state.hospitalized_fraction - state.hospitalized_fraction);
    
    // Penalty for high ICU occupancy
    double icu_penalty = -20.0 * std::max(0.0, next_state.icu_occupancy - 0.8);
    
    // Penalty for low hospital capacity
    double capacity_penalty = -30.0 * std::max(0.0, 0.2 - next_state.hospital_capacity);
    
    // Bonus for keeping infection rate low
    double low_infection_bonus = (next_state.infected_fraction < 0.05) ? 10.0 : 0.0;
    
    return infection_penalty + hospitalization_penalty + icu_penalty + 
           capacity_penalty + low_infection_bonus;
}

double EpidemicRewardFunction::calculateEconomicReward(const EpidemicAction& action,
                                                     const EpidemicState& state) const {
    // Economic cost based on intervention stringency
    double intervention_cost = 0.0;
    
    // Calculate costs for each intervention type using C++14 compatible loop
    for (const auto& intervention : action.interventions) {
        EpidemicAction::ActionType action_type = intervention.first;
        double value = intervention.second;
        
        switch (action_type) {
            case EpidemicAction::LOCKDOWN_INTENSITY:
                intervention_cost += value * 30.0;
                break;
            case EpidemicAction::SCHOOL_CLOSURE:
                intervention_cost += value * 15.0;
                break;
            case EpidemicAction::MASK_MANDATE:
                intervention_cost += value * 2.0;
                break;
            case EpidemicAction::SOCIAL_DISTANCING:
                intervention_cost += value * 10.0;
                break;
            case EpidemicAction::TESTING_EXPANSION:
                intervention_cost += value * 5.0;
                break;
            case EpidemicAction::CONTACT_TRACING:
                intervention_cost += value * 8.0;
                break;
            case EpidemicAction::VACCINATION_PRIORITY:
                intervention_cost += value * 3.0;
                break;
            case EpidemicAction::BORDER_CONTROL:
                intervention_cost += value * 20.0;
                break;
        }
    }
    
    // Economic activity bonus
    double economic_bonus = state.economic_activity * 20.0;
    
    return economic_bonus - intervention_cost;
}

double EpidemicRewardFunction::calculateSocialReward(const EpidemicAction& action,
                                                   const EpidemicState& state) const {
    // Social cost based on mobility restrictions
    double mobility_bonus = state.mobility_index * 10.0;
    
    // Penalty for policy fatigue
    double fatigue_penalty = -5.0 * state.policy_fatigue;
    
    // Penalty for severe restrictions
    double restriction_penalty = 0.0;
    auto it = action.interventions.find(EpidemicAction::LOCKDOWN_INTENSITY);
    if (it != action.interventions.end() && it->second > 0.5) {
        restriction_penalty -= 15.0;
    }
    
    it = action.interventions.find(EpidemicAction::SCHOOL_CLOSURE);
    if (it != action.interventions.end() && it->second > 0.5) {
        restriction_penalty -= 10.0;
    }
    
    it = action.interventions.find(EpidemicAction::BORDER_CONTROL);
    if (it != action.interventions.end() && it->second > 0.5) {
        restriction_penalty -= 8.0;
    }
    
    return mobility_bonus + fatigue_penalty + restriction_penalty;
}

// ExperienceReplayBuffer implementation
ExperienceReplayBuffer::ExperienceReplayBuffer(size_t capacity)
    : capacity(capacity), current_size(0), next_idx(0), rng(std::random_device{}()) {
    buffer.reserve(capacity);
}

void ExperienceReplayBuffer::add(const EpidemicState& state, const EpidemicAction& action,
                                double reward, const EpidemicState& next_state, bool done) {
    if (buffer.size() < capacity) {
        buffer.emplace_back(state, action, reward, next_state, done);
        current_size++;
    } else {
        buffer[next_idx] = Experience(state, action, reward, next_state, done);
    }
    next_idx = (next_idx + 1) % capacity;
}

std::vector<ExperienceReplayBuffer::Experience> ExperienceReplayBuffer::sample(size_t batch_size) const {
    if (batch_size > current_size) {
        throw std::invalid_argument("Batch size larger than buffer size");
    }
    
    std::vector<Experience> batch;
    batch.reserve(batch_size);
    
    std::uniform_int_distribution<size_t> dist(0, current_size - 1);
    for (size_t i = 0; i < batch_size; ++i) {
        batch.push_back(buffer[dist(rng)]);
    }
    
    return batch;
}

// NeuralNetwork implementation
NeuralNetwork::NeuralNetwork(const std::vector<int>& layers)
    : layer_sizes(layers), rng(std::random_device{}()) {
    initializeWeights();
}

void NeuralNetwork::initializeWeights() {
    weights.clear();
    biases.clear();
    
    std::normal_distribution<double> weight_dist(0.0, 0.1);
    
    for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
        std::vector<std::vector<double>> layer_weights(layer_sizes[i], 
                                                      std::vector<double>(layer_sizes[i + 1]));
        std::vector<double> layer_biases(layer_sizes[i + 1]);
        
        for (int j = 0; j < layer_sizes[i]; ++j) {
            for (int k = 0; k < layer_sizes[i + 1]; ++k) {
                layer_weights[j][k] = weight_dist(rng);
            }
        }
        
        for (int k = 0; k < layer_sizes[i + 1]; ++k) {
            layer_biases[k] = weight_dist(rng);
        }
        
        weights.push_back(layer_weights);
        biases.push_back(layer_biases);
    }
}

std::vector<double> NeuralNetwork::forward(const std::vector<double>& input) const {
    if (input.size() != static_cast<size_t>(layer_sizes[0])) {
        throw std::invalid_argument("Input size doesn't match network input layer size");
    }
    
    std::vector<double> current_layer = input;
    
    for (size_t layer = 0; layer < weights.size(); ++layer) {
        std::vector<double> next_layer(layer_sizes[layer + 1], 0.0);
        
        for (int j = 0; j < layer_sizes[layer + 1]; ++j) {
            for (int i = 0; i < layer_sizes[layer]; ++i) {
                next_layer[j] += current_layer[i] * weights[layer][i][j];
            }
            next_layer[j] += biases[layer][j];
            
            // Apply ReLU activation (except for output layer)
            if (layer < weights.size() - 1) {
                next_layer[j] = relu(next_layer[j]);
            }
        }
        
        current_layer = next_layer;
    }
    
    return current_layer;
}

void NeuralNetwork::backward(const std::vector<double>& input, 
                           const std::vector<double>& target,
                           double learning_rate) {
    // Simple gradient descent implementation
    // In practice, would use more sophisticated optimization
    
    std::vector<double> output = forward(input);
    std::vector<double> error(output.size());
    
    for (size_t i = 0; i < output.size(); ++i) {
        error[i] = target[i] - output[i];
    }
    
    // Update weights and biases (simplified backpropagation)
    for (size_t layer = weights.size(); layer > 0; --layer) {
        size_t idx = layer - 1;
        for (int j = 0; j < layer_sizes[idx + 1]; ++j) {
            biases[idx][j] += learning_rate * error[j];
            for (int i = 0; i < layer_sizes[idx]; ++i) {
                double input_val = (idx == 0) ? input[i] : 1.0; // Simplified
                weights[idx][i][j] += learning_rate * error[j] * input_val;
            }
        }
    }
}

void NeuralNetwork::saveParameters(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for saving: " + filename);
    }
    
    // Save layer sizes
    file << layer_sizes.size() << "\n";
    for (int size : layer_sizes) {
        file << size << " ";
    }
    file << "\n";
    
    // Save weights and biases
    for (size_t layer = 0; layer < weights.size(); ++layer) {
        for (int i = 0; i < layer_sizes[layer]; ++i) {
            for (int j = 0; j < layer_sizes[layer + 1]; ++j) {
                file << weights[layer][i][j] << " ";
            }
        }
        file << "\n";
        
        for (int j = 0; j < layer_sizes[layer + 1]; ++j) {
            file << biases[layer][j] << " ";
        }
        file << "\n";
    }
}

void NeuralNetwork::loadParameters(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for loading: " + filename);
    }
    
    // Load layer sizes
    size_t num_layers;
    file >> num_layers;
    layer_sizes.resize(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        file >> layer_sizes[i];
    }
    
    // Initialize weights and biases with correct sizes
    weights.resize(num_layers - 1);
    biases.resize(num_layers - 1);
    
    // Load weights and biases
    for (size_t layer = 0; layer < weights.size(); ++layer) {
        weights[layer].resize(layer_sizes[layer], std::vector<double>(layer_sizes[layer + 1]));
        biases[layer].resize(layer_sizes[layer + 1]);
        
        for (int i = 0; i < layer_sizes[layer]; ++i) {
            for (int j = 0; j < layer_sizes[layer + 1]; ++j) {
                file >> weights[layer][i][j];
            }
        }
        
        for (int j = 0; j < layer_sizes[layer + 1]; ++j) {
            file >> biases[layer][j];
        }
    }
}

// DQNAgent implementation
DQNAgent::DQNAgent(int state_size, int action_size, double learning_rate, double gamma,
                   double epsilon, double epsilon_decay, double epsilon_min)
    : epsilon(epsilon), epsilon_decay(epsilon_decay), epsilon_min(epsilon_min),
      gamma(gamma), learning_rate(learning_rate), target_update_freq(100),
      batch_size(32), training_step(0), rng(std::random_device{}()) {
    
    // Create neural networks
    std::vector<int> network_architecture = {state_size, 64, 64, action_size};
    q_network = std::make_unique<NeuralNetwork>(network_architecture);
    target_network = std::make_unique<NeuralNetwork>(network_architecture);
    
    // Create experience replay buffer
    replay_buffer = std::make_unique<ExperienceReplayBuffer>(10000);
}

EpidemicAction DQNAgent::selectAction(const EpidemicState& state) const {
    // ε-greedy action selection
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    
    if (uniform(rng) < epsilon) {
        // Random action
        std::uniform_int_distribution<int> action_dist(0, 7);
        return indexToAction(action_dist(rng));
    } else {
        // Greedy action
        std::vector<double> state_vec = stateToVector(state);
        std::vector<double> q_values = q_network->forward(state_vec);
        
        int best_action = std::distance(q_values.begin(), 
                                       std::max_element(q_values.begin(), q_values.end()));
        return indexToAction(best_action);
    }
}

void DQNAgent::train() {
    if (!replay_buffer->canSample(batch_size)) {
        return;
    }
    
    auto batch = replay_buffer->sample(batch_size);
    
    for (const auto& experience : batch) {
        std::vector<double> state_vec = stateToVector(experience.state);
        std::vector<double> next_state_vec = stateToVector(experience.next_state);
        
        std::vector<double> current_q = q_network->forward(state_vec);
        std::vector<double> next_q = target_network->forward(next_state_vec);
        
        double target_value = experience.reward;
        if (!experience.done) {
            target_value += gamma * *std::max_element(next_q.begin(), next_q.end());
        }
        
        // Update Q-value for taken action (simplified implementation)
        std::vector<double> target_q = current_q;
        // In a complete implementation, would need to identify the action index
        // and update: target_q[action_index] = target_value;
        
        q_network->backward(state_vec, target_q, learning_rate);
    }
    
    // Update target network periodically
    training_step++;
    if (training_step % target_update_freq == 0) {
        updateTargetNetwork();
    }
    
    // Decay epsilon
    if (epsilon > epsilon_min) {
        epsilon *= epsilon_decay;
    }
}

void DQNAgent::remember(const EpidemicState& state, const EpidemicAction& action,
                       double reward, const EpidemicState& next_state, bool done) {
    replay_buffer->add(state, action, reward, next_state, done);
}

void DQNAgent::updateTargetNetwork() {
    // Copy weights from main network to target network
    // Simplified implementation
    *target_network = *q_network;
}

std::vector<double> DQNAgent::stateToVector(const EpidemicState& state) const {
    return state.toFeatureVector();
}

EpidemicAction DQNAgent::indexToAction(int action_index) const {
    // Simple mapping from action index to EpidemicAction
    EpidemicAction action;
    
    switch (action_index) {
        case 0: // No intervention
            break;
        case 1: // Light social distancing
            action.interventions[EpidemicAction::SOCIAL_DISTANCING] = 0.3;
            break;
        case 2: // Moderate social distancing
            action.interventions[EpidemicAction::SOCIAL_DISTANCING] = 0.6;
            action.interventions[EpidemicAction::MASK_MANDATE] = 1.0;
            break;
        case 3: // Heavy social distancing
            action.interventions[EpidemicAction::SOCIAL_DISTANCING] = 0.9;
            action.interventions[EpidemicAction::MASK_MANDATE] = 1.0;
            action.interventions[EpidemicAction::SCHOOL_CLOSURE] = 1.0;
            break;
        case 4: // Lockdown
            action.interventions[EpidemicAction::LOCKDOWN_INTENSITY] = 1.0;
            action.interventions[EpidemicAction::MASK_MANDATE] = 1.0;
            action.interventions[EpidemicAction::SCHOOL_CLOSURE] = 1.0;
            action.interventions[EpidemicAction::BORDER_CONTROL] = 1.0;
            break;
        case 5: // Enhanced testing
            action.interventions[EpidemicAction::TESTING_EXPANSION] = 1.5;
            action.interventions[EpidemicAction::CONTACT_TRACING] = 0.7;
            break;
        case 6: // Vaccination campaign
            action.interventions[EpidemicAction::VACCINATION_PRIORITY] = 4.0;
            break;
        case 7: // Combined intervention
            action.interventions[EpidemicAction::SOCIAL_DISTANCING] = 0.5;
            action.interventions[EpidemicAction::MASK_MANDATE] = 1.0;
            action.interventions[EpidemicAction::TESTING_EXPANSION] = 1.2;
            action.interventions[EpidemicAction::VACCINATION_PRIORITY] = 3.0;
            break;
    }
    
    return action;
}

void DQNAgent::saveModel(const std::string& filepath) const {
    q_network->saveParameters(filepath);
}

void DQNAgent::loadModel(const std::string& filepath) {
    q_network->loadParameters(filepath);
    updateTargetNetwork();
}

// EpidemicAction implementation
EpidemicAction::EpidemicAction() {
    // Default constructor - no interventions
}

EpidemicAction::EpidemicAction(const std::map<ActionType, double>& actions) 
    : interventions(actions) {
    clipToValidRange();
}

void EpidemicAction::clipToValidRange() {
    for (auto& intervention : interventions) {
        switch (intervention.first) {
            case LOCKDOWN_INTENSITY:
            case SCHOOL_CLOSURE:
            case MASK_MANDATE:
            case SOCIAL_DISTANCING:
            case CONTACT_TRACING:
            case BORDER_CONTROL:
                intervention.second = std::max(0.0, std::min(1.0, intervention.second));
                break;
            case TESTING_EXPANSION:
                intervention.second = std::max(0.0, std::min(2.0, intervention.second));
                break;
            case VACCINATION_PRIORITY:
                intervention.second = std::max(0.0, std::min(4.0, intervention.second));
                break;
        }
    }
}

std::vector<double> EpidemicAction::toVector() const {
    std::vector<double> vec(8, 0.0);  // 8 action types
    
    for (const auto& intervention : interventions) {
        vec[static_cast<int>(intervention.first)] = intervention.second;
    }
    
    return vec;
}

EpidemicAction EpidemicAction::fromVector(const std::vector<double>& actionVector) {
    EpidemicAction action;
    
    for (size_t i = 0; i < actionVector.size() && i < 8; ++i) {
        if (actionVector[i] > 0.0) {
            action.interventions[static_cast<ActionType>(i)] = actionVector[i];
        }
    }
    
    action.clipToValidRange();
    return action;
}

double EpidemicAction::getEconomicCost() const {
    double cost = 0.0;
    
    for (const auto& intervention : interventions) {
        switch (intervention.first) {
            case LOCKDOWN_INTENSITY:
                cost += intervention.second * 30.0;
                break;
            case SCHOOL_CLOSURE:
                cost += intervention.second * 15.0;
                break;
            case MASK_MANDATE:
                cost += intervention.second * 2.0;
                break;
            case SOCIAL_DISTANCING:
                cost += intervention.second * 10.0;
                break;
            case TESTING_EXPANSION:
                cost += intervention.second * 5.0;
                break;
            case CONTACT_TRACING:
                cost += intervention.second * 8.0;
                break;
            case VACCINATION_PRIORITY:
                cost += intervention.second * 3.0;
                break;
            case BORDER_CONTROL:
                cost += intervention.second * 20.0;
                break;
        }
    }
    
    return cost;
}

double EpidemicAction::getSocialCost() const {
    double cost = 0.0;
    
    // Social costs are primarily from restrictive measures
    auto it = interventions.find(LOCKDOWN_INTENSITY);
    if (it != interventions.end()) {
        cost += it->second * 15.0;
    }
    
    it = interventions.find(SCHOOL_CLOSURE);
    if (it != interventions.end()) {
        cost += it->second * 10.0;
    }
    
    it = interventions.find(BORDER_CONTROL);
    if (it != interventions.end()) {
        cost += it->second * 8.0;
    }
    
    return cost;
}

// EpidemicEnvironment implementation
EpidemicEnvironment::EpidemicEnvironment(int population_size, double transmission_rate,
                                       double recovery_rate, int max_episode_length)
    : max_episode_length(max_episode_length), current_step(0), episode_done(false),
      population_size(population_size), base_transmission_rate(transmission_rate),
      recovery_rate(recovery_rate), rng(std::random_device{}()) {
    
    // Initialize population (simplified)
    population = std::make_unique<Population>(population_size);
    
    // Initialize with a basic SIR state
    current_state.susceptible_fraction = 0.99;
    current_state.infected_fraction = 0.01;
    current_state.recovered_fraction = 0.0;
    current_state.hospitalized_fraction = 0.002;
    current_state.icu_occupancy = 0.1;
    current_state.hospital_capacity = 0.8;
    current_state.testing_rate = 0.1;
    current_state.economic_activity = 1.0;
    current_state.mobility_index = 1.0;
    current_state.policy_fatigue = 0.0;
    current_state.day = 0;
    current_state.reproduction_number = 2.5;
    current_state.vaccination_coverage = 0.0;
}

EpidemicState EpidemicEnvironment::reset() {
    current_step = 0;
    episode_done = false;
    
    // Reset to initial epidemic state
    current_state.susceptible_fraction = 0.99;
    current_state.infected_fraction = 0.01;
    current_state.recovered_fraction = 0.0;
    current_state.hospitalized_fraction = 0.002;
    current_state.icu_occupancy = 0.1;
    current_state.hospital_capacity = 0.8;
    current_state.testing_rate = 0.1;
    current_state.economic_activity = 1.0;
    current_state.mobility_index = 1.0;
    current_state.policy_fatigue = 0.0;
    current_state.day = 0;
    current_state.reproduction_number = 2.5;
    current_state.vaccination_coverage = 0.0;
    
    return current_state;
}

std::tuple<EpidemicState, double, bool> EpidemicEnvironment::step(const EpidemicAction& action) {
    EpidemicState prev_state = current_state;
    
    // Simple epidemic dynamics simulation
    double effective_R = base_transmission_rate / recovery_rate;
    
    // Modify transmission based on interventions
    for (const auto& intervention : action.interventions) {
        switch (intervention.first) {
            case EpidemicAction::LOCKDOWN_INTENSITY:
                effective_R *= (1.0 - 0.8 * intervention.second);
                break;
            case EpidemicAction::SOCIAL_DISTANCING:
                effective_R *= (1.0 - 0.6 * intervention.second);
                break;
            case EpidemicAction::MASK_MANDATE:
                effective_R *= (1.0 - 0.3 * intervention.second);
                break;
            case EpidemicAction::SCHOOL_CLOSURE:
                effective_R *= (1.0 - 0.4 * intervention.second);
                break;
            case EpidemicAction::BORDER_CONTROL:
                effective_R *= (1.0 - 0.2 * intervention.second);
                break;
            default:
                break;
        }
    }
    
    current_state.reproduction_number = effective_R;
    
    // Update epidemic state (simplified SIR dynamics)
    double dt = 1.0;  // Daily time step
    double new_infections = effective_R * current_state.infected_fraction * current_state.susceptible_fraction * dt;
    double new_recoveries = recovery_rate * current_state.infected_fraction * dt;
    
    // Bounds checking
    new_infections = std::min(new_infections, current_state.susceptible_fraction);
    new_recoveries = std::min(new_recoveries, current_state.infected_fraction);
    
    current_state.susceptible_fraction -= new_infections;
    current_state.infected_fraction += new_infections - new_recoveries;
    current_state.recovered_fraction += new_recoveries;
    
    // Update other state variables
    current_state.hospitalized_fraction = current_state.infected_fraction * 0.2;
    current_state.icu_occupancy = current_state.hospitalized_fraction * 0.3;
    current_state.day++;
    current_step++;
    
    // Update economic and social metrics based on interventions
    current_state.economic_activity = 1.0 - action.getEconomicCost() / 100.0;
    current_state.economic_activity = std::max(0.1, std::min(1.0, current_state.economic_activity));
    
    current_state.mobility_index = 1.0 - action.getSocialCost() / 50.0;
    current_state.mobility_index = std::max(0.1, std::min(1.0, current_state.mobility_index));
    
    current_state.policy_fatigue = std::min(1.0, current_state.policy_fatigue + 0.01);
    
    // Check termination conditions
    episode_done = (current_step >= max_episode_length) || 
                   (current_state.infected_fraction < 0.001) ||
                   (current_state.icu_occupancy > 1.0);
    
    // Calculate reward
    double reward = 0.0;
    if (reward_function) {
        reward = reward_function->calculateReward(prev_state, action, current_state);
    }
    
    return std::make_tuple(current_state, reward, episode_done);
}

void EpidemicEnvironment::setRewardFunction(std::unique_ptr<EpidemicRewardFunction> reward_func) {
    reward_function = std::move(reward_func);
}

void EpidemicEnvironment::render() const {
    std::cout << "Day " << current_state.day 
              << " | S: " << std::fixed << std::setprecision(3) << current_state.susceptible_fraction
              << " | I: " << current_state.infected_fraction 
              << " | R: " << current_state.recovered_fraction
              << " | R_t: " << current_state.reproduction_number
              << std::endl;
}
