/**
 * @file RLSIR.h
 * @brief Reinforcement Learning framework for epidemic control policy optimization
 * @author Scientific Computing Team
 * @date 2025
 * 
 * This file implements Reinforcement Learning (RL) for optimizing epidemic control
 * policies, including Deep Q-Networks (DQN) for adaptive non-pharmaceutical interventions (NPIs).
 */

#ifndef RL_EPIDEMIC_CONTROL_H
#define RL_EPIDEMIC_CONTROL_H

#include "Simulation.h"
#include <vector>
#include <random>
#include <functional>
#include <memory>
#include <map>
#include <array>

/**
 * @brief Epidemic state representation for RL environment
 */
struct EpidemicState {
    int susceptible;           ///< Number of susceptible individuals
    int exposed;              ///< Number of exposed individuals
    int infected;             ///< Number of infected individuals
    int recovered;            ///< Number of recovered individuals
    int hospitalized;         ///< Number of hospitalized individuals
    int icu;                  ///< Number of ICU patients
    int deaths;               ///< Number of deaths
    int vaccinated;           ///< Number of vaccinated individuals
    
    double effective_reproduction_number;    ///< Effective reproduction number R_t
    double hospital_capacity_utilization;   ///< Hospital capacity utilization (0-1)
    double icu_capacity_utilization;        ///< ICU capacity utilization (0-1)
    double economic_cost;                   ///< Economic cost of interventions
    double social_mobility_index;           ///< Population mobility index (0-1)
    
    /**
     * @brief Convert state to vector for neural networks
     */
    std::vector<double> toVector() const;
    
    /**
     * @brief Restore state from vector
     */
    void fromVector(const std::vector<double>& vec);
};

/**
 * @brief Action space for epidemic control interventions
 */
struct EpidemicAction {
    enum ActionType {
        LOCKDOWN_INTENSITY,     ///< Lockdown strictness (0-1)
        SCHOOL_CLOSURE,         ///< School closure level (0-1)
        MASK_MANDATE,           ///< Mask wearing requirement (0-1)
        SOCIAL_DISTANCING,      ///< Social distancing measures (0-1)
        TESTING_EXPANSION,      ///< Testing capacity scaling (0-2)
        CONTACT_TRACING,        ///< Contact tracing intensity (0-1)
        VACCINATION_PRIORITY,   ///< Vaccination prioritization strategy (0-4)
        BORDER_CONTROL         ///< Travel restrictions (0-1)
    };
    
    std::map<ActionType, double> interventions;  ///< Intervention levels
    
    /**
     * @brief Default constructor with no interventions
     */
    EpidemicAction();
    
    /**
     * @brief Constructor with specific intervention levels
     */
    EpidemicAction(const std::map<ActionType, double>& actions);
    
    /**
     * @brief Clip actions to valid ranges
     */
    void clipToValidRange();
    
    /**
     * @brief Convert action to vector for neural networks
     */
    std::vector<double> toVector() const;
    
    /**
     * @brief Apply actions to modify transmission rates
     */
    double getTransmissionModifier() const;
    
    /**
     * @brief Calculate economic cost of actions
     */
    double getEconomicCost() const;
};

/**
 * @brief Experience tuple for DQN replay buffer
 */
struct Experience {
    std::vector<double> state;      ///< Current state
    std::vector<double> action;     ///< Action taken
    double reward;                  ///< Reward received
    std::vector<double> next_state; ///< Next state
    bool done;                      ///< Episode termination flag
};

/**
 * @brief Deep Q-Network agent for epidemic control
 */
class DQNAgent {
private:
    static constexpr size_t STATE_SIZE = 13;
    static constexpr size_t ACTION_SIZE = 8;
    static constexpr size_t HIDDEN_SIZE = 256;
    static constexpr size_t MEMORY_SIZE = 10000;
    static constexpr size_t BATCH_SIZE = 32;
    
    std::vector<std::vector<double>> q_network_;      ///< Main Q-network weights
    std::vector<std::vector<double>> target_network_; ///< Target network weights
    std::vector<Experience> memory_;                  ///< Experience replay buffer
    std::mt19937 rng_;                               ///< Random number generator
    
    double epsilon_;                                 ///< Exploration rate
    double learning_rate_;                          ///< Learning rate
    double gamma_;                                  ///< Discount factor
    int target_update_frequency_;                   ///< Target network update frequency
    int update_counter_;                           ///< Update counter
    
public:
    /**
     * @brief Constructor
     */
    DQNAgent(double learning_rate = 0.001, double gamma = 0.99, 
             double epsilon = 1.0, int target_update_freq = 100);
    
    /**
     * @brief Select action using epsilon-greedy policy
     */
    EpidemicAction selectAction(const EpidemicState& state);
    
    /**
     * @brief Store experience in replay buffer
     */
    void storeExperience(const EpidemicState& state, const EpidemicAction& action,
                        double reward, const EpidemicState& next_state, bool done);
    
    /**
     * @brief Train the network on a batch of experiences
     */
    void train();
    
    /**
     * @brief Update target network
     */
    void updateTargetNetwork();
    
    /**
     * @brief Decay exploration rate
     */
    void decayEpsilon(double decay_rate = 0.995);
    
    /**
     * @brief Save model to file
     */
    void saveModel(const std::string& filename) const;
    
    /**
     * @brief Load model from file
     */
    void loadModel(const std::string& filename);
    
private:
    /**
     * @brief Forward pass through network
     */
    std::vector<double> forward(const std::vector<double>& input, 
                               const std::vector<std::vector<double>>& network) const;
    
    /**
     * @brief Sample random batch from memory
     */
    std::vector<Experience> sampleBatch() const;
};

/**
 * @brief Epidemic simulation environment for RL
 */
class EpidemicEnvironment {
private:
    std::unique_ptr<Population> population_;    ///< Population simulation
    EpidemicState current_state_;              ///< Current epidemic state
    int day_;                                  ///< Current simulation day
    int max_days_;                            ///< Maximum simulation days
    std::mt19937 rng_;                        ///< Random number generator
    
    // Reward function weights
    double health_weight_;                     ///< Weight for health outcomes
    double economic_weight_;                   ///< Weight for economic outcomes
    double social_weight_;                     ///< Weight for social outcomes
    
public:
    /**
     * @brief Constructor
     */
    EpidemicEnvironment(int population_size = 100000, int max_days = 365);
    
    /**
     * @brief Reset environment to initial state
     */
    EpidemicState reset();
    
    /**
     * @brief Take action and return new state, reward, and done flag
     */
    std::tuple<EpidemicState, double, bool> step(const EpidemicAction& action);
    
    /**
     * @brief Get current state
     */
    const EpidemicState& getCurrentState() const { return current_state_; }
    
    /**
     * @brief Check if episode is done
     */
    bool isDone() const;
    
    /**
     * @brief Set reward function weights
     */
    void setRewardWeights(double health_weight, double economic_weight, double social_weight);
    
private:
    /**
     * @brief Update epidemic state from population
     */
    void updateState();
    
    /**
     * @brief Calculate reward for current state and action
     */
    double calculateReward(const EpidemicAction& action) const;
    
    /**
     * @brief Apply action effects to population
     */
    void applyAction(const EpidemicAction& action);
};

/**
 * @brief Utility functions for RL framework
 */
namespace RLUtils {
    /**
     * @brief Normalize vector to [0, 1] range
     */
    std::vector<double> normalize(const std::vector<double>& vec, 
                                 double min_val = 0.0, double max_val = 1.0);
    
    /**
     * @brief Calculate moving average
     */
    double movingAverage(const std::vector<double>& values, size_t window_size);
    
    /**
     * @brief Save training statistics to CSV
     */
    void saveTrainingStats(const std::vector<double>& rewards,
                          const std::vector<double>& losses,
                          const std::string& filename);
}

#endif // RL_EPIDEMIC_CONTROL_H
