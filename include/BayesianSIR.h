/**
 * @file BayesianSIR.h
 * @brief Reinforcement Learning framework for epidemic control policy optimization
 * @author Scientific Computing Team
 * @date 2025
 * 
 * This file implements Reinforcement Learning (RL) for optimizing epidemic control
 * policies, including Deep Q-Networks (DQN), Policy Gradient Methods (PPO), and
 * Actor-Critic algorithms for adaptive non-pharmaceutical interventions (NPIs).
 * 
 * The framework combines Bayesian uncertainty quantification with RL-based policy
 * optimization to balance multiple objectives: minimizing infections, reducing
 * economic impact, and preserving individual freedoms.
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
 * 
 * Represents the current state of the epidemic for RL decision-making,
 * including compartmental model states, healthcare capacity, and policy metrics.
 */
struct EpidemicState {
    // Compartmental model states
    double susceptible_fraction;    ///< S/N - Fraction of susceptible population
    double infected_fraction;       ///< I/N - Fraction of infected population  
    double recovered_fraction;      ///< R/N - Fraction of recovered population
    double hospitalized_fraction;   ///< H/N - Fraction requiring hospitalization
    
    // Healthcare system metrics
    double icu_occupancy;          ///< ICU bed occupancy (0-1)
    double hospital_capacity;      ///< Available hospital capacity (0-1)
    double testing_rate;           ///< Current testing rate per capita
    
    // Economic and social metrics
    double economic_activity;      ///< Current economic activity level (0-1)
    double mobility_index;         ///< Population mobility index (0-1)
    double policy_fatigue;         ///< Public compliance fatigue (0-1)
    
    // Temporal information
    int day;                       ///< Current simulation day
    double reproduction_number;    ///< Effective reproduction number R_t
    double vaccination_coverage;   ///< Fraction of population vaccinated
    
    /**
     * @brief Convert state to feature vector for neural networks
     */
    std::vector<double> toFeatureVector() const;
    
    /**
     * @brief Check if state represents epidemic emergency
     */
    bool isEmergencyState() const;
};

/**
 * @brief Action space for epidemic control interventions
 * 
 * Defines the possible non-pharmaceutical interventions (NPIs) that
 * can be applied by the RL agent to control epidemic spread.
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
     * @brief Convert action to vector representation
     */
    std::vector<double> toVector() const;
    
    /**
     * @brief Create action from vector representation
     */
    static EpidemicAction fromVector(const std::vector<double>& actionVector);
    
    /**
     * @brief Calculate economic cost of interventions
     */
    double getEconomicCost() const;
    
    /**
     * @brief Calculate social cost of interventions
     */
    double getSocialCost() const;
};

/**
 * @brief Reward function for epidemic control RL
 * 
 * Multi-objective reward function balancing health outcomes,
 * economic impact, and social considerations.
 */
class EpidemicRewardFunction {
private:
    // Reward weights for different objectives
    double health_weight;          ///< Weight for health outcomes
    double economic_weight;        ///< Weight for economic considerations
    double social_weight;          ///< Weight for social/freedom aspects
    double policy_consistency_weight; ///< Weight for policy stability
    
public:
    /**
     * @brief Constructor with customizable weights
     */
    EpidemicRewardFunction(double health_w = 1.0, double economic_w = 0.5, 
                          double social_w = 0.3, double consistency_w = 0.2);
    
    /**
     * @brief Calculate reward for state-action-next_state transition
     */
    double calculateReward(const EpidemicState& state, 
                          const EpidemicAction& action,
                          const EpidemicState& next_state) const;
    
    /**
     * @brief Calculate health component of reward
     */
    double calculateHealthReward(const EpidemicState& state, 
                                const EpidemicState& next_state) const;
    
    /**
     * @brief Calculate economic component of reward
     */
    double calculateEconomicReward(const EpidemicAction& action,
                                  const EpidemicState& state) const;
    
    /**
     * @brief Calculate social component of reward
     */
    double calculateSocialReward(const EpidemicAction& action,
                                const EpidemicState& state) const;
    
    /**
     * @brief Set reward weights for multi-objective optimization
     */
    void setWeights(double health_w, double economic_w, double social_w, double consistency_w);
};

/**
 * @brief Experience replay buffer for deep RL training
 * 
 * Stores state-action-reward-next_state transitions for off-policy learning
 * algorithms like DQN and DDPG.
 */
class ExperienceReplayBuffer {
private:
    struct Experience {
        EpidemicState state;
        EpidemicAction action;
        double reward;
        EpidemicState next_state;
        bool done;
        
        Experience(const EpidemicState& s, const EpidemicAction& a, double r,
                  const EpidemicState& ns, bool d)
            : state(s), action(a), reward(r), next_state(ns), done(d) {}
    };
    
    std::vector<Experience> buffer;
    size_t capacity;
    size_t current_size;
    size_t next_idx;
    mutable std::mt19937 rng;
    
public:
    /**
     * @brief Constructor with buffer capacity
     */
    ExperienceReplayBuffer(size_t capacity = 10000);
    
    /**
     * @brief Add experience to buffer
     */
    void add(const EpidemicState& state, const EpidemicAction& action, 
             double reward, const EpidemicState& next_state, bool done);
    
    /**
     * @brief Sample batch of experiences for training
     */
    std::vector<Experience> sample(size_t batch_size) const;
    
    /**
     * @brief Get current buffer size
     */
    size_t size() const { return current_size; }
    
    /**
     * @brief Check if buffer has enough samples for training
     */
    bool canSample(size_t batch_size) const { return current_size >= batch_size; }
};

/**
 * @brief Neural network for RL policy and value function approximation
 * 
 * Simple multi-layer perceptron for epidemic control RL.
 * In practice, would integrate with TensorFlow/PyTorch.
 */
class NeuralNetwork {
private:
    std::vector<std::vector<std::vector<double>>> weights;  ///< Layer weights
    std::vector<std::vector<double>> biases;                ///< Layer biases
    std::vector<int> layer_sizes;                          ///< Neurons per layer
    mutable std::mt19937 rng;
    
    /**
     * @brief Activation function (ReLU)
     */
    double relu(double x) const { return std::max(0.0, x); }
    
    /**
     * @brief Derivative of ReLU
     */
    double relu_derivative(double x) const { return x > 0 ? 1.0 : 0.0; }
    
public:
    /**
     * @brief Constructor with layer architecture
     */
    NeuralNetwork(const std::vector<int>& layers);
    
    /**
     * @brief Forward pass through network
     */
    std::vector<double> forward(const std::vector<double>& input) const;
    
    /**
     * @brief Backward pass for gradient computation
     */
    void backward(const std::vector<double>& input, 
                  const std::vector<double>& target,
                  double learning_rate);
    
    /**
     * @brief Initialize weights randomly
     */
    void initializeWeights();
    
    /**
     * @brief Save network parameters
     */
    void saveParameters(const std::string& filename) const;
    
    /**
     * @brief Load network parameters
     */
    void loadParameters(const std::string& filename);
};

/**
 * @brief Observed epidemic data for parameter estimation
 * 
 * Contains time series observations of the epidemic state
 * for comparison with model predictions.
 */
struct EpidemicData {
    std::vector<int> days;           ///< Observation days
    std::vector<int> susceptible;    ///< Number of susceptible individuals
    std::vector<int> infected;       ///< Number of infected individuals
    std::vector<int> recovered;      ///< Number of recovered individuals
    std::vector<double> observationError; ///< Measurement error variance
    
    int totalPopulation;             ///< Total population size
    
    /**
     * @brief Add an observation to the dataset
     */
    void addObservation(int day, int S, int I, int R, double error = 1.0) {
        days.push_back(day);
        susceptible.push_back(S);
        infected.push_back(I);
        recovered.push_back(R);
        observationError.push_back(error);
    }
    
    /**
     * @brief Get the number of observations
     */
    size_t size() const { return days.size(); }
};

/**
 * @brief Deep Q-Network (DQN) agent for epidemic control
 * 
 * Implements DQN with experience replay and target networks for
 * learning optimal epidemic intervention policies.
 */
class DQNAgent {
private:
    std::unique_ptr<NeuralNetwork> q_network;      ///< Main Q-network
    std::unique_ptr<NeuralNetwork> target_network; ///< Target Q-network for stable training
    std::unique_ptr<ExperienceReplayBuffer> replay_buffer;
    
    double epsilon;              ///< Exploration rate for ε-greedy policy
    double epsilon_decay;        ///< Decay rate for exploration
    double epsilon_min;          ///< Minimum exploration rate
    double gamma;               ///< Discount factor for future rewards
    double learning_rate;       ///< Learning rate for neural network
    int target_update_freq;     ///< Frequency to update target network
    int batch_size;            ///< Batch size for training
    int training_step;         ///< Current training step counter
    
    mutable std::mt19937 rng;
    
public:
    /**
     * @brief Constructor for DQN agent
     */
    DQNAgent(int state_size = 13, int action_size = 8, 
             double learning_rate = 0.001, double gamma = 0.99, 
             double epsilon = 1.0, double epsilon_decay = 0.995, 
             double epsilon_min = 0.01);
    
    /**
     * @brief Select action using ε-greedy policy
     */
    EpidemicAction selectAction(const EpidemicState& state) const;
    
    /**
     * @brief Train the agent on a batch of experiences
     */
    void train();
    
    /**
     * @brief Store experience in replay buffer
     */
    void remember(const EpidemicState& state, const EpidemicAction& action,
                  double reward, const EpidemicState& next_state, bool done);
    
    /**
     * @brief Update target network weights
     */
    void updateTargetNetwork();
    
    /**
     * @brief Convert state to vector for neural network input
     */
    std::vector<double> stateToVector(const EpidemicState& state) const;
    
    /**
     * @brief Convert action index to EpidemicAction
     */
    EpidemicAction indexToAction(int action_index) const;
    
    /**
     * @brief Save trained model
     */
    void saveModel(const std::string& filepath) const;
    
    /**
     * @brief Load trained model
     */
    void loadModel(const std::string& filepath);
};

/**
 * @brief Proximal Policy Optimization (PPO) agent for epidemic control
 * 
 * Implements PPO algorithm with actor-critic architecture for
 * continuous and discrete action spaces in epidemic control.
 */
class PPOAgent {
private:
    std::unique_ptr<NeuralNetwork> actor_network;   ///< Policy network
    std::unique_ptr<NeuralNetwork> critic_network;  ///< Value network
    
    double learning_rate;
    double gamma;              ///< Discount factor
    double lambda;             ///< GAE parameter
    double clip_epsilon;       ///< PPO clipping parameter
    double entropy_coeff;      ///< Entropy regularization coefficient
    int update_epochs;         ///< Number of epochs per update
    int batch_size;
    
    mutable std::mt19937 rng;
    
public:
    /**
     * @brief Constructor for PPO agent
     */
    PPOAgent(int state_size = 13, int action_size = 8,
             double learning_rate = 0.0003, double gamma = 0.99,
             double lambda = 0.95, double clip_epsilon = 0.2);
    
    /**
     * @brief Select action from policy
     */
    EpidemicAction selectAction(const EpidemicState& state) const;
    
    /**
     * @brief Update policy using collected trajectories
     */
    void update(const std::vector<EpidemicState>& states,
                const std::vector<EpidemicAction>& actions,
                const std::vector<double>& rewards,
                const std::vector<bool>& dones);
    
    /**
     * @brief Calculate value estimate for state
     */
    double getValue(const EpidemicState& state) const;
    
    /**
     * @brief Calculate advantages using GAE
     */
    std::vector<double> calculateAdvantages(
        const std::vector<double>& rewards,
        const std::vector<double>& values,
        const std::vector<bool>& dones) const;
};

/**
 * @brief RL training environment for epidemic control
 * 
 * Manages the MDP environment for training RL agents on epidemic
 * control policies with configurable objectives and constraints.
 */
class EpidemicEnvironment {
private:
    std::unique_ptr<Population> population;
    std::unique_ptr<EpidemicRewardFunction> reward_function;
    EpidemicState current_state;
    
    int max_episode_length;
    int current_step;
    bool episode_done;
    
    // Environment parameters
    int population_size;
    double base_transmission_rate;
    double recovery_rate;
    
    mutable std::mt19937 rng;
    
public:
    /**
     * @brief Constructor for epidemic environment
     */
    EpidemicEnvironment(int population_size = 1000,
                       double transmission_rate = 0.3,
                       double recovery_rate = 0.1,
                       int max_episode_length = 365);
    
    /**
     * @brief Reset environment to initial state
     */
    EpidemicState reset();
    
    /**
     * @brief Execute action and return next state, reward, done
     */
    std::tuple<EpidemicState, double, bool> step(const EpidemicAction& action);
    
    /**
     * @brief Get current state
     */
    const EpidemicState& getCurrentState() const { return current_state; }
    
    /**
     * @brief Check if episode is complete
     */
    bool isDone() const { return episode_done; }
    
    /**
     * @brief Set reward function
     */
    void setRewardFunction(std::unique_ptr<EpidemicRewardFunction> reward_func);
    
    /**
     * @brief Render current state for visualization
     */
    void render() const;
};

/**
 * @brief RL training utilities and analysis
 * 
 * Tools for training RL agents and analyzing learning performance.
 */
class RLTraining {
public:
    /**
     * @brief Training configuration
     */
    struct TrainingConfig {
        int total_episodes;
        int max_episode_length;
        double learning_rate;
        int batch_size;
        int update_frequency;
        std::string save_path;
    };
    
    /**
     * @brief Train DQN agent on epidemic environment
     */
    static void trainDQN(DQNAgent& agent, EpidemicEnvironment& env,
                        const TrainingConfig& config);
    
    /**
     * @brief Train PPO agent on epidemic environment
     */
    static void trainPPO(PPOAgent& agent, EpidemicEnvironment& env,
                        const TrainingConfig& config);
    
    /**
     * @brief Evaluate trained agent performance
     */
    static double evaluateAgent(const DQNAgent& agent, EpidemicEnvironment& env,
                               int num_episodes = 10);
    
    /**
     * @brief Calculate learning curve metrics
     */
    static std::vector<double> calculateReturns(const std::vector<double>& rewards);
    
    /**
     * @brief Save training metrics
     */
    static void saveTrainingMetrics(const std::vector<double>& episode_rewards,
                                   const std::string& filepath);
};

#endif // BAYESIAN_SIR_H
