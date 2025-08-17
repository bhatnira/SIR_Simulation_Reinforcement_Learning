/**
 * @file SimpleRLExperiments.cpp
 * @brief Simple RL experiments for epidemic control
 * @author Scientific Computing Team
 * @date 2025
 */

#include "Population.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <map>
#include <limits>
#include <memory>
#include <tuple>

// Simple epidemic state
struct SimpleState {
    double infected_fraction;
    double hospitalized_fraction;
    double economic_activity;
    int day;
    
    std::vector<double> toVector() const {
        return {infected_fraction, hospitalized_fraction, economic_activity, static_cast<double>(day)};
    }
};

// Simple action (just lockdown intensity)
struct SimpleAction {
    double lockdown_intensity; // 0 = no lockdown, 1 = full lockdown
    
    SimpleAction(double intensity = 0.0) : lockdown_intensity(intensity) {}
};

// Simple Q-learning agent
class SimpleQLearning {
private:
    std::map<std::pair<int, int>, std::map<int, double>> q_table_;
    double learning_rate_;
    double gamma_;
    double epsilon_;
    std::mt19937 rng_;
    
public:
    SimpleQLearning(double lr = 0.1, double gamma = 0.9, double eps = 0.1)
        : learning_rate_(lr), gamma_(gamma), epsilon_(eps), rng_(std::random_device{}()) {}
    
    SimpleAction selectAction(const SimpleState& state) {
        auto state_key = std::make_pair(
            static_cast<int>(state.infected_fraction * 100),
            static_cast<int>(state.hospitalized_fraction * 100)
        );
        
        // Epsilon-greedy policy
        std::uniform_real_distribution<> dis(0.0, 1.0);
        if (dis(rng_) < epsilon_) {
            // Random action
            std::uniform_real_distribution<> action_dis(0.0, 1.0);
            return SimpleAction(action_dis(rng_));
        }
        
        // Greedy action
        double best_value = -std::numeric_limits<double>::infinity();
        int best_action = 0;
        for (int a = 0; a <= 10; ++a) {
            double q_value = q_table_[state_key][a];
            if (q_value > best_value) {
                best_value = q_value;
                best_action = a;
            }
        }
        
        return SimpleAction(best_action / 10.0);
    }
    
    void update(const SimpleState& state, const SimpleAction& action, 
                double reward, const SimpleState& next_state) {
        auto state_key = std::make_pair(
            static_cast<int>(state.infected_fraction * 100),
            static_cast<int>(state.hospitalized_fraction * 100)
        );
        auto next_state_key = std::make_pair(
            static_cast<int>(next_state.infected_fraction * 100),
            static_cast<int>(next_state.hospitalized_fraction * 100)
        );
        
        int action_idx = static_cast<int>(action.lockdown_intensity * 10);
        
        // Find max Q-value for next state
        double max_next_q = -std::numeric_limits<double>::infinity();
        for (int a = 0; a <= 10; ++a) {
            max_next_q = std::max(max_next_q, q_table_[next_state_key][a]);
        }
        
        // Q-learning update
        double current_q = q_table_[state_key][action_idx];
        double target = reward + gamma_ * max_next_q;
        q_table_[state_key][action_idx] = current_q + learning_rate_ * (target - current_q);
    }
    
    void decayEpsilon(double decay = 0.995) {
        epsilon_ *= decay;
        epsilon_ = std::max(epsilon_, 0.01);
    }
};

// Simple environment
class SimpleEnvironment {
private:
    std::unique_ptr<Population> pop_;
    int day_;
    int max_days_;
    std::mt19937 rng_;
    
public:
    SimpleEnvironment(int pop_size = 10000, int max_days = 100) 
        : day_(0), max_days_(max_days), rng_(std::random_device{}()) {
        pop_ = std::make_unique<Population>(pop_size);
        // Start with some infections
        for (int i = 0; i < pop_size / 100; ++i) {
            pop_->infectRandomPerson();
        }
    }
    
    SimpleState reset() {
        day_ = 0;
        pop_ = std::make_unique<Population>(pop_->getPopulationSize());
        // Start with some infections
        for (int i = 0; i < pop_->getPopulationSize() / 100; ++i) {
            pop_->infectRandomPerson();
        }
        return getCurrentState();
    }
    
    std::tuple<SimpleState, double, bool> step(const SimpleAction& action) {
        // Apply lockdown effect (reduce transmission)
        double transmission_reduction = action.lockdown_intensity * 0.7;
        
        // Simulate one day with reduced transmission
        pop_->simulateDay(transmission_reduction);
        day_++;
        
        SimpleState new_state = getCurrentState();
        double reward = calculateReward(new_state, action);
        bool done = (day_ >= max_days_) || (new_state.infected_fraction < 0.001);
        
        return std::make_tuple(new_state, reward, done);
    }
    
    SimpleState getCurrentState() const {
        SimpleState state;
        int infected = 0, hospitalized = 0;
        
        for (const auto& person : pop_->getPeople()) {
            if (person->isInfected()) infected++;
            if (person->isHospitalized()) hospitalized++;
        }
        
        state.infected_fraction = static_cast<double>(infected) / pop_->getSize();
        state.hospitalized_fraction = static_cast<double>(hospitalized) / pop_->getSize();
        state.economic_activity = 1.0; // Simplified
        state.day = day_;
        
        return state;
    }
    
private:
    double calculateReward(const SimpleState& state, const SimpleAction& action) const {
        // Multi-objective reward: minimize infections and economic cost
        double health_penalty = state.infected_fraction + 10.0 * state.hospitalized_fraction;
        double economic_penalty = action.lockdown_intensity * action.lockdown_intensity;
        
        return -(health_penalty + 0.1 * economic_penalty);
    }
};

// Experiment functions
void experiment1_learning_curve() {
    std::cout << "Experiment 1: Learning Curve Analysis\\n";
    std::cout << "=====================================\\n";
    
    SimpleQLearning agent(0.1, 0.9, 0.5);
    SimpleEnvironment env(5000, 50);
    
    std::vector<double> episode_rewards;
    std::vector<double> episode_infections;
    
    for (int episode = 0; episode < 100; ++episode) {
        SimpleState state = env.reset();
        double total_reward = 0;
        double total_infections = 0;
        
        for (int step = 0; step < 50; ++step) {
            SimpleAction action = agent.selectAction(state);
            auto result = env.step(action);
            SimpleState next_state = std::get<0>(result);
            double reward = std::get<1>(result);
            bool done = std::get<2>(result);
            
            agent.update(state, action, reward, next_state);
            
            total_reward += reward;
            total_infections += next_state.infected_fraction;
            state = next_state;
            
            if (done) break;
        }
        
        agent.decayEpsilon();
        episode_rewards.push_back(total_reward);
        episode_infections.push_back(total_infections);
        
        if (episode % 10 == 0) {
            std::cout << "Episode " << episode << ": Reward = " << std::fixed << std::setprecision(2) 
                     << total_reward << ", Avg Infection = " << total_infections/50 << "\\n";
        }
    }
    
    // Save results
    std::ofstream file("experiment1_learning_curve.csv");
    file << "Episode,Reward,AvgInfection\\n";
    for (size_t i = 0; i < episode_rewards.size(); ++i) {
        file << i << "," << episode_rewards[i] << "," << episode_infections[i]/50 << "\\n";
    }
    file.close();
    std::cout << "Results saved to experiment1_learning_curve.csv\\n\\n";
}

void experiment2_hyperparameter_sensitivity() {
    std::cout << "Experiment 2: Hyperparameter Sensitivity\\n";
    std::cout << "========================================\\n";
    
    std::vector<double> learning_rates = {0.01, 0.1, 0.5};
    std::vector<double> gamma_values = {0.8, 0.9, 0.95};
    
    std::ofstream file("experiment2_hyperparameters.csv");
    file << "LearningRate,Gamma,FinalReward,FinalInfection\\n";
    
    for (double lr : learning_rates) {
        for (double gamma : gamma_values) {
            std::cout << "Testing LR=" << lr << ", Gamma=" << gamma << "\\n";
            
            SimpleQLearning agent(lr, gamma, 0.3);
            SimpleEnvironment env(3000, 30);
            
            double final_reward = 0;
            double final_infection = 0;
            
            // Train for several episodes
            for (int episode = 0; episode < 50; ++episode) {
                SimpleState state = env.reset();
                double total_reward = 0;
                
                for (int step = 0; step < 30; ++step) {
                    SimpleAction action = agent.selectAction(state);
                    auto result = env.step(action);
                    SimpleState next_state = std::get<0>(result);
                    double reward = std::get<1>(result);
                    bool done = std::get<2>(result);
                    
                    agent.update(state, action, reward, next_state);
                    total_reward += reward;
                    state = next_state;
                    
                    if (done) break;
                }
                
                agent.decayEpsilon();
                
                if (episode == 49) {
                    final_reward = total_reward;
                    final_infection = state.infected_fraction;
                }
            }
            
            file << lr << "," << gamma << "," << final_reward << "," << final_infection << "\\n";
        }
    }
    
    file.close();
    std::cout << "Results saved to experiment2_hyperparameters.csv\\n\\n";
}

void experiment3_policy_comparison() {
    std::cout << "Experiment 3: Policy Comparison\\n";
    std::cout << "===============================\\n";
    
    SimpleEnvironment env(8000, 60);
    
    std::ofstream file("experiment3_policies.csv");
    file << "Policy,TotalReward,TotalInfections,TotalEconomicCost\\n";
    
    // Policy 1: No intervention
    {
        SimpleState state = env.reset();
        double total_reward = 0, total_infections = 0, total_economic_cost = 0;
        
        for (int step = 0; step < 60; ++step) {
            SimpleAction action(0.0); // No lockdown
            auto result = env.step(action);
            SimpleState next_state = std::get<0>(result);
            double reward = std::get<1>(result);
            bool done = std::get<2>(result);
            
            total_reward += reward;
            total_infections += next_state.infected_fraction;
            total_economic_cost += action.lockdown_intensity;
            state = next_state;
            
            if (done) break;
        }
        
        file << "NoIntervention," << total_reward << "," << total_infections << "," << total_economic_cost << "\\n";
        std::cout << "No Intervention: Reward=" << total_reward << ", Infections=" << total_infections << "\\n";
    }
    
    // Policy 2: Always moderate lockdown
    {
        SimpleState state = env.reset();
        double total_reward = 0, total_infections = 0, total_economic_cost = 0;
        
        for (int step = 0; step < 60; ++step) {
            SimpleAction action(0.5); // Moderate lockdown
            auto result = env.step(action);
            SimpleState next_state = std::get<0>(result);
            double reward = std::get<1>(result);
            bool done = std::get<2>(result);
            
            total_reward += reward;
            total_infections += next_state.infected_fraction;
            total_economic_cost += action.lockdown_intensity;
            state = next_state;
            
            if (done) break;
        }
        
        file << "ModerateLockdown," << total_reward << "," << total_infections << "," << total_economic_cost << "\\n";
        std::cout << "Moderate Lockdown: Reward=" << total_reward << ", Infections=" << total_infections << "\\n";
    }
    
    // Policy 3: Trained RL agent
    {
        SimpleQLearning agent(0.1, 0.9, 0.1);
        
        // Quick training
        for (int episode = 0; episode < 30; ++episode) {
            SimpleState train_state = env.reset();
            for (int step = 0; step < 30; ++step) {
                SimpleAction action = agent.selectAction(train_state);
                auto result = env.step(action);
                SimpleState next_state = std::get<0>(result);
                double reward = std::get<1>(result);
                bool done = std::get<2>(result);
                
                agent.update(train_state, action, reward, next_state);
                train_state = next_state;
                if (done) break;
            }
            agent.decayEpsilon();
        }
        
        // Evaluation
        SimpleState state = env.reset();
        double total_reward = 0, total_infections = 0, total_economic_cost = 0;
        
        for (int step = 0; step < 60; ++step) {
            SimpleAction action = agent.selectAction(state);
            auto result = env.step(action);
            SimpleState next_state = std::get<0>(result);
            double reward = std::get<1>(result);
            bool done = std::get<2>(result);
            
            total_reward += reward;
            total_infections += next_state.infected_fraction;
            total_economic_cost += action.lockdown_intensity;
            state = next_state;
            
            if (done) break;
        }
        
        file << "RLAgent," << total_reward << "," << total_infections << "," << total_economic_cost << "\\n";
        std::cout << "RL Agent: Reward=" << total_reward << ", Infections=" << total_infections << "\\n";
    }
    
    file.close();
    std::cout << "Results saved to experiment3_policies.csv\\n\\n";
}

int main() {
    std::cout << "Simple RL Experiments for Epidemic Control\\n";
    std::cout << "==========================================\\n\\n";
    
    try {
        experiment1_learning_curve();
        experiment2_hyperparameter_sensitivity();
        experiment3_policy_comparison();
        
        std::cout << "All experiments completed successfully!\\n";
        std::cout << "Generated files:\\n";
        std::cout << "- experiment1_learning_curve.csv\\n";
        std::cout << "- experiment2_hyperparameters.csv\\n";
        std::cout << "- experiment3_policies.csv\\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\\n";
        return 1;
    }
    
    return 0;
}
