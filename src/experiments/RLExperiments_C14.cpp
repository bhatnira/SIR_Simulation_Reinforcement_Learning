/**
 * @file RLExperiments_fixed.cpp
 * @brief C++14 compatible experimental suite for RL epidemic control framework
 * @author Scientific Computing Team
 * @date 2025
 */

#include "BayesianSIR.h"
#include <iostream>
#include <iomanip>
#include <memory>
#include <fstream>
#include <chrono>
#include <cmath>
#include <tuple>

/**
 * @brief Experiment 1: Learning Curve Analysis
 * Tests how quickly the DQN agent learns optimal policies
 */
void experiment1_learningCurveAnalysis() {
    std::cout << "\n=== EXPERIMENT 1: Learning Curve Analysis ===\n";
    
    std::ofstream results("experiment1_learning_curves.csv");
    results << "Episode,Reward,Steps,FinalInfected,EconomicCost,SocialCost\n";
    
    // Create environment with moderate epidemic parameters
    EpidemicEnvironment env(1000, 0.4, 0.15, 300);
    auto reward_func = std::make_unique<EpidemicRewardFunction>(0.6, 0.25, 0.15);
    env.setRewardFunction(std::move(reward_func));
    
    // Create DQN agent with detailed logging
    DQNAgent agent(13, 8, 0.001, 0.95, 1.0, 0.998, 0.05);
    
    std::cout << "Training agent for 200 episodes with learning curve tracking...\n";
    
    int num_episodes = 200;
    double best_reward = -std::numeric_limits<double>::infinity();
    int episodes_without_improvement = 0;
    
    for (int episode = 0; episode < num_episodes; ++episode) {
        EpidemicState state = env.reset();
        double total_reward = 0.0;
        double economic_cost = 0.0;
        double social_cost = 0.0;
        int steps = 0;
        
        while (!env.isDone() && steps < 300) {
            EpidemicAction action = agent.selectAction(state);
            
            // C++14 compatible tuple unpacking
            std::tuple<EpidemicState, double, bool> step_result = env.step(action);
            EpidemicState next_state = std::get<0>(step_result);
            double reward = std::get<1>(step_result);
            bool done = std::get<2>(step_result);
            
            agent.remember(state, action, reward, next_state, done);
            
            if (steps > 64) {
                agent.train();
            }
            
            total_reward += reward;
            economic_cost += (1.0 - next_state.economic_activity);
            social_cost += (1.0 - next_state.mobility_index);
            
            state = next_state;
            steps++;
        }
        
        // Track improvement
        if (total_reward > best_reward) {
            best_reward = total_reward;
            episodes_without_improvement = 0;
        } else {
            episodes_without_improvement++;
        }
        
        // Log results
        results << episode << "," << total_reward << "," << steps << ","
                << state.infected_fraction << "," << economic_cost << "," 
                << social_cost << "\n";
        
        if (episode % 25 == 0) {
            std::cout << "Episode " << std::setw(3) << episode 
                     << " | Reward: " << std::fixed << std::setprecision(1) << total_reward
                     << " | Best: " << best_reward
                     << " | Final infected: " << std::fixed << std::setprecision(3) 
                     << state.infected_fraction << std::endl;
        }
        
        // Early stopping if no improvement for 50 episodes
        if (episodes_without_improvement > 50) {
            std::cout << "Early stopping at episode " << episode << " (no improvement)\n";
            break;
        }
    }
    
    results.close();
    agent.saveModel("experiment1_trained_model.model");
    std::cout << "Experiment 1 complete. Results saved to experiment1_learning_curves.csv\n";
}

/**
 * @brief Experiment 2: Hyperparameter Sensitivity Analysis (Simplified)
 * Tests key hyperparameters with reduced combinations for faster execution
 */
void experiment2_hyperparameterSensitivity() {
    std::cout << "\n=== EXPERIMENT 2: Hyperparameter Sensitivity Analysis ===\n";
    
    std::ofstream results("experiment2_hyperparameters.csv");
    results << "LearningRate,Gamma,EpsilonDecay,HealthWeight,EconomicWeight,SocialWeight,FinalReward,FinalInfected\n";
    
    // Reduced test combinations for faster execution
    std::vector<double> learning_rates = {0.001, 0.01};
    std::vector<double> gammas = {0.95, 0.99};
    std::vector<std::vector<double>> reward_weights = {
        {0.7, 0.2, 0.1},  // Health-focused
        {0.4, 0.4, 0.2}   // Balanced
    };
    
    int config_count = 0;
    int total_configs = learning_rates.size() * gammas.size() * reward_weights.size();
    
    for (double lr : learning_rates) {
        for (double gamma : gammas) {
            for (const auto& weights : reward_weights) {
                config_count++;
                std::cout << "Testing configuration " << config_count << "/" << total_configs 
                         << " (lr=" << lr << ", gamma=" << gamma << ")\n";
                
                // Create environment and agent with current hyperparameters
                EpidemicEnvironment env(1000, 0.35, 0.12, 200);
                auto reward_func = std::make_unique<EpidemicRewardFunction>(weights[0], weights[1], weights[2]);
                env.setRewardFunction(std::move(reward_func));
                
                DQNAgent agent(13, 8, lr, gamma, 1.0, 0.997, 0.01);
                
                // Train for fewer episodes due to multiple configurations
                double best_reward = -std::numeric_limits<double>::infinity();
                EpidemicState final_state;
                
                for (int episode = 0; episode < 50; ++episode) {
                    EpidemicState state = env.reset();
                    double total_reward = 0.0;
                    int steps = 0;
                    
                    while (!env.isDone() && steps < 200) {
                        EpidemicAction action = agent.selectAction(state);
                        
                        std::tuple<EpidemicState, double, bool> step_result = env.step(action);
                        EpidemicState next_state = std::get<0>(step_result);
                        double reward = std::get<1>(step_result);
                        bool done = std::get<2>(step_result);
                        
                        agent.remember(state, action, reward, next_state, done);
                        if (steps > 32) agent.train();
                        
                        total_reward += reward;
                        state = next_state;
                        steps++;
                    }
                    
                    if (total_reward > best_reward) {
                        best_reward = total_reward;
                        final_state = state;
                    }
                }
                
                results << lr << "," << gamma << ",0.997,"
                       << weights[0] << "," << weights[1] << "," << weights[2] << ","
                       << best_reward << "," << final_state.infected_fraction << "\n";
            }
        }
    }
    
    results.close();
    std::cout << "Experiment 2 complete. Results saved to experiment2_hyperparameters.csv\n";
}

/**
 * @brief Experiment 3: Epidemic Scenario Robustness Testing
 * Tests agent performance across different epidemic parameters
 */
void experiment3_scenarioRobustness() {
    std::cout << "\n=== EXPERIMENT 3: Epidemic Scenario Robustness Testing ===\n";
    
    std::ofstream results("experiment3_robustness.csv");
    results << "TransmissionRate,RecoveryRate,PopulationSize,Scenario,AgentReward,BaselineReward,Improvement,FinalInfected,PeakInfected\n";
    
    // Create and train a basic agent for testing
    DQNAgent agent(13, 8);
    std::cout << "Training agent for robustness testing...\n";
    
    EpidemicEnvironment train_env(1000, 0.3, 0.1, 200);
    auto train_reward = std::make_unique<EpidemicRewardFunction>(0.6, 0.25, 0.15);
    train_env.setRewardFunction(std::move(train_reward));
    
    for (int ep = 0; ep < 50; ++ep) {
        EpidemicState state = train_env.reset();
        int steps = 0;
        while (!train_env.isDone() && steps < 200) {
            EpidemicAction action = agent.selectAction(state);
            
            std::tuple<EpidemicState, double, bool> step_result = train_env.step(action);
            EpidemicState next_state = std::get<0>(step_result);
            double reward = std::get<1>(step_result);
            bool done = std::get<2>(step_result);
            
            agent.remember(state, action, reward, next_state, done);
            if (steps > 32) agent.train();
            state = next_state;
            steps++;
        }
    }
    
    // Test scenarios with different epidemic parameters
    struct Scenario {
        std::string name;
        double transmission_rate;
        double recovery_rate;
        int population_size;
    };
    
    std::vector<Scenario> scenarios = {
        {"Low Transmission", 0.2, 0.15, 1000},
        {"Moderate Transmission", 0.35, 0.12, 1000},
        {"High Transmission", 0.5, 0.1, 1000},
        {"Slow Recovery", 0.3, 0.08, 1000},
        {"Fast Recovery", 0.3, 0.2, 1000},
        {"Small Population", 0.3, 0.12, 500},
        {"Large Population", 0.3, 0.12, 2000}
    };
    
    for (const auto& scenario : scenarios) {
        std::cout << "Testing scenario: " << scenario.name << std::endl;
        
        // Test with RL agent
        EpidemicEnvironment env_rl(scenario.population_size, scenario.transmission_rate, 
                                  scenario.recovery_rate, 250);
        auto reward_func_rl = std::make_unique<EpidemicRewardFunction>(0.6, 0.25, 0.15);
        env_rl.setRewardFunction(std::move(reward_func_rl));
        
        EpidemicState state_rl = env_rl.reset();
        double agent_reward = 0.0;
        double peak_infected_rl = 0.0;
        
        while (!env_rl.isDone()) {
            EpidemicAction action = agent.selectAction(state_rl);
            
            std::tuple<EpidemicState, double, bool> step_result = env_rl.step(action);
            EpidemicState next_state = std::get<0>(step_result);
            double reward = std::get<1>(step_result);
            bool done = std::get<2>(step_result);
            
            agent_reward += reward;
            peak_infected_rl = std::max(peak_infected_rl, next_state.infected_fraction);
            state_rl = next_state;
        }
        
        // Test with baseline (no intervention)
        EpidemicEnvironment env_baseline(scenario.population_size, scenario.transmission_rate,
                                        scenario.recovery_rate, 250);
        auto reward_func_baseline = std::make_unique<EpidemicRewardFunction>(0.6, 0.25, 0.15);
        env_baseline.setRewardFunction(std::move(reward_func_baseline));
        
        EpidemicState state_baseline = env_baseline.reset();
        double baseline_reward = 0.0;
        double peak_infected_baseline = 0.0;
        EpidemicAction no_action;  // Default constructor gives no intervention
        
        while (!env_baseline.isDone()) {
            std::tuple<EpidemicState, double, bool> step_result = env_baseline.step(no_action);
            EpidemicState next_state = std::get<0>(step_result);
            double reward = std::get<1>(step_result);
            bool done = std::get<2>(step_result);
            
            baseline_reward += reward;
            peak_infected_baseline = std::max(peak_infected_baseline, next_state.infected_fraction);
            state_baseline = next_state;
        }
        
        double improvement = agent_reward - baseline_reward;
        
        results << scenario.transmission_rate << "," << scenario.recovery_rate << ","
               << scenario.population_size << "," << scenario.name << ","
               << agent_reward << "," << baseline_reward << "," << improvement << ","
               << state_rl.infected_fraction << "," << peak_infected_rl << "\n";
        
        std::cout << "  Agent reward: " << std::fixed << std::setprecision(1) << agent_reward
                 << " | Baseline: " << baseline_reward 
                 << " | Improvement: " << improvement << std::endl;
    }
    
    results.close();
    std::cout << "Experiment 3 complete. Results saved to experiment3_robustness.csv\n";
}

/**
 * @brief Experiment 4: Multi-Objective Optimization Analysis
 * Tests different reward weight combinations and their effects
 */
void experiment4_multiObjectiveOptimization() {
    std::cout << "\n=== EXPERIMENT 4: Multi-Objective Optimization Analysis ===\n";
    
    std::ofstream results("experiment4_multiobjective.csv");
    results << "HealthWeight,EconomicWeight,SocialWeight,TotalReward,FinalInfected,EconomicActivity,MobilityIndex\n";
    
    // Test different weight combinations
    std::vector<std::vector<double>> weight_combinations = {
        {1.0, 0.0, 0.0},    // Pure health focus
        {0.8, 0.2, 0.0},    // Health + economy
        {0.6, 0.3, 0.1},    // Health-focused balanced
        {0.5, 0.5, 0.0},    // Health-economy balance
        {0.33, 0.33, 0.34}, // Equal weights
        {0.3, 0.7, 0.0},    // Economy-focused
        {0.4, 0.2, 0.4},    // Health-social balance
    };
    
    for (const auto& weights : weight_combinations) {
        std::cout << "Testing weights: Health=" << weights[0] 
                 << ", Economic=" << weights[1] << ", Social=" << weights[2] << std::endl;
        
        // Create environment with specific reward weights
        EpidemicEnvironment env(1000, 0.35, 0.12, 200);
        auto reward_func = std::make_unique<EpidemicRewardFunction>(weights[0], weights[1], weights[2]);
        env.setRewardFunction(std::move(reward_func));
        
        // Train agent with these specific weights
        DQNAgent agent(13, 8, 0.001, 0.95, 1.0, 0.997, 0.02);
        
        double best_total_reward = -std::numeric_limits<double>::infinity();
        EpidemicState best_final_state;
        
        // Train for moderate number of episodes
        for (int episode = 0; episode < 40; ++episode) {
            EpidemicState state = env.reset();
            double total_reward = 0.0;
            int steps = 0;
            
            while (!env.isDone() && steps < 200) {
                EpidemicAction action = agent.selectAction(state);
                
                std::tuple<EpidemicState, double, bool> step_result = env.step(action);
                EpidemicState next_state = std::get<0>(step_result);
                double reward = std::get<1>(step_result);
                bool done = std::get<2>(step_result);
                
                agent.remember(state, action, reward, next_state, done);
                if (steps > 32) agent.train();
                
                total_reward += reward;
                state = next_state;
                steps++;
            }
            
            if (total_reward > best_total_reward) {
                best_total_reward = total_reward;
                best_final_state = state;
            }
        }
        
        results << weights[0] << "," << weights[1] << "," << weights[2] << ","
               << best_total_reward << ","
               << best_final_state.infected_fraction << ","
               << best_final_state.economic_activity << ","
               << best_final_state.mobility_index << "\n";
        
        std::cout << "  Best reward: " << std::fixed << std::setprecision(1) << best_total_reward
                 << " | Final infected: " << std::setprecision(3) << best_final_state.infected_fraction
                 << " | Economic activity: " << best_final_state.economic_activity << std::endl;
    }
    
    results.close();
    std::cout << "Experiment 4 complete. Results saved to experiment4_multiobjective.csv\n";
}

/**
 * @brief Experiment 5: Quick Performance Comparison
 * Compares different intervention strategies
 */
void experiment5_performanceComparison() {
    std::cout << "\n=== EXPERIMENT 5: Performance Comparison ===\n";
    
    std::ofstream results("experiment5_comparison.csv");
    results << "Strategy,FinalInfected,EconomicActivity,MobilityIndex,TotalReward\n";
    
    // Define different strategies
    std::vector<std::pair<std::string, EpidemicAction>> strategies;
    
    // No intervention
    strategies.emplace_back("No Intervention", EpidemicAction());
    
    // Light social distancing
    EpidemicAction light_action;
    light_action.interventions[EpidemicAction::SOCIAL_DISTANCING] = 0.3;
    strategies.emplace_back("Light Social Distancing", light_action);
    
    // Moderate intervention
    EpidemicAction moderate_action;
    moderate_action.interventions[EpidemicAction::SOCIAL_DISTANCING] = 0.6;
    moderate_action.interventions[EpidemicAction::MASK_MANDATE] = 1.0;
    strategies.emplace_back("Moderate Intervention", moderate_action);
    
    // Strict lockdown
    EpidemicAction strict_action;
    strict_action.interventions[EpidemicAction::LOCKDOWN_INTENSITY] = 1.0;
    strict_action.interventions[EpidemicAction::MASK_MANDATE] = 1.0;
    strategies.emplace_back("Strict Lockdown", strict_action);
    
    std::cout << "Strategy               |Inf_frac| Economic | Mobility | Total Reward\n";
    std::cout << "----------------------+--------+----------+----------+-------------\n";
    
    for (const auto& strategy_pair : strategies) {
        const std::string& strategy_name = strategy_pair.first;
        const EpidemicAction& base_action = strategy_pair.second;
        
        // Create environment
        EpidemicEnvironment env(1000, 0.3, 0.1, 200);
        auto reward_func = std::make_unique<EpidemicRewardFunction>(0.6, 0.25, 0.15);
        env.setRewardFunction(std::move(reward_func));
        
        // Simulate with fixed strategy
        EpidemicState state = env.reset();
        double total_reward = 0.0;
        
        while (!env.isDone()) {
            std::tuple<EpidemicState, double, bool> step_result = env.step(base_action);
            EpidemicState next_state = std::get<0>(step_result);
            double reward = std::get<1>(step_result);
            bool done = std::get<2>(step_result);
            
            total_reward += reward;
            state = next_state;
        }
        
        results << strategy_name << ","
               << state.infected_fraction << ","
               << state.economic_activity << ","
               << state.mobility_index << ","
               << total_reward << "\n";
        
        std::cout << std::setw(22) << strategy_name << "|"
                 << std::setw(7) << std::fixed << std::setprecision(3) << state.infected_fraction << "|"
                 << std::setw(9) << std::fixed << std::setprecision(2) << state.economic_activity << "|"
                 << std::setw(9) << std::fixed << std::setprecision(2) << state.mobility_index << "|"
                 << std::setw(12) << std::fixed << std::setprecision(1) << total_reward
                 << std::endl;
    }
    
    results.close();
    std::cout << "Experiment 5 complete. Results saved to experiment5_comparison.csv\n";
}

int main() {
    std::cout << "===========================================================\n";
    std::cout << "   RL Epidemic Control - Experimental Suite\n";
    std::cout << "   Scientific Computing Project - Version 2.0\n";
    std::cout << "===========================================================\n";
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        experiment1_learningCurveAnalysis();
        experiment2_hyperparameterSensitivity();
        experiment3_scenarioRobustness();
        experiment4_multiObjectiveOptimization();
        experiment5_performanceComparison();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::minutes>(end_time - start_time);
        
        std::cout << "\n=== All Experiments Complete ===\n";
        std::cout << "Total execution time: " << duration.count() << " minutes\n";
        std::cout << "\nGenerated files:\n";
        std::cout << "• experiment1_learning_curves.csv - Learning progression data\n";
        std::cout << "• experiment2_hyperparameters.csv - Hyperparameter sensitivity analysis\n";
        std::cout << "• experiment3_robustness.csv - Robustness across epidemic scenarios\n";
        std::cout << "• experiment4_multiobjective.csv - Multi-objective optimization results\n";
        std::cout << "• experiment5_comparison.csv - Strategy comparison results\n";
        std::cout << "• experiment1_trained_model.model - Best trained model\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Experimental error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
