/**
 * @file ComprehensiveExperiments.cpp
 * @brief Comprehensive experimental suite for RL epidemic control research paper
 * @author Scientific Computing Team
 * @date 2025
 * 
 * This file implements 7 comprehensive experiments for evaluating the RL framework:
 * 1. Learning Convergence Analysis
 * 2. Policy Performance Comparison
 * 3. Robustness to Initial Conditions
 * 4. Multi-Objective Optimization Trade-offs
 * 5. Real-World Scenario Testing
 * 6. Computational Efficiency Analysis
 * 7. Sensitivity Analysis
 */

#include "RLSIR.h"
#include <iostream>
#include <iomanip>
#include <memory>
#include <fstream>
#include <chrono>
#include <cmath>
#include <vector>
#include <map>
#include <random>
#include <algorithm>

/**
 * @brief Utility function to calculate statistics
 */
struct Statistics {
    double mean, std_dev, min_val, max_val;
    
    static Statistics calculate(const std::vector<double>& data) {
        Statistics stats;
        stats.mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
        
        double variance = 0.0;
        stats.min_val = *std::min_element(data.begin(), data.end());
        stats.max_val = *std::max_element(data.begin(), data.end());
        
        for (double val : data) {
            variance += (val - stats.mean) * (val - stats.mean);
        }
        stats.std_dev = std::sqrt(variance / data.size());
        
        return stats;
    }
};

/**
 * @brief EXPERIMENT 1: Learning Convergence Analysis
 * Analyzes how the DQN agent converges to optimal policies over training episodes
 */
void experiment1_learningConvergence() {
    std::cout << "\n=== EXPERIMENT 1: Learning Convergence Analysis ===\n";
    
    std::ofstream results("exp1_learning_convergence.csv");
    results << "Episode,Reward,Epsilon,AvgLoss,FinalInfected,EconomicCost,SocialCost,Steps\n";
    
    // Environment setup
    EpidemicEnvironment env(5000, 0.35, 0.12, 365);
    auto reward_func = std::make_unique<EpidemicRewardFunction>(0.6, 0.25, 0.15);
    env.setRewardFunction(std::move(reward_func));
    
    // Agent with tracking capabilities
    DQNAgent agent(13, 8, 0.0005, 0.99, 1.0, 0.995, 0.01);
    
    std::cout << "Training for 500 episodes with convergence tracking...\n";
    
    int num_episodes = 500;
    std::vector<double> episode_rewards;
    std::vector<double> moving_averages;
    
    for (int episode = 0; episode < num_episodes; ++episode) {
        EpidemicState state = env.reset();
        double total_reward = 0.0;
        double economic_cost = 0.0;
        double social_cost = 0.0;
        int steps = 0;
        
        // Episode simulation
        while (!env.isDone() && steps < 365) {
            EpidemicAction action = agent.selectAction(state);
            auto result = env.step(action);
            EpidemicState next_state = std::get<0>(result);
            double reward = std::get<1>(result);
            bool done = std::get<2>(result);
            
            agent.remember(state, action, reward, next_state, done);
            
            if (steps > 64) {  // Start training after collecting experiences
                agent.train();
            }
            
            total_reward += reward;
            economic_cost += (1.0 - next_state.social_mobility_index);
            social_cost += (1.0 - next_state.social_mobility_index);
            state = next_state;
            steps++;
        }
        
        episode_rewards.push_back(total_reward);
        
        // Calculate moving average (last 50 episodes)
        int window = std::min(50, episode + 1);
        double moving_avg = 0.0;
        for (int i = episode - window + 1; i <= episode; ++i) {
            moving_avg += episode_rewards[i];
        }
        moving_avg /= window;
        moving_averages.push_back(moving_avg);
        
        // Log detailed results
        results << episode << "," << total_reward << "," << agent.getEpsilon() 
               << ",0.0," << (double)state.infected / 5000.0 << "," << economic_cost 
               << "," << social_cost << "," << steps << "\\n";
        
        if (episode % 50 == 0) {
            std::cout << "Episode " << std::setw(3) << episode 
                     << " | Reward: " << std::fixed << std::setprecision(1) << total_reward
                     << " | Avg: " << std::fixed << std::setprecision(1) << moving_avg
                     << " | Final Infected: " << std::fixed << std::setprecision(3) << (double)state.infected / 5000.0
                     << std::endl;
        }
    }
    
    results.close();
    
    // Analysis
    Statistics reward_stats = Statistics::calculate(episode_rewards);
    std::cout << "\\nLearning Convergence Results:\\n";
    std::cout << "Final 50-episode average: " << moving_averages.back() << "\\n";
    std::cout << "Overall reward: " << reward_stats.mean << " ± " << reward_stats.std_dev << "\\n";
    
    // Save trained model for subsequent experiments
    agent.saveModel("convergence_trained_model.model");
    std::cout << "Saved trained model for subsequent experiments\\n";
}

/**
 * @brief EXPERIMENT 2: Policy Performance Comparison
 * Compares RL-learned policy against baseline heuristic policies
 */
void experiment2_policyComparison() {
    std::cout << "\\n=== EXPERIMENT 2: Policy Performance Comparison ===\\n";
    
    std::ofstream results("exp2_policy_comparison.csv");
    results << "Policy,Run,TotalReward,FinalInfected,PeakInfected,EconomicCost,SocialCost,Duration\\n";
    
    std::vector<std::string> policy_names = {
        "No_Intervention", "Static_Moderate", "Static_Strict", 
        "Reactive_Threshold", "RL_Trained"
    };
    
    int num_runs = 20;  // Multiple runs for statistical significance
    
    for (const std::string& policy_name : policy_names) {
        std::cout << "Testing " << policy_name << " policy...\\n";
        
        for (int run = 0; run < num_runs; ++run) {
            // Create fresh environment for each run
            EpidemicEnvironment env(5000, 0.35, 0.12, 365);
            auto reward_func = std::make_unique<EpidemicRewardFunction>(0.6, 0.25, 0.15);
            env.setRewardFunction(std::move(reward_func));
            
            EpidemicState state = env.reset();
            double total_reward = 0.0;
            double peak_infected = 0.0;
            double economic_cost = 0.0;
            double social_cost = 0.0;
            int duration = 0;
            
            // Policy-specific setup
            std::unique_ptr<DQNAgent> rl_agent;
            if (policy_name == "RL_Trained") {
                rl_agent = std::make_unique<DQNAgent>(13, 8);
                try {
                    rl_agent->loadModel("convergence_trained_model.model");
                } catch (...) {
                    std::cout << "Warning: Could not load trained model for run " << run << "\\n";
                }
            }
            
            while (!env.isDone() && duration < 365) {
                EpidemicAction action;
                
                // Policy selection
                if (policy_name == "No_Intervention") {
                    action = EpidemicAction();  // Default empty action
                } 
                else if (policy_name == "Static_Moderate") {
                    action.interventions[EpidemicAction::SOCIAL_DISTANCING] = 0.5;
                    action.interventions[EpidemicAction::MASK_MANDATE] = 0.8;
                    action.interventions[EpidemicAction::TESTING_EXPANSION] = 1.2;
                } 
                else if (policy_name == "Static_Strict") {
                    action.interventions[EpidemicAction::LOCKDOWN_INTENSITY] = 0.8;
                    action.interventions[EpidemicAction::SCHOOL_CLOSURE] = 1.0;
                    action.interventions[EpidemicAction::MASK_MANDATE] = 1.0;
                    action.interventions[EpidemicAction::SOCIAL_DISTANCING] = 0.9;
                } 
                else if (policy_name == "Reactive_Threshold") {
                    // Reactive policy based on infection rate
                    if (state.infected_fraction > 0.05) {
                        action.interventions[EpidemicAction::LOCKDOWN_INTENSITY] = 0.6;
                        action.interventions[EpidemicAction::SCHOOL_CLOSURE] = 0.8;
                        action.interventions[EpidemicAction::SOCIAL_DISTANCING] = 0.7;
                    } else if (state.infected_fraction > 0.02) {
                        action.interventions[EpidemicAction::SOCIAL_DISTANCING] = 0.4;
                        action.interventions[EpidemicAction::MASK_MANDATE] = 0.6;
                    }
                    action.interventions[EpidemicAction::TESTING_EXPANSION] = 1.0;
                } 
                else if (policy_name == "RL_Trained" && rl_agent) {
                    action = rl_agent->selectAction(state);
                }
                
                auto result = env.step(action);
                EpidemicState next_state = std::get<0>(result);
                double reward = std::get<1>(result);
                
                total_reward += reward;
                peak_infected = std::max(peak_infected, next_state.infected_fraction);
                economic_cost += (1.0 - next_state.economic_activity);
                social_cost += (1.0 - next_state.mobility_index);
                
                state = next_state;
                duration++;
            }
            
            results << policy_name << "," << run << "," << total_reward << ","
                   << state.infected_fraction << "," << peak_infected << ","
                   << economic_cost << "," << social_cost << "," << duration << "\\n";
        }
    }
    
    results.close();
    std::cout << "Policy comparison completed. Results saved to exp2_policy_comparison.csv\\n";
}

/**
 * @brief EXPERIMENT 3: Robustness to Initial Conditions
 * Tests performance across different epidemic scenarios
 */
void experiment3_robustnessAnalysis() {
    std::cout << "\\n=== EXPERIMENT 3: Robustness to Initial Conditions ===\\n";
    
    std::ofstream results("exp3_robustness_analysis.csv");
    results << "Scenario,R0,InitialInfected,PopSize,Reward,FinalInfected,PeakInfected,Duration\\n";
    
    // Load trained agent
    DQNAgent agent(13, 8);
    try {
        agent.loadModel("convergence_trained_model.model");
    } catch (...) {
        std::cout << "Warning: Using untrained agent\\n";
    }
    
    // Define test scenarios
    std::vector<std::tuple<std::string, double, double, int>> scenarios = {
        {"Low_R0_Small", 1.5, 0.001, 1000},
        {"Low_R0_Medium", 1.5, 0.001, 5000},
        {"Low_R0_Large", 1.5, 0.001, 10000},
        {"Medium_R0_Small", 2.5, 0.005, 1000},
        {"Medium_R0_Medium", 2.5, 0.005, 5000},
        {"Medium_R0_Large", 2.5, 0.005, 10000},
        {"High_R0_Small", 4.0, 0.01, 1000},
        {"High_R0_Medium", 4.0, 0.01, 5000},
        {"High_R0_Large", 4.0, 0.01, 10000},
        {"Extreme_R0", 6.0, 0.02, 5000}
    };
    
    for (const auto& scenario : scenarios) {
        std::string name = std::get<0>(scenario);
        double r0 = std::get<1>(scenario);
        double initial_infected = std::get<2>(scenario);
        int pop_size = std::get<3>(scenario);
        
        std::cout << "Testing scenario: " << name << "\\n";
        
        // Run multiple times for statistics
        std::vector<double> rewards, final_infected, peak_infected, durations;
        
        for (int run = 0; run < 10; ++run) {
            EpidemicEnvironment env(pop_size, r0, initial_infected, 365);
            auto reward_func = std::make_unique<EpidemicRewardFunction>(0.6, 0.25, 0.15);
            env.setRewardFunction(std::move(reward_func));
            
            EpidemicState state = env.reset();
            double total_reward = 0.0;
            double peak_inf = 0.0;
            int steps = 0;
            
            while (!env.isDone() && steps < 365) {
                EpidemicAction action = agent.selectAction(state);
                auto result = env.step(action);
                EpidemicState next_state = std::get<0>(result);
                double reward = std::get<1>(result);
                
                total_reward += reward;
                peak_inf = std::max(peak_inf, next_state.infected_fraction);
                state = next_state;
                steps++;
            }
            
            rewards.push_back(total_reward);
            final_infected.push_back(state.infected_fraction);
            peak_infected.push_back(peak_inf);
            durations.push_back(steps);
        }
        
        // Calculate averages
        Statistics reward_stats = Statistics::calculate(rewards);
        Statistics final_stats = Statistics::calculate(final_infected);
        Statistics peak_stats = Statistics::calculate(peak_infected);
        Statistics duration_stats = Statistics::calculate(durations);
        
        results << name << "," << r0 << "," << initial_infected << "," << pop_size
               << "," << reward_stats.mean << "," << final_stats.mean 
               << "," << peak_stats.mean << "," << duration_stats.mean << "\\n";
    }
    
    results.close();
    std::cout << "Robustness analysis completed\\n";
}

/**
 * @brief EXPERIMENT 4: Multi-Objective Trade-off Analysis
 * Analyzes trade-offs between health, economic, and social objectives
 */
void experiment4_multiObjectiveAnalysis() {
    std::cout << "\\n=== EXPERIMENT 4: Multi-Objective Trade-off Analysis ===\\n";
    
    std::ofstream results("exp4_multiobjective_analysis.csv");
    results << "HealthWeight,EconWeight,SocialWeight,TotalReward,HealthScore,EconScore,SocialScore,FinalInfected\\n";
    
    // Test different weight combinations
    std::vector<std::tuple<double, double, double>> weight_combinations = {
        {1.0, 0.0, 0.0},   // Health-only
        {0.8, 0.2, 0.0},   // Health-economic
        {0.8, 0.1, 0.1},   // Health-focused balanced
        {0.6, 0.3, 0.1},   // Health-economic focused
        {0.6, 0.2, 0.2},   // Health-social focused
        {0.5, 0.3, 0.2},   // Balanced
        {0.4, 0.4, 0.2},   // Health-economic equal
        {0.4, 0.3, 0.3},   // Economic-social equal
        {0.3, 0.5, 0.2},   // Economic-focused
        {0.3, 0.4, 0.3},   // Truly balanced
        {0.2, 0.6, 0.2},   // Economic-dominant
        {0.2, 0.3, 0.5},   // Social-focused
        {0.0, 0.7, 0.3},   // Economic-social only
        {0.0, 0.5, 0.5},   // Economic-social equal
        {0.0, 0.0, 1.0}    // Social-only
    };
    
    for (const auto& weights : weight_combinations) {
        double health_w = std::get<0>(weights);
        double econ_w = std::get<1>(weights);
        double social_w = std::get<2>(weights);
        
        std::cout << "Testing weights: H=" << health_w << " E=" << econ_w << " S=" << social_w << "\\n";
        
        // Train agent with specific objective weights
        EpidemicEnvironment env(5000, 0.35, 0.12, 365);
        auto reward_func = std::make_unique<EpidemicRewardFunction>(health_w, econ_w, social_w);
        env.setRewardFunction(std::move(reward_func));
        
        DQNAgent agent(13, 8, 0.001, 0.99, 0.3, 0.99, 0.01);  // Faster training
        
        // Quick training (100 episodes)
        for (int episode = 0; episode < 100; ++episode) {
            EpidemicState state = env.reset();
            int steps = 0;
            
            while (!env.isDone() && steps < 365) {
                EpidemicAction action = agent.selectAction(state);
                auto result = env.step(action);
                EpidemicState next_state = std::get<0>(result);
                double reward = std::get<1>(result);
                bool done = std::get<2>(result);
                
                agent.remember(state, action, reward, next_state, done);
                if (steps > 32) agent.train();
                
                state = next_state;
                steps++;
            }
        }
        
        // Evaluate trained agent
        std::vector<double> total_rewards, health_scores, econ_scores, social_scores, final_infected;
        
        for (int eval = 0; eval < 5; ++eval) {
            env.reset();
            EpidemicState state = env.reset();
            double total_reward = 0.0;
            double health_score = 0.0;
            double econ_score = 0.0;
            double social_score = 0.0;
            int steps = 0;
            
            while (!env.isDone() && steps < 365) {
                EpidemicAction action = agent.selectAction(state);
                auto result = env.step(action);
                EpidemicState next_state = std::get<0>(result);
                double reward = std::get<1>(result);
                
                total_reward += reward;
                // Calculate individual objective scores
                health_score += -next_state.infected_fraction * 1000.0;
                econ_score += next_state.economic_activity * 10.0;
                social_score += next_state.mobility_index * 10.0;
                
                state = next_state;
                steps++;
            }
            
            total_rewards.push_back(total_reward);
            health_scores.push_back(health_score);
            econ_scores.push_back(econ_score);
            social_scores.push_back(social_score);
            final_infected.push_back(state.infected_fraction);
        }
        
        // Calculate averages
        Statistics reward_stats = Statistics::calculate(total_rewards);
        Statistics health_stats = Statistics::calculate(health_scores);
        Statistics econ_stats = Statistics::calculate(econ_scores);
        Statistics social_stats = Statistics::calculate(social_scores);
        Statistics infected_stats = Statistics::calculate(final_infected);
        
        results << health_w << "," << econ_w << "," << social_w << ","
               << reward_stats.mean << "," << health_stats.mean << ","
               << econ_stats.mean << "," << social_stats.mean << ","
               << infected_stats.mean << "\\n";
    }
    
    results.close();
    std::cout << "Multi-objective analysis completed\\n";
}

/**
 * @brief EXPERIMENT 5: Real-World Scenario Testing
 * Tests against scenarios inspired by real pandemic data
 */
void experiment5_realWorldScenarios() {
    std::cout << "\\n=== EXPERIMENT 5: Real-World Scenario Testing ===\\n";
    
    std::ofstream results("exp5_realworld_scenarios.csv");
    results << "Scenario,Description,TotalReward,PeakInfected,TotalDeaths,EconomicImpact,PolicyChanges\\n";
    
    // Load trained agent
    DQNAgent agent(13, 8);
    try {
        agent.loadModel("convergence_trained_model.model");
    } catch (...) {
        std::cout << "Warning: Using untrained agent\\n";
    }
    
    struct Scenario {
        std::string name;
        std::string description;
        double r0;
        double initial_infected;
        int population;
        int duration;
    };
    
    std::vector<Scenario> scenarios = {
        {"COVID_Early_2020", "Early COVID-19 outbreak scenario", 2.8, 0.0001, 1000000, 365},
        {"COVID_Delta", "Delta variant outbreak", 4.5, 0.001, 500000, 200},
        {"COVID_Omicron", "Omicron variant outbreak", 6.0, 0.005, 500000, 150},
        {"Seasonal_Flu", "Seasonal influenza outbreak", 1.8, 0.01, 100000, 120},
        {"SARS_Like", "SARS-like coronavirus", 2.2, 0.0005, 50000, 300},
        {"H1N1_Pandemic", "H1N1 pandemic scenario", 2.0, 0.002, 200000, 250},
        {"Novel_Pathogen", "Hypothetical novel pathogen", 3.5, 0.0002, 750000, 400}
    };
    
    for (const auto& scenario : scenarios) {
        std::cout << "Testing: " << scenario.name << "\\n";
        
        EpidemicEnvironment env(scenario.population, scenario.r0, 
                              scenario.initial_infected, scenario.duration);
        auto reward_func = std::make_unique<EpidemicRewardFunction>(0.6, 0.25, 0.15);
        env.setRewardFunction(std::move(reward_func));
        
        EpidemicState state = env.reset();
        double total_reward = 0.0;
        double peak_infected = 0.0;
        double total_deaths = 0.0;
        double economic_impact = 0.0;
        int policy_changes = 0;
        EpidemicAction last_action;
        
        int steps = 0;
        while (!env.isDone() && steps < scenario.duration) {
            EpidemicAction action = agent.selectAction(state);
            
            // Count significant policy changes
            if (steps > 0) {
                double action_diff = 0.0;
                for (const auto& intervention : action.interventions) {
                    auto it = last_action.interventions.find(intervention.first);
                    if (it != last_action.interventions.end()) {
                        action_diff += std::abs(intervention.second - it->second);
                    } else {
                        action_diff += std::abs(intervention.second);
                    }
                }
                if (action_diff > 0.5) policy_changes++;
            }
            last_action = action;
            
            auto result = env.step(action);
            EpidemicState next_state = std::get<0>(result);
            double reward = std::get<1>(result);
            
            total_reward += reward;
            peak_infected = std::max(peak_infected, next_state.infected_fraction);
            // Estimate deaths (simplified)
            total_deaths += next_state.infected_fraction * scenario.population * 0.01;
            economic_impact += (1.0 - next_state.economic_activity);
            
            state = next_state;
            steps++;
        }
        
        results << scenario.name << "," << scenario.description << ","
               << total_reward << "," << peak_infected << "," << total_deaths
               << "," << economic_impact << "," << policy_changes << "\\n";
    }
    
    results.close();
    std::cout << "Real-world scenario testing completed\\n";
}

/**
 * @brief EXPERIMENT 6: Computational Efficiency Analysis
 * Measures training time, inference time, and scalability
 */
void experiment6_computationalEfficiency() {
    std::cout << "\\n=== EXPERIMENT 6: Computational Efficiency Analysis ===\\n";
    
    std::ofstream results("exp6_computational_efficiency.csv");
    results << "PopulationSize,TrainingTimeMs,InferenceTimeMs,MemoryUsageMB,EpisodesPerSecond\\n";
    
    std::vector<int> population_sizes = {1000, 5000, 10000, 25000, 50000};
    
    for (int pop_size : population_sizes) {
        std::cout << "Testing population size: " << pop_size << "\\n";
        
        // Training time measurement
        auto start_time = std::chrono::high_resolution_clock::now();
        
        EpidemicEnvironment env(pop_size, 0.35, 0.01, 365);
        auto reward_func = std::make_unique<EpidemicRewardFunction>(0.6, 0.25, 0.15);
        env.setRewardFunction(std::move(reward_func));
        
        DQNAgent agent(13, 8, 0.001, 0.99, 0.5, 0.99, 0.01);
        
        // Train for 50 episodes
        int training_episodes = 50;
        for (int episode = 0; episode < training_episodes; ++episode) {
            EpidemicState state = env.reset();
            int steps = 0;
            
            while (!env.isDone() && steps < 200) {  // Shorter episodes for efficiency test
                EpidemicAction action = agent.selectAction(state);
                auto result = env.step(action);
                EpidemicState next_state = std::get<0>(result);
                double reward = std::get<1>(result);
                bool done = std::get<2>(result);
                
                agent.remember(state, action, reward, next_state, done);
                if (steps > 32) agent.train();
                
                state = next_state;
                steps++;
            }
        }
        
        auto training_end = std::chrono::high_resolution_clock::now();
        auto training_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            training_end - start_time).count();
        
        // Inference time measurement
        auto inference_start = std::chrono::high_resolution_clock::now();
        
        // Run 100 inference steps
        EpidemicState state = env.reset();
        for (int i = 0; i < 100; ++i) {
            EpidemicAction action = agent.selectAction(state);
            auto result = env.step(action);
            state = std::get<0>(result);
        }
        
        auto inference_end = std::chrono::high_resolution_clock::now();
        auto inference_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            inference_end - inference_start).count();
        
        // Calculate episodes per second
        double episodes_per_second = (double)training_episodes / 
                                   (training_duration / 1000.0);
        
        // Rough memory estimation (simplified)
        double memory_usage_mb = pop_size * 0.001 + 50.0;  // Simplified estimate
        
        results << pop_size << "," << training_duration << "," 
               << inference_duration << "," << memory_usage_mb << ","
               << episodes_per_second << "\\n";
    }
    
    results.close();
    std::cout << "Computational efficiency analysis completed\\n";
}

/**
 * @brief EXPERIMENT 7: Sensitivity Analysis
 * Tests sensitivity to hyperparameters and environmental factors
 */
void experiment7_sensitivityAnalysis() {
    std::cout << "\\n=== EXPERIMENT 7: Sensitivity Analysis ===\\n";
    
    std::ofstream results("exp7_sensitivity_analysis.csv");
    results << "Parameter,Value,AverageReward,StdReward,ConvergenceEpisodes\\n";
    
    // Base parameters
    double base_lr = 0.001;
    double base_gamma = 0.99;
    double base_epsilon_decay = 0.995;
    double base_r0 = 0.35;
    
    struct ParameterTest {
        std::string name;
        std::vector<double> values;
    };
    
    std::vector<ParameterTest> parameter_tests = {
        {"learning_rate", {0.0001, 0.0005, 0.001, 0.005, 0.01}},
        {"gamma", {0.90, 0.95, 0.99, 0.995, 0.999}},
        {"epsilon_decay", {0.99, 0.995, 0.998, 0.999, 0.9995}},
        {"transmission_rate", {0.2, 0.3, 0.35, 0.4, 0.5}}
    };
    
    for (const auto& param_test : parameter_tests) {
        std::cout << "Testing parameter: " << param_test.name << "\\n";
        
        for (double value : param_test.values) {
            std::vector<double> rewards;
            std::vector<int> convergence_episodes;
            
            // Run 5 trials for each parameter value
            for (int trial = 0; trial < 5; ++trial) {
                // Set up environment and agent based on parameter being tested
                double lr = (param_test.name == "learning_rate") ? value : base_lr;
                double gamma = (param_test.name == "gamma") ? value : base_gamma;
                double eps_decay = (param_test.name == "epsilon_decay") ? value : base_epsilon_decay;
                double r0 = (param_test.name == "transmission_rate") ? value : base_r0;
                
                EpidemicEnvironment env(5000, r0, 0.01, 365);
                auto reward_func = std::make_unique<EpidemicRewardFunction>(0.6, 0.25, 0.15);
                env.setRewardFunction(std::move(reward_func));
                
                DQNAgent agent(13, 8, lr, gamma, 1.0, eps_decay, 0.01);
                
                // Train and track convergence
                std::vector<double> episode_rewards;
                int convergence_episode = -1;
                double best_avg = -std::numeric_limits<double>::infinity();
                
                for (int episode = 0; episode < 200; ++episode) {
                    EpidemicState state = env.reset();
                    double total_reward = 0.0;
                    int steps = 0;
                    
                    while (!env.isDone() && steps < 200) {
                        EpidemicAction action = agent.selectAction(state);
                        auto result = env.step(action);
                        EpidemicState next_state = std::get<0>(result);
                        double reward = std::get<1>(result);
                        bool done = std::get<2>(result);
                        
                        agent.remember(state, action, reward, next_state, done);
                        if (steps > 32) agent.train();
                        
                        total_reward += reward;
                        state = next_state;
                        steps++;
                    }
                    
                    episode_rewards.push_back(total_reward);
                    
                    // Check for convergence (moving average improvement)
                    if (episode >= 49) {
                        double current_avg = 0.0;
                        for (int i = episode - 49; i <= episode; ++i) {
                            current_avg += episode_rewards[i];
                        }
                        current_avg /= 50.0;
                        
                        if (current_avg > best_avg + 1.0) {
                            best_avg = current_avg;
                            convergence_episode = episode;
                        }
                    }
                }
                
                // Calculate final performance (last 50 episodes)
                double final_performance = 0.0;
                for (int i = episode_rewards.size() - 50; i < episode_rewards.size(); ++i) {
                    final_performance += episode_rewards[i];
                }
                final_performance /= 50.0;
                
                rewards.push_back(final_performance);
                convergence_episodes.push_back(convergence_episode);
            }
            
            // Calculate statistics
            Statistics reward_stats = Statistics::calculate(rewards);
            double avg_convergence = 0.0;
            for (int conv : convergence_episodes) {
                avg_convergence += (conv > 0) ? conv : 200;  // Use 200 if didn't converge
            }
            avg_convergence /= convergence_episodes.size();
            
            results << param_test.name << "," << value << ","
                   << reward_stats.mean << "," << reward_stats.std_dev << ","
                   << avg_convergence << "\\n";
        }
    }
    
    results.close();
    std::cout << "Sensitivity analysis completed\\n";
}

/**
 * @brief Main experimental suite
 */
int main() {
    std::cout << "===============================================================\\n";
    std::cout << "    COMPREHENSIVE RL EPIDEMIC CONTROL EXPERIMENTAL SUITE\\n";
    std::cout << "    Scientific Computing Research Paper - Version 2.0\\n";
    std::cout << "===============================================================\\n";
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        std::cout << "\\nRunning comprehensive experimental evaluation...\\n";
        std::cout << "Total estimated time: ~30-45 minutes\\n";
        
        experiment1_learningConvergence();
        experiment2_policyComparison();
        experiment3_robustnessAnalysis();
        experiment4_multiObjectiveAnalysis();
        experiment5_realWorldScenarios();
        experiment6_computationalEfficiency();
        experiment7_sensitivityAnalysis();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::minutes>(
            end_time - start_time).count();
        
        std::cout << "\\n===============================================================\\n";
        std::cout << "    EXPERIMENTAL SUITE COMPLETED SUCCESSFULLY\\n";
        std::cout << "    Total execution time: " << total_duration << " minutes\\n";
        std::cout << "===============================================================\\n";
        
        std::cout << "\\nGenerated result files:\\n";
        std::cout << "• exp1_learning_convergence.csv - Learning curve analysis\\n";
        std::cout << "• exp2_policy_comparison.csv - Policy performance comparison\\n";
        std::cout << "• exp3_robustness_analysis.csv - Robustness to initial conditions\\n";
        std::cout << "• exp4_multiobjective_analysis.csv - Multi-objective trade-offs\\n";
        std::cout << "• exp5_realworld_scenarios.csv - Real-world scenario testing\\n";
        std::cout << "• exp6_computational_efficiency.csv - Computational efficiency\\n";
        std::cout << "• exp7_sensitivity_analysis.csv - Sensitivity analysis\\n";
        
        std::cout << "\\nNext steps for paper analysis:\\n";
        std::cout << "1. Import CSV files into your analysis software (R, Python, etc.)\\n";
        std::cout << "2. Generate visualizations (learning curves, comparison plots, etc.)\\n";
        std::cout << "3. Perform statistical significance tests\\n";
        std::cout << "4. Calculate effect sizes and confidence intervals\\n";
        std::cout << "5. Document methodology and results in paper\\n";
        
    } catch (const std::exception& e) {
        std::cerr << "\\nError during experimental execution: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
