/**
 * @file PaperExperiments.cpp
 * @brief Comprehensive experimental suite for RL epidemic control research paper
 * @author Scientific Computing Team
 * @date 2025
 * 
 * This file implements comprehensive experiments specifically designed for the research paper,
 * working with the current RLSIR framework structure.
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
#include <numeric>

/**
 * @brief Utility function to calculate statistics
 */
struct Statistics {
    double mean, std_dev, min_val, max_val;
    
    static Statistics calculate(const std::vector<double>& data) {
        if (data.empty()) return {0.0, 0.0, 0.0, 0.0};
        
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
 * @brief Helper function to calculate infection fraction from state
 */
double getInfectedFraction(const EpidemicState& state, int population) {
    return static_cast<double>(state.infected) / population;
}

/**
 * @brief EXPERIMENT 1: Learning Convergence Analysis
 */
void experiment1_learningConvergence() {
    std::cout << "\n=== EXPERIMENT 1: Learning Convergence Analysis ===\n";
    
    std::ofstream results("exp1_learning_convergence.csv");
    results << "Episode,Reward,FinalInfected,EconomicCost,Steps\n";
    
    const int population = 5000;
    EpidemicEnvironment env(population, 0.35, 0.12, 365);
    auto reward_func = std::make_unique<EpidemicRewardFunction>(0.6, 0.25, 0.15);
    env.setRewardFunction(std::move(reward_func));
    
    DQNAgent agent(13, 8, 0.001, 0.99, 1.0, 0.995, 0.01);
    
    std::cout << "Training for 300 episodes with convergence tracking...\n";
    
    int num_episodes = 300;
    std::vector<double> episode_rewards;
    
    for (int episode = 0; episode < num_episodes; ++episode) {
        EpidemicState state = env.reset();
        double total_reward = 0.0;
        int steps = 0;
        
        // Episode simulation
        while (!env.isDone() && steps < 365) {
            EpidemicAction action = agent.selectAction(state);
            auto result = env.step(action);
            EpidemicState next_state = std::get<0>(result);
            double reward = std::get<1>(result);
            bool done = std::get<2>(result);
            
            agent.remember(state, action, reward, next_state, done);
            
            if (steps > 64) {
                agent.train();
            }
            
            total_reward += reward;
            state = next_state;
            steps++;
        }
        
        episode_rewards.push_back(total_reward);
        
        // Log results every 10 episodes
        if (episode % 10 == 0) {
            double infected_fraction = getInfectedFraction(state, population);
            results << episode << "," << total_reward << "," 
                   << infected_fraction << "," << state.economic_cost << "," 
                   << steps << "\n";
        }
        
        if (episode % 50 == 0) {
            // Calculate moving average (last 20 episodes)
            int window = std::min(20, episode + 1);
            double moving_avg = 0.0;
            for (int i = episode - window + 1; i <= episode; ++i) {
                moving_avg += episode_rewards[i];
            }
            moving_avg /= window;
            
            std::cout << "Episode " << std::setw(3) << episode 
                     << " | Reward: " << std::fixed << std::setprecision(1) << total_reward
                     << " | Avg: " << std::fixed << std::setprecision(1) << moving_avg
                     << " | Final Infected: " << std::fixed << std::setprecision(3) 
                     << getInfectedFraction(state, population) << std::endl;
        }
    }
    
    results.close();
    
    // Save trained model
    agent.saveModel("paper_trained_model.model");
    
    Statistics reward_stats = Statistics::calculate(episode_rewards);
    std::cout << "\nLearning Convergence Results:\n";
    std::cout << "Mean reward: " << reward_stats.mean << " ± " << reward_stats.std_dev << "\n";
    std::cout << "Model saved for subsequent experiments\n";
}

/**
 * @brief EXPERIMENT 2: Policy Performance Comparison
 */
void experiment2_policyComparison() {
    std::cout << "\n=== EXPERIMENT 2: Policy Performance Comparison ===\n";
    
    std::ofstream results("exp2_policy_comparison.csv");
    results << "Policy,Run,TotalReward,FinalInfected,PeakInfected,Duration\n";
    
    const int population = 5000;
    std::vector<std::string> policy_names = {
        "No_Intervention", "Static_Moderate", "Static_Strict", "RL_Trained"
    };
    
    int num_runs = 15;
    
    for (const std::string& policy_name : policy_names) {
        std::cout << "Testing " << policy_name << " policy...\n";
        
        for (int run = 0; run < num_runs; ++run) {
            EpidemicEnvironment env(population, 0.35, 0.12, 365);
            auto reward_func = std::make_unique<EpidemicRewardFunction>(0.6, 0.25, 0.15);
            env.setRewardFunction(std::move(reward_func));
            
            EpidemicState state = env.reset();
            double total_reward = 0.0;
            double peak_infected = 0.0;
            int duration = 0;
            
            // Policy-specific setup
            std::unique_ptr<DQNAgent> rl_agent;
            if (policy_name == "RL_Trained") {
                rl_agent = std::make_unique<DQNAgent>(13, 8);
                try {
                    rl_agent->loadModel("paper_trained_model.model");
                } catch (...) {
                    std::cout << "Warning: Could not load trained model for run " << run << "\n";
                }
            }
            
            while (!env.isDone() && duration < 365) {
                EpidemicAction action;
                
                // Policy selection
                if (policy_name == "No_Intervention") {
                    // Empty action - no interventions
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
                else if (policy_name == "RL_Trained" && rl_agent) {
                    action = rl_agent->selectAction(state);
                }
                
                auto result = env.step(action);
                EpidemicState next_state = std::get<0>(result);
                double reward = std::get<1>(result);
                
                total_reward += reward;
                double current_infected = getInfectedFraction(next_state, population);
                peak_infected = std::max(peak_infected, current_infected);
                
                state = next_state;
                duration++;
            }
            
            double final_infected = getInfectedFraction(state, population);
            results << policy_name << "," << run << "," << total_reward << ","
                   << final_infected << "," << peak_infected << "," << duration << "\n";
        }
    }
    
    results.close();
    std::cout << "Policy comparison completed\n";
}

/**
 * @brief EXPERIMENT 3: Robustness Analysis
 */
void experiment3_robustnessAnalysis() {
    std::cout << "\n=== EXPERIMENT 3: Robustness Analysis ===\n";
    
    std::ofstream results("exp3_robustness_analysis.csv");
    results << "Scenario,R0,InitialInfected,PopSize,AvgReward,StdReward,AvgFinalInfected\n";
    
    // Load trained agent
    DQNAgent agent(13, 8);
    try {
        agent.loadModel("paper_trained_model.model");
        std::cout << "Loaded trained model\n";
    } catch (...) {
        std::cout << "Warning: Using untrained agent\n";
    }
    
    // Define test scenarios
    std::vector<std::tuple<std::string, double, double, int>> scenarios = {
        {"Low_R0_Small", 1.8, 0.001, 1000},
        {"Low_R0_Large", 1.8, 0.001, 10000},
        {"Medium_R0_Small", 2.5, 0.005, 1000},
        {"Medium_R0_Large", 2.5, 0.005, 10000},
        {"High_R0_Small", 4.0, 0.01, 1000},
        {"High_R0_Large", 4.0, 0.01, 10000}
    };
    
    for (const auto& scenario : scenarios) {
        std::string name = std::get<0>(scenario);
        double r0 = std::get<1>(scenario);
        double initial_infected = std::get<2>(scenario);
        int pop_size = std::get<3>(scenario);
        
        std::cout << "Testing scenario: " << name << "\n";
        
        std::vector<double> rewards, final_infected;
        
        for (int run = 0; run < 10; ++run) {
            EpidemicEnvironment env(pop_size, r0, initial_infected, 365);
            auto reward_func = std::make_unique<EpidemicRewardFunction>(0.6, 0.25, 0.15);
            env.setRewardFunction(std::move(reward_func));
            
            EpidemicState state = env.reset();
            double total_reward = 0.0;
            int steps = 0;
            
            while (!env.isDone() && steps < 365) {
                EpidemicAction action = agent.selectAction(state);
                auto result = env.step(action);
                EpidemicState next_state = std::get<0>(result);
                double reward = std::get<1>(result);
                
                total_reward += reward;
                state = next_state;
                steps++;
            }
            
            rewards.push_back(total_reward);
            final_infected.push_back(getInfectedFraction(state, pop_size));
        }
        
        Statistics reward_stats = Statistics::calculate(rewards);
        Statistics infected_stats = Statistics::calculate(final_infected);
        
        results << name << "," << r0 << "," << initial_infected << "," << pop_size
               << "," << reward_stats.mean << "," << reward_stats.std_dev 
               << "," << infected_stats.mean << "\n";
    }
    
    results.close();
    std::cout << "Robustness analysis completed\n";
}

/**
 * @brief EXPERIMENT 4: Multi-Objective Analysis
 */
void experiment4_multiObjectiveAnalysis() {
    std::cout << "\n=== EXPERIMENT 4: Multi-Objective Analysis ===\n";
    
    std::ofstream results("exp4_multiobjective_analysis.csv");
    results << "HealthWeight,EconWeight,SocialWeight,AvgReward,AvgFinalInfected\n";
    
    const int population = 5000;
    
    // Test different weight combinations
    std::vector<std::tuple<double, double, double>> weight_combinations = {
        {1.0, 0.0, 0.0},   // Health-only
        {0.8, 0.2, 0.0},   // Health-economic
        {0.6, 0.3, 0.1},   // Health-economic focused
        {0.5, 0.3, 0.2},   // Balanced
        {0.4, 0.4, 0.2},   // Health-economic equal
        {0.3, 0.5, 0.2},   // Economic-focused
        {0.2, 0.6, 0.2}    // Economic-dominant
    };
    
    for (const auto& weights : weight_combinations) {
        double health_w = std::get<0>(weights);
        double econ_w = std::get<1>(weights);
        double social_w = std::get<2>(weights);
        
        std::cout << "Testing weights: H=" << health_w << " E=" << econ_w << " S=" << social_w << "\n";
        
        // Train agent with specific objective weights
        EpidemicEnvironment env(population, 0.35, 0.12, 365);
        auto reward_func = std::make_unique<EpidemicRewardFunction>(health_w, econ_w, social_w);
        env.setRewardFunction(std::move(reward_func));
        
        DQNAgent agent(13, 8, 0.002, 0.99, 0.5, 0.995, 0.01);
        
        // Quick training (80 episodes)
        for (int episode = 0; episode < 80; ++episode) {
            EpidemicState state = env.reset();
            int steps = 0;
            
            while (!env.isDone() && steps < 200) {
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
        std::vector<double> total_rewards, final_infected;
        
        for (int eval = 0; eval < 5; ++eval) {
            EpidemicState state = env.reset();
            double total_reward = 0.0;
            int steps = 0;
            
            while (!env.isDone() && steps < 200) {
                EpidemicAction action = agent.selectAction(state);
                auto result = env.step(action);
                EpidemicState next_state = std::get<0>(result);
                double reward = std::get<1>(result);
                
                total_reward += reward;
                state = next_state;
                steps++;
            }
            
            total_rewards.push_back(total_reward);
            final_infected.push_back(getInfectedFraction(state, population));
        }
        
        Statistics reward_stats = Statistics::calculate(total_rewards);
        Statistics infected_stats = Statistics::calculate(final_infected);
        
        results << health_w << "," << econ_w << "," << social_w << ","
               << reward_stats.mean << "," << infected_stats.mean << "\n";
    }
    
    results.close();
    std::cout << "Multi-objective analysis completed\n";
}

/**
 * @brief EXPERIMENT 5: Computational Efficiency
 */
void experiment5_computationalEfficiency() {
    std::cout << "\n=== EXPERIMENT 5: Computational Efficiency ===\n";
    
    std::ofstream results("exp5_computational_efficiency.csv");
    results << "PopulationSize,TrainingTimeMs,InferenceTimeMs,EpisodesPerSecond\n";
    
    std::vector<int> population_sizes = {1000, 3000, 5000, 7000, 10000};
    
    for (int pop_size : population_sizes) {
        std::cout << "Testing population size: " << pop_size << "\n";
        
        // Training time measurement
        auto start_time = std::chrono::high_resolution_clock::now();
        
        EpidemicEnvironment env(pop_size, 0.35, 0.01, 365);
        auto reward_func = std::make_unique<EpidemicRewardFunction>(0.6, 0.25, 0.15);
        env.setRewardFunction(std::move(reward_func));
        
        DQNAgent agent(13, 8, 0.001, 0.99, 0.5, 0.99, 0.01);
        
        // Train for 30 episodes
        int training_episodes = 30;
        for (int episode = 0; episode < training_episodes; ++episode) {
            EpidemicState state = env.reset();
            int steps = 0;
            
            while (!env.isDone() && steps < 150) {
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
        
        EpidemicState state = env.reset();
        for (int i = 0; i < 50; ++i) {
            EpidemicAction action = agent.selectAction(state);
            auto result = env.step(action);
            state = std::get<0>(result);
        }
        
        auto inference_end = std::chrono::high_resolution_clock::now();
        auto inference_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            inference_end - inference_start).count();
        
        double episodes_per_second = (double)training_episodes / (training_duration / 1000.0);
        
        results << pop_size << "," << training_duration << "," 
               << inference_duration << "," << episodes_per_second << "\n";
    }
    
    results.close();
    std::cout << "Computational efficiency analysis completed\n";
}

/**
 * @brief Main experimental suite
 */
int main() {
    std::cout << "===============================================================\n";
    std::cout << "    COMPREHENSIVE RL EPIDEMIC CONTROL PAPER EXPERIMENTS\n";
    std::cout << "    Scientific Computing Research Paper - Version 2.0\n";
    std::cout << "===============================================================\n";
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        std::cout << "\nRunning experimental evaluation for research paper...\n";
        std::cout << "Estimated total time: ~15-20 minutes\n";
        
        experiment1_learningConvergence();
        experiment2_policyComparison();
        experiment3_robustnessAnalysis();
        experiment4_multiObjectiveAnalysis();
        experiment5_computationalEfficiency();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::minutes>(
            end_time - start_time).count();
        
        std::cout << "\n===============================================================\n";
        std::cout << "    PAPER EXPERIMENTS COMPLETED SUCCESSFULLY\n";
        std::cout << "    Total execution time: " << total_duration << " minutes\n";
        std::cout << "===============================================================\n";
        
        std::cout << "\nGenerated result files for paper analysis:\n";
        std::cout << "• exp1_learning_convergence.csv - Learning curve analysis\n";
        std::cout << "• exp2_policy_comparison.csv - Policy performance comparison\n";
        std::cout << "• exp3_robustness_analysis.csv - Robustness to scenarios\n";
        std::cout << "• exp4_multiobjective_analysis.csv - Multi-objective trade-offs\n";
        std::cout << "• exp5_computational_efficiency.csv - Computational performance\n";
        
        std::cout << "\nRecommended paper sections:\n";
        std::cout << "1. Results: Learning Convergence (Exp 1)\n";
        std::cout << "2. Results: Policy Comparison (Exp 2)\n";
        std::cout << "3. Results: Robustness Analysis (Exp 3)\n";
        std::cout << "4. Results: Multi-Objective Optimization (Exp 4)\n";
        std::cout << "5. Discussion: Computational Efficiency (Exp 5)\n";
        
        std::cout << "\nNext steps for paper:\n";
        std::cout << "• Import CSV files into analysis software (R/Python/MATLAB)\n";
        std::cout << "• Generate publication-quality figures\n";
        std::cout << "• Perform statistical significance tests\n";
        std::cout << "• Calculate confidence intervals\n";
        
    } catch (const std::exception& e) {
        std::cerr << "\nError during experimental execution: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
