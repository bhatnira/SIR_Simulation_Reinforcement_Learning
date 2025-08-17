/**
 * @file RLExample.cpp
 * @brief Example of using RL framework for epidemic control policy optimization
 * @author Scientific Computing Team
 * @date 2025
 */

#include "BayesianSIR.h"
#include <iostream>
#include <iomanip>
#include <memory>

/**
 * @brief Demonstrate DQN training for epidemic control
 */
void demonstrateDQNTraining() {
    std::cout << "\n=== DQN Agent Training for Epidemic Control ===\n";
    
    // Create environment
    EpidemicEnvironment env(1000, 0.3, 0.1, 200);
    
    // Create DQN agent
    DQNAgent agent(13, 8, 0.001, 0.99, 1.0, 0.995, 0.01);
    
    // Create reward function
    auto reward_func = std::make_unique<EpidemicRewardFunction>(0.7, 0.2, 0.1);
    env.setRewardFunction(std::move(reward_func));
    
    std::cout << "Training DQN agent for 100 episodes...\n";
    
    int num_episodes = 100;
    std::vector<double> episode_rewards;
    
    for (int episode = 0; episode < num_episodes; ++episode) {
        EpidemicState state = env.reset();
        double total_reward = 0.0;
        int steps = 0;
        
        while (!env.isDone() && steps < 200) {
            // Select action
            EpidemicAction action = agent.selectAction(state);
            
            // Take step in environment
            auto [next_state, reward, done] = env.step(action);
            
            // Store experience
            agent.remember(state, action, reward, next_state, done);
            
            // Train agent
            if (steps > 32) {  // Start training after collecting some experiences
                agent.train();
            }
            
            total_reward += reward;
            state = next_state;
            steps++;
        }
        
        episode_rewards.push_back(total_reward);
        
        if (episode % 10 == 0) {
            double avg_reward = 0.0;
            for (int i = std::max(0, episode - 9); i <= episode; ++i) {
                avg_reward += episode_rewards[i];
            }
            avg_reward /= std::min(episode + 1, 10);
            
            std::cout << "Episode " << std::setw(3) << episode 
                     << " | Avg Reward: " << std::fixed << std::setprecision(2) 
                     << avg_reward << " | Steps: " << steps << std::endl;
        }
    }
    
    std::cout << "Training completed!\n";
    
    // Save trained model
    agent.saveModel("dqn_epidemic_control.model");
    std::cout << "Model saved to dqn_epidemic_control.model\n";
}

/**
 * @brief Demonstrate policy evaluation
 */
void demonstratePolicyEvaluation() {
    std::cout << "\n=== Policy Evaluation ===\n";
    
    // Create environment
    EpidemicEnvironment env(1000, 0.3, 0.1, 200);
    
    // Create and load trained agent
    DQNAgent agent(13, 8);
    try {
        agent.loadModel("dqn_epidemic_control.model");
        std::cout << "Loaded trained model\n";
    } catch (const std::exception& e) {
        std::cout << "Could not load model, using untrained agent\n";
    }
    
    // Create reward function
    auto reward_func = std::make_unique<EpidemicRewardFunction>(0.7, 0.2, 0.1);
    env.setRewardFunction(std::move(reward_func));
    
    std::cout << "Evaluating policy over 10 episodes...\n";
    
    std::vector<double> evaluation_rewards;
    
    for (int episode = 0; episode < 10; ++episode) {
        EpidemicState state = env.reset();
        double total_reward = 0.0;
        int steps = 0;
        
        std::cout << "\nEpisode " << episode + 1 << ":\n";
        std::cout << "Step | S_frac| I_frac| R_frac| H_frac|ICU_occ|Hosp_c| Day | Action\n";
        std::cout << "-----+-------+-------+-------+-------+-------+------+-----+-------\n";
        
        while (!env.isDone() && steps < 200) {
            // Select action using trained policy (no exploration)
            EpidemicAction action = agent.selectAction(state);
            
            // Display state and action
            if (steps % 20 == 0) {  // Display every 20 steps
                std::cout << std::setw(4) << steps << " |"
                         << std::setw(6) << std::fixed << std::setprecision(3) << state.susceptible_fraction << "|"
                         << std::setw(6) << std::fixed << std::setprecision(3) << state.infected_fraction << "|"
                         << std::setw(6) << std::fixed << std::setprecision(3) << state.recovered_fraction << "|"
                         << std::setw(6) << std::fixed << std::setprecision(3) << state.hospitalized_fraction << "|"
                         << std::setw(6) << std::fixed << std::setprecision(3) << state.icu_occupancy << "|"
                         << std::setw(6) << std::fixed << std::setprecision(3) << state.hospital_capacity << "|"
                         << std::setw(4) << state.day << "|"
                         << " Actions: ";
                
                // Display action summary
                for (const auto& intervention : action.interventions) {
                    if (intervention.second > 0.1) {
                        std::cout << intervention.first << ":" << std::fixed << std::setprecision(1) << intervention.second << " ";
                    }
                }
                std::cout << std::endl;
            }
            
            // Take step in environment
            auto [next_state, reward, done] = env.step(action);
            
            total_reward += reward;
            state = next_state;
            steps++;
        }
        
        evaluation_rewards.push_back(total_reward);
        std::cout << "Total reward: " << std::fixed << std::setprecision(2) 
                 << total_reward << " | Final infected: " << std::fixed << std::setprecision(3) 
                 << state.infected_fraction << std::endl;
    }
    
    // Calculate statistics
    double mean_reward = 0.0;
    for (double reward : evaluation_rewards) {
        mean_reward += reward;
    }
    mean_reward /= evaluation_rewards.size();
    
    double std_reward = 0.0;
    for (double reward : evaluation_rewards) {
        std_reward += (reward - mean_reward) * (reward - mean_reward);
    }
    std_reward = std::sqrt(std_reward / evaluation_rewards.size());
    
    std::cout << "\nEvaluation Results:\n";
    std::cout << "Mean reward: " << std::fixed << std::setprecision(2) << mean_reward << " ± " << std_reward << std::endl;
}

/**
 * @brief Compare different intervention strategies
 */
void compareInterventionStrategies() {
    std::cout << "\n=== Intervention Strategy Comparison ===\n";
    
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
    moderate_action.interventions[EpidemicAction::VACCINATION_PRIORITY] = 2.0;
    moderate_action.interventions[EpidemicAction::CONTACT_TRACING] = 0.5;
    strategies.emplace_back("Moderate Intervention", moderate_action);
    
    // Strict lockdown
    EpidemicAction strict_action;
    strict_action.interventions[EpidemicAction::LOCKDOWN_INTENSITY] = 1.0;
    strict_action.interventions[EpidemicAction::MASK_MANDATE] = 1.0;
    strict_action.interventions[EpidemicAction::SCHOOL_CLOSURE] = 1.0;
    strict_action.interventions[EpidemicAction::BORDER_CONTROL] = 1.0;
    strict_action.interventions[EpidemicAction::CONTACT_TRACING] = 0.8;
    strategies.emplace_back("Strict Lockdown", strict_action);
    
    // Vaccination focus
    EpidemicAction vacc_action;
    vacc_action.interventions[EpidemicAction::VACCINATION_PRIORITY] = 4.0;
    vacc_action.interventions[EpidemicAction::MASK_MANDATE] = 1.0;
    vacc_action.interventions[EpidemicAction::TESTING_EXPANSION] = 1.5;
    strategies.emplace_back("Vaccination Focus", vacc_action);
    
    std::cout << "Strategy               |Inf_frac| Economic Cost | Social Impact | Total Reward\n";
    std::cout << "----------------------+--------+---------------+---------------+-------------\n";
    
    for (const auto& [strategy_name, base_action] : strategies) {
        // Create environment
        EpidemicEnvironment env(1000, 0.3, 0.1, 200);
        auto reward_func = std::make_unique<EpidemicRewardFunction>(0.7, 0.2, 0.1);
        env.setRewardFunction(std::move(reward_func));
        
        // Simulate with fixed strategy
        EpidemicState state = env.reset();
        double total_reward = 0.0;
        double total_economic_cost = 0.0;
        double total_social_impact = 0.0;
        
        while (!env.isDone()) {
            auto [next_state, reward, done] = env.step(base_action);
            total_reward += reward;
            total_economic_cost += (1.0 - next_state.economic_activity);
            total_social_impact += (1.0 - next_state.mobility_index);
            state = next_state;
        }
        
        std::cout << std::setw(22) << strategy_name << "|"
                 << std::setw(7) << std::fixed << std::setprecision(3) << state.infected_fraction << "|"
                 << std::setw(14) << std::fixed << std::setprecision(1) << total_economic_cost << "|"
                 << std::setw(14) << std::fixed << std::setprecision(1) << total_social_impact << "|"
                 << std::setw(12) << std::fixed << std::setprecision(1) << total_reward
                 << std::endl;
    }
}

int main() {
    std::cout << "=======================================================\n";
    std::cout << "   Reinforcement Learning for Epidemic Control\n";
    std::cout << "   Scientific Computing Project - Version 2.0\n";
    std::cout << "=======================================================\n";
    
    try {
        // Demonstrate different aspects of the RL framework
        demonstrateDQNTraining();
        demonstratePolicyEvaluation();
        compareInterventionStrategies();
        
        std::cout << "\n=== Analysis Complete ===\n";
        std::cout << "The RL framework provides:\n";
        std::cout << "• Adaptive policy learning for epidemic control\n";
        std::cout << "• Multi-objective optimization (health, economic, social)\n";
        std::cout << "• Comparison of intervention strategies\n";
        std::cout << "• Real-time policy adaptation based on epidemic state\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
