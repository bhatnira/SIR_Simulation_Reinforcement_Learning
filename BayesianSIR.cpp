/**
 * @file BayesianSIR.cpp
 * @brief Implementation of Reinforcement Learning framework for epidemic control
 * @author Scientific Computing Team
 * @date 2025
 */

#include "BayesianSIR.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cassert>
#include <fstream>
#include <iomanip>

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

double BayesianSIRSampler::logPrior(const BayesianParameters& params) const {
    double logPriorProb = 0.0;
    
    // Check bounds and calculate log prior for each parameter
    logPriorProb += logPriorDensity(params.transmissionRate, transmissionPrior);
    logPriorProb += logPriorDensity(params.recoveryRate, recoveryPrior);
    logPriorProb += logPriorDensity(params.initialInfectedFrac, initialInfectedPrior);
    
    return logPriorProb;
}

double BayesianSIRSampler::logLikelihood(const BayesianParameters& params, 
                                        const EpidemicData& data) const {
    // Run SIR simulation with current parameters
    auto simResults = simulateSIR(params, data.totalPopulation, 
                                 *std::max_element(data.days.begin(), data.days.end()) + 1);
    
    double logLik = 0.0;
    
    // Calculate likelihood assuming Gaussian observation error
    for (size_t i = 0; i < data.size(); ++i) {
        int day = data.days[i];
        if (day < static_cast<int>(simResults.size())) {
            // Compare simulation results with observations
            double errorS = data.susceptible[i] - simResults[day][0];
            double errorI = data.infected[i] - simResults[day][1];
            double errorR = data.recovered[i] - simResults[day][2];
            
            double variance = data.observationError[i];
            
            // Log-likelihood contribution (assuming independence)
            logLik -= 0.5 * (errorS * errorS + errorI * errorI + errorR * errorR) / variance;
            logLik -= 1.5 * std::log(2.0 * M_PI * variance); // Normalization constant
        }
    }
    
    return logLik;
}

std::vector<std::array<int, 3>> BayesianSIRSampler::simulateSIR(
    const BayesianParameters& params, int populationSize, int days) const {
    
    std::vector<std::array<int, 3>> results(days);
    
    // Initialize compartments
    int initialInfected = static_cast<int>(params.initialInfectedFrac * populationSize);
    int susceptible = populationSize - initialInfected;
    int infected = initialInfected;
    int recovered = 0;
    
    results[0] = {susceptible, infected, recovered};
    
    // Simple deterministic SIR dynamics for likelihood calculation
    for (int day = 1; day < days; ++day) {
        double beta = params.transmissionRate;
        double gamma = params.recoveryRate;
        
        // Calculate rates
        double infectionRate = beta * susceptible * infected / static_cast<double>(populationSize);
        double recoveryRate = gamma * infected;
        
        // Update compartments (Euler method for ODE approximation)
        int newInfections = static_cast<int>(std::round(infectionRate));
        int newRecoveries = static_cast<int>(std::round(recoveryRate));
        
        // Ensure we don't exceed population bounds
        newInfections = std::min(newInfections, susceptible);
        newRecoveries = std::min(newRecoveries, infected);
        
        susceptible -= newInfections;
        infected += newInfections - newRecoveries;
        recovered += newRecoveries;
        
        results[day] = {susceptible, infected, recovered};
    }
    
    return results;
}

BayesianParameters BayesianSIRSampler::proposeParameters(const BayesianParameters& current) const {
    BayesianParameters proposed = current;
    
    // Add random walk proposals
    proposed.transmissionRate += normal_dist(rng) * std::sqrt(current.transmissionRateVar);
    proposed.recoveryRate += normal_dist(rng) * std::sqrt(current.recoveryRateVar);
    proposed.initialInfectedFrac += normal_dist(rng) * std::sqrt(current.initialInfectedFracVar);
    
    // Ensure parameters stay in valid ranges
    proposed.transmissionRate = std::max(0.0, proposed.transmissionRate);
    proposed.recoveryRate = std::max(0.0, proposed.recoveryRate);
    proposed.initialInfectedFrac = std::max(0.0, std::min(1.0, proposed.initialInfectedFrac));
    
    return proposed;
}

double BayesianSIRSampler::sampleFromPrior(const PriorDistribution& prior) const {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    
    switch (prior.type) {
        case PriorDistribution::UNIFORM:
            return uniform(rng) * (prior.param2 - prior.param1) + prior.param1;
            
        case PriorDistribution::NORMAL: {
            std::normal_distribution<double> normal(prior.param1, std::sqrt(prior.param2));
            return normal(rng);
        }
        
        case PriorDistribution::BETA: {
            std::gamma_distribution<double> gamma1(prior.param1, 1.0);
            std::gamma_distribution<double> gamma2(prior.param2, 1.0);
            double x = gamma1(rng);
            double y = gamma2(rng);
            return x / (x + y);
        }
        
        case PriorDistribution::GAMMA: {
            std::gamma_distribution<double> gamma(prior.param1, 1.0 / prior.param2);
            return gamma(rng);
        }
        
        case PriorDistribution::EXPONENTIAL: {
            std::exponential_distribution<double> exp(prior.param1);
            return exp(rng);
        }
        
        default:
            return 0.0;
    }
}

double BayesianSIRSampler::logPriorDensity(double value, const PriorDistribution& prior) const {
    switch (prior.type) {
        case PriorDistribution::UNIFORM:
            if (value >= prior.param1 && value <= prior.param2) {
                return -std::log(prior.param2 - prior.param1);
            }
            return -std::numeric_limits<double>::infinity();
            
        case PriorDistribution::NORMAL: {
            double variance = prior.param2;
            double diff = value - prior.param1;
            return -0.5 * diff * diff / variance - 0.5 * std::log(2.0 * M_PI * variance);
        }
        
        case PriorDistribution::BETA: {
            if (value <= 0.0 || value >= 1.0) {
                return -std::numeric_limits<double>::infinity();
            }
            double alpha = prior.param1;
            double beta = prior.param2;
            return (alpha - 1.0) * std::log(value) + (beta - 1.0) * std::log(1.0 - value) +
                   std::lgamma(alpha + beta) - std::lgamma(alpha) - std::lgamma(beta);
        }
        
        case PriorDistribution::GAMMA: {
            if (value <= 0.0) {
                return -std::numeric_limits<double>::infinity();
            }
            double shape = prior.param1;
            double rate = prior.param2;
            return (shape - 1.0) * std::log(value) - rate * value + 
                   shape * std::log(rate) - std::lgamma(shape);
        }
        
        case PriorDistribution::EXPONENTIAL: {
            if (value < 0.0) {
                return -std::numeric_limits<double>::infinity();
            }
            return std::log(prior.param1) - prior.param1 * value;
        }
        
        default:
            return 0.0;
    }
}

std::vector<BayesianParameters> BayesianSIRSampler::sample(const EpidemicData& data, 
                                                          const BayesianParameters& initial) {
    std::vector<BayesianParameters> samples;
    samples.reserve((totalSamples - burnInSamples) / thinning);
    
    BayesianParameters current = initial;
    double currentLogPosterior = logPrior(current) + logLikelihood(current, data);
    
    acceptedTransmission = acceptedRecovery = acceptedInitialInfected = totalProposals = 0;
    
    std::cout << "Starting MCMC sampling..." << std::endl;
    std::cout << "Burn-in: " << burnInSamples << " samples" << std::endl;
    std::cout << "Total: " << totalSamples << " samples" << std::endl;
    std::cout << "Thinning: " << thinning << std::endl;
    
    for (int iter = 0; iter < totalSamples; ++iter) {
        // Propose new parameters
        BayesianParameters proposed = proposeParameters(current);
        double proposedLogPosterior = logPrior(proposed) + logLikelihood(proposed, data);
        
        // Metropolis-Hastings acceptance
        double logAcceptanceRatio = proposedLogPosterior - currentLogPosterior;
        
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        if (std::log(uniform(rng)) < logAcceptanceRatio) {
            // Track acceptance for individual parameters before updating
            if (proposed.transmissionRate != current.transmissionRate) acceptedTransmission++;
            if (proposed.recoveryRate != current.recoveryRate) acceptedRecovery++;
            if (proposed.initialInfectedFrac != current.initialInfectedFrac) acceptedInitialInfected++;
            
            current = proposed;
            currentLogPosterior = proposedLogPosterior;
        }
        
        totalProposals++;
        
        // Store sample after burn-in and according to thinning
        if (iter >= burnInSamples && (iter - burnInSamples) % thinning == 0) {
            samples.push_back(current);
        }
        
        // Progress reporting
        if ((iter + 1) % 1000 == 0) {
            std::cout << "Iteration " << (iter + 1) << "/" << totalSamples 
                     << " (acceptance rate: " << 100.0 * (acceptedTransmission + acceptedRecovery + acceptedInitialInfected) / (3.0 * totalProposals) 
                     << "%)" << std::endl;
        }
    }
    
    std::cout << "MCMC sampling completed!" << std::endl;
    std::cout << "Collected " << samples.size() << " posterior samples" << std::endl;
    
    return samples;
}

std::array<double, 3> BayesianSIRSampler::getAcceptanceRates() const {
    if (totalProposals == 0) return {0.0, 0.0, 0.0};
    
    return {
        static_cast<double>(acceptedTransmission) / totalProposals,
        static_cast<double>(acceptedRecovery) / totalProposals,
        static_cast<double>(acceptedInitialInfected) / totalProposals
    };
}

double BayesianSIRSampler::calculateModelEvidence(const EpidemicData& data, int numSamples) const {
    // Importance sampling estimate of marginal likelihood
    double logSum = -std::numeric_limits<double>::infinity();
    
    for (int i = 0; i < numSamples; ++i) {
        BayesianParameters params;
        params.transmissionRate = sampleFromPrior(transmissionPrior);
        params.recoveryRate = sampleFromPrior(recoveryPrior);
        params.initialInfectedFrac = sampleFromPrior(initialInfectedPrior);
        
        double logLik = logLikelihood(params, data);
        
        // Log-sum-exp trick for numerical stability
        if (std::isfinite(logLik)) {
            if (logSum == -std::numeric_limits<double>::infinity()) {
                logSum = logLik;
            } else {
                double maxLog = std::max(logSum, logLik);
                logSum = maxLog + std::log(std::exp(logSum - maxLog) + std::exp(logLik - maxLog));
            }
        }
    }
    
    return logSum - std::log(numSamples);
}

// Posterior Analysis implementation
PosteriorAnalysis::SummaryStats PosteriorAnalysis::summarizeParameter(const std::vector<double>& samples) {
    SummaryStats stats;
    
    if (samples.empty()) {
        return stats;
    }
    
    // Calculate basic statistics
    stats.mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    
    // Calculate variance and standard deviation
    double variance = 0.0;
    for (double sample : samples) {
        variance += (sample - stats.mean) * (sample - stats.mean);
    }
    variance /= (samples.size() - 1);
    stats.std = std::sqrt(variance);
    
    // Calculate quantiles
    std::vector<double> sortedSamples = samples;
    std::sort(sortedSamples.begin(), sortedSamples.end());
    
    stats.median = sortedSamples[sortedSamples.size() / 2];
    stats.q025 = sortedSamples[static_cast<size_t>(0.025 * sortedSamples.size())];
    stats.q975 = sortedSamples[static_cast<size_t>(0.975 * sortedSamples.size())];
    
    // Calculate effective sample size (simplified estimate)
    stats.effectiveN = effectiveSampleSize(samples);
    stats.rHat = 1.0; // Would need multiple chains for proper calculation
    
    return stats;
}

double PosteriorAnalysis::effectiveSampleSize(const std::vector<double>& chain) {
    // Simplified effective sample size calculation
    // In practice, would use more sophisticated autocorrelation analysis
    
    if (chain.size() < 10) return chain.size();
    
    // Calculate autocorrelation at lag 1
    double mean = std::accumulate(chain.begin(), chain.end(), 0.0) / chain.size();
    
    double autoCorr = 0.0;
    double variance = 0.0;
    
    for (size_t i = 0; i < chain.size() - 1; ++i) {
        autoCorr += (chain[i] - mean) * (chain[i + 1] - mean);
        variance += (chain[i] - mean) * (chain[i] - mean);
    }
    
    if (variance > 0) {
        autoCorr /= variance;
        return chain.size() / (1.0 + 2.0 * std::max(0.0, autoCorr));
    }
    
    return chain.size();
}

std::pair<double, double> PosteriorAnalysis::credibleInterval(const std::vector<double>& samples, 
                                                            double probability) {
    std::vector<double> sortedSamples = samples;
    std::sort(sortedSamples.begin(), sortedSamples.end());
    
    double alpha = (1.0 - probability) / 2.0;
    size_t lowerIndex = static_cast<size_t>(alpha * sortedSamples.size());
    size_t upperIndex = static_cast<size_t>((1.0 - alpha) * sortedSamples.size());
    
    upperIndex = std::min(upperIndex, sortedSamples.size() - 1);
    
    return {sortedSamples[lowerIndex], sortedSamples[upperIndex]};
}

double PosteriorAnalysis::posteriorProbability(const std::vector<double>& samples, 
                                              double threshold, bool greater) {
    int count = 0;
    for (double sample : samples) {
        if ((greater && sample > threshold) || (!greater && sample < threshold)) {
            count++;
        }
    }
    return static_cast<double>(count) / samples.size();
}
