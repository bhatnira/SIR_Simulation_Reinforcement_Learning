/**
 * @file test_mcmc.cpp
 * @brief Simple MCMC diagnostic test
 */

#include "BayesianSIR.h"
#include <iostream>

int main() {
    std::cout << "Testing MCMC components..." << std::endl;
    
    // Create simple priors
    PriorDistribution betaPrior(PriorDistribution::GAMMA, 2.0, 1.0);      
    PriorDistribution gammaPrior(PriorDistribution::GAMMA, 2.0, 10.0);    
    PriorDistribution initPrior(PriorDistribution::BETA, 1.0, 99.0);      
    
    BayesianSIRSampler sampler(betaPrior, gammaPrior, initPrior, 100, 500, 1);
    
    // Test parameter proposal
    BayesianParameters test;
    test.transmissionRate = 2.0;
    test.recoveryRate = 0.2;
    test.initialInfectedFrac = 0.01;
    test.transmissionRateVar = 0.1;
    test.recoveryRateVar = 0.01;
    test.initialInfectedFracVar = 0.001;
    
    std::cout << "\nTesting parameter proposals:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        auto proposed = sampler.proposeParameters(test);
        std::cout << "Proposal " << i+1 << ": β=" << proposed.transmissionRate 
                  << ", γ=" << proposed.recoveryRate 
                  << ", init=" << proposed.initialInfectedFrac << std::endl;
    }
    
    // Test likelihood with simple data
    EpidemicData simpleData;
    simpleData.totalPopulation = 1000;
    simpleData.days = {0, 1, 2};
    simpleData.susceptible = {995, 990, 980};
    simpleData.infected = {5, 10, 15};
    simpleData.recovered = {0, 0, 5};
    simpleData.observationError = {100.0, 100.0, 100.0};
    
    double likelihood = sampler.logLikelihood(test, simpleData);
    std::cout << "\nLog-likelihood: " << likelihood << std::endl;
    
    double prior = sampler.logPrior(test);
    std::cout << "Log-prior: " << prior << std::endl;
    
    return 0;
}
