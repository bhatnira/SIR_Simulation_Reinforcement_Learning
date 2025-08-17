# Makefile for SIR Epidemic Simulation with RL
CXX = g++
CXXFLAGS = -std=c++14 -Wall -Wextra -O2

paper_experiments: Person.o Population.o RLSIR.o PaperExperiments.o
	$(CXX) $(CXXFLAGS) -o paper_experiments Person.o Population.o RLSIR.o PaperExperiments.o

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o paper_experiments *.csv *.model

run-paper: paper_experiments
	@echo "Running paper experiments..."
	./paper_experiments

.PHONY: clean run-paper
