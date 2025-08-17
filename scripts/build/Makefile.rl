# Simple Makefile for RL Experiments
CXX = g++
CXXFLAGS = -std=c++14 -Wall -Wextra -Wpedantic -O2

# Source files
RL_EXPERIMENTS_SOURCES = Person.cpp Population.cpp RLSIR.cpp RLExperiments_C14.cpp
RL_EXPERIMENTS_OBJECTS = $(RL_EXPERIMENTS_SOURCES:.cpp=.o)
RL_EXPERIMENTS_TARGET = rl_experiments

# Build target
rl-experiments: $(RL_EXPERIMENTS_OBJECTS)
	@echo "Linking $(RL_EXPERIMENTS_TARGET)..."
	$(CXX) $(CXXFLAGS) -o $(RL_EXPERIMENTS_TARGET) $(RL_EXPERIMENTS_OBJECTS)
	@echo "Build complete: $(RL_EXPERIMENTS_TARGET)"

# Object files
%.o: %.cpp
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Run experiments
run-experiments: $(RL_EXPERIMENTS_TARGET)
	@echo "Running RL experiments suite..."
	./$(RL_EXPERIMENTS_TARGET)

# Clean
clean:
	rm -f $(RL_EXPERIMENTS_OBJECTS) $(RL_EXPERIMENTS_TARGET)

.PHONY: rl-experiments run-experiments clean
