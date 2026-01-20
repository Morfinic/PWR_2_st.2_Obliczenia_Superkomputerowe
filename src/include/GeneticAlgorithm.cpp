#include "GeneticAlgorithm.h"
#include <omp.h>

GeneticAlgorithm::GeneticAlgorithm(
    Knapsack& knapsack,
    int populationSize,
    double mutationRate,
    int tournamentSize,
    int maxGenerations
) :
    knapsack(knapsack),
    populationSize(populationSize),
    chromosomeLength(knapsack.getItemCount()),
    mutationRate(mutationRate),
    tournamentSize(tournamentSize),
    maxGenerations(maxGenerations),
    generator(std::random_device{}())
{
    this->capacity = knapsack.getCapacity();
    auto items = knapsack.getItems();
    this->itemWeights.reserve(items.size());
    this->itemValues.reserve(items.size());

    for(const auto& item : items) {
        this->itemWeights.push_back(item.weight);
        this->itemValues.push_back(item.value);
    }
};

// G - Liczba generacji
// P - Rozmiar populacji
// N - Rozmiar problemu
// T - Rozmiar turnieju
// p - współczynnik mutacji
// Złożoność całkowita: O(G * P * N)

// O(P * N)
std::vector<KnapsackGA::Solution> GeneticAlgorithm::generateInitialPopulation(){
    std::vector<KnapsackGA::Solution> population(this->populationSize);
    std::uniform_int_distribution<> boolDist(0, 1);

    for (int i = 0; i < this->populationSize; i++)
    {
        KnapsackGA::Solution chromosome(this->chromosomeLength);
        for (int j = 0; j < this->chromosomeLength; j++)
        {
            chromosome[j] = boolDist(this->generator);
        }
        population[i] = chromosome;
    }
    
    return population;
}

// O(N)
int GeneticAlgorithm::calculateFitness(KnapsackGA::Solution& chromosome){
    int totalWeight = 0;
    int totalValue = 0;

    for (int i = 0; i < this->chromosomeLength; i++)
    {
        if(chromosome[i]) {
            totalWeight += this->itemWeights[i];
            totalValue += this->itemValues[i];
        }
    }

    if(totalWeight > this->capacity){
        return 0;
    }

    return totalValue;
}

// O(T)
KnapsackGA::Solution GeneticAlgorithm::tournamentSelection(std::vector<KnapsackGA::Solution>& population, std::vector<int> fitnesses){
    static thread_local std::mt19937 threadGen(std::random_device{}() + omp_get_thread_num());
    std::uniform_int_distribution<> dist(0, this->populationSize - 1);
    
    int bestFitness = -1;
    int bestIndex = -1;

    for(int i = 0; i < this->tournamentSize; i++){
        int index = dist(threadGen);
        if(fitnesses[index] > bestFitness){
            bestFitness = fitnesses[index];
            bestIndex = index;
        }
    }
    
    return population[bestIndex];
}

// O(N)
std::pair<KnapsackGA::Solution, KnapsackGA::Solution> GeneticAlgorithm::crossover(KnapsackGA::Solution& parent1, KnapsackGA::Solution& parent2){
    static thread_local std::mt19937 threadGen(std::random_device{}() + omp_get_thread_num());
    std::uniform_int_distribution<> dist(0, this->chromosomeLength - 1);
    int crossoverPoint = dist(threadGen);

    KnapsackGA::Solution child1 = parent1;
    KnapsackGA::Solution child2 = parent2;

    for(int i = crossoverPoint; i < this->chromosomeLength; i++){
        child1[i] = parent2[i];
        child2[i] = parent1[i];
    }

    return {child1, child2};
}

// O(N * p)
void GeneticAlgorithm::mutate(KnapsackGA::Solution& chromosome){
    static thread_local std::mt19937 threadGen(std::random_device{}() + omp_get_thread_num());
    std::geometric_distribution<int> jumpDist(this->mutationRate);
    
    int index = 0;
    while(index < this->chromosomeLength){
        index += jumpDist(threadGen);
        if(index < this->chromosomeLength){
            chromosome[index] = !chromosome[index];
            index++;
        }
    }
}

std::pair<KnapsackGA::Solution, KnapsackGA::Solution> GeneticAlgorithm::solve(bool verbose, bool log, bool isParallel){
    std::vector<KnapsackGA::Solution> population = this->generateInitialPopulation();
    
    std::vector<KnapsackGA::Solution> newPopulation(this->populationSize);
    std::vector<int> fitnesses(this->populationSize);
    std::vector<int> convergence;

    KnapsackGA::Solution bestSolution = population[0];
    KnapsackGA::Solution bestLastGenSolution = population[0];
    int bestFitness = -1;

    for(int g = 0; g < this->maxGenerations; g++){
        
        int bestGenFitness = -1;
        KnapsackGA::Solution bestGenSolution = population[0];

        // O(P * N)
        #pragma omp parallel for if(isParallel)
        for(int i = 0; i < this->populationSize; i++){
            fitnesses[i] = this->calculateFitness(population[i]);
            
            if (isParallel) {
                if(fitnesses[i] > bestGenFitness){
                    #pragma omp critical
                    {
                        if(fitnesses[i] > bestGenFitness){
                            bestGenFitness = fitnesses[i];
                            bestGenSolution = population[i];
                        }
                    }
                }
            } else {
                if(fitnesses[i] > bestGenFitness){
                    bestGenFitness = fitnesses[i];
                    bestGenSolution = population[i];
                }
            }
        }

        if(bestGenFitness > bestFitness){
            bestFitness = bestGenFitness;
            bestSolution = bestGenSolution;
        }

        newPopulation[0] = bestGenSolution;

        // Pętla Ewolucji: P/2 iteracji
        #pragma omp parallel for if(isParallel)
        for(int i = 1; i < this->populationSize; i += 2){
            // Selekcja 2 * O(T)
            KnapsackGA::Solution parent1 = this->tournamentSelection(population, fitnesses);
            KnapsackGA::Solution parent2 = this->tournamentSelection(population, fitnesses);

            // Krzyżowanie O(N)
            auto [child1, child2] = this->crossover(parent1, parent2);

            // Mutacja 2 * O(N * p)
            mutate(child1);
            mutate(child2);

            newPopulation[i] = child1;
            if(i + 1 < this->populationSize){
                newPopulation[i+1] = child2;
            }
        }
        
        // Double buffering O(1)
        std::swap(population, newPopulation);
        
        bestLastGenSolution = bestGenSolution;
        convergence.push_back(bestGenFitness);
    }

    return {bestSolution, bestLastGenSolution};
}