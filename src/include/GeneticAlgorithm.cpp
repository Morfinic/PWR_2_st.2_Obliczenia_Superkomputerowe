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
{};

// G - Liczba generacji
// P - Rozmiar populacji
// N - Rozmiar problemu
// T - Rozmiar turnieju
// k - liczba wątków procesora

std::vector<KnapsackGA::Solution>GeneticAlgorithm::generateInitialPopulation(){
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

int GeneticAlgorithm::calculateFitness(KnapsackGA::Solution& chromosome){
    auto [totalWeight, totalValue] = this->knapsack.evaluate(chromosome);

    if(totalWeight > knapsack.getCapacity()){
        return 0;
    }

    return totalValue;
}

KnapsackGA::Solution GeneticAlgorithm::tournamentSelection(std::vector<KnapsackGA::Solution>& population, std::vector<int> fitnesses){
    std::uniform_int_distribution<> indexDist(0, this->populationSize - 1);
    KnapsackGA::Solution bestChromosome;
    int bestFitness = -1;

    for (int i = 0; i < this->tournamentSize; i++)
    {
        int index = indexDist(this->generator);
        KnapsackGA::Solution chromosome = population[index];
        int fitness = fitnesses[i];
        if (fitness <= bestFitness)
            continue;
        bestFitness = fitness;
        bestChromosome = chromosome;
    }
    return bestChromosome;
}

std::pair<KnapsackGA::Solution, KnapsackGA::Solution> GeneticAlgorithm::crossover(KnapsackGA::Solution& parent1, KnapsackGA::Solution& parent2){
    std::uniform_int_distribution<> pointDist(1, this->chromosomeLength);

    int point = pointDist(this->generator);
    KnapsackGA::Solution child1 = parent1;
    KnapsackGA::Solution child2 = parent2;

    for (int i = point; i < this->chromosomeLength; i++)
    {
        // std::swap(child1[i], child2[i]);
        std::vector<bool>::swap(child1[i], child2[i]);
    }
    return {child1, child2};
}

void GeneticAlgorithm::mutate(KnapsackGA::Solution& chromosome){
    std::uniform_real_distribution<> probabilityDist(0.0, 1.0);
    for (int i = 0; i < this->chromosomeLength; i++)
    {
        if(probabilityDist(this->generator) >= this->mutationRate)
            continue;
        chromosome[i] = !chromosome[i];
    }
}

std::pair<KnapsackGA::Solution, KnapsackGA::Solution> GeneticAlgorithm::solve(bool verbose, bool log, bool isParallel){
    std::vector<KnapsackGA::Solution> population = this->generateInitialPopulation();
    KnapsackGA::Solution bestSolution;
    KnapsackGA::Solution bestLastGenSolution;
    int bestFitness = -1;
    std::vector<int> convergence;

    // Działa G razy
    for (int generation = 0; generation < this->maxGenerations; generation++)
    {
        std::vector<int> fitnesses(this->populationSize);
        int bestGenFitness = -1;
        KnapsackGA::Solution bestGenSolution;

        // Tu można dać schedule(typ, chunk_size)
        // np. schedule(static, 64)
        // Działa P * N
        #pragma omp parallel for schedule(static) if(isParallel)
        for (int i = 0; i < this->populationSize; ++i) // O(P)
        {
            int fitness = this->calculateFitness(population[i]); // O(N)
            fitnesses[i] = fitness;
        }

        int bestGenIndex = -1;
        // Działa P razy
        for (int i = 0; i < this->populationSize; ++i) {
            if (fitnesses[i] > bestGenFitness) {
                bestGenFitness = fitnesses[i];
                bestGenIndex = i;
            }
        }

        bestGenSolution = population[bestGenIndex];

        if(bestGenFitness > bestFitness){
            bestFitness = bestGenFitness;
            bestSolution = bestGenSolution;
        }

        std::vector<KnapsackGA::Solution> newPopulation(this->populationSize);
        newPopulation[0] = bestGenSolution;

        // Działa P/2 razy
        #pragma omp parallel for if(isParallel)
        for(int i = 1; i < this->populationSize; i += 2){
            // Działa T razy
            KnapsackGA::Solution parent1 = this->tournamentSelection(population, fitnesses);
            // Działa T razy
            KnapsackGA::Solution parent2 = this->tournamentSelection(population, fitnesses);
            // 2*O(T)

            // O(N)
            auto [child1, child2] = this->crossover(parent1, parent2);


            // N
            mutate(child1);
            // N
            mutate(child2);
            // mutate - 2*O(N)

            newPopulation[i] = child1;
            if(i + 1 < this->populationSize){
                newPopulation[i+1] = child2;
            }
        }

        population = newPopulation;
        bestLastGenSolution = bestGenSolution;
        convergence.push_back(bestGenFitness);
    }

    return {bestSolution, bestLastGenSolution};
}
