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
    // Optymalizacja: Kopiujemy dane do prostych wektorów int, 
    // aby calculateFitness nie skakało po pamięci.
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
// k - liczba wątków procesora

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

int GeneticAlgorithm::calculateFitness(KnapsackGA::Solution& chromosome){
    // Zoptymalizowana wersja: operuje na lokalnych wektorach int
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

KnapsackGA::Solution GeneticAlgorithm::tournamentSelection(std::vector<KnapsackGA::Solution>& population, std::vector<int> fitnesses){
    // Używamy thread_local na wszelki wypadek, to nie szkodzi wersji single-thread,
    // a zabezpiecza przed błędami w przyszłości.
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

void GeneticAlgorithm::mutate(KnapsackGA::Solution& chromosome){
    static thread_local std::mt19937 threadGen(std::random_device{}() + omp_get_thread_num());
    std::uniform_real_distribution<> dist(0.0, 1.0);
    
    for(int i = 0; i < this->chromosomeLength; i++){
        if(dist(threadGen) < this->mutationRate){
            chromosome[i] = !chromosome[i];
        }
    }
}

std::pair<KnapsackGA::Solution, KnapsackGA::Solution> GeneticAlgorithm::solve(bool verbose, bool log, bool isParallel){
    std::vector<KnapsackGA::Solution> population = this->generateInitialPopulation();
    
    // Alokujemy raz, używamy wielokrotnie
    std::vector<KnapsackGA::Solution> newPopulation(this->populationSize);
    std::vector<int> fitnesses(this->populationSize);
    std::vector<int> convergence;

    // Inicjalizujemy czymkolwiek poprawnym, żeby uniknąć pustych wektorów
    KnapsackGA::Solution bestSolution = population[0];
    KnapsackGA::Solution bestLastGenSolution = population[0];
    int bestFitness = -1;

    for(int g = 0; g < this->maxGenerations; g++){
        
        // ZMIANA: -1 zamiast 0. Jeśli wszyscy mają fitness 0 (przepełnienie),
        // to i tak musimy kogoś wybrać jako "najlepszego z generacji", 
        // żeby nie przypisać pustego wektora.
        int bestGenFitness = -1;
        
        // Zabezpieczenie: domyślnie bierzemy pierwszego, żeby wektor nigdy nie był pusty
        KnapsackGA::Solution bestGenSolution = population[0];

        #pragma omp parallel for if(isParallel)
        for(int i = 0; i < this->populationSize; i++){
            fitnesses[i] = this->calculateFitness(population[i]);
            
            // Critical jest potrzebny tylko przy włączonym parallel
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
                // Wersja Single Thread (bez narzutu critical)
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
        // Tworzenie nowej Generacji
        
        population = newPopulation;
        bestLastGenSolution = bestGenSolution;
        convergence.push_back(bestGenFitness);
    }

    return {bestSolution, bestLastGenSolution};
}