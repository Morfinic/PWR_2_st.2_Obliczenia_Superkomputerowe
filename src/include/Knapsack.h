#pragma once
#include <vector>
#include <random>
#include "Item.h"
#include "KnapsackGA.h"

class Knapsack
{
    private:
        KnapsackGA::Items items;
        int capacity;
    public:
        Knapsack(int capacity);
        void addItem(Item& item);
        int getCapacity();
        int getItemCount();
        KnapsackGA::Items getItems();
        KnapsackGA::KnapsackEvaluation evaluate(KnapsackGA::Solution& solution);
        static Knapsack generateRandom(int numItems, int maxWeight, int maxValue, int capacity, bool fixed=false);
        void printSolution(KnapsackGA::Solution& solution);
};
