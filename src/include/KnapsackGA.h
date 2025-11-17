#pragma once

#include <vector>
#include "Item.h"

namespace KnapsackGA{
    typedef std::vector<Item> Items;
    typedef std::pair<int, int> KnapsackEvaluation;
    typedef std::vector<bool> Solution;
}