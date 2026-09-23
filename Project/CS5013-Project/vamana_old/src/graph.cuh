
#pragma once

#include "vamana.h"

struct Graph_t {
    // probably needed just for the transfer at the start
    // since the graph will reside on gpu the whole time
    uint8_t* graph          = nullptr;
    uint     graph_size     = 0;  // number of graphEntrySize
    uint     graph_capacity = 0;

    uint8_t* d_graph          = nullptr;
    uint     d_graph_size     = 0;
    uint     d_graph_capacity = 0;

    // uint dimension = D;

    // static constexpr unsigned graphEntrySize =
    //     D * sizeof(float) + sizeof(unsigned) + R * sizeof(unsigned);
};
