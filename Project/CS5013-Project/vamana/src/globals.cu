#include "globals.cuh"

namespace FreshVamana::Globals {

__managed__ uint d_graph_capacity_g = 0;
__managed__ uint d_graph_size_g     = 0;

uint*                         d_delete_list_g = nullptr;
FreshVamana::Consts::dtype_g* d_insert_list_g = nullptr;

}  // namespace FreshVamana::Globals
