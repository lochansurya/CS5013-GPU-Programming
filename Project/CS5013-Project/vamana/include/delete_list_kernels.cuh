#pragma once

__global__ void checkIfNodeDeletedKernel(const uint* d_delete_list,
                                         uint        size,
                                         uint        node_id,
                                         bool*       d_found);
