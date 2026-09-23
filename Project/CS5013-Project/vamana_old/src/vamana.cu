#include <time.h>
#include <cstring>
#include <iostream>

#ifndef VAMANA_H
#include "vamana.h"
#endif

#include "graph.cuh"

#include <stdio.h>
#include <stdlib.h>

// handles the batch vamana
// d_queryVecs contains 10000 128 dim vectors
void vamanaInner(uint8_t* d_graph, float* d_queryVecs, float alpha, unsigned batchStart,
                 unsigned batchSize) {
    unsigned* d_visitedSets;
    unsigned* d_visitedSetCount;
    uint8_t*  d_reverseEdgeIndex;

    CPUTimer cputimer;

    cputimer.Start();
    gpuErrchk(cudaMalloc(&d_visitedSets, batchSize * MAX_PARENTS_PERQUERY * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_visitedSetCount, batchSize * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_reverseEdgeIndex, N * reverseIndexEntrySize * sizeof(uint8_t)));
    gpuErrchk(cudaMemset(d_visitedSetCount, 0, batchSize * sizeof(unsigned)));
    cputimer.Stop();
    printf("vamanaInner mallocs: %f sec\n", cputimer.Elapsed());

    cputimer.Start();
    greedySearch(d_graph, d_queryVecs, d_visitedSets, d_visitedSetCount, batchStart, batchSize);
    cputimer.Stop();
    printf("greedySearch: %f sec\n", cputimer.Elapsed());

    cputimer.Start();

    computeOutNeighbors(d_graph, d_queryVecs, d_visitedSets, d_visitedSetCount, alpha,
                        d_reverseEdgeIndex, batchStart, batchSize);
    cputimer.Stop();
    printf("computeOutNeighbors: %f sec\n", cputimer.Elapsed());

    // if (batchStart == 100000) {
    //     uint8_t *reverseEdgeIndex = (uint8_t*)malloc(N * reverseIndexEntrySize *
    //     sizeof(uint8_t)); cudaMemcpy(reverseEdgeIndex, d_reverseEdgeIndex, N *
    //     reverseIndexEntrySize, cudaMemcpyDeviceToHost); FILE *outFile =
    //     fopen("build/reverseIndex.bin", "wb"); if (!outFile) {
    //         printf("Could not open output file.\n");
    //         return;
    //     }
    //     fwrite(reverseEdgeIndex, reverseIndexEntrySize, N, outFile);
    //     free(reverseEdgeIndex);
    // }

    cputimer.Start();
    computeReverseEdges(d_graph, d_reverseEdgeIndex, alpha);
    cputimer.Stop();
    printf("computeReverseEdges: %f sec\n", cputimer.Elapsed());

    cputimer.Start();
    gpuErrchk(cudaFree(d_visitedSets));
    gpuErrchk(cudaFree(d_visitedSetCount));
    gpuErrchk(cudaFree(d_reverseEdgeIndex));
    cputimer.Stop();
    printf("vamanaInner frees: %f sec\n", cputimer.Elapsed());
}

void vamanaOuter(Graph_t& graph_struct, float alpha) {
    uint8_t* graph = graph_struct.graph;

    int batchSize = 10000;

    // uint8_t* d_graph;
    uint8_t* d_graph = graph_struct.d_graph;
    float*   queryVecs;
    float*   d_queryVecs;

    gpuErrchk(cudaMalloc(&d_graph, N * graphEntrySize * sizeof(uint8_t)));
    graph_struct.d_graph_capacity = N;
    graph_struct.d_graph_size     = N;

    gpuErrchk(cudaMalloc(&d_queryVecs, batchSize * D * sizeof(float)));
    queryVecs = (float*)malloc(batchSize * D * sizeof(float));
    gpuErrchk(
        cudaMemcpy(d_graph, graph, N * graphEntrySize * sizeof(uint8_t), cudaMemcpyHostToDevice));

    for (int batchStart = 0; batchStart < N; batchStart += batchSize) {
        if (batchStart + batchSize > N)
            batchSize = N - batchStart;
        printf("Starting iteration: %d to %d.\n", batchStart, batchStart + batchSize - 1);

        // Load query vectors
        for (int i = 0; i < batchSize; i++) {
            float* queryVec = (float*)(graph + (batchStart + i) * graphEntrySize);
            for (int j = 0; j < D; j++) {
                queryVecs[i * D + j] = queryVec[j];
            }
        }
        // Copy query vectors to GPU
        gpuErrchk(cudaMemcpy(d_queryVecs, queryVecs, batchSize * D * sizeof(float),
                             cudaMemcpyHostToDevice));
        vamanaInner(d_graph, d_queryVecs, alpha, batchStart, batchSize);
    }

    // Copy graph back onto CPU
    gpuErrchk(
        cudaMemcpy(graph, d_graph, N * graphEntrySize * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    // cudaFree(d_graph);
    cudaFree(d_queryVecs);
}

void driverFn(char* graphFilePath, char* basePointsPath, char* outFilePath, Graph_t& graph_struct) {
    CPUTimer cputimer;
    cputimer.Start();

    graph_struct.graph          = (uint8_t*)calloc(N, graphEntrySize);
    graph_struct.graph_capacity = N;

    if (!graph_struct.graph) {
        printf("Could not allocate memory for graph.\n");
        return;
    }
    // already zeroed out
    // memset(graph_struct.graph, 0, N * graphEntrySize);  // Zero out the graph

    // Read graph from file
    FILE* graphFile = fopen(graphFilePath, "rb");
    if (!graphFile) {
        printf("Could not open graph file.\n");
        return;
    }
    fread(graph_struct.graph, graphEntrySize, NUM_QUERIES, graphFile);
    fclose(graphFile);

    graph_struct.graph_size = NUM_QUERIES;

    /*
    graph is a random graph of size 10000
    graph has format

    struct node {
        float vec[128];
        uint degree;
        uint neighbors[degree];
    } node_t;

    basepoints is all the points in the graph
    float basepoinst[N][128];
    */

    // #define get_base
#if defined(get_base)
    // Read basepoints
    unsigned numBasePoints  = N - NUM_QUERIES;
    float*   basePoints     = (float*)malloc(numBasePoints * D * sizeof(float));
    FILE*    basePointsFile = fopen(basePointsPath, "rb");
    if (!basePointsFile) {
        printf("Could not open basepoints file.\n");
        return;
    }
    fseek(basePointsFile, 8, SEEK_CUR);  // SKip first 8 bytes
    fread(basePoints, D * sizeof(float), numBasePoints, basePointsFile);
    fclose(basePointsFile);

    cputimer.Stop();
    printf("Reading graph and basepoints: %f sec\n", cputimer.Elapsed());

    cputimer.Start();

    // Copy coordinates to graph
    // since NUM_QUERIES = N = 10,000 this loop doesnt run
    // It just puts the extra basepoints into the 'graph' without adding neighbors.
    for (int i = NUM_QUERIES; i < N; i++) {
        float* queryVec = (float*)(graph + i * graphEntrySize);
        for (int j = 0; j < D; j++) {
            queryVec[j] = basePoints[i * D + j];
        }
    }

#endif

    cputimer.Stop();
    printf("Copying query vectors: %f sec\n", cputimer.Elapsed());

    // generateRandomGraph(graph, 0, 10000);

    // unsigned batchSize = 10000;
    // for (int batchStart = 0; batchStart < N; batchStart += batchSize) {
    //     generateRandomGraph(graph, batchStart, batchSize);
    // }

    cputimer.Start();
    vamanaOuter(graph_struct, 1.5);
    cputimer.Stop();
    printf("Vamana: %f sec\n", cputimer.Elapsed());

    cputimer.Start();

    FILE* outFile = fopen(outFilePath, "wb");
    if (!outFile) {
        printf("Could not open output file.\n");
        return;
    }

    fwrite(graph_struct.graph, graphEntrySize, N, outFile);

    cputimer.Stop();
    printf("Writing graph to file: %f sec\n", cputimer.Elapsed());
    fclose(outFile);
}

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("Usage: %s <graph> <basepoints> <output>\n", argv[0]);
        return 1;
    }
    Graph_t graph_struct;
    driverFn(argv[1], argv[2], argv[3], graph_struct);

    std::cout << graph_struct.graph_capacity << " " << graph_struct.graph_size << '\n';
    std::cout << graph_struct.d_graph_capacity << " " << graph_struct.d_graph_size << '\n';

    // we have the graph_struct with all data. we just need to implement insert/delete methods.
    // add workfloat handler here;

    return 0;
}
