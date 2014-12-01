// Author @ Eric Reinsmidt
// Date @ 2014.11.23
// Version 0.1

/*

  Driver cuda program for gameOfLife.cu

*/

#include <stdio.h>
#include <iostream>
#include "gameOfLife.cu"
using namespace std;

int main() {

    // Flag to output all generations to stdout
    bool showAll = false;

    // /Init number of generations to run
    int numGenerations = 100;

    // Create arrays for current and next generation on host
    char currentGeneration[65536];
    char theNextGeneration[65536];

    // Create pointers to current and next generation on device
    char *currentGenerationOnDevice;
    char *theNextGenerationOnDevice;

    // Fill automaton with empty cells
    for (int i = 0; i < 65536; i++) {
      currentGeneration[i] = 0;
    }

    // Place a strip of 10 vertical cells in middle of automaton
    // This is what will become the pentadecathlon oscillator
    for (int i = 123; i < 133; i++) {
      currentGeneration[i * 256 + 128] = 1;
    }

    // Output starting generation to stdout if flag set
    if (showAll) {
      cout << "Starting Generation:" << endl;
      outputAutomaton(currentGeneration);
    }

    // Allocate memory for current generation on device
    if (cudaMalloc((void **)&currentGenerationOnDevice, 65536 * sizeof(char)) != cudaSuccess) {
      cout << "cudaMalloc() failed!" << endl;
      exit(0);
    }

    // Allocate memory for the next generation on device
    if (cudaMalloc((void **)&theNextGenerationOnDevice, 65536 * sizeof(char)) != cudaSuccess) {
      cout << "cudaMalloc() failed!" << endl;
      exit(0);
    }

    // Copy initial generation from host to device
    if (cudaMemcpy(currentGenerationOnDevice, currentGeneration, 65536 * sizeof(char), cudaMemcpyHostToDevice) != cudaSuccess) {
      cout << "cudaMemcpy() failed!" << endl;
      exit(0);
    }

    // Continue calculating next generation until desired number of generations
    for(int i = 0; i < numGenerations; i++) {
      
      changeCellState <<<64, 1024>>>(currentGenerationOnDevice, theNextGenerationOnDevice, 256, 256);

      // Block until the device has completed all preceding requested tasks
      if (cudaDeviceSynchronize() != cudaSuccess) {
        cout << "cudaDeviceSynchronize() failed!" << endl;
        exit(0);
      }

      // Output current generation to stdout if flag set
      if (showAll) {
        cudaMemcpy(currentGeneration, theNextGenerationOnDevice, 65536 * sizeof(char), cudaMemcpyDeviceToHost);
        outputAutomaton(currentGeneration);
      }

      // Copy calculated generation to current generation on device
      cudaMemcpy(currentGenerationOnDevice, theNextGenerationOnDevice, 65536 * sizeof(char), cudaMemcpyDeviceToDevice);
    }

    // Copy final generation on device to host
    cudaMemcpy(theNextGeneration, theNextGenerationOnDevice, 65536 * sizeof(char), cudaMemcpyDeviceToHost);
    
    // Output final generation to stdout
    cout << "Final Generation:" << endl;
    outputAutomaton(theNextGeneration);

    // Explicitly destroy and clean up all resources associated
    // with the current device in the current process.
    if (cudaDeviceReset() != cudaSuccess) {
        cout << "cudaDeviceReset() failed!" << endl;
        return 1;
    }

    return 0;
}