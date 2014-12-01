// Author @ Eric Reinsmidt
// Date @ 2014.11.23
// Version 0.1

/*

  Hi Eduardo, I've made an assumption in the code. I don't set the device using cudaSetDevice().
  So, on a multi-GPU system this code will default to device 0, which is whatever
  device is in the first slot. So please keep that in mind if testing on a multi-GPU
  system.

  I did not assume a finite grid.

  I've prepopulated the automaton with a row of 10 cells. This leads to the creation of
  a pentadecathlon oscillator:
  http://www.conwaylife.com/wiki/Pentadecathlon

  Also if you are bored, I coded the Game of Life four or five years ago in JavaScript
  that is fun to play with:
  http://reinsmidt.com/snippets/life/

*/

#include <stdio.h>
#include <iostream>
using namespace std;

void outputAutomaton(char *automaton) {
  int cellNum = 0;
  for(int i = 0; i < 256; i++) {
    for(int j = 0; j < 256; j++) {
      if(automaton[cellNum] == 1) {
        cout << "@";
      } else {
        cout << " ";
      }
      cellNum++;
    }
    cout << endl;
  }
  return;
}

// Translate from index into threads to rows and colums
__device__ void translateToRowAndCol(int index, int *row, int *col, int rows, int cols) {
  *row = index / rows;
  *col = index % cols;
  return;
}

// Translate from rows and columns to index into threads
__device__ void translateToIndex(int row, int col, int *index, int rows, int cols) {
  *index = __umul24(row, cols) + col;
}

// Check neighbor's health and update cell accordingly
__global__ void changeCellState(char *currGen, char *nextGen, int rows, int cols) {
  
  // Calculate index into array
  int index = __umul24(blockDim.x, blockIdx.x) + threadIdx.x;
  
  int colIndex;
  int rowIndex;
  int newIndex;

  translateToRowAndCol(index, &rowIndex, &colIndex, rows, cols);
  translateToIndex(rowIndex, colIndex, &newIndex, rows, cols);

  int cellNeighbors = 0;
  int tempIndex;

  //////////////////////////////////
  // Normal cases around neighbor //
  //////////////////////////////////

  // Check upper neighbor
  if (rowIndex != 0) {
    translateToIndex((rowIndex - 1), colIndex, &tempIndex, rows, cols);
    if (currGen[tempIndex] == 1) {
      cellNeighbors++;
    }
  }

  // Check upper right neighbor
  if (rowIndex != 0 && colIndex != cols - 1) {
    translateToIndex((rowIndex - 1), (colIndex + 1), &tempIndex, rows, cols);
    if (currGen[tempIndex] == 1) {
      cellNeighbors++;
    }
  }

  // Check right neighbor
  if (colIndex != cols - 1) {
    translateToIndex(rowIndex, (colIndex + 1), &tempIndex, rows, cols);
    if(currGen[tempIndex] == 1) {
      cellNeighbors++;
    }
  }

  // Check bottom right neighbor
  if (colIndex != cols - 1 && rowIndex != rows - 1) {
    translateToIndex((rowIndex + 1), (colIndex + 1), &tempIndex, rows, cols);
    if (currGen[tempIndex] == 1) {
      cellNeighbors++;
    }
  }

  // Check bottom neighbor
  if (rowIndex != rows - 1) {
    translateToIndex((rowIndex + 1), colIndex, &tempIndex, rows, cols);
    if (currGen[tempIndex] == 1) {
      cellNeighbors++;
    }
  }

  // Check bottom left neighbor
  if (rowIndex != rows - 1 && colIndex != 0) {
    translateToIndex((rowIndex + 1), (colIndex - 1), &tempIndex, rows, cols);
    if (currGen[tempIndex] == 1) {
      cellNeighbors++;
    }
  }

  // Check left neighbor
  if (colIndex != 0) {
    translateToIndex(rowIndex, (colIndex - 1), &tempIndex, rows, cols);
    if (currGen[tempIndex] == 1) {
      cellNeighbors++;
    }
  }

  // Check upper left neighbor
  if (colIndex != 0 && rowIndex != 0) {
    translateToIndex((rowIndex - 1), (colIndex - 1), &tempIndex, rows, cols);
    if (currGen[tempIndex] == 1) {
      cellNeighbors++;
    }
  }

  //////////////////////////////////////////
  // Special cases like edges and corners //
  //////////////////////////////////////////
  
  // Upper row, wrap to bottom row
  if (rowIndex == 0) {
    translateToIndex((rows - 1), colIndex, &tempIndex, rows, cols);
    if (currGen[tempIndex]) {
      cellNeighbors++;
    }
  }

  // Bottom row, wrap to top row
  if (rowIndex == (rows - 1)) {
    translateToIndex(0, colIndex, &tempIndex, rows, cols);
    if (currGen[tempIndex]) {
      cellNeighbors++;
    }
  }

  // Left column, wrap to right column
  if (colIndex == 0) {
    translateToIndex(rowIndex, (cols - 1), &tempIndex, rows, cols);
    if (currGen[tempIndex]) {
      cellNeighbors++;
    }
  }

  // Right column, wrap to left column
  if (colIndex == (cols - 1)) {
    translateToIndex(rowIndex, 0, &tempIndex, rows, cols);
    if (currGen[tempIndex]) {
      cellNeighbors++;
    }
  }

  // Upper left, wrap to bottom right
  if (rowIndex == 0 && colIndex == 0) {
    translateToIndex((rows - 1), (cols - 1), &tempIndex, rows, cols);
    if (currGen[tempIndex]) {
      cellNeighbors++;
    }
  }

  // Upper right, wrap to bottom left
  if (rowIndex == 0 && colIndex == (cols - 1)) {
    translateToIndex((rows - 1), 0, &tempIndex, rows, cols);
    if (currGen[tempIndex]) {
      cellNeighbors++;
    }
  }

  // Bottom right, wrap to upper left
  if (rowIndex == (rows - 1) && colIndex == (cols - 1)) {
    translateToIndex(0, 0, &tempIndex, rows, cols);
    if (currGen[tempIndex]) {
      cellNeighbors++;
    }
  }

  // Bottom left, wrap to upper right
  if (rowIndex == (rows - 1) && colIndex == 0) {
    translateToIndex(0, (cols - 1), &tempIndex, rows, cols);
    if (currGen[tempIndex]) {
      cellNeighbors++;
    }
  }

  //__syncthreads();

  // Determine if cell lives, dies, or is born by evaluating how many neighbors it has
  if (currGen[index] == 1) { // Live cell
    if (cellNeighbors < 2 || cellNeighbors > 3) {
      nextGen[index] = 0; // Died from underpopulation or overcrowding
    } else {
      nextGen[index] = 1; // Still alive
    }
  } else { // Dead cell
    if (cellNeighbors == 3) {
      nextGen[index] = 1; // Born
    } else {
      nextGen[index] = 0; // Still dead
    }
  }
}