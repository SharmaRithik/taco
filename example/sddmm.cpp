#include <random>
#include <iostream>
#include "taco.h"

using namespace taco;

// Function to generate random dense tensor
Tensor<double> generateRandomDenseTensor(int dim1, int dim2, Format format) {
  std::default_random_engine gen(0);
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  Tensor<double> tensor({dim1, dim2}, format);

  for (int i = 0; i < dim1; ++i) {
    for (int j = 0; j < dim2; ++j) {
      tensor.insert({i, j}, unif(gen));
    }
  }

  tensor.pack();
  return tensor;
}

int main(int argc, char* argv[]) {
  // Define storage formats
  Format dcsr({Sparse, Sparse});
  Format rm({Dense, Dense});
  Format cm({Dense, Dense}, {1, 0});

  // Load the sparse matrix B from file
  Tensor<double> B = read("webbase-1M/webbase-1M.mtx", dcsr);

  // Generate random dense matrices C and D
  Tensor<double> C = generateRandomDenseTensor(B.getDimension(0), 1000, rm);
  Tensor<double> D = generateRandomDenseTensor(1000, B.getDimension(1), cm);

  // Declare the output tensor A with the same dimensions as B
  Tensor<double> A(B.getDimensions(), dcsr);

  // Define the SDDMM computation
  IndexVar i, j, k;
  A(i, j) = B(i, j) * C(i, k) * D(k, j);

  // Compile, assemble, and compute the tensor A
  A.compile();
  A.assemble();
  A.compute();

  // Write the output tensor A to file
  write("A.mtx", A);

  std::cout << "SDDMM computation complete. Output written to A.mtx." << std::endl;

  return 0;
}

