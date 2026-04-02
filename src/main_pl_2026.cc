#include <Eigen/Dense>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

#include "structurecomputer.h"

int main(int argc, char** argv) {
  // Demonstrate least squares solution based on a measurement model of the
  // form z = H*x + w, with R = E[w*w'] being the measurement noise
  // covariance matrix.
  constexpr size_t nx = 3;
  constexpr size_t nz = 4;
  Eigen::VectorXd z(nz), xHat(nx);
  Eigen::MatrixXd H(nz, nx), R(nz, nz);
  // Fill in z, H, and R with example values
  z << 4, 10, 6, 5;
  H << 1, 0, 3, 0, 2, 6, 0, 0, 0, 0, 0, 1e-6;
  R = 3 * Eigen::MatrixXd::Identity(nz, nz);
  const auto Rinv = R.inverse();
  // This is the straightforward way to solve the normal equations
  xHat = (H.transpose() * Rinv * H).inverse() * (H.transpose() * Rinv * z);
  std::cout << "The straightforward solution is \n" << xHat << std::endl;
  // This method is similar but more numerically stable
  xHat = (H.transpose() * Rinv * H).ldlt().solve(H.transpose() * Rinv * z);
  std::cout << "The ldlt-based solution is \n" << xHat << std::endl;

  // Create an instance of a StructureComputer object
  StructureComputer structureComputer;
  // Create shared pointers to two CameraBundle objects.  The make_shared
  // function creates the objects and returns a shared pointer to each.
  auto cb1 = std::make_shared<CameraBundle>();
  auto cb2 = std::make_shared<CameraBundle>();
  // Fill cb1's and cb2's data members with dummy contents
  cb1->rx << 23, 56;
  cb1->RCI = Eigen::MatrixXd::Identity(3, 3);
  cb2->RCI = Eigen::MatrixXd::Identity(3, 3);
  cb1->rc_I(0) = 1;
  cb1->rc_I(1) = 0.4;
  
  return EXIT_SUCCESS;
}
