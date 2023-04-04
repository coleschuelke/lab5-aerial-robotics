#include <Eigen/Dense>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

#include "structurecomputer.h"

int main(int argc, char** argv) {
  // Demonstrate least squares solution based on a measurement model of the
  // form z = Hr*x + w, with R = E[w*w'] being the measurement noise
  // covariance matrix.
  constexpr size_t nx = 3;
  constexpr size_t nz = 4;
  Eigen::VectorXd z(nz), xHat(nx);
  Eigen::MatrixXd Hr(nz, nx), R(nz, nz);
  // Fill in z, Hr, and R with example values
  z << 4, 10, 6, 5;
  Hr << 1, 0, 3, 0, 2, 6, 0, 0, 0, 0, 0, 1e-6;
  R = 3 * Eigen::MatrixXd::Identity(nz, nz);
  Eigen::MatrixXd Rinv = R.inverse();
  // This is the straightforward way to solve the the normal equations:
  xHat = (Hr.transpose() * Rinv * Hr).inverse() * (Hr.transpose() * Rinv * z);
  std::cout << "The straightforward solution is \n" << xHat << std::endl;
  // This method is similar but more numerically stable:
  xHat = (Hr.transpose() * Rinv * Hr).ldlt().solve(Hr.transpose() * Rinv * z);
  std::cout << "The ldlt-based solution is \n" << xHat << std::endl;

  // Create an instance of a StructureComputer object
  StructureComputer structureComputer;
  // Create shared pointers to two CameraBundle objects.  The make_shared
  // function creates the objects and returns a shared pointer to each.
  std::shared_ptr<CameraBundle> cb1 = std::make_shared<CameraBundle>();
  std::shared_ptr<CameraBundle> cb2 = std::make_shared<CameraBundle>();
  // Fill cb1's and cb2's data members with example contents.
  cb1->rx << 23, 56;
  cb1->RCI.fill(3);
  cb1->rc_I(0) = 1;
  cb1->rc_I(1) = 0.4;
  // Clear out structureComputer
  structureComputer.clear();
  // Push cb1 and cb2 to structureComputer
  structureComputer.push(cb1);
  structureComputer.push(cb2);
  // Estimate 3D location of feature.  The try and catch blocks are for
  // exception handling -- they handle any errors that might be thrown during a
  // call to computeStructure().  You can use this to protect any portion of
  // your code in main.cc.
  try {
    Point p1 = structureComputer.computeStructure();
  } catch (std::exception& e) {
    std::cout << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  } catch (...) {
    std::cout << "Unhandled error" << std::endl;
    return EXIT_FAILURE;
  }

  // Display an image
  if (argc != 2) {
    std::cout << "No path to image provided." << std::endl;
    return EXIT_FAILURE;
  }
  cv::Mat image = cv::imread(std::string(argv[1]));
  if (!image.data) {
    std::cout << "Could not open or find the image" << std::endl;
    return EXIT_FAILURE;
  }
  cv::namedWindow("Display window", cv::WINDOW_NORMAL);
  cv::resizeWindow("Display window", 600, 600);
  cv::imshow("Display window", image);
  cv::waitKey(0);
  return EXIT_SUCCESS;
}
