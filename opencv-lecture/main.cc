#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {

  using namespace cv;

  // Open video file via a VideoCapture object named camera
  std::string filename = "./input.m4v";
  VideoCapture camera(filename);
  // Check to ensure the video file opened properly
  if(!camera.isOpened()) {
    std::cout << "Failed to open " << filename << std::endl; 
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}



