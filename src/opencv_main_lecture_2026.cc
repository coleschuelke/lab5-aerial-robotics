#include <Eigen/Dense>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

int main(int argc, char** argv) {
  // Open video file via a VideoCapture object named camera
  std::string filename =
      "/home/aeronaut/Workspace/lab5-aerial-robotics/opencv-lecture/input.m4v";
  cv::VideoCapture camera(filename);
  // Check to ensure the video file opened properly
  if (!camera.isOpened()) {
    std::cout << "Failed to open " << filename << std::endl;
    return EXIT_FAILURE;
  }

  // create output window
  cv::namedWindow("Image", cv::WINDOW_AUTOSIZE);

  // === find and track red square in each frame of the vid ===
  while (true) {
    // == read in frame ==
    cv::Mat frame, framep;
    bool readSuccess{camera.read(frame)};
    if (!readSuccess) break;

    // == clean frame using our image processing tools ==

    // blur the image to reduce sharp corners and noise
    cv::GaussianBlur(frame, framep, cv::Size(11, 11), 3);

    // convert image from BGR to HSV -> color in one parameter H
    // filter for red color: H elemof [170, 180] or [0, 10]
    cv::cvtColor(framep, framep, cv::COLOR_BGR2HSV);
    cv::Scalar colorLower_l{0, 120, 100};
    cv::Scalar colorLower_h{10, 255, 255};
    cv::Scalar colorUpper_l{170, 120, 100};
    cv::Scalar colorUpper_h{180, 255, 255};

    // create binary 'frames' that show white (1) if in the red range
    cv::Mat matLower, matUpper;
    cv::inRange(framep, colorLower_l, colorLower_h, matLower);
    cv::inRange(framep, colorUpper_l, colorUpper_h, matUpper);
    framep = matLower | matUpper;

    // erode and dialate to 'open' the image (gets rid of stray wisps of red and
    // then restores original size)
    int iters{5};
    cv::erode(framep, framep, cv::Mat(), cv::Point(-1, -1), iters);
    cv::dilate(framep, framep, cv::Mat(), cv::Point(-1, -1), iters);

    // == find contours in the image and determine if it is our red square ==
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(framep, contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);
    // loop through contours found and determine if red square
    float maxAR{1.5};  // aspect ratio bound
    float minR{35};    // radius bound
    for (int ii{}; ii < contours.size(); ii++) {
      // find the enclosing circle
      cv::Point2f center{};
      float radius{};
      cv::minEnclosingCircle(contours[ii], center, radius);

      // determine the aspect ratio of the bounding box
      float aspectRatio{};
      int minPointsForEllipse{5};
      if (contours[ii].size() >= minPointsForEllipse) {
        cv::RotatedRect boundingRectangle{cv::fitEllipse(contours[ii])};
        float width{boundingRectangle.size.width};
        float height{boundingRectangle.size.height};
        aspectRatio = std::max(width, height) / std::min(width, height);
      }

      // debug print
      std::cout << "AR = " << aspectRatio << "  ; radius = " << radius << '\n';

      // if AR and radius are within bounds declare red square and draw a circle
      cv::RNG rng(234);
      cv::Scalar color{rng.uniform(0.f, 255.f), rng.uniform(0.f, 255.f),
                       rng.uniform(0.f, 255.f)};
      if (aspectRatio < maxAR && radius > minR)
        cv::circle(frame, center, radius, color, 3);
    }

    // == display frame ==
    cv::imshow("Image", frame);
    int keycode{cv::waitKey()};
    // break if 'q' was entered
    if (keycode == 'q') break;
  }

  return EXIT_SUCCESS;
}

