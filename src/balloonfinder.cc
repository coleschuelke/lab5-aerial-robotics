#include "balloonfinder.h"

#include <Eigen/Dense>
#include <cassert>
#include <numeric>
#include <opencv2/core/eigen.hpp>
#include <ratio>
#include <tuple>
#include <vector>

#include "navtoolbox.h"

BalloonFinder::BalloonFinder(bool debuggingEnabled, bool calibrationEnabled,
                             const Eigen::Vector3d &blueTrue_I,
                             const Eigen::Vector3d &redTrue_I) {
  debuggingEnabled_ = debuggingEnabled;
  calibrationEnabled_ = calibrationEnabled;
  blueTrue_I_ = blueTrue_I;
  redTrue_I_ = redTrue_I;
  V_.resize(3, 0);
  W_.resize(3, 0);
}

// Returns true if the input contour touches the edge of the input image;
// otherwise returns false.
//
bool touchesEdge(const cv::Mat &image, const std::vector<cv::Point> &contour) {
  const size_t borderWidth = static_cast<size_t>(0.01 * image.rows);

  for (const auto &pt : contour) {
    if (pt.x <= borderWidth || pt.x >= (image.cols - borderWidth) ||
        pt.y <= borderWidth || pt.y >= (image.rows - borderWidth))
      return true;
  }
  return false;
}

Eigen::Vector3d BalloonFinder::eCB_calibrated() const {
  using namespace Eigen;
  const SensorParams sp;
  const size_t N = V_.cols();
  if (N < 2 || !calibrationEnabled_) {
    return Vector3d::Zero();
  }
  const VectorXd aVec = VectorXd::Ones(N);
  const Matrix3d dRCB = navtbx::wahbaSolver(aVec, W_, V_);
  const Matrix3d RCB = navtbx::euler2dc(sp.eCB());
  return navtbx::dc2euler(dRCB * RCB);
}

bool BalloonFinder::findBalloonsOfSpecifiedColor(
    const cv::Mat *image, const Eigen::Matrix3d RCI, const Eigen::Vector3d rc_I,
    const BalloonFinder::BalloonColor color,
    std::vector<Eigen::Vector2d> *rxVec) {
  using namespace cv;
  bool returnValue = false;
  rxVec->clear();
  Mat original;
  // Clone the original image for debugging purposes
  if (debuggingEnabled_)
    original = image->clone();
  const size_t nCols_m1 = image->cols - 1;
  const size_t nRows_m1 = image->rows - 1;
  // Blur the image to reduce small-scale noise
  Mat framep;
  GaussianBlur(*image, framep, Size(21, 21), 0, 0);

  // *************************************************************************
  //
  // Implement the rest of the function here.  Your goal is to find a balloon of
  // the color specified by the input 'color', and find its center in image
  // plane coordinates (see the comments below for a discussion on image plane
  // coordinates), expressed in pixels.  Suppose rx is an Eigen::Vector2d object
  // that holds the x and y position of a balloon center in image plane
  // coordinates.  You can push rx onto rxVec as follows: rxVec->push_back(rx)
  //
  // *************************************************************************
  // Convert to HSV and set color bounds
  cvtColor(framep, framep, COLOR_BGR2HSV);
  // RED
  Scalar redLower_l{0, 120, 100};
  Scalar redLower_u{10, 255, 255};
  Scalar redUpper_l{170, 120, 100};
  Scalar redUpper_u{180, 255, 255};
  // BLUE (Will need to tune this)
  Scalar blue_l{100, 180, 190};
  Scalar blue_u{130, 255, 255};

  // Binary frame for selected color (RED or BLUE)
  Mat matLower, matUpper;
  if (color == BalloonColor::RED) {
    inRange(framep, redLower_l, redLower_u, matLower);
    inRange(framep, redUpper_l, redUpper_u, matUpper);
    framep = matLower | matUpper;
  } else { // Color is BLUE
    inRange(framep, blue_l, blue_u, framep);
  }
  // Erode and dilate to open the image
  int iters{5}; // Can be adjusted, but should be plenty
  erode(framep, framep, Mat(), cv::Point(-1, -1), iters);
  dilate(framep, framep, Mat(), cv::Point(-1, -1), iters);

  // Find contours and determine if they are a red balloon
  float maxAR{1.5};
  float minR{25};
  std::vector<std::vector<cv::Point>> contours;
  findContours(framep, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
  for (int i = 0; i < contours.size(); i++) {
    // Circle for now, may try other bounding options
    Point2f center;
    float radius;
    minEnclosingCircle(contours[i], center, radius);

    float aspectRatio;
    int minPointsForEllipse{5};
    if (contours[i].size() >= minPointsForEllipse) {
      RotatedRect boundingRectangle{fitEllipse(contours[i])};
      float width{boundingRectangle.size.width};
      float height{boundingRectangle.size.height};
      aspectRatio = std::max(width, height) / std::min(width, height);
    }
    // If the detection looks like a balloon, set the return value and push the
    // point
    if (aspectRatio < maxAR && radius > minR) {
      rxVec->push_back(Eigen::Vector2d(center.x, center.y));
      returnValue = true;
    }
    // Find the center of the outline in pixels and pass it to rxVec
  }

  // The debugging section below plots the back-projection of the true balloon
  // 3d location on the original image.  The balloon centers you find should be
  // close to the back-projected coordinates in xc_pixels.  Feel free to alter
  // the code in the debugging section below, or add other such sections, so
  // that you can see how your estimated balloon centers compare with the
  // back-projected centers.
  if (debuggingEnabled_) {
    Eigen::Vector2d xc_pixels;
    Scalar trueProjectionColor;
    if (color == BalloonColor::BLUE) {
      xc_pixels = backProject(RCI, rc_I, blueTrue_I_);
      trueProjectionColor = Scalar(255, 0, 0);
    } else {
      xc_pixels = backProject(RCI, rc_I, redTrue_I_);
      trueProjectionColor = Scalar(0, 0, 255);
    }
    Point2f center;
    // The image plane coordinate system, in which xc_pixels is expressed, has
    // its origin at the lower-right of the image, x axis pointing left and y
    // axis pointing up, whereas the variable 'center' below, used by OpenCV for
    // plotting on the image, is referenced to the image's top left corner and
    // has the opposite x and y directions.  The measurements returned in rxVec
    // should be given in the image plane coordinate system like xc_pixels.
    // Hence, once you've found a balloon center from your image processing
    // techniques, you'll need to convert it to the image plane coordinate
    // system using an inverse of the mapping below.
    center.x = nCols_m1 - xc_pixels(0);
    center.y = nRows_m1 - xc_pixels(1);
    circle(original, center, 20, trueProjectionColor, FILLED);
    auto &points = *rxVec;
    for (int i{}; i < points.size(); i++) { // Draw the found balloon centers
      cv::Point2f found_center(points[i](0, 0), points[i](1, 0));
      circle(original, found_center, 25, Scalar(60, 255, 255), 3);
    }
    namedWindow("Display", WINDOW_NORMAL);
    resizeWindow("Display", 1000, 1000);
    imshow("Display", original);
    waitKey(0);
  }
  return returnValue;
}

void BalloonFinder::findBalloons(
    const cv::Mat *image, const Eigen::Matrix3d RCI, const Eigen::Vector3d rc_I,
    std::vector<std::shared_ptr<const CameraBundle>> *bundles,
    std::vector<BalloonColor> *colors) {
  // Crop image to 4k size.  This removes the bottom 16 rows of the image,
  // which are an artifact of the camera API.
  const cv::Rect croppedRegion(0, 0, sensorParams_.imageWidthPixels(),
                               sensorParams_.imageHeightPixels());
  cv::Mat croppedImage = (*image)(croppedRegion);
  // Convert camera instrinsic matrix K and distortion parameters to OpenCV
  // format
  cv::Mat K, distortionCoeffs, undistortedImage;
  Eigen::Matrix3d Kpixels = sensorParams_.K() / sensorParams_.pixelSize();
  Kpixels(2, 2) = 1;
  cv::eigen2cv(Kpixels, K);
  cv::eigen2cv(sensorParams_.distortionCoeffs(), distortionCoeffs);
  // Undistort image
  cv::undistort(croppedImage, undistortedImage, K, distortionCoeffs);

  // Find balloons of specified color
  std::vector<BalloonColor> candidateColors = {BalloonColor::RED,
                                               BalloonColor::BLUE};
  for (auto color : candidateColors) {
    std::vector<Eigen::Vector2d> rxVec;
    if (findBalloonsOfSpecifiedColor(&undistortedImage, RCI, rc_I, color,
                                     &rxVec)) {
      for (const auto &rx : rxVec) {
        std::shared_ptr<CameraBundle> cb = std::make_shared<CameraBundle>();
        cb->RCI = RCI;
        cb->rc_I = rc_I;
        cb->rx = rx;
        bundles->push_back(cb);
        colors->push_back(color);
      }
    }
  }
}
