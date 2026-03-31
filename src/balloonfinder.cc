#include "balloonfinder.h"

#include <Eigen/Dense>
#include <cassert>
#include <opencv2/core/eigen.hpp>

#include "navtoolbox.h"

BalloonFinder::BalloonFinder(bool debuggingEnabled, bool calibrationEnabled,
                             const Eigen::Vector3d& blueTrue_I,
                             const Eigen::Vector3d& redTrue_I) {
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
bool touchesEdge(const cv::Mat& image, const std::vector<cv::Point>& contour) {
  const size_t borderWidth = static_cast<size_t>(0.01 * image.rows);

  for (const auto& pt : contour) {
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
    const cv::Mat* image, const Eigen::Matrix3d RCI, const Eigen::Vector3d rc_I,
    const BalloonFinder::BalloonColor color,
    std::vector<Eigen::Vector2d>* rxVec) {
  using namespace cv;
  bool returnValue = false;
  rxVec->clear();
  Mat original;
  // Clone the original image for debugging purposes
  if (debuggingEnabled_) original = image->clone();
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
    namedWindow("Display", WINDOW_NORMAL);
    resizeWindow("Display", 1000, 1000);
    imshow("Display", original);
    waitKey(0);
  }
  return returnValue;
}

void BalloonFinder::findBalloons(
    const cv::Mat* image, const Eigen::Matrix3d RCI, const Eigen::Vector3d rc_I,
    std::vector<std::shared_ptr<const CameraBundle>>* bundles,
    std::vector<BalloonColor>* colors) {
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
      for (const auto& rx : rxVec) {
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
