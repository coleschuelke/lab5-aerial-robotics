#include "balloonfinder.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <fstream>
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
// TODO: Maybe I should do something with this
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

void BalloonFinder::trainBalloonsOfSpecifiedColor(
    const cv::Mat *image, const Eigen::Matrix3d RCI, const Eigen::Vector3d rc_I,
    const BalloonFinder::BalloonColor color,
    std::vector<Eigen::Vector2d> *rxVec, std::ostream &os) {
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

  // Immediately calculate the true BP center for labeling purposes
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
  // ==== Preprocessing ===
  // Even out the lighting
  cvtColor(framep, framep, COLOR_BGR2Lab);
  std::vector<Mat> channels;
  split(framep, channels);
  Ptr<CLAHE> clahe = createCLAHE();
  clahe->setClipLimit(2.5);
  clahe->apply(channels[0], channels[0]);
  merge(channels, framep);
  cvtColor(framep, framep, COLOR_Lab2BGR);
  const Mat colorGraded = framep.clone();
  // Convert to HSV and set color bounds
  cvtColor(framep, framep, COLOR_BGR2HSV);
  // RED
  // TODO: Err on the side of detecting everything
  const float hsv_scale = 2.55;
  Scalar redLower_l{0 / 2, 50 * hsv_scale, 50 * hsv_scale};
  Scalar redLower_u{30 / 2, 100 * hsv_scale, 100 * hsv_scale};
  Scalar redUpper_l{320 / 2, 40 * hsv_scale, 40 * hsv_scale};
  Scalar redUpper_u{360 / 2, 100 * hsv_scale, 100 * hsv_scale};
  // BLUE (Will need to tune this)
  Scalar blue_l{180 / 2, 40 * hsv_scale, 40 * hsv_scale};
  Scalar blue_u{265 / 2, 100 * hsv_scale, 100 * hsv_scale};

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
  cv::Mat mask = framep;

  // Find contours and determine if they are a balloon
  float maxAR{1.55};
  float minAR{1};
  float minR{47};
  float minES{0.975};
  // TODO: Could used to add some more features
  std::vector<std::vector<cv::Point>> contours;
  findContours(framep, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
  std::vector<cv::RotatedRect> rects;
  std::vector<cv::RotatedRect> ellipses;
  for (int i = 0; i < contours.size(); i++) {
    // Circle for now, may try other bounding options
    // TODO: Here is where we will export to a csv
    Point2f cont_center;
    float radius;
    minEnclosingCircle(contours[i], cont_center, radius);
    // If the center is close enough to the back-projected one: label it
    float aspectRatio;
    float ellipseScore;
    int minPointsForEllipse{5};
    if (contours[i].size() >= minPointsForEllipse) {
      RotatedRect boundingRectangle{fitEllipse(contours[i])};
      ellipses.push_back(boundingRectangle);
      RotatedRect minRect{minAreaRect(contours[i])};
      rects.push_back(minRect);
      float width{boundingRectangle.size.width};
      float height{boundingRectangle.size.height};
      double ellipseArea{width * height * 0.785};
      double contArea{contourArea(contours[i])};
      aspectRatio = std::max(width, height) / std::min(width, height);
      ellipseScore = contArea / ellipseArea; // Should be close to one
    }
    // WARN: I'm realizing that one issue is going to be that actual balloons
    // that aren't the true balloons could confuse the model because they will
    // be labeled as incorrect, but should have the exact same properties as
    // what we want to detect. In that case we will have to be doing robust
    // estimation.

    // If the contour corresponds to a real balloon, label it with a 1 and write
    // out the features Else label it with a zero
    float detection_radius = 50; // Criteria for positive label in pixels
    Point2f dist = center - cont_center;
    Eigen::Vector2d dist_eig;
    dist_eig << dist.x, dist.y;

    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision,
                                           Eigen::DontAlignCols, ", ", "\n");

    if (dist_eig.norm() < detection_radius) {
      // Write out with a 1
    } else {
      // Write out with a 0
    }
    os << row.format(CSVFormat);
  }

  // Drawing images for debugging purposes
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
    center.x = nCols_m1 - xc_pixels(0);
    center.y = nRows_m1 - xc_pixels(1);
    circle(original, center, 20, trueProjectionColor,
           FILLED); // Draw the true center
    auto &points = *rxVec;
    for (int i{}; i < points.size(); i++) { // Draw the found balloon centers
      cv::Point2f found_center(nCols_m1 - points[i](0, 0),
                               nRows_m1 - points[i](1, 0));
      circle(original, found_center, 25, trueProjectionColor, 3);
    }
    // Draw the contours in green
    drawContours(original, contours, -1, Scalar(0, 255, 0), 3);
    namedWindow("Display", WINDOW_NORMAL);
    resizeWindow("Display", 1000, 1000);
    imshow("Display", colorGraded); // Show the binary mask
    waitKey(0);
    imshow("Display", mask); // Show the binary mask
    waitKey(0);
    imshow("Display", original);
    waitKey(0);
  }
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
  // ==== Preprocessing ===
  // Even out the lighting
  cvtColor(framep, framep, COLOR_BGR2Lab);
  std::vector<Mat> channels;
  split(framep, channels);
  Ptr<CLAHE> clahe = createCLAHE();
  clahe->setClipLimit(2.5);
  clahe->apply(channels[0], channels[0]);
  merge(channels, framep);
  cvtColor(framep, framep, COLOR_Lab2BGR);
  const Mat colorGraded = framep.clone();
  // Convert to HSV and set color bounds
  cvtColor(framep, framep, COLOR_BGR2HSV);
  // RED
  // TODO: Err on the side of detecting everything
  const float hsv_scale = 2.55;
  Scalar redLower_l{0 / 2, 50 * hsv_scale, 50 * hsv_scale};
  Scalar redLower_u{30 / 2, 100 * hsv_scale, 100 * hsv_scale};
  Scalar redUpper_l{320 / 2, 40 * hsv_scale, 40 * hsv_scale};
  Scalar redUpper_u{360 / 2, 100 * hsv_scale, 100 * hsv_scale};
  // BLUE (Will need to tune this)
  Scalar blue_l{180 / 2, 40 * hsv_scale, 40 * hsv_scale};
  Scalar blue_u{265 / 2, 100 * hsv_scale, 100 * hsv_scale};

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
  cv::Mat mask = framep;

  // Find contours and determine if they are a balloon
  float maxAR{1.55};
  float minAR{1};
  float minR{47};
  float minES{0.975};
  // TODO: Could used to add some more features
  std::vector<std::vector<cv::Point>> contours;
  findContours(framep, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
  std::vector<cv::RotatedRect> rects;
  std::vector<cv::RotatedRect> ellipses;
  for (int i = 0; i < contours.size(); i++) {
    // Circle for now, may try other bounding options
    // TODO: Here is where we will export to a csv
    Point2f center;
    float radius;
    minEnclosingCircle(contours[i], center, radius);
    // If the center is close enough to the back-projected one: label it
    float aspectRatio;
    float ellipseScore;
    int minPointsForEllipse{5};
    if (contours[i].size() >= minPointsForEllipse) {
      RotatedRect boundingRectangle{fitEllipse(contours[i])};
      ellipses.push_back(boundingRectangle);
      RotatedRect minRect{minAreaRect(contours[i])};
      rects.push_back(minRect);
      float width{boundingRectangle.size.width};
      float height{boundingRectangle.size.height};
      double ellipseArea{width * height * 0.785};
      double contArea{contourArea(contours[i])};
      aspectRatio = std::max(width, height) / std::min(width, height);
      ellipseScore = contArea / ellipseArea; // Should be close to one
    }
    // If the detection looks like a balloon, set the return value and push the
    // point
    if (aspectRatio < maxAR && radius > minR && aspectRatio > minAR &&
        ellipseScore > minES) {
      std::cout << "Aspect Ratio " << aspectRatio << std::endl;
      std::cout << "EllipseScore: " << ellipseScore << std::endl;

      rxVec->push_back(
          Eigen::Vector2d(nCols_m1 - center.x, nRows_m1 - center.y));
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
    circle(original, center, 20, trueProjectionColor,
           FILLED); // Draw the true center
    auto &points = *rxVec;
    for (int i{}; i < points.size(); i++) { // Draw the found balloon centers
      cv::Point2f found_center(nCols_m1 - points[i](0, 0),
                               nRows_m1 - points[i](1, 0));
      circle(original, found_center, 25, trueProjectionColor, 3);
    }
    // // Draw the bounding shapes on the masked image
    // for (int i{}; i < contours.size(); i++) { // Draw the found balloon
    // centers
    //   Point2f vertices[4];
    //   rects[i].points(vertices);
    //   for (int j{}; j < 4; j++) {
    //     line(original, vertices[j], vertices[(j + 1) % 4], Scalar(255, 0, 0),
    //          3);
    //   }
    //   ellipse(original, ellipses[i], Scalar(0, 255, 0), 3);
    // }
    // Draw the contours in green
    drawContours(original, contours, -1, Scalar(0, 255, 0), 3);
    namedWindow("Display", WINDOW_NORMAL);
    resizeWindow("Display", 1000, 1000);
    imshow("Display", colorGraded); // Show the binary mask
    waitKey(0);
    imshow("Display", mask); // Show the binary mask
    waitKey(0);
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

void BalloonFinder::trainBalloons(
    const cv::Mat *image, const Eigen::Matrix3d RCI, const Eigen::Vector3d rc_I,
    std::vector<std::shared_ptr<const CameraBundle>> *bundles,
    std::vector<BalloonColor> *colors, std::ostream &os) {
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
    trainBalloonsOfSpecifiedColor(&undistortedImage, RCI, rc_I, color, &rxVec,
                                  os);
  }
}
