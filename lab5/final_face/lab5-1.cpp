// Include required header files from OpenCV directory
#include <fcntl.h>
#include <linux/fb.h>
#include <sys/ioctl.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace cv::face;
using namespace std;

string cascadeName, nestedCascadeName;

void showFeatureAndSaveInformation(Ptr<FisherFaceRecognizer> model, int argc,
                                   int height, string output_folder);
static Mat norm_0_255(InputArray _src);
static void read_csv(const string &filename, vector<Mat> &images,
                     vector<int> &labels, char separator);

struct framebuffer_info {
  uint32_t bits_per_pixel;  // depth of framebuffer
  uint32_t xres_virtual;    // how many pixel in a row in virtual screen
  uint32_t yres_virtual;
};

struct framebuffer_info get_framebuffer_info(
    const char *framebuffer_device_path);

namespace patch {
template <typename T>
std::string to_string(const T &n) {
  std::ostringstream stm;
  stm << n;
  return stm.str();
}
}  // namespace patch

static Mat norm_0_255(InputArray _src) {
  Mat src = _src.getMat();
  // Create and return normalized image:
  Mat dst;
  switch (src.channels()) {
    case 1:
      cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
      break;
    case 3:
      cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
      break;
    default:
      src.copyTo(dst);
      break;
  }
  return dst;
}

static void read_csv(const string &filename, vector<Mat> &images,
                     vector<int> &labels, char separator = ';') {
  std::ifstream file(filename.c_str(), ifstream::in);
  string line, path, classlabel;
  while (getline(file, line)) {
    stringstream liness(line);
    getline(liness, path, separator);
    getline(liness, classlabel);

    if (!path.empty() && !classlabel.empty()) {
      images.push_back(imread(path, 0));
      labels.push_back(atoi(classlabel.c_str()));
    }
  }
}

void showFeatureAndSaveInformation(Ptr<FisherFaceRecognizer> model, int argc,
                                   int height, string output_folder) {
  Mat eigenvalues = model->getEigenValues();
  // And we can do the same to display the Eigenvectors (read Eigenfaces):
  Mat W = model->getEigenVectors();
  // Get the sample mean from the training data
  Mat mean = model->getMean();
  // Display or save the image reconstruction at some predefined steps:
  ofstream saveEigenValueFile(output_folder + "/eigenValue.txt", ios::out);
  for (int i = 0; i < min(16, W.cols); i++) {
    saveEigenValueFile << "eigenValue:" << endl;
    saveEigenValueFile << eigenvalues.at<double>(i) << endl;
    string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));

    cout << msg << endl;
    // get eigenvector #i
    Mat ev = W.col(i).clone();
    // Reshape to original size & normalize to [0...255] for imshow.
    Mat grayscale = norm_0_255(ev.reshape(1, height));
    // Show the image & apply a Bone colormap for better sensing.
    Mat cgrayscale;
    applyColorMap(grayscale, cgrayscale, COLORMAP_BONE);
    // Display or save:
    if (argc != 2) {
      imwrite(format("%s/fisherface_%d.png", output_folder.c_str(), i),
              norm_0_255(cgrayscale));
    }
  }
  saveEigenValueFile.close();
  /// save model
  model->write(output_folder + "/model.xml");
}

class oldFeaturePredict {
 public:
  vector<int> label_list;
  vector<cv::Mat> old_projection_list;
  string projectionFileName = "record/faceProjection.txt";
  cv::Mat model_eigenvectors;
  cv::Mat model_mean;
  int threshold = 400;
  oldFeaturePredict(cv::Mat model_eigenvectors, cv::Mat model_mean)
      : model_eigenvectors(model_eigenvectors), model_mean(model_mean) {
    readProjectionFromFile();
  }
  void readProjectionFromFile() {
    ifstream file(projectionFileName, ios::in);
    cout << " start readProjectionFromFile -----" << endl;
    fflush(stdout);
    string line;
    regex re(
        "(\\d),\\[([+-]?[0-9]*[.][0-9]+), ([+-]?[0-9]*[.][0-9]+), "
        "([+-]?[0-9]*[.][0-9]+)\\]");
    smatch sm;
    int count = 0;
    map<int, int> eachClassOccureCount;
    while (getline(file, line)) {
      regex_search(line, sm, re);
      int label = stoi(sm[1].str());
      if (eachClassOccureCount[label]++ > 15) {
        continue;
      }
      if (count++ > 50) {
        break;
      }
      label_list.push_back(label);
      Mat old_projection_item = Mat(1, 3, CV_64F);
      old_projection_item.at<double>(0, 0) = stod(sm[2].str());
      old_projection_item.at<double>(0, 1) = stod(sm[3].str());
      old_projection_item.at<double>(0, 2) = stod(sm[4].str());
      old_projection_list.push_back(old_projection_item);
    }
    cout << "saved projection values" << endl;
    for (auto item : label_list) {
      cout << item << endl;
    }
    for (auto item : old_projection_list) {
      cout << item << endl;
    }
  }
  void predict(InputArray _src, int &label, double &confidence) const {
    // get data
    Mat src = _src.getMat();
    // project into PCA subspace
    Mat q =
        LDA::subspaceProject(model_eigenvectors, model_mean, src.reshape(1, 1));
    for (size_t sampleIdx = 0; sampleIdx < old_projection_list.size();
         sampleIdx++) {
      double dist = norm(old_projection_list[sampleIdx], q, NORM_L2);
      int tmp_label = label_list[sampleIdx];
      label = tmp_label;
      confidence = dist;
      if (label != 2 && label != 3) {
        if (dist < 600) {
          cout << "****saved Feature predict label:unknown(" << label
               << ") confidence:" << confidence << endl;
          return;
        }
      } else {
        if (dist < 600) {
          cout << "****saved Feature predict label:" << label
               << " confidence:" << confidence << endl;
          return;
        }
      }
    }
    cout << "**** saved Feature predict label : Unknow" << endl;
    label = -1;
  }
};

int main(int argc, const char *argv[]) {
  // ----------------------------------
  // Initialization
  // ----------------------------------

  Mat frame, image;

  double scale = 1;

  CascadeClassifier cascade, nestedCascade;
  nestedCascade.load("haarcascades/haarcascade_eye_tree_eyeglasses.xml");
  cascade.load("haarcascades/haarcascade_frontalface_alt2.xml");
  cout << "Load CascadeClassifier Done" << endl;

  // ----------------------------------
  // Prepare Face Recognition Model
  // ----------------------------------

  string output_folder = ".";
  if (argc == 3) {
    output_folder = string(argv[2]);
  }
  string fn_csv = string(argv[1]);
  vector<Mat> images;
  vector<int> labels;

  read_csv(fn_csv, images, labels);

  cout << "Training EigenFaceRecognizer..." << endl;
  Ptr<FisherFaceRecognizer> model = FisherFaceRecognizer::create();
  model->train(images, labels);
  Mat eigenvalues = model->getEigenValues();
  // And we can do the same to display the Eigenvectors (read Eigenfaces):
  Mat eigenVectors = model->getEigenVectors();
  // Get the sample mean from the training data
  Mat mean = model->getMean();

  // build old projection class
  oldFeaturePredict oldFeaturePredict_obj(eigenVectors, mean);
  // ----------------------------------
  // open camera device
  // ----------------------------------

  const int frame_rate = 10;
  cv::VideoCapture camera(2);
  framebuffer_info fb_info = get_framebuffer_info("/dev/fb0");
  std::ofstream ofs("/dev/fb0");

  // streaming
  cout << "Get Webcam, Start Streaming and Face Detection..." << endl;
  int height = images[0].rows;
  showFeatureAndSaveInformation(model, argc, height, output_folder);
  ofstream saveProjectionFile(output_folder + "/faceProjection.txt",
                              ios::out | ios::app);
  fflush(stdout);
  while (true) {
    camera >> frame;

    if (frame.empty()) break;

    Mat img = frame.clone();

    // ----------------------------------
    // Face Detection
    // ----------------------------------
    vector<Rect> faces, faces2;
    Mat gray, smallImg;

    cvtColor(img, gray, COLOR_BGR2GRAY);  // Convert to Gray Scale
    double fx = 1 / scale;

    // Resize the Grayscale Image
    resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR);
    equalizeHist(smallImg, smallImg);

    // Detect faces of different sizes using cascade classifier
    cascade.detectMultiScale(smallImg, faces, 1.3, 3, 0 | CASCADE_SCALE_IMAGE,
                             Size(30, 30));

    // Draw circles around the faces
    for (size_t i = 0; i < faces.size(); i++) {
      Rect r = faces[i];
      Mat smallImgROI;
      vector<Rect> nestedObjects;
      Point center;
      Scalar color = Scalar(0, 255, 0);  // Color for Drawing tool
      int radius;

      double aspect_ratio = (double)r.width / r.height;

      rectangle(img, cv::Point(cvRound(r.x * scale), cvRound(r.y * scale)),
                cv::Point(cvRound((r.x + r.width - 1) * scale),
                          cvRound((r.y + r.height - 1) * scale)),
                color, 3, 8, 0);
      if (nestedCascade.empty()) continue;
      smallImgROI = smallImg(r);

      Mat resized;
      resize(gray(r), resized, Size(92, 112), 0, 0, INTER_CUBIC);

      // ----------------------------------
      // Face Recognition
      // ----------------------------------

      int predictedLabel = -1;
      double confidence = 0.0;
      // cout << "Get Face! Start Face Recognition..." << endl;
      model->predict(resized, predictedLabel, confidence);
      /// show current face projection to eigenvector
      Mat projection =
          LDA::subspaceProject(eigenVectors, mean, resized.reshape(1, 1));
      cout << "current Face Projection to eigenvetors space:" << projection
           << endl;
      // save current face feature ,let we can load this feature after to
      // predict face
      if (confidence < 800) {
        saveProjectionFile << predictedLabel << "," << projection << endl;
      }
      /////////////use saved Feature to prediction
      int oldProjectionPredictLabel = -1;
      double oldProjectionConfidence = 0.0;
      oldFeaturePredict_obj.predict(resized, oldProjectionPredictLabel,
                                    oldProjectionConfidence);

      string result_message =
          format("!!!!Predicted class = %d, Confidence = %f", predictedLabel,
                 confidence);
      cout << result_message << endl;

      string name = "";
      if (confidence > 1000) {
        name = "Unknown:((";
      } else {
        if (predictedLabel == 1) {
          name = "312553040";
        } else if (predictedLabel == 2) {
          name = "312551172";
        } else {
          name = "Unknown:)";
        }
      }

      // Draw name on image
      int font_face = cv::FONT_HERSHEY_COMPLEX;
      double font_scale = 1;
      int thickness = 1;
      int baseline;
      cv::Size text_size =
          cv::getTextSize(name, font_face, font_scale, thickness, &baseline);

      cv::Point origin;
      origin.x = cvRound(r.x * scale);
      origin.y = cvRound(r.y * scale) - text_size.height / 2;
      cv::putText(img, name, origin, font_face, font_scale,
                  cv::Scalar(0, 255, 255), thickness, 8, 0);

      // Detect and draw eyes on image
      nestedCascade.detectMultiScale(smallImgROI, nestedObjects, 1.1, 2,
                                     0 | CASCADE_SCALE_IMAGE, Size(30, 30));
      for (size_t j = 0; j < nestedObjects.size(); j++) {
        Rect nr = nestedObjects[j];
        center.x = cvRound((r.x + nr.x + nr.width * 0.5) * scale);
        center.y = cvRound((r.y + nr.y + nr.height * 0.5) * scale);
        radius = cvRound((nr.width + nr.height) * 0.25 * scale);
        circle(img, center, radius, color, 3, 8, 0);
      }
    }

    // Display with imshow
    cv::Size2f frame_size = img.size();
    cv::Mat framebuffer_compat;
    cv::cvtColor(img, framebuffer_compat, cv::COLOR_BGR2BGR565);
    for (int y = 0; y < frame_size.height; y++) {
      ofs.seekp(y * fb_info.xres_virtual * fb_info.bits_per_pixel / 8);
      ofs.write(reinterpret_cast<char *>(framebuffer_compat.ptr(y)),
                frame_size.width * fb_info.bits_per_pixel / 8);
    }
  }

  return 0;
}

struct framebuffer_info get_framebuffer_info(
    const char *framebuffer_device_path) {
  struct framebuffer_info fb_info;
  struct fb_var_screeninfo screen_info;
  int fd = open(framebuffer_device_path, O_RDWR);
  if (!ioctl(fd, FBIOGET_VSCREENINFO, &screen_info)) {
    fb_info.xres_virtual = screen_info.xres_virtual;
    fb_info.yres_virtual = screen_info.yres_virtual;
    fb_info.bits_per_pixel = screen_info.bits_per_pixel;
  }
  return fb_info;
};