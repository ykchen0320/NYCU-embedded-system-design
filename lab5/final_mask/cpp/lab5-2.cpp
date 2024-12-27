#include <fcntl.h>
#include <linux/fb.h>
#include <sys/ioctl.h>

#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

struct framebuffer_info {
  uint32_t bits_per_pixel;  // framebuffer depth
  uint32_t xres_virtual;    // how many pixel in a row in virtual screen
};

struct Detection {
  int class_id;
  float confidence;
  cv::Rect box;
};

struct framebuffer_info get_framebuffer_info(
    const char *framebuffer_device_path);

const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.2;
const float NMS_THRESHOLD = 0.4;
const float CONFIDENCE_THRESHOLD = 0.4;

void load_net(cv::dnn::Net &net) {
  auto result = cv::dnn::readNet("config_files/sim_mask_yolov5_m.onnx");
  std::cout << "Running on CPU\n";
  result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
  result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
  net = result;
}

const std::vector<cv::Scalar> colors = {cv::Scalar(0, 255, 0),
                                        cv::Scalar(0, 0, 255)};

cv::Mat format_yolov5(const cv::Mat &source) {
  int col = source.cols;
  int row = source.rows;
  int _max = MAX(col, row);
  cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
  source.copyTo(result(cv::Rect(0, 0, col, row)));
  return result;
}

void detect(cv::Mat &image, cv::dnn::Net &net, std::vector<Detection> &output,
            const std::vector<std::string> &className) {
  cv::Mat blob;

  auto input_image = format_yolov5(image);

  cv::dnn::blobFromImage(input_image, blob, 1. / 255.,
                         cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(),
                         true, false);
  net.setInput(blob);
  std::vector<cv::Mat> outputs;
  net.forward(outputs, net.getUnconnectedOutLayersNames());

  float x_factor = input_image.cols / INPUT_WIDTH;
  float y_factor = input_image.rows / INPUT_HEIGHT;

  float *data = (float *)outputs[0].data;

  const int dimensions = 8;
  const int rows = 25200;

  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;

  for (int i = 0; i < rows; ++i) {
    float confidence = data[4];
    if (confidence >= CONFIDENCE_THRESHOLD) {
      float *classes_scores = data + 5;
      cv::Mat scores(1, className.size(), CV_32FC1, classes_scores);
      cv::Point class_id;
      double max_class_score;
      minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
      if (max_class_score > SCORE_THRESHOLD) {
        confidences.push_back(confidence);

        class_ids.push_back(class_id.x);

        float x = data[0];
        float y = data[1];
        float w = data[2];
        float h = data[3];
        int left = int((x - 0.5 * w) * x_factor);
        int top = int((y - 0.5 * h) * y_factor);
        int width = int(w * x_factor);
        int height = int(h * y_factor);
        boxes.push_back(cv::Rect(left, top, width, height));
      }
    }

    data += dimensions;
  }

  std::vector<int> nms_result;
  cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD,
                    nms_result);
  for (int i = 0; i < nms_result.size(); i++) {
    int idx = nms_result[i];
    Detection result;
    result.class_id = class_ids[idx];
    result.confidence = confidences[idx];
    result.box = boxes[idx];
    output.push_back(result);
  }
}

std::vector<std::string> load_class_list() {
  std::vector<std::string> class_list;
  std::ifstream ifs("config_files/mask_class.txt");
  std::string line;
  while (getline(ifs, line)) {
    class_list.push_back(line);
  }
  return class_list;
}

int main(int argc, char **argv) {
  std::vector<std::string> class_list = load_class_list();

  // Load image
  std::string image_path = "example/demo_mask.png";  // Path to the image
  cv::Size2f image_size;
  cv::Mat image;

  framebuffer_info fb_info = get_framebuffer_info("/dev/fb0");
  std::ofstream ofs("/dev/fb0");

  // read image file (sample.bmp) from opencv libs.
  // https://docs.opencv.org/3.4.7/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56
  // image = .......
  image = cv::imread(image_path);

  // get image size of the image.
  // https://docs.opencv.org/3.4.7/d3/d63/classcv_1_1Mat.html#a146f8e8dda07d1365a575ab83d9828d1
  // image_size = ......
  image_size = image.size();

  cv::dnn::Net net;
  load_net(net);

  std::vector<Detection> output;
  detect(image, net, output, class_list);
  int detections = output.size();

  for (int i = 0; i < detections; ++i) {
    auto detection = output[i];
    auto box = detection.box;
    auto classId = detection.class_id;
    if (classId == 1) {
      cv::rectangle(image, box, colors[1], 0.5);
    } else {
      cv::rectangle(image, box, colors[0], 0.5);
    }
  }

  cv::imwrite("example/output.png", image);

  image_size = image.size();
  cv::cvtColor(image, image_bgr565, cv::COLOR_BGR2BGR565);

  // output to framebufer row by row
  for (int y = 0; y < image_size.height; y++) {
    // move to the next written position of output device framebuffer by
    // "std::ostream::seekp()". posisiotn can be calcluated by "y",
    // "fb_info.xres_virtual", and "fb_info.bits_per_pixel".
    // http://www.cplusplus.com/reference/ostream/ostream/seekp/
    ofs.seekp(y * fb_info.xres_virtual * fb_info.bits_per_pixel / 8);

    // write to the framebuffer by "std::ostream::write()".
    // you could use "cv::Mat::ptr()" to get the pointer of the corresponding
    // row. you also have to count how many bytes to write to the buffer
    // http://www.cplusplus.com/reference/ostream/ostream/write/
    // https://docs.opencv.org/3.4.7/d3/d63/classcv_1_1Mat.html#a13acd320291229615ef15f96ff1ff738
    ofs.write(image_bgr565.ptr<const char>(y),
              image_size.width * fb_info.bits_per_pixel / 8);
  }

  return 0;
}

struct framebuffer_info get_framebuffer_info(
    const char *framebuffer_device_path) {
  struct framebuffer_info fb_info;  // Used to return the required attrs.
  struct fb_var_screeninfo
      screen_info;  // Used to get attributes of the device from OS kernel.

  // open device with linux system call "open()"
  // https://man7.org/linux/man-pages/man2/open.2.html
  int fd = open(framebuffer_device_path, O_RDWR);

  // get attributes of the framebuffer device thorugh linux system call
  // "ioctl()". the command you would need is "FBIOGET_VSCREENINFO"
  // https://man7.org/linux/man-pages/man2/ioctl.2.html
  // https://www.kernel.org/doc/Documentation/fb/api.txt
  ioctl(fd, FBIOGET_VSCREENINFO, &screen_info);

  // put the required attributes in variable "fb_info" you found with "ioctl()
  // and return it."
  // fb_info.xres_virtual =       // 8
  // fb_info.bits_per_pixel =     // 16
  fb_info.xres_virtual = screen_info.xres;              // 8
  fb_info.bits_per_pixel = screen_info.bits_per_pixel;  // 16

  return fb_info;
};
