#include <fcntl.h>
#include <linux/fb.h>
#include <pthread.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <thread>

#include "yolo-fastestv2.h"

struct framebuffer_info {
  uint32_t bits_per_pixel;  // depth of framebuffer
  uint32_t xres_virtual;    // how many pixel in a row in virtual screen
  uint32_t yres_virtual;
};

struct framebuffer_info get_framebuffer_info(
    const char *framebuffer_device_path);

int main(int argc, const char *argv[]) {
  static const char *class_names[] = {
      "person",        "bicycle",      "car",
      "motorcycle",    "airplane",     "bus",
      "train",         "truck",        "boat",
      "traffic light", "fire hydrant", "stop sign",
      "parking meter", "bench",        "bird",
      "cat",           "dog",          "horse",
      "sheep",         "cow",          "elephant",
      "bear",          "zebra",        "giraffe",
      "backpack",      "umbrella",     "handbag",
      "tie",           "suitcase",     "frisbee",
      "skis",          "snowboard",    "sports ball",
      "kite",          "baseball bat", "baseball glove",
      "skateboard",    "surfboard",    "tennis racket",
      "bottle",        "wine glass",   "cup",
      "fork",          "knife",        "spoon",
      "bowl",          "banana",       "apple",
      "sandwich",      "orange",       "broccoli",
      "carrot",        "hot dog",      "pizza",
      "donut",         "cake",         "chair",
      "couch",         "potted plant", "bed",
      "dining table",  "toilet",       "tv",
      "laptop",        "mouse",        "remote",
      "keyboard",      "cell phone",   "microwave",
      "oven",          "toaster",      "sink",
      "refrigerator",  "book",         "clock",
      "vase",          "scissors",     "teddy bear",
      "hair drier",    "toothbrush"};

  yoloFastestv2 api;

  api.loadModel("./model/yolo-fastestv2-opt.param",
                "./model/yolo-fastestv2-opt.bin");

  std::vector<TargetBox> boxes;
  // variable to store the frame get from video stream
  int frame_rate = 10;
  cv::Mat frame;

  // open video stream device
  // https://docs.opencv.org/3.4.7/d8/dfe/classcv_1_1VideoCapture.html#a5d5f5dacb77bbebdcbfb341e3d4355c1
  cv::VideoCapture camera(2);

  // get info of the framebuffer
  // fb_info = ......
  framebuffer_info fb_info = get_framebuffer_info("/dev/fb0");

  // open the framebuffer device
  // http://www.cplusplus.com/reference/fstream/ofstream/ofstream/
  // ofs = ......
  std::ofstream ofs("/dev/fb0");

  // check if video stream device is opened success or not
  // https://docs.opencv.org/3.4.7/d8/dfe/classcv_1_1VideoCapture.html#a9d2ca36789e7fcfe7a7be3b328038585
  // if( !check )
  // {
  //     std::cerr << "Could not open video device." << std::endl;
  //     return 1;
  // }
  if (!camera.isOpened()) {
    std::cerr << "camera can't open" << std::endl;
    return 1;
  }

  camera.set(CV_CAP_PROP_FRAME_HEIGHT, fb_info.yres_virtual);
  camera.set(CV_CAP_PROP_FPS, frame_rate);
  cv::Size2f image_size;

  // set propety of the frame
  // https://docs.opencv.org/3.4.7/d8/dfe/classcv_1_1VideoCapture.html#a8c6d8c2d37505b5ca61ffd4bb54e9a7c
  // https://docs.opencv.org/3.4.7/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
  int framebuffer_width = fb_info.xres_virtual;
  int framebuffer_depth = fb_info.bits_per_pixel;
  int w = camera.get(cv::CAP_PROP_FRAME_WIDTH);
  int h = camera.get(cv::CAP_PROP_FRAME_HEIGHT);

  while (true) {
    // get video frame from stream
    // https://docs.opencv.org/3.4.7/d8/dfe/classcv_1_1VideoCapture.html#a473055e77dd7faa4d26d686226b292c1
    // https://docs.opencv.org/3.4.7/d8/dfe/classcv_1_1VideoCapture.html#a199844fb74226a28b3ce3a39d1ff6765
    // frame = ......
    if (!camera.read(frame)) {
      std::cerr << "Failed to capture frame." << std::endl;
      break;
    }

    api.detection(frame, boxes);

    for (int i = 0; i < boxes.size(); i++) {
      std::cout << boxes[i].x1 << " " << boxes[i].y1 << " " << boxes[i].x2
                << " " << boxes[i].y2 << " " << boxes[i].score << " "
                << boxes[i].cate << std::endl;

      char text[256];
      sprintf(text, "%s %.1f%%", class_names[boxes[i].cate],
              boxes[i].score * 100);

      int baseLine = 0;
      cv::Size label_size =
          cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

      int x = boxes[i].x1;
      int y = boxes[i].y1 - label_size.height - baseLine;
      if (y < 0) y = 0;
      if (x + label_size.width > frame.cols) x = frame.cols - label_size.width;

      cv::rectangle(
          frame,
          cv::Rect(cv::Point(x, y),
                   cv::Size(label_size.width, label_size.height + baseLine)),
          cv::Scalar(255, 255, 255), -1);

      cv::putText(frame, text, cv::Point(x, y + label_size.height),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

      cv::rectangle(frame, cv::Point(boxes[i].x1, boxes[i].y1),
                    cv::Point(boxes[i].x2, boxes[i].y2),
                    cv::Scalar(255, 255, 0), 2, 2, 0);
    }

    // get size of the video frame
    // https://docs.opencv.org/3.4.7/d3/d63/classcv_1_1Mat.html#a146f8e8dda07d1365a575ab83d9828d1
    // frame_size = ......
    cv::Size2f frame_size = frame.size();
    cv::Mat frame_bgr565;

    // transfer color space from BGR to BGR565 (16-bit image) to fit the
    // requirement of the LCD
    // https://docs.opencv.org/3.4.7/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab
    // https://docs.opencv.org/3.4.7/d8/d01/group__imgproc__color__conversions.html#ga4e0972be5de079fed4e3a10e24ef5ef0
    cv::cvtColor(frame, frame_bgr565, cv::COLOR_BGR2BGR565);

    // output the video frame to framebufer row by row
    for (int y = 0; y < frame_size.height; y++) {
      // move to the next written position of output device framebuffer by
      // "std::ostream::seekp()"
      // http://www.cplusplus.com/reference/ostream/ostream/seekp/
      ofs.seekp(y * framebuffer_width * 2 +
                (framebuffer_width - frame_size.width));

      // write to the framebuffer by "std::ostream::write()"
      // you could use "cv::Mat::ptr()" to get the pointer of the corresponding
      // row. you also need to cacluate how many bytes required to write to the
      // buffer http://www.cplusplus.com/reference/ostream/ostream/write/
      // https://docs.opencv.org/3.4.7/d3/d63/classcv_1_1Mat.html#a13acd320291229615ef15f96ff1ff738
      ofs.write(frame_bgr565.ptr<const char>(y),
                frame_size.width * fb_info.bits_per_pixel / 8);
    }
  }

  // closing video stream
  // https://docs.opencv.org/3.4.7/d8/dfe/classcv_1_1VideoCapture.html#afb4ab689e553ba2c8f0fec41b9344ae6
  camera.release();

  return 0;
}

struct framebuffer_info get_framebuffer_info(
    const char *framebuffer_device_path) {
  struct framebuffer_info fb_info;  // Used to return the required attrs.
  struct fb_var_screeninfo
      screen_info;  // Used to get attributes of the device from OS kernel.

  // open deive with linux system call "open( )"
  // https://man7.org/linux/man-pages/man2/open.2.html
  int fd = open(framebuffer_device_path, O_RDWR);

  // get attributes of the framebuffer device thorugh linux system call
  // "ioctl()" the command you would need is "FBIOGET_VSCREENINFO"
  // https://man7.org/linux/man-pages/man2/ioctl.2.html
  // https://www.kernel.org/doc/Documentation/fb/api.txt
  ioctl(fd, FBIOGET_VSCREENINFO, &screen_info);

  // put the required attributes in variable "fb_info" you found with "ioctl()
  // and return it." fb_info.xres_virtual = ...... fb_info.bits_per_pixel =
  // ......
  fb_info.xres_virtual = screen_info.xres;
  fb_info.bits_per_pixel = screen_info.bits_per_pixel;
  fb_info.yres_virtual = screen_info.yres;

  return fb_info;
};