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

pthread_mutex_t mutex;
cv::Mat frame1;

void nonblocking() {
  struct termios t;
  tcgetattr(0, &t);
  t.c_lflag &= ~ICANON;
  tcsetattr(0, TCSANOW, &t);
  int flags = fcntl(0, F_GETFL, 0);
  fcntl(0, F_SETFL, flags | O_NONBLOCK);
}

// Function to capture 'c' and save the frame
void *save_frame_thread(void *arg) {
  int i = 0;  // To number screenshots

  while (true) {
    char key = getchar();  // Non-blocking key press
    if (key == 'c') {
      cv::Mat save_frame;

      pthread_mutex_lock(&mutex);  // Lock the mutex
      save_frame = frame1;
      pthread_mutex_unlock(&mutex);  // Unlock the mutex

      if (!save_frame.empty()) {
        std::string filename = "screenshot/" + std::to_string(i++) + ".bmp";
        cv::imwrite(filename, save_frame);
        std::cout << "Saved frame: " << filename << std::endl;
      }
    }

    usleep(1000);  // Sleep to avoid busy-waiting
  }

  return nullptr;
}

struct framebuffer_info {
  uint32_t bits_per_pixel;  // depth of framebuffer
  uint32_t xres_virtual;    // how many pixel in a row in virtual screen
  uint32_t yres_virtual;
};

struct framebuffer_info get_framebuffer_info(
    const char *framebuffer_device_path);

int main(int argc, const char *argv[]) {
  pthread_mutex_init(&mutex, nullptr);
  nonblocking();
  // variable to store the frame get from video stream
  cv::Mat frame, frame_bgr565;

  // open video stream device
  // https://docs.opencv.org/3.4.7/d8/dfe/classcv_1_1VideoCapture.html#a5d5f5dacb77bbebdcbfb341e3d4355c1
  cv::VideoCapture camera(2);

  // get info of the framebuffer
  framebuffer_info fb_info = get_framebuffer_info("/dev/fb0");

  // open the framebuffer device
  // http://www.cplusplus.com/reference/fstream/ofstream/ofstream/
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

  // set propety of the frame
  // https://docs.opencv.org/3.4.7/d8/dfe/classcv_1_1VideoCapture.html#a8c6d8c2d37505b5ca61ffd4bb54e9a7c
  // https://docs.opencv.org/3.4.7/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
  camera.set(CV_CAP_PROP_FRAME_WIDTH, 800);
  camera.set(CV_CAP_PROP_FRAME_HEIGHT, 600);
  camera.set(CV_CAP_PROP_FPS, 30);

  int i = 0;
  char name[16] = {0};

  pthread_t save_thread;
  if (pthread_create(&save_thread, nullptr, save_frame_thread, nullptr) != 0) {
    std::cerr << "Error: pthread create" << std::endl;
    return -1;
  }

  while (true) {
    // get video frame from stream
    // https://docs.opencv.org/3.4.7/d8/dfe/classcv_1_1VideoCapture.html#a473055e77dd7faa4d26d686226b292c1
    // https://docs.opencv.org/3.4.7/d8/dfe/classcv_1_1VideoCapture.html#a199844fb74226a28b3ce3a39d1ff6765
    // frame = ......
    if (!camera.read(frame)) {
      std::cerr << "Failed to capture frame." << std::endl;
      break;
    }

    pthread_mutex_lock(&mutex);
    frame1 = frame.clone();
    pthread_mutex_unlock(&mutex);

    // get size of the video frame
    // https://docs.opencv.org/3.4.7/d3/d63/classcv_1_1Mat.html#a146f8e8dda07d1365a575ab83d9828d1
    // frame_size = ......
    cv::Size2f frame_size = frame.size();
    int frame_width = frame_size.width;
    int frame_height = frame_size.height;
    int new_frame_width = (frame_width * fb_info.yres_virtual) / frame_height;
    cv::resize(frame, frame_bgr565,
               cv::Size(new_frame_width, fb_info.yres_virtual));
    int x_offset = (fb_info.xres_virtual - new_frame_width) / 2;

    // transfer color space from BGR to BGR565 (16-bit image) to fit the
    // requirement of the LCD
    // https://docs.opencv.org/3.4.7/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab
    // https://docs.opencv.org/3.4.7/d8/d01/group__imgproc__color__conversions.html#ga4e0972be5de079fed4e3a10e24ef5ef0
    cv::cvtColor(frame_bgr565, frame_bgr565, cv::COLOR_BGR2BGR565);

    // output the video frame to framebufer row by row
    for (int y = 0; y < frame_size.height; y++) {
      // move to the next written position of output device framebuffer by
      // "std::ostream::seekp()"
      // http://www.cplusplus.com/reference/ostream/ostream/seekp/
      ofs.seekp((y * fb_info.xres_virtual + x_offset) * fb_info.bits_per_pixel /
                8);

      // write to the framebuffer by "std::ostream::write()"
      // you could use "cv::Mat::ptr()" to get the pointer of the corresponding
      // row. you also need to cacluate how many bytes required to write to the
      // buffer http://www.cplusplus.com/reference/ostream/ostream/write/
      // https://docs.opencv.org/3.4.7/d3/d63/classcv_1_1Mat.html#a13acd320291229615ef15f96ff1ff738
      ofs.write(frame_bgr565.ptr<const char>(y),
                new_frame_width * fb_info.bits_per_pixel / 8);
    }
  }

  pthread_join(save_thread, nullptr);
  pthread_mutex_destroy(&mutex);

  // closing video stream
  // https://docs.opencv.org/3.4.7/d8/dfe/classcv_1_1VideoCapture.html#afb4ab689e553ba2c8f0fec41b9344ae6
  camera.release();
  ofs.close();
  cv::destroyAllWindows();

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