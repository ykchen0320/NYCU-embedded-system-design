#include <fcntl.h>
#include <linux/fb.h>
#include <sys/ioctl.h>

#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

std::vector<std::string> load_class_list() {
    std::vector<std::string> class_list;
    std::ifstream ifs("config_files/mask_class.txt");
    std::string line;
    while (getline(ifs, line)) {
        class_list.push_back(line);
    }
    return class_list;
}

struct framebuffer_info {
    uint32_t bits_per_pixel;  // framebuffer depth
    uint32_t xres_virtual;    // how many pixel in a row in virtual screen
};

struct framebuffer_info get_framebuffer_info(const char *framebuffer_device_path);

void load_net(cv::dnn::Net &net, bool is_cuda) {
    /*
    yolov5m
    yolov5n
    yolov5s
    yolov5l
    yolov5x
    */
    auto result = cv::dnn::readNet("config_files/sim_mask_yolov5_x.onnx");
    if (is_cuda) {
        std::cout << "Attempty to use CUDA\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    } else {
        std::cout << "Running on CPU\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    net = result;
}

const std::vector<cv::Scalar> colors = {cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255)};

const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.2;
const float NMS_THRESHOLD = 0.4;
const float CONFIDENCE_THRESHOLD = 0.4;

struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
};

cv::Mat format_yolov5(const cv::Mat &source) {
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

void detect(cv::Mat &image, cv::dnn::Net &net, std::vector<Detection> &output, const std::vector<std::string> &className) {
    cv::Mat blob;

    auto input_image = format_yolov5(image);

    cv::dnn::blobFromImage(input_image, blob, 1. / 255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
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
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    for (int i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }
}

int main(int argc, char **argv) {
    std::vector<std::string> class_list = load_class_list();

    // Load image
    std::string image_path = "example/demo_mask.png";  // Path to the image
    cv::Size2f image_size;
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Error: Unable to load image " << image_path << std::endl;
        return -1;
    }

    framebuffer_info fb_info = get_framebuffer_info("/dev/fb0");
    std::ofstream ofs("/dev/fb0");

    // bool is_cuda = argc > 1 && strcmp(argv[1], "cuda") == 0;
    bool is_cuda = 0;

    cv::dnn::Net net;
    load_net(net, is_cuda);

    std::vector<Detection> output;
    detect(image, net, output, class_list);
    int detections = output.size();

    for (int i = 0; i < detections; ++i) {
        auto detection = output[i];
        auto box = detection.box;
        auto classId = detection.class_id;
        //const auto color = colors[classId % colors.size()];
        if(classId == 1){
            cv::rectangle(image, box, colors[1], 0.5); // 改細
        }
        else{
            cv::rectangle(image, box, colors[0], 0.5); // 改細
        }
        //cv::rectangle(image, box, color, 3);

        //cv::rectangle(image, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
        //cv::putText(image, class_list[classId].c_str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::imwrite("example/output.png", image);

    image_size = image.size();
    cv::cvtColor(image, image, cv::COLOR_BGR2BGR565);

    // output to framebufer row by row
    for (int y = 0; y < image_size.height; y++) {
        ofs.seekp(y * fb_info.xres_virtual * fb_info.bits_per_pixel / 8);
        ofs.write(image.ptr<const char>(y), image_size.width * fb_info.bits_per_pixel / 8);
    }

    return 0;

    // auto start = std::chrono::high_resolution_clock::now();
    // int frame_count = 0;
    // float fps = -1;
    // int total_frames = 0;

    // while (true)
    // {
    //     capture.read(frame);
    //     if (frame.empty())
    //     {
    //         std::cout << "End of stream\n";
    //         break;
    //     }

    //     std::vector<Detection> output;
    //     detect(frame, net, output, class_list);

    //     frame_count++;
    //     total_frames++;

    //     int detections = output.size();

    //     for (int i = 0; i < detections; ++i)
    //     {

    //         auto detection = output[i];
    //         auto box = detection.box;
    //         auto classId = detection.class_id;
    //         const auto color = colors[classId % colors.size()];
    //         cv::rectangle(frame, box, color, 3);

    //         cv::rectangle(frame, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
    //         cv::putText(frame, class_list[classId].c_str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    //     }

    //     if (frame_count >= 30)
    //     {

    //         auto end = std::chrono::high_resolution_clock::now();
    //         fps = frame_count * 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    //         frame_count = 0;
    //         start = std::chrono::high_resolution_clock::now();
    //     }

    //     if (fps > 0)
    //     {

    //         std::ostringstream fps_label;
    //         fps_label << std::fixed << std::setprecision(2);
    //         fps_label << "FPS: " << fps;
    //         std::string fps_label_str = fps_label.str();

    //         cv::putText(frame, fps_label_str.c_str(), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
    //     }

    //     cv::imshow("output", frame);

    //     if (cv::waitKey(1) != -1)
    //     {
    //         capture.release();
    //         std::cout << "finished by user\n";
    //         break;
    //     }
    // }

    // std::cout << "Total frames: " << total_frames << "\n";

    // return 0;
}

struct framebuffer_info get_framebuffer_info(const char *framebuffer_device_path) {
    struct framebuffer_info fb_info;       // Used to return the required attrs.
    struct fb_var_screeninfo screen_info;  // Used to get attributes of the device from OS kernel.

    int buffer_fd = open(framebuffer_device_path, O_RDWR);
    ioctl(buffer_fd, FBIOGET_VSCREENINFO, &screen_info);

    // put the required attributes in variable "fb_info" you found with "ioctl() and return it."
    fb_info.xres_virtual = screen_info.xres_virtual;      // it is = 8
    fb_info.bits_per_pixel = screen_info.bits_per_pixel;  // it is = 16

    return fb_info;
};
