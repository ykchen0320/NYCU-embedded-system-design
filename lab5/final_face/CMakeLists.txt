CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

set(name "facerec")
project(facerec_cpp_samples)

#SET(OpenCV_DIR /path/to/your/opencv/installation)
SET(OpenCV_DIR /usr/local/arm-opencv/build)

# packages
find_package(OpenCV REQUIRED) # http://opencv.org

# probably you should loop through the sample files here


add_executable(facerec_eigenfaces facerec_eigenfaces.cpp)
target_link_libraries(facerec_eigenfaces opencv_face opencv_core opencv_imgproc opencv_highgui)

