#! /bin/bash
set -e
git clone https://github.com/dog-qiuqiu/Yolo-FastestV2.git
cd Yolo-FastestV2
pip3 install -r requirements.txt
cd ..
git clone https://github.com/Tencent/ncnn.git
cd ncnn
echo "set(CMAKE_SYSTEM_NAME Linux)" > ./toolchains/toolchain-arm-linux.cmake
echo "set(CMAKE_SYSTEM_PROCESSOR arm)" >> ./toolchains/toolchain-arm-linux.cmake
echo "set(CMAKE_C_COMPILER arm-linux-gnueabihf-gcc)" >> ./toolchains/toolchain-arm-linux.cmake
echo "set(CMAKE_CXX_COMPILER arm-linux-gnueabihf-g++)" >> ./toolchains/toolchain-arm-linux.cmake
echo "set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)" >> ./toolchains/toolchain-arm-linux.cmake
echo "set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)" >> ./toolchains/toolchain-arm-linux.cmake
echo "set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)" >> ./toolchains/toolchain-arm-linux.cmake
echo "set(THREADS_PTHREAD_ARG \"-pthread\")" >> ./toolchains/toolchain-arm-linux.cmake
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=./../toolchains/toolchain-arm-linux.cmake
make -j$(nproc)
make install
cp -rf ./install/* ../../Yolo-FastestV2/sample/ncnn
