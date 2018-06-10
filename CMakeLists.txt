cmake_minimum_required(VERSION 3.7)
project(danznect)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g --no-warn")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 --no-warn")

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
find_package(libfreenect REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED core imgproc photo highgui)

include_directories("/usr/include/libusb-1.0/")

set(SOURCE_FILES
    danznect.cpp
    glWindowPos.h
    libfreenect_cpp11.hpp)

add_executable(danznect ${SOURCE_FILES})
target_link_libraries(danznect
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARY}
    ${GLEW_LIBRARY}
    ${FREENECT_LIBRARIES}
    Threads::Threads
    ${OpenCV_LIBS})