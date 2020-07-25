cmake_minimum_required(VERSION 3.0)

set(PROJ_NAME "robot_camera")
project(${PROJ_NAME})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")
string(STRIP "${CMAKE_C_FLAGS}" CMAKE_C_FLAGS)
string(STRIP "${CMAKE_CXX_FLAGS}" CMAKE_CXX_FLAGS)
message(STATUS "C_FLAGS: ${CMAKE_C_FLAGS}")
message(STATUS "CXX_FLAGS: ${CMAKE_CXX_FLAGS}")


find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBS})

add_definitions(-DLOG_TAG=MYNTEYE)
find_package(mynteyed REQUIRED)
message(STATUS "Found mynteye: ${mynteyed_VERSION}")

include_directories(
  "/usr/local/include/mynteyed"
  "src/util"
  ${OpenCV_INCLUDE_DIRS}
  ${CMAKE_CURRENT_LIST_DIR}/src
)

find_package(pybind11 REQUIRED)
pybind11_add_module(${PROJ_NAME} src/main.cpp src/util/mat_warper.cpp src/util/mat_warper.h src/util/cam_utils.cc)

target_link_libraries(${PROJ_NAME} PRIVATE pybind11::module mynteye_depth ${OpenCV_LIBS})

find_package(Python)
find_path(PYTHON_SITE_PACKAGES site-packages ${PYTHON_INCLUDE_PATH}/..)

install(TARGETS ${PROJ_NAME} RUNTIME DESTINATION ${PYTHON_SITE_PACKAGES}
							 LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES}
							 ARCHIVE DESTINATION ${PYTHON_SITE_PACKAGES}
							 )