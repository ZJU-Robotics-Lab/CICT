cmake_minimum_required(VERSION 3.0)

set(PROJ_NAME "get_image")

project(${PROJ_NAME})
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")
string(STRIP "${CMAKE_C_FLAGS}" CMAKE_C_FLAGS)
string(STRIP "${CMAKE_CXX_FLAGS}" CMAKE_CXX_FLAGS)
message(STATUS "C_FLAGS: ${CMAKE_C_FLAGS}")
message(STATUS "CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

set(OUTPUT_DIR "${CMAKE_CURRENT_LIST_DIR}/output")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBS})


add_definitions(-DLOG_TAG=MYNTEYE)

find_package(mynteyed REQUIRED)
message(STATUS "Found mynteye: ${mynteyed_VERSION}")
if(NOT mynteyed_WITH_OPENCV)
  message(FATAL_ERROR "Please install OpenCV, then rebuild mynteye with it."
      " Because these samples use OpenCV to show images.")
endif()


include_directories(
  "/usr/local/include/mynteyed"
  "/usr/local/include/mynteyed"
  "src/util"
  ${OpenCV_INCLUDE_DIRS}
  ${CMAKE_CURRENT_LIST_DIR}/src
)

#set(MYNTEYE_DLL_SEARCH_PATHS
#  ${PRO_DIR}/_install/bin
#  ${PRO_DIR}/_install/bin/3rdparty
#  ${OpenCV_LIB_SEARCH_PATH}
#)

add_executable(${PROJ_NAME}
    src/get_image.cc
    src/util/cam_utils.cc
    src/util/cv_painter.cc
)

target_link_libraries(${PROJ_NAME} mynteye_depth ${OpenCV_LIBS})

install(TARGETS ${PROJ_NAME} RUNTIME DESTINATION ${OUTPUT_DIR}
               LIBRARY DESTINATION ${OUTPUT_DIR}
               ARCHIVE DESTINATION ${OUTPUT_DIR}
               )