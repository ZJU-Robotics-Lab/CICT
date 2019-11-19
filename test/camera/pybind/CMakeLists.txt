cmake_minimum_required(VERSION 3.0)

set(PROJ_NAME "robot_camera")
project(${PROJ_NAME})

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBS})

find_package(pybind11 REQUIRED)
pybind11_add_module(${PROJ_NAME} main.cpp mat_warper.cpp mat_warper.h)

target_link_libraries(${PROJ_NAME} PRIVATE pybind11::module ${OpenCV_LIBS})

find_package(Python)
find_path(PYTHON_SITE_PACKAGES site-packages ${PYTHON_INCLUDE_PATH}/..)

install(TARGETS ${PROJ_NAME} RUNTIME DESTINATION ${PYTHON_SITE_PACKAGES}
							 LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES}
							 ARCHIVE DESTINATION ${PYTHON_SITE_PACKAGES}
							 )