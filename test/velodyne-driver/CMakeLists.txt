cmake_minimum_required(VERSION 3.0)

set(PROJ_NAME "velodyne")
project(${PROJ_NAME})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")

find_package(pybind11 REQUIRED)
pybind11_add_module(${PROJ_NAME} src/velodyne.cpp)

target_link_libraries(${PROJ_NAME} PRIVATE pybind11::module)

find_package(Python)
find_path(PYTHON_SITE_PACKAGES site-packages ${PYTHON_INCLUDE_PATH}/..)

install(TARGETS ${PROJ_NAME} RUNTIME DESTINATION ${PYTHON_SITE_PACKAGES}
							 LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES}
							 ARCHIVE DESTINATION ${PYTHON_SITE_PACKAGES}
							 )