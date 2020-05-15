cmake_minimum_required(VERSION 3.15)
project(yalbb_example)
# Set compiler params
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_COMPILER mpic++)
# Look for my files
file(GLOB SRC src/*.cpp)
file(GLOB HEADERS src/*.hpp)
# Produce YALBB library
execute_process(COMMAND mkdir -p build WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/yalbb)
execute_process(COMMAND cmake -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_LIST_DIR}/yalbb/build
        -DCMAKE_BUILD_TYPE=Release . WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/yalbb)
execute_process(COMMAND make -j 4 yalbb WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/yalbb)
execute_process(COMMAND make install yalbb WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/yalbb)
find_package(Threads)
# find MPI
find_package(MPI REQUIRED)
# find YALBB
find_library(YALBB_LIBRARY_PATH NAMES yalbb HINTS ${CMAKE_CURRENT_LIST_DIR}/yalbb/lib)
find_path(YALBB_INCLUDE_DIRECTORY NAMES yalbb HINTS ${CMAKE_CURRENT_LIST_DIR}/yalbb/includes)


# find my LB library (zoltan as an example)
set(ZOLTAN_INSTALL_FOLDER /usr/local CACHE PATH "zoltan install folder")
find_library(ZOLTAN_LIBRARY_PATH NAMES zoltan HINTS ${ZOLTAN_INSTALL_FOLDER}/lib)
find_path(ZOLTAN_INCLUDE_DIRECTORY NAMES zoltan.h HINTS ${ZOLTAN_INSTALL_FOLDER}/include)
# Build
add_executable(example ${SRC} ${HEADERS})
target_include_directories(example PRIVATE ${MPI_C_INCLUDE_PATH} ${YALBB_INCLUDE_DIRECTORY} ${ZOLTAN_INCLUDE_DIRECTORY} )
target_link_libraries(example PRIVATE ${MPI_C_LIBRARY} ${ZOLTAN_LIBRARY_PATH}  ${YALBB_LIBRARY_PATH} stdc++fs ${ZOLTAN_LIBRARY_PATH} ${CMAKE_THREAD_LIBS_INIT})
