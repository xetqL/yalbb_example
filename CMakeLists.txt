cmake_minimum_required(VERSION 3.16)
project(yalbb_example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_COMPILER mpic++)

file(GLOB SRC include/*.cpp)
file(GLOB HEADERS include/*.hpp)

find_package(MPI REQUIRED)

find_library(YALBB_LIBRARY_PATH NAMES yalbb HINTS /home/xetql/ljmpi/build/lib)
find_path(YALBB_INCLUDE_DIRECTORY NAMES yalbb HINTS /home/xetql/ljmpi/build/include)

find_library(ZOLTAN_LIBRARY_PATH NAMES zoltan HINTS ${CMAKE_INSTALL_PREFIX}/lib)
find_path(ZOLTAN_INCLUDE_DIRECTORY NAMES zoltan HINTS ${CMAKE_INSTALL_PREFIX}/include)

message(${ZOLTAN_LIBRARY_PATH})

add_executable(example example.cpp ${SRC})

target_include_directories(example PRIVATE
        ${MPI_C_INCLUDE_PATH}
        ${YALBB_INCLUDE_DIRECTORY})
target_link_libraries(example PRIVATE
        ${MPI_C_LIBRARY}
        ${ZOLTAN_LIBRARY_PATH}
        ${YALBB_LIBRARY_PATH}
        stdc++fs)