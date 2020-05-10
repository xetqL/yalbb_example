cmake_minimum_required(VERSION 3.16)
project(yalbb_example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_COMPILER mpic++)

file(GLOB SRC src/*.cpp)
file(GLOB HEADERS src/*.hpp)

find_package(MPI REQUIRED)

execute_process(COMMAND mkdir -p build WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/yalbb)
execute_process(COMMAND cmake -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_LIST_DIR}/yalbb/build -DCMAKE_BUILD_TYPE=Release . WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/yalbb)
execute_process(COMMAND make -j 4 yalbb WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/yalbb)
execute_process(COMMAND make install yalbb WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/yalbb)

#find_library(YALBB_LIBRARY_PATH NAMES yalbb HINTS ${CMAKE_CURRENT_LIST_DIR}/yalbb/build/lib)
#find_path(YALBB_INCLUDE_DIRECTORY NAMES yalbb HINTS ${CMAKE_CURRENT_LIST_DIR}/yalbb/build/include)
find_library(YALBB_LIBRARY_PATH NAMES yalbb HINTS /home/xetql/yalbb/lib)
find_path(YALBB_INCLUDE_DIRECTORY NAMES yalbb HINTS /home/xetql/yalbb/include)
message(${YALBB_LIBRARY_PATH})
message(${YALBB_INCLUDE_DIRECTORY})

find_library(ZOLTAN_LIBRARY_PATH NAMES zoltan HINTS ${CMAKE_INSTALL_PREFIX}/lib)
find_path(ZOLTAN_INCLUDE_DIRECTORY NAMES zoltan HINTS ${CMAKE_INSTALL_PREFIX}/include)

add_executable(example ${SRC} ${HEADERS})

target_include_directories(example PRIVATE
        ${MPI_C_INCLUDE_PATH}
        ${YALBB_INCLUDE_DIRECTORY})

target_link_libraries(example PRIVATE
        ${MPI_C_LIBRARY}
        ${ZOLTAN_LIBRARY_PATH}
        ${YALBB_LIBRARY_PATH}
        stdc++fs)