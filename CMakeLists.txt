cmake_minimum_required(VERSION 2.8)
project( LDBMImageBackgroundRemover )
set(CMAKE_VERBOSE_MAKEFILE TRUE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -v")
aux_source_directory(src SOURCE_FILES)
#set(SOURCE_FILES src/main.cpp src/LDBM.h src/LDBM.cpp)
find_package( OpenCV REQUIRED )
find_package( Eigen3 REQUIRED )

add_executable( LDBMImageBackgroundRemover ${SOURCE_FILES} )

target_link_libraries( LDBMImageBackgroundRemover ${OpenCV_LIBS} )
target_link_libraries( LDBMImageBackgroundRemover ${Eigen_LIBS} )
