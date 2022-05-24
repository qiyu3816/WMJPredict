project(Predict)
set(CMAKE_CXX_STANDARD 17)
cmake_minimum_required(VERSION 3.1)

if(EXISTS /usr/bin/clang AND EXISTS /usr/bin/clang++)
   set(CMAKE_C_COMPILER /usr/bin/clang)
   set(CMAKE_CXX_COMPILER /usr/bin/clang++)
endif()

find_package(OpenCV REQUIRED)
include_directories(include ${OpenCV_INCLUDE_DIRS})
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
include_directories(include ${Python3_INCLUDE_DIRS})
include_directories("/usr/local/include/matplotlibcpp17/")

include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
get_property(include_dir DIRECTORY ${CMAKE_CURRENT_LIST_DIR} PROPERTY INCLUDE_DIRECTORIES)
set(PREDICT_INCLUDE ${include_dir})

add_library(KF SHARED ./src/KF.cpp)
target_link_libraries(KF ${OpenCV_LIBS})

add_library(EKF SHARED ./src/EKF.cpp)
target_link_libraries(EKF ${OpenCV_LIBS})

add_definitions(-DKF_CFG="${CMAKE_CURRENT_LIST_DIR}/KF.yaml")
exec_program(ln ARGS -s ${CMAKE_CURRENT_LIST_DIR}/KF.yaml ${PROJECT_BINARY_DIRR} OUTPUT_VARIABLE LN_OUT RETURN_VALUE LN_RET)
exec_program(ln ARGS -s ${CMAKE_CURRENT_LIST_DIR}/EKF.yaml ${PROJECT_BINARY_DIRR} OUTPUT_VARIABLE LN_OUT RETURN_VALUE LN_RET)

set(TEST_PATH ${CMAKE_CURRENT_LIST_DIR}/test)
macro(add_test_file TEST_FILE)
    add_executable(${TEST_FILE} ${TEST_PATH}/${TEST_FILE}.cpp)
    target_link_libraries(${TEST_FILE} EKF KF ${Python3_LIBRARIES})
endmacro()

add_test_file(EKFTest)
add_test_file(KFTest)
add_test_file(BarTest)