cmake_minimum_required(VERSION 3.15)
project(minimalist_compliance_control LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_DYNAMIXEL "Build Dynamixel C++ extension" OFF)

if(BUILD_DYNAMIXEL)
  # ==================== Include Directories ====================
  set(DYNAMIXEL_DIR ${CMAKE_SOURCE_DIR}/real_world/dynamixel)
  include_directories(${DYNAMIXEL_DIR}/include)

  # ==================== Dynamixel SDK ====================
  file(GLOB SDK_SRCS ${DYNAMIXEL_DIR}/src/dynamixel_sdk/*.cpp)

  add_library(dynamixel_sdk STATIC ${SDK_SRCS})
  set_target_properties(dynamixel_sdk PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(
    dynamixel_sdk PUBLIC ${DYNAMIXEL_DIR}/include/dynamixel_sdk
  )

  # ==================== Dynamixel Control ====================
  file(GLOB CTRL_SRCS ${DYNAMIXEL_DIR}/src/dynamixel_control/*.cpp)

  add_library(dynamixel_control STATIC ${CTRL_SRCS})
  set_target_properties(dynamixel_control PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(
    dynamixel_control PUBLIC ${DYNAMIXEL_DIR}/include/dynamixel_control
  )
  target_link_libraries(dynamixel_control PUBLIC dynamixel_sdk)

  # ==================== Python Binding Module ====================
  find_package(pybind11 REQUIRED)

  pybind11_add_module(
    dynamixel_cpp
    real_world/dynamixel/src/dynamixel_mch.cpp
  )
  target_include_directories(dynamixel_cpp PRIVATE ${DYNAMIXEL_DIR}/include)
  target_link_libraries(dynamixel_cpp PRIVATE dynamixel_control)

  install(
    TARGETS dynamixel_cpp
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
    ARCHIVE DESTINATION .
  )
endif()
