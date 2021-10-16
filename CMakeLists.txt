cmake_minimum_required(VERSION 2.8.11)
project(simExtDDPGquad)
set(CMAKE_CXX_STANDARD 17)
set(LIBPLUGIN_DIR libPlugin)
set(COPPELIASIM_ROOT_DIR /opt/vrep)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libPlugin/cmake/")
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_MACOSX_RPATH 1)

# uncomment following if plugin uses Qt:
#find_package(Qt5 COMPONENTS Core Gui Widgets PrintSupport REQUIRED)
#set(CMAKE_AUTOMOC ON)
#add_definitions(-DQT_COMPIL)

if(NOT LIBPLUGIN_DIR)
    if(DEFINED ENV{COPPELIASIM_ROOT_DIR})
        set(LIBPLUGIN_DIR $ENV{COPPELIASIM_ROOT_DIR}/programming/libPlugin)
    else()
        set(LIBPLUGIN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../libPlugin)
    endif()
endif()
list(APPEND CMAKE_MODULE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules
    ${LIBPLUGIN_DIR}/cmake)
find_package(CoppeliaSim 4.1.0 REQUIRED)
include_directories(include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/generated)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external)

coppeliasim_generate_stubs(${CMAKE_CURRENT_BINARY_DIR}/generated XML_FILE ${CMAKE_CURRENT_SOURCE_DIR}/callbacks.xml LUA_FILE ${CMAKE_CURRENT_SOURCE_DIR}/simSkel.lua)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/config.h ESCAPE_QUOTES)



set(SOURCES
        libPlugin/simPlusPlus/Plugin.cpp
        ddpg_plugin/generate_sample.cpp
        ddpg_plugin/keras_model.cc
        )

set(HEADER

        include/header.h
        libPlugin/simPlusPlus/Plugin.h
        ddpg_plugin/generate_sample.h
        ddpg_plugin/keras_model.h

        )





set(TRAJ_SOURCE
        motiongeneration.cpp
        trajrunnable.cpp
        TrajGenerator/RapidTrajectoryGenerator.cpp
        TrajGenerator/SingleAxisTrajectory.cpp
        )

set(TRAJ_HEADER
        include/motiongeneration.h
        include/trajrunnable.h

        TrajGenerator/RootFinder/cubic.hpp
        TrajGenerator/RootFinder/quadratic.hpp
        TrajGenerator/RootFinder/quartic.hpp
        TrajGenerator/RootFinder/quartic_descartes.hpp
        TrajGenerator/RootFinder/quartic_error.hpp
        TrajGenerator/RootFinder/quartic_ferrari.hpp
        TrajGenerator/RootFinder/quartic_neumark.hpp
        TrajGenerator/RootFinder/quartic_yacfraid.hpp
        TrajGenerator/RapidTrajectoryGenerator.h
        TrajGenerator/SingleAxisTrajectory.h
        TrajGenerator/Vec3.h
        )




coppeliasim_add_plugin(${PROJECT_NAME} SOURCES
        plugin.cpp
        motiongeneration.cpp
        trajrunnable.cpp

        ${SOURCES}
        ${HEADER}
        ${TRAJ_SOURCE}
        ${TRAJ_HEADER}
        )
#target_link_libraries(simExtSkel Qt5::Core Qt5::Gui Qt5::Widgets Qt5::PrintSupport)
