cmake_minimum_required(VERSION 3.26)
project(SS_tree)


include_directories("C:/Users/rivas/OneDrive/Documentos/EDA/Lab/eigen-3.4.0/eigen-3.4.0")

set(CMAKE_CXX_STANDARD 20)

add_executable(SS_tree main.cpp
        Point.h Point.cpp Data.h SSTree.cpp SSTree.h)
