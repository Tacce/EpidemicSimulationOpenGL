cmake_minimum_required(VERSION 3.10)
project(SIR_sequential)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(external/glfw)

# Crea la libreria GLAD
add_library(glad external/glad/src/gl.c)
target_include_directories(glad PUBLIC external/glad/include)

# Crea l'eseguibile
add_executable(SIR_sequential main.cpp)

# Collega le librerie e specifica i percorsi di inclusione
target_link_libraries(SIR_sequential glfw glad)
target_include_directories(SIR_sequential PRIVATE 
    external/glad/include
    external/glfw/include
)
