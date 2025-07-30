cmake_minimum_required(VERSION 3.10)
project(GL_SIR_sequential)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(${CMAKE_SOURCE_DIR}/../external/glfw ${CMAKE_BINARY_DIR}/glfw)

# Crea la libreria GLAD
add_library(glad ${CMAKE_SOURCE_DIR}/../external/glad/src/gl.c)
target_include_directories(glad PUBLIC ${CMAKE_SOURCE_DIR}/../external/glad/include)

# Crea l'eseguibile
add_executable(GL_SIR_sequential main.cpp)

# Collega le librerie e specifica i percorsi di inclusione
target_link_libraries(GL_SIR_sequential glfw glad)
target_include_directories(GL_SIR_sequential PRIVATE 
    ${CMAKE_SOURCE_DIR}/../external/glad/include
    ${CMAKE_SOURCE_DIR}/../external/glfw/include
)
