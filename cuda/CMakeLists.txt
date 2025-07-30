cmake_minimum_required(VERSION 3.18)
project(EpidemicCUDA VERSION 1.0 LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Abilita CUDA
enable_language(CUDA)

# Trova CUDA
find_package(CUDAToolkit REQUIRED)

# Aggiungi le directory esterne
add_subdirectory(${CMAKE_SOURCE_DIR}/../external/glfw ${CMAKE_BINARY_DIR}/glfw)

# Crea la libreria GLAD
add_library(glad ${CMAKE_SOURCE_DIR}/../external/glad/src/gl.c)
target_include_directories(glad PUBLIC ${CMAKE_SOURCE_DIR}/../external/glad/include)

# Crea l'eseguibile CUDA
add_executable(epidemic_cuda main.cu)

# Imposta proprietà CUDA
set_target_properties(epidemic_cuda PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75;86" # Modifica in base alla tua GPU
)

# Collega le librerie
target_link_libraries(epidemic_cuda 
    glfw 
    glad 
    CUDA::cudart
)

# Specifica i percorsi di inclusione
target_include_directories(epidemic_cuda PRIVATE 
    ${CMAKE_SOURCE_DIR}/../external/glad/include
    ${CMAKE_SOURCE_DIR}/../external/glfw/include
    ${CUDAToolkit_INCLUDE_DIRS}
)
