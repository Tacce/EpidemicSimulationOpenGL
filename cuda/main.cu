#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define WIDTH 800
#define HEIGHT 800
#define GRID_SIZE 50

struct Vertex {
    float x, y;
    float r, g, b;
};

GLuint vbo;
GLuint vao;
cudaGraphicsResource* cudaVbo;

__global__ void updateColors(Vertex* vertices, float time) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= GRID_SIZE * GRID_SIZE * 6) return;

    int quad = i / 6;
    int x = quad % GRID_SIZE;
    int y = quad / GRID_SIZE;

    float r = 0.5f + 0.5f * sinf(time + x * 0.1f);
    float g = 0.5f + 0.5f * cosf(time + y * 0.1f);
    float b = 0.5f + 0.5f * sinf(time + y * 0.1f);

    vertices[i].r = r;
    vertices[i].g = g;
    vertices[i].b = b;
}

void createVBO() {
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * GRID_SIZE * GRID_SIZE * 6, nullptr, GL_DYNAMIC_DRAW);
    
    // Configura attributi
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    
    cudaError_t err = cudaGraphicsGLRegisterBuffer(&cudaVbo, vbo, cudaGraphicsMapFlagsWriteDiscard);
    if (err != cudaSuccess) {
        std::cerr << "Errore registrazione buffer CUDA: " << cudaGetErrorString(err) << std::endl;
    }
}

void fillVertices() {
    // Crea i dati vertices in memoria CPU
    std::vector<Vertex> vertices(GRID_SIZE * GRID_SIZE * 6);
    
    for (int y = 0; y < GRID_SIZE; y++) {
        for (int x = 0; x < GRID_SIZE; x++) {
            float fx = -1.0f + 2.0f * x / GRID_SIZE;
            float fy = -1.0f + 2.0f * y / GRID_SIZE;
            float w = 2.0f / GRID_SIZE;

            int idx = (y * GRID_SIZE + x) * 6;
            vertices[idx + 0] = {fx, fy, 1, 0, 0};
            vertices[idx + 1] = {fx + w, fy, 1, 0, 0};
            vertices[idx + 2] = {fx + w, fy + w, 1, 0, 0};
            vertices[idx + 3] = {fx, fy, 1, 0, 0};
            vertices[idx + 4] = {fx + w, fy + w, 1, 0, 0};
            vertices[idx + 5] = {fx, fy + w, 1, 0, 0};
        }
    }
    
    // Carica i dati nel buffer OpenGL
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size() * sizeof(Vertex), vertices.data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

GLuint compileShader() {
    const char* vSrc = R"(#version 330 core
    layout(location = 0) in vec2 aPos;
    layout(location = 1) in vec3 aColor;
    out vec3 vColor;
    void main() {
        gl_Position = vec4(aPos, 0.0, 1.0);
        vColor = aColor;
    })";
    const char* fSrc = R"(#version 330 core
    in vec3 vColor;
    out vec4 FragColor;
    void main() {
        FragColor = vec4(vColor, 1.0);
    })";

    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vSrc, nullptr);
    glCompileShader(vs);
    
    // Controlla errori vertex shader
    GLint success;
    GLchar infoLog[512];
    glGetShaderiv(vs, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vs, 512, nullptr, infoLog);
        std::cerr << "Errore compilazione vertex shader: " << infoLog << std::endl;
    }
    
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fSrc, nullptr);
    glCompileShader(fs);
    
    // Controlla errori fragment shader
    glGetShaderiv(fs, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fs, 512, nullptr, infoLog);
        std::cerr << "Errore compilazione fragment shader: " << infoLog << std::endl;
    }
    
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    
    // Controlla errori linking
    glGetProgramiv(prog, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(prog, 512, nullptr, infoLog);
        std::cerr << "Errore linking shader program: " << infoLog << std::endl;
    }
    
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

int main() {
    if (!glfwInit()) {
        std::cerr << "Errore inizializzazione GLFW" << std::endl;
        return -1;
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* win = glfwCreateWindow(WIDTH, HEIGHT, "CUDA OpenGL Demo", nullptr, nullptr);
    if (!win) {
        std::cerr << "Errore creazione finestra GLFW" << std::endl;
        glfwTerminate();
        return -1;
    }
    
    glfwMakeContextCurrent(win);
    if (!gladLoadGL((GLADloadfunc)glfwGetProcAddress)) {
        std::cerr << "Errore inizializzando GLAD" << std::endl;
        glfwTerminate();
        return -1;
    }

    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    
    int runtimeVersion;
    cudaRuntimeGetVersion(&runtimeVersion);
    std::cout << "CUDA Runtime Version: " << runtimeVersion << std::endl;

    // Prima imposta l'interoperabilità, poi verifica i device
    cudaError_t err = cudaGLSetGLDevice(0);
    if (err != cudaSuccess) {
        std::cerr << "Errore configurazione interoperabilità CUDA-OpenGL: " << cudaGetErrorString(err) << std::endl;
        glfwTerminate();
        return -1;
    }

    // Verifica che CUDA sia disponibile
    int deviceCount;
    err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        std::cerr << "Nessuna GPU CUDA disponibile! Errore: " << cudaGetErrorString(err) << std::endl;
        glfwTerminate();
        return -1;
    }
    std::cout << "GPU CUDA trovate: " << deviceCount << std::endl;

    createVBO();
    fillVertices();
    GLuint shader = compileShader();
    std::cout << "Inizializzazione completata!" << std::endl;

    while (!glfwWindowShouldClose(win)) {
        float t = glfwGetTime();
        Vertex* ptr;
        size_t size;
        cudaError_t err = cudaGraphicsMapResources(1, &cudaVbo);
        if (err != cudaSuccess) {
            std::cerr << "Errore mapping nel loop: " << cudaGetErrorString(err) << std::endl;
            break;
        }
        
        err = cudaGraphicsResourceGetMappedPointer((void**)&ptr, &size, cudaVbo);
        if (err != cudaSuccess) {
            std::cerr << "Errore getting pointer nel loop: " << cudaGetErrorString(err) << std::endl;
            cudaGraphicsUnmapResources(1, &cudaVbo);
            break;
        }

        int N = GRID_SIZE * GRID_SIZE * 6;
        int blockSize = 256;
        int gridSize = (N + blockSize - 1) / blockSize;
        updateColors<<<gridSize, blockSize>>>(ptr, t);
        
        // Sincronizza CUDA prima di unmappare
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "Errore sincronizzazione CUDA: " << cudaGetErrorString(err) << std::endl;
        }
        
        err = cudaGraphicsUnmapResources(1, &cudaVbo);
        if (err != cudaSuccess) {
            std::cerr << "Errore unmapping nel loop: " << cudaGetErrorString(err) << std::endl;
        }

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(shader);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, N);

        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    cudaGraphicsUnregisterResource(cudaVbo);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glfwTerminate();
    return 0;
}
