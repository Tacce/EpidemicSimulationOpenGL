#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>


const int GRID_SIZE_X = 1024;
const int GRID_SIZE_Y = 288;

const float CELL_WIDTH  = 2.0f / GRID_SIZE_X;
const float CELL_HEIGHT = 2.0f / GRID_SIZE_Y;

const int VERTICES_COUNT = GRID_SIZE_X * GRID_SIZE_Y * 6;
const int NUM_NODES = GRID_SIZE_X * GRID_SIZE_Y;

const int STEP_TIMER = 10; // Milliseconds

struct Vertex {
    float x, y;
    float r, g, b;
};

// Initialize GLFW and create a window

GLFWwindow* initWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);

    GLFWwindow* window = glfwCreateWindow(
        mode->width, mode->height,
        "SIR Epidemic Simulation CUDA",
        monitor,
        nullptr
    );
    if (!window) {
        std::cerr << "Errore nella creazione della finestra\n";
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(window);
    if (!gladLoadGL((GLADloadfunc)glfwGetProcAddress)) {
        std::cerr << "Errore inizializzando GLAD\n";
        exit(EXIT_FAILURE);
    }
    return window;
}

GLuint vbo;
GLuint vao;
cudaGraphicsResource* cudaVbo;

__device__ void updateColors(int node_index, float r, float g, float b, Vertex* vertices) {
    int base = node_index * 6;    
    for (int k = 0; k < 6; ++k) {
        vertices[base + k].r = r;
        vertices[base + k].g = g;
        vertices[base + k].b = b;
    }
}

__device__ uint32_t xorshift32(uint32_t &state){
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

__device__ float rand_uniform(uint32_t &state){
    return (xorshift32(state) & 0xFFFFFF) / float(0x1000000);
}

__global__ void simulate_step(double p, double q,int si, int step, int active_infections,
                   int* Levels, int* ImmuneCountdown, int* ImmuneStep, Vertex* vertices, int base_seed) {
    
    const int dx[] = {-1, 1, 0, 0};
    const int dy[] = {0, 0, -1, 1};
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= GRID_SIZE_X * GRID_SIZE_Y) return; // indice fuori dai limiti

    uint32_t prng_state = base_seed
                        ^ (i * 0x9E3779B9u)   // Spread by node index
                        ^ (step * 0x85EBCA6Bu) // Spread by simulation step
                        ^ ((i << 16) ^ (step << 8)); // Mix shifting
    if (Levels[i] == step) {
            int row = i / GRID_SIZE_X;
            int col = i % GRID_SIZE_X;            
            // Infezione dei vicini
            for (int j = 0; j < 4; j++) {
                int new_row = row + dx[j];
                int new_col = col + dy[j];
                // Controlla se il vicino è dentro i confini della griglia
                if (new_row >= 0 && new_row < GRID_SIZE_Y && 
                    new_col >= 0 && new_col < GRID_SIZE_X) {

                    int neighbor = new_row * GRID_SIZE_X + new_col;
                    
                    if (Levels[neighbor] < step && ImmuneStep[neighbor] <step && rand_uniform(prng_state) < p) {
                        Levels[neighbor] = step + 1;
                        active_infections++;
                        updateColors(neighbor, 1.0f, 0.0f, 0.0f, vertices); // Aggiorna colore a rosso
                    }
                }
            }
            // Simula la guarigione
            if (rand_uniform(prng_state) < q) {
                ImmuneCountdown[i] = si; // Nodo recuperato - imposta countdown immunità
                ImmuneStep[i] = step + 1; // Nodo immunizzato al prossimo step
                active_infections--;
                updateColors(i, 0.0f, 0.0f, 1.0f, vertices); 
            } else {
                Levels[i] = step + 1; // Nodo può infettare anche al prossimo step
            }
        }
        else if (ImmuneCountdown[i] > 0) {
            ImmuneCountdown[i]--; // Decrementa il contatore di immunità
            if (ImmuneCountdown[i] > 0) {
                ImmuneStep[i] = step + 1; // Aggiorna lo step di immunizzazione
            } 
            updateColors(i, 0.0f, 0.0f, ((float)ImmuneCountdown[i]/si), vertices); // Aggiorna colore a blu
        }
}

void createVBO() {
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * GRID_SIZE_X * GRID_SIZE_Y * 6, nullptr, GL_DYNAMIC_DRAW);

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

void initVertices(Vertex* vertices, int first_infected_node = 0) {
    int index = 0;
    for (int j = 0; j < GRID_SIZE_Y; ++j) {
        for (int i = 0; i < GRID_SIZE_X; ++i){
            float x = -1.0f + i * CELL_WIDTH;
            float y = 1.0f - (j + 1) * CELL_HEIGHT;

            //float r = (i + j) % 2 == 0 ? 1.0f : 0.0f; // Alterna i colori tra rosso e nero
            float r = index/6 ==first_infected_node ? 1.0f : 0.0f; // Solo il primo nodo è rosso
            
            // 2 triangoli per formare un quadrato
            vertices[index++] = { x, y, r, 0, 0 };
            vertices[index++] = { x + CELL_WIDTH, y, r, 0, 0 };
            vertices[index++] = { x + CELL_WIDTH, y + CELL_HEIGHT, r, 0, 0 };

            vertices[index++] = { x, y, r, 0, 0 };
            vertices[index++] = { x + CELL_WIDTH, y + CELL_HEIGHT, r, 0, 0 };
            vertices[index++] = { x, y + CELL_HEIGHT, r, 0, 0 };
        }
    }

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, VERTICES_COUNT * sizeof(Vertex), vertices);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void initLevelsAndImmune(int* Levels, int* ImmuneCountdown, int* ImmuneStep, int first_infected_node = 0) {
    for (int i = 0; i < NUM_NODES; ++i) {
        Levels[i] = -1; // Inizializza tutti i nodi come non infetti
        ImmuneCountdown[i] = 0; // Inizializza tutti i nodi come non immuni
        ImmuneStep[i] = -1; // Inizializza tutti i nodi come mai stati immuni
    }
    Levels[first_infected_node] = 0; // Il primo nodo è infetto all'inizio
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
    GLFWwindow* window = initWindow();
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

    int first_infected_node = 0; // Nodo inizialmente infetto

    // Allocazione dinamica degli array
    Vertex* vertices = new Vertex[VERTICES_COUNT];
    int* Levels = new int[NUM_NODES];
    int* ImmuneCountdown = new int[NUM_NODES];
    int* ImmuneStep = new int[NUM_NODES];
    
    initVertices(vertices, first_infected_node);
    initLevelsAndImmune(Levels, ImmuneCountdown, ImmuneStep, first_infected_node);

    // Copia i dati dei vertici nella memoria GPU
    int* d_levels;
    int* d_immuneCountdown;
    int* d_immuneStep;

    cudaMalloc((void**)&d_levels, sizeof(int) * NUM_NODES);
    cudaMalloc((void**)&d_immuneCountdown, sizeof(int) * NUM_NODES);
    cudaMalloc((void**)&d_immuneStep, sizeof(int) * NUM_NODES);
    
    cudaMemcpy(d_levels, Levels, sizeof(int) * NUM_NODES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_immuneCountdown, ImmuneCountdown, sizeof(int) * NUM_NODES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_immuneStep, ImmuneStep, sizeof(int) * NUM_NODES, cudaMemcpyHostToDevice); 

    GLuint shader = compileShader();
    std::cout << "Inizializzazione completata!" << std::endl;

    int active_infections = 1; // Inizio con un'infezione
    int step = 0;

    float p = 0.5; // Probabilità di infezione
    float q = 0.5; // Probabilità di guarigione
    int si = 50; // Durata dell'immunità

    int N = GRID_SIZE_X * GRID_SIZE_Y * 6;

    auto lastUpdate = std::chrono::steady_clock::now();

    while (!glfwWindowShouldClose(window)) {
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastUpdate).count() > STEP_TIMER) {
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

            int blockSize = 64;
            int gridSize = (GRID_SIZE_X*GRID_SIZE_Y + blockSize - 1) / blockSize;
            unsigned int base_seed = static_cast<unsigned int>(time(nullptr)) ^ rand();

            simulate_step<<<gridSize, blockSize>>>(p, q, si, step, active_infections, d_levels, d_immuneCountdown, d_immuneStep, ptr, base_seed);
            step++;
            // Sincronizza CUDA prima di unmappare
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                std::cerr << "Errore sincronizzazione CUDA: " << cudaGetErrorString(err) << std::endl;
            }
            
            err = cudaGraphicsUnmapResources(1, &cudaVbo);
            if (err != cudaSuccess) {
                std::cerr << "Errore unmapping nel loop: " << cudaGetErrorString(err) << std::endl;
            }

            lastUpdate = now; // Aggiorna il timestamp dell'ultimo step
        }

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(shader);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, N);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    cudaGraphicsUnregisterResource(cudaVbo);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glfwTerminate();
    return 0;
}
