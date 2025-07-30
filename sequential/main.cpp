#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>

const int GRID_SIZE_X = 1024;
const int GRID_SIZE_Y = 288;

const float CELL_WIDTH  = 2.0f / GRID_SIZE_X;
const float CELL_HEIGHT = 2.0f / GRID_SIZE_Y;

const int VERTICES_COUNT = GRID_SIZE_X * GRID_SIZE_Y * 6;
const int NUM_NODES = GRID_SIZE_X * GRID_SIZE_Y;

const int STEP_TIMER = 10; // Milliseconds


const int dx[] = {-1, 1, 0, 0};
const int dy[] = {0, 0, -1, 1};

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
        "SIR Epidemic Simulation",
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

// SHADER PROGRAMS

const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 aColor;
out vec3 vColor;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    vColor = aColor;
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
in vec3 vColor;
out vec4 FragColor;
void main() {
    FragColor = vec4(vColor, 1.0);
}
)";

unsigned int createShaderProgram() {
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);

    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    return shaderProgram;
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
}

void initLevelsAndImmune(int* Levels, int* ImmuneCountdown, int* ImmuneStep, int first_infected_node = 0) {
    for (int i = 0; i < NUM_NODES; ++i) {
        Levels[i] = -1; // Inizializza tutti i nodi come non infetti
        ImmuneCountdown[i] = 0; // Inizializza tutti i nodi come non immuni
        ImmuneStep[i] = -1; // Inizializza tutti i nodi come mai stati immuni
    }
    Levels[first_infected_node] = 0; // Il primo nodo è infetto all'inizio
}

void updateColors(int node_index, float r, float g, float b, Vertex* vertices) {
    int base = node_index * 6;    
    for (int k = 0; k < 6; ++k) {
        vertices[base + k].r = r;
        vertices[base + k].g = g;
        vertices[base + k].b = b;
    }
}

void simulate_step(double p, double q,int si, int& step, int& active_infections,
                   int* Levels, int* ImmuneCountdown, int* ImmuneStep, Vertex* vertices) {
    for (int i = 0; i < NUM_NODES; i++) {
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
                    
                    if (Levels[neighbor] < step && ImmuneStep[neighbor] <step && ((double)rand() / RAND_MAX) < p) {
                        Levels[neighbor] = step + 1;
                        active_infections++;
                        updateColors(neighbor, 1.0f, 0.0f, 0.0f, vertices); // Aggiorna colore a rosso
                    }
                }
            }
            // Simula la guarigione
            if (((double)rand() / RAND_MAX) < q) {
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
    step++;
    // Stampa lo stato della simulazione
    /*std::cout << "Step " << step << ": " << active_infections << " active infections\n";
    if (active_infections > 0) {
        std::cout << "Infected nodes: ";
        for (int i = 0; i < NUM_NODES; i++) {
            if (Levels[i] == step) {
                std::cout << i << " ";
            }
        }
        std::cout << "\n";
        std::cout << "Immune nodes: ";
        for (int i = 0; i < NUM_NODES; i++) {
            if (Immune[i] > 0) {
                std::cout << i << " ";
            }
        }
    }*/
}

int main() {
    srand(static_cast<unsigned>(time(0)));
    GLFWwindow* window = initWindow();
    unsigned int shaderProgram = createShaderProgram();

    unsigned int VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    int first_infected_node = 0; // Nodo inizialmente infetto

    // Allocazione dinamica degli array
    Vertex* vertices = new Vertex[VERTICES_COUNT];
    int* Levels = new int[NUM_NODES];
    int* ImmuneCountdown = new int[NUM_NODES];
    int* ImmuneStep = new int[NUM_NODES];
    
    initVertices(vertices, first_infected_node);
    initLevelsAndImmune(Levels, ImmuneCountdown, ImmuneStep, first_infected_node);

    int active_infections = 1; // Inizio con un'infezione
    int step = 0;

    float p = 0.5; // Probabilità di infezione
    float q = 0.5; // Probabilità di guarigione
    int si = 50; // Durata dell'immunità

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * VERTICES_COUNT, vertices, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    auto lastUpdate = std::chrono::steady_clock::now();

    while (!glfwWindowShouldClose(window)) {
        auto now = std::chrono::steady_clock::now();
        // active_infections > 0 &&
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastUpdate).count() > STEP_TIMER) {
            simulate_step(p, q, si, step, active_infections, Levels, ImmuneCountdown, ImmuneStep, vertices);
            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(Vertex) * VERTICES_COUNT, vertices);
            lastUpdate = now;
        }

        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, VERTICES_COUNT);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Deallocazione della memoria dinamica
    delete[] vertices;
    delete[] Levels;
    delete[] ImmuneCountdown;
    delete[] ImmuneStep;

    glfwTerminate();
    return 0;
}
