#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>

const int GRID_SIZE_X = 32;
const int GRID_SIZE_Y = 18;
const float CELL_WIDTH  = 2.0f / GRID_SIZE_X;
const float CELL_HEIGHT = 2.0f / GRID_SIZE_Y;

struct Vertex {
    float x, y;
    float r, g, b;
};

GLFWwindow* initWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);

    GLFWwindow* window = glfwCreateWindow(
        mode->width, mode->height,
        "Griglia Random",
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

void initVertices(std::vector<Vertex>& vertices) {
    vertices.reserve(GRID_SIZE_X * GRID_SIZE_Y * 6);
    for (int i = 0; i < GRID_SIZE_X; ++i) {
        for (int j = 0; j < GRID_SIZE_Y; ++j) {
            float x = -1.0f + i * CELL_WIDTH;
            float y = 1.0f - (j + 1) * CELL_HEIGHT;

            float r = (i + j) % 2 == 0 ? 1.0f : 0.0f; // Alterna i colori tra rosso e nero

            // 2 triangoli per formare un quadrato
            // sostituire r con 0.0f se non si vuole scacchiera

            vertices.push_back({ x, y, r, 0, 0 });
            vertices.push_back({ x + CELL_WIDTH, y, r/2, 0, 0 });
            vertices.push_back({ x + CELL_WIDTH, y + CELL_HEIGHT, r/4, 0, 0 });

            vertices.push_back({ x, y, r, 0, 0 });
            vertices.push_back({ x + CELL_WIDTH, y + CELL_HEIGHT, r/2, 0, 0 });
            vertices.push_back({ x, y + CELL_HEIGHT, r/4, 0, 0 });
        }
    }
}

void updateColors(std::vector<Vertex>& vertices) {
    for (int i = 0; i < GRID_SIZE_X; ++i) {
        for (int j = 0; j < GRID_SIZE_Y; ++j) {
            
            int base = (i * GRID_SIZE_Y + j) * 6;
            
            float r;
            if (vertices[base].r)
            r = 0.0f;
            else    
            r = 1.0f;

            //float r = rand() % 2 ? 1.0f : 0.0f;
            //float g = 0.0f;
            //float b = 0.0f;

            for (int k = 0; k < 6; ++k) {
                vertices[base + k].r = r/((k%3)+1);
                //vertices[base + k].g = g;
                //vertices[base + k].b = b;
            }
        }
    }
}

int main() {
    srand(static_cast<unsigned>(time(0)));
    GLFWwindow* window = initWindow();
    unsigned int shaderProgram = createShaderProgram();

    unsigned int VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    std::vector<Vertex> vertices;
    initVertices(vertices);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * vertices.size(), vertices.data(), GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    auto lastUpdate = std::chrono::steady_clock::now();

    while (!glfwWindowShouldClose(window)) {
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastUpdate).count() > 500) {
            updateColors(vertices);
            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(Vertex) * vertices.size(), vertices.data());
            lastUpdate = now;
        }

        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, vertices.size());

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}
