#include <iostream>
#include <vector>
#include <algorithm>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <cmath>

// Simulation parameters
const int N = 1024;         // simulation grid size
const float dt = 0.025f;    // time step

// Global textures for simulation fields:
// Density (ping–pong), Velocity (and temporary for advection),
// Pressure (ping–pong) and Divergence.
GLuint densityTexA, densityTexB;
GLuint velocityTex;
GLuint velocityTexTemp;
GLuint pressureTexA, pressureTexB;
GLuint divergenceTex;

// Global compute shader program objects:
GLuint advectVelocityProgram;
GLuint divergenceProgram;
GLuint pressureProgram;
GLuint gradientProgram;
GLuint advectDensityProgram;

// Global contrast parameter (for rendering density)
float gContrast = 1.0f;

// Global mouse state (for injecting forces/density)
bool mousePressed = false;
double lastMouseX = 0.0, lastMouseY = 0.0;

// Forward declarations
GLuint compileShader(const char* source, GLenum shaderType);
GLuint createShaderProgram(const char* vertexSource, const char* fragmentSource);
GLuint createComputeProgram(const char* computeSource);
void initSimulationTextures();
void initComputeShaders();
void simulationStep();
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);

// ------------------- Compute Shader Sources ------------------- //

// 1. Advect Velocity (backtrace velocity field)
const char* advectVelocitySource = R"(
// Modified Advect Velocity Compute Shader with Inertia
#version 430
layout (local_size_x = 16, local_size_y = 16) in;
layout (binding = 0, rg32f) uniform image2D velocityIn;
layout (binding = 1, rg32f) uniform image2D velocityOut;
uniform float dt;
uniform float gridSize;
uniform float inertia; // New uniform controlling how much old velocity to keep
void main(){
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(velocityIn);
    if(pos.x >= size.x || pos.y >= size.y) return;
    
    // Get the current (old) velocity
    vec2 oldVel = imageLoad(velocityIn, pos).rg;
    
    // Compute the advected velocity by backtracing the old velocity
    vec2 posCenter = vec2(pos) + 0.5;
    vec2 prevPos = posCenter - dt * gridSize * oldVel;
    vec2 coord = prevPos - 0.5;
    ivec2 i0 = ivec2(floor(coord));
    ivec2 i1 = i0 + ivec2(1);
    vec2 f = fract(coord);
    i0 = clamp(i0, ivec2(0), size - ivec2(1));
    i1 = clamp(i1, ivec2(0), size - ivec2(1));
    vec2 v00 = imageLoad(velocityIn, i0).rg;
    vec2 v10 = imageLoad(velocityIn, ivec2(i1.x, i0.y)).rg;
    vec2 v01 = imageLoad(velocityIn, ivec2(i0.x, i1.y)).rg;
    vec2 v11 = imageLoad(velocityIn, i1).rg;
    vec2 advectedVel = mix(mix(v00, v10, f.x), mix(v01, v11, f.x), f.y);
    
    // Blend the advected velocity with the old velocity
    vec2 newVel = mix(advectedVel, oldVel, inertia);
    
    imageStore(velocityOut, pos, vec4(newVel, 0.0, 1.0));
}
)";

// 2. Compute Divergence of velocity
const char* divergenceSource = R"(
#version 430
layout (local_size_x = 16, local_size_y = 16) in;
layout (binding = 0, rg32f) uniform image2D velocity;
layout (binding = 1, r32f) uniform image2D divergence;
uniform float gridSize;
void main(){
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(velocity);
    if(pos.x <= 0 || pos.y <= 0 || pos.x >= size.x-1 || pos.y >= size.y-1) {
       imageStore(divergence, pos, vec4(0.0,0.0,0.0,1.0));
       return;
    }
    float left   = imageLoad(velocity, pos - ivec2(1,0)).r;
    float right  = imageLoad(velocity, pos + ivec2(1,0)).r;
    float bottom = imageLoad(velocity, pos - ivec2(0,1)).g;
    float top    = imageLoad(velocity, pos + ivec2(0,1)).g;
    float div = (right - left + top - bottom) * 0.5 / gridSize;
    imageStore(divergence, pos, vec4(div,0.0,0.0,1.0));
}
)";

// 3. Pressure Solve (Jacobi Iteration)
const char* pressureSolveSource = R"(
#version 430
layout (local_size_x = 16, local_size_y = 16) in;
layout (binding = 0, r32f) uniform image2D pressureIn;
layout (binding = 1, r32f) uniform image2D pressureOut;
layout (binding = 2, r32f) uniform image2D divergence;
uniform float alpha;
uniform float reciprocalBeta;
void main(){
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(pressureIn);
    if(pos.x <= 0 || pos.y <= 0 || pos.x >= size.x-1 || pos.y >= size.y-1){
       imageStore(pressureOut, pos, vec4(0.0,0.0,0.0,1.0));
       return;
    }
    float pL = imageLoad(pressureIn, pos - ivec2(1, 0)).r;
    float pR = imageLoad(pressureIn, pos + ivec2(1, 0)).r;
    float pB = imageLoad(pressureIn, pos - ivec2(0, 1)).r;
    float pT = imageLoad(pressureIn, pos + ivec2(0, 1)).r;
    float div = imageLoad(divergence, pos).r;
    float newP = (pL + pR + pB + pT + alpha * div) * reciprocalBeta;
    imageStore(pressureOut, pos, vec4(newP,0.0,0.0,1.0));
}
)";

// 4. Subtract Pressure Gradient (Projection)
const char* subtractGradientSource = R"(
#version 430
layout (local_size_x = 16, local_size_y = 16) in;
layout (binding = 0, rg32f) uniform image2D velocity;
layout (binding = 1, r32f) uniform image2D pressure;
uniform float gridSize;
void main(){
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(velocity);
    if(pos.x <= 0 || pos.y <= 0 || pos.x >= size.x-1 || pos.y >= size.y-1)
        return;
    float pL = imageLoad(pressure, pos - ivec2(1,0)).r;
    float pR = imageLoad(pressure, pos + ivec2(1,0)).r;
    float pB = imageLoad(pressure, pos - ivec2(0,1)).r;
    float pT = imageLoad(pressure, pos + ivec2(0,1)).r;
    vec2 vel = imageLoad(velocity, pos).rg;
    vel.r -= (pR - pL) * 0.5 / gridSize;
    vel.g -= (pT - pB) * 0.5 / gridSize;
    imageStore(velocity, pos, vec4(vel, 0.0, 1.0));
}
)";

// 5. Advect Density with Dissipation (added uniform "dissipation")
const char* advectDensitySource = R"(
#version 430
layout (local_size_x = 16, local_size_y = 16) in;
layout (binding = 0, r32f) uniform image2D densityIn;
layout (binding = 1, r32f) uniform image2D densityOut;
layout (binding = 2, rg32f) uniform image2D velocity;
uniform float dt;
uniform float gridSize;
uniform float dissipation;  // New uniform for decay
void main(){
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(densityIn);
    if(pos.x >= size.x || pos.y >= size.y) return;
    vec2 posCenter = vec2(pos) + 0.5;
    vec2 vel = imageLoad(velocity, pos).rg;
    vec2 prevPos = posCenter - dt * gridSize * vel;
    vec2 coord = prevPos - 0.5;
    ivec2 i0 = ivec2(floor(coord));
    ivec2 i1 = i0 + ivec2(1);
    vec2 f = fract(coord);
    i0 = clamp(i0, ivec2(0), size - ivec2(1));
    i1 = clamp(i1, ivec2(0), size - ivec2(1));
    float d00 = imageLoad(densityIn, i0).r;
    float d10 = imageLoad(densityIn, ivec2(i1.x, i0.y)).r;
    float d01 = imageLoad(densityIn, ivec2(i0.x, i1.y)).r;
    float d11 = imageLoad(densityIn, i1).r;
    float newDensity = mix(mix(d00, d10, f.x), mix(d01, d11, f.x), f.y);
    newDensity *= dissipation;  // Apply dissipation
    imageStore(densityOut, pos, vec4(newDensity, 0.0, 0.0, 1.0));
}
)";

// ------------------- Renderer for Density ------------------- //
class Renderer {
public:
    GLuint vao, vbo, ebo;
    GLuint shaderProgram;
    int gridSize;
    Renderer(int gridSize) : gridSize(gridSize) {
        float vertices[] = {
            // positions    // texCoords
            -1.0f, -1.0f,    0.0f, 0.0f,
             1.0f, -1.0f,    1.0f, 0.0f,
             1.0f,  1.0f,    1.0f, 1.0f,
            -1.0f,  1.0f,    0.0f, 1.0f
        };
        unsigned int indices[] = { 0, 1, 2, 2, 3, 0 };
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glGenBuffers(1, &ebo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);
        glBindVertexArray(0);

        const char* vertexShaderSource = R"(
            #version 430 core
            layout (location = 0) in vec2 aPos;
            layout (location = 1) in vec2 aTexCoord;
            out vec2 TexCoord;
            void main(){
                gl_Position = vec4(aPos, 0.0, 1.0);
                TexCoord = aTexCoord;
            }
        )";
        const char* fragmentShaderSource = R"(
            #version 430 core
            out vec4 FragColor;
            in vec2 TexCoord;
            uniform sampler2D densityMap;
            uniform float contrast;
            void main(){
                float d = texture(densityMap, TexCoord).r;
                d = d / 50.0;
                d = clamp((d - 0.5) * contrast + 0.5, 0.0, 1.0);
                FragColor = vec4(d, d, d, 1.0);
            }
        )";
        shaderProgram = createShaderProgram(vertexShaderSource, fragmentShaderSource);
    }
    void draw(GLuint simTexture) {
        glUseProgram(shaderProgram);
        glBindVertexArray(vao);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, simTexture);
        glUniform1i(glGetUniformLocation(shaderProgram, "densityMap"), 0);
        glUniform1f(glGetUniformLocation(shaderProgram, "contrast"), gContrast);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
        glUseProgram(0);
    }
};

// ------------------- Shader Utility Functions ------------------- //
GLuint compileShader(const char* source, GLenum shaderType) {
    GLuint shader = glCreateShader(shaderType);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Shader compile error:\n" << infoLog << std::endl;
    }
    return shader;
}
GLuint createShaderProgram(const char* vertexSource, const char* fragmentSource) {
    GLuint vertexShader = compileShader(vertexSource, GL_VERTEX_SHADER);
    GLuint fragmentShader = compileShader(fragmentSource, GL_FRAGMENT_SHADER);
    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::cerr << "Program linking error:\n" << infoLog << std::endl;
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    return program;
}
GLuint createComputeProgram(const char* computeSource) {
    GLuint computeShader = compileShader(computeSource, GL_COMPUTE_SHADER);
    GLuint program = glCreateProgram();
    glAttachShader(program, computeShader);
    glLinkProgram(program);
    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::cerr << "Compute program linking error:\n" << infoLog << std::endl;
    }
    glDeleteShader(computeShader);
    return program;
}

// ------------------- Initialization ------------------- //
void initSimulationTextures() {
    // Density textures:
    glGenTextures(1, &densityTexA);
    glBindTexture(GL_TEXTURE_2D, densityTexA);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, N, N, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glGenTextures(1, &densityTexB);
    glBindTexture(GL_TEXTURE_2D, densityTexB);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, N, N, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    // Velocity textures:
    glGenTextures(1, &velocityTex);
    glBindTexture(GL_TEXTURE_2D, velocityTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, N, N, 0, GL_RG, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glGenTextures(1, &velocityTexTemp);
    glBindTexture(GL_TEXTURE_2D, velocityTexTemp);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, N, N, 0, GL_RG, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    // Pressure textures:
    glGenTextures(1, &pressureTexA);
    glBindTexture(GL_TEXTURE_2D, pressureTexA);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, N, N, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glGenTextures(1, &pressureTexB);
    glBindTexture(GL_TEXTURE_2D, pressureTexB);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, N, N, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    // Divergence texture:
    glGenTextures(1, &divergenceTex);
    glBindTexture(GL_TEXTURE_2D, divergenceTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, N, N, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
}
void initComputeShaders() {
    advectVelocityProgram = createComputeProgram(advectVelocitySource);
    divergenceProgram = createComputeProgram(divergenceSource);
    pressureProgram = createComputeProgram(pressureSolveSource);
    gradientProgram = createComputeProgram(subtractGradientSource);
    advectDensityProgram = createComputeProgram(advectDensitySource);
}

// ------------------- Simulation Step ------------------- //
void simulationStep() {
    int groupsX = (N + 15) / 16;
    int groupsY = (N + 15) / 16;
    // 1. Advect Velocity:
    glBindImageTexture(0, velocityTex, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RG32F);
    glBindImageTexture(1, velocityTexTemp, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG32F);
    glUseProgram(advectVelocityProgram);
    glUniform1f(glGetUniformLocation(advectVelocityProgram, "dt"), dt);
    glUniform1f(glGetUniformLocation(advectVelocityProgram, "gridSize"), float(N));
    glUniform1f(glGetUniformLocation(advectVelocityProgram, "inertia"), 0.75f);
    glDispatchCompute(groupsX, groupsY, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    std::swap(velocityTex, velocityTexTemp);

    // 2. Compute Divergence:
    glBindImageTexture(0, velocityTex, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RG32F);
    glBindImageTexture(1, divergenceTex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
    glUseProgram(divergenceProgram);
    glUniform1f(glGetUniformLocation(divergenceProgram, "gridSize"), float(N));
    glDispatchCompute(groupsX, groupsY, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    // 3. Pressure Solve (Jacobi Iterations, 30 times)
    float alpha = -float(N) * float(N);
    float reciprocalBeta = 0.25f;
    for (int i = 0; i < 30; i++) {
        glBindImageTexture(0, pressureTexA, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
        glBindImageTexture(1, pressureTexB, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
        glBindImageTexture(2, divergenceTex, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
        glUseProgram(pressureProgram);
        glUniform1f(glGetUniformLocation(pressureProgram, "alpha"), alpha);
        glUniform1f(glGetUniformLocation(pressureProgram, "reciprocalBeta"), reciprocalBeta);
        glDispatchCompute(groupsX, groupsY, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        std::swap(pressureTexA, pressureTexB);
    }

    // 4. Subtract Gradient (Project Velocity)
    glBindImageTexture(0, velocityTex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RG32F);
    glBindImageTexture(1, pressureTexA, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
    glUseProgram(gradientProgram);
    glUniform1f(glGetUniformLocation(gradientProgram, "gridSize"), float(N));
    glDispatchCompute(groupsX, groupsY, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    // 5. Advect Density with Dissipation:
    glBindImageTexture(0, densityTexA, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(1, densityTexB, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
    glBindImageTexture(2, velocityTex, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RG32F);
    glUseProgram(advectDensityProgram);
    glUniform1f(glGetUniformLocation(advectDensityProgram, "dt"), dt);
    glUniform1f(glGetUniformLocation(advectDensityProgram, "gridSize"), float(N));
    glUniform1f(glGetUniformLocation(advectDensityProgram, "dissipation"), 0.999f);
    glDispatchCompute(groupsX, groupsY, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    std::swap(densityTexA, densityTexB);
}

// ------------------- Mouse Callbacks ------------------- //
// In the mouse button callback (for initial injection):
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            mousePressed = true;
            glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
            int width, height;
            glfwGetWindowSize(window, &width, &height);
            int centerX = int((lastMouseX / double(width)) * N);
            int centerY = int(((height - lastMouseY) / double(height)) * N);
            std::cout << "Mouse pressed at (" << lastMouseX << ", " << lastMouseY
                << ") -> grid (" << centerX << ", " << centerY << ")\n";
            // Inject density and velocity over a radius
            const int radius = 5; // Adjust for larger/smaller effect area
            for (int dj = -radius; dj <= radius; dj++) {
                for (int di = -radius; di <= radius; di++) {
                    int cellX = centerX + di;
                    int cellY = centerY + dj;
                    // Check bounds:
                    if (cellX < 0 || cellX >= N || cellY < 0 || cellY >= N)
                        continue;
                    // Compute a weight (linear falloff)
                    float dist = sqrtf(float(di * di + dj * dj));
                    if (dist > radius) continue;  // outside the circle
                    float weight = 1.0f - (dist / float(radius));
                    float densityValue = 100.0f * weight;
                    // For initial injection, you might choose a default velocity impulse:
                    glm::vec2 velocityValue(0.5f * weight, 0.5f * weight);
                    glBindTexture(GL_TEXTURE_2D, densityTexA);
                    glTexSubImage2D(GL_TEXTURE_2D, 0, cellX, cellY, 1, 1, GL_RED, GL_FLOAT, &densityValue);
                    glBindTexture(GL_TEXTURE_2D, 0);
                    glBindTexture(GL_TEXTURE_2D, velocityTex);
                    glTexSubImage2D(GL_TEXTURE_2D, 0, cellX, cellY, 1, 1, GL_RG, GL_FLOAT, &velocityValue);
                    glBindTexture(GL_TEXTURE_2D, 0);
                }
            }
        }
        else if (action == GLFW_RELEASE) {
            mousePressed = false;
        }
    }
}

// In the cursor position callback (for drag injection):
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    if (!mousePressed)
        return;

    int width, height;
    glfwGetWindowSize(window, &width, &height);
    int centerX = int((xpos / double(width)) * N);
    int centerY = int(((height - ypos) / double(height)) * N);

    // Calculate drag delta from previous mouse position (if desired)
    double dx = xpos - lastMouseX;
    double dy = ypos - lastMouseY;
    lastMouseX = xpos;
    lastMouseY = ypos;

    std::cout << "Dragging at (" << xpos << ", " << ypos
        << ") -> grid (" << centerX << ", " << centerY
        << ") with delta (" << dx << ", " << dy << ")\n";

    // Convert the drag delta into a velocity impulse.
    float velocityScale = 0.25f;
    glm::vec2 velocityValue(dx * velocityScale, -dy * velocityScale);

    // Inject over a radius.
    const int radius = 5;  // same as above
    for (int dj = -radius; dj <= radius; dj++) {
        for (int di = -radius; di <= radius; di++) {
            int cellX = centerX + di;
            int cellY = centerY + dj;
            if (cellX < 0 || cellX >= N || cellY < 0 || cellY >= N)
                continue;
            float dist = sqrtf(float(di * di + dj * dj));
            if (dist > radius) continue;
            float weight = 1.0f - (dist / float(radius));
            float densityValue = 100.0f * weight;
            glm::vec2 localVelocity = velocityValue * weight;
            glBindTexture(GL_TEXTURE_2D, densityTexA);
            glTexSubImage2D(GL_TEXTURE_2D, 0, cellX, cellY, 1, 1, GL_RED, GL_FLOAT, &densityValue);
            glBindTexture(GL_TEXTURE_2D, 0);
            glBindTexture(GL_TEXTURE_2D, velocityTex);
            glTexSubImage2D(GL_TEXTURE_2D, 0, cellX, cellY, 1, 1, GL_RG, GL_FLOAT, &localVelocity);
            glBindTexture(GL_TEXTURE_2D, 0);
        }
    }
}

// ------------------- Main ------------------- //
int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(800, 600, "Realistic Fluid Simulation", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW\n";
        return -1;
    }
    glViewport(0, 0, 800, 600);

    initSimulationTextures();
    initComputeShaders();

    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);

    Renderer renderer(N);

    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT);
        simulationStep();
        renderer.draw(densityTexA);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwTerminate();
    return 0;
}
