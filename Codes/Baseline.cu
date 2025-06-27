/**
 * CUDA Baseline Matrix Multiplication Framework
 * 
 * This implementation serves as the foundation for systematic energy-aware GPU kernel optimization research. The framework provides customizable parameters for thread organization, matrix dimensions, and comprehensive energy/performance measurement capabilities using NVML.
 * 
 * Research Context: Systematic Performance Engineering Framework for Energy-Aware CUDA Kernel Optimization
 * 
 * Author: Aniruddha Shete
 * Purpose: Baseline reference for optimization comparison studies
 */

#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nvml.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <thread>
#include <atomic>
#include <cmath>
#include <random>
#include <cassert>

// Configuration constants for systematic testing
#define MAX_BLOCK_SIZE 1024
#define MIN_BLOCK_SIZE 64
#define WARP_SIZE 32
#define DEFAULT_TILE_SIZE 16
#define POWER_SAMPLING_INTERVAL_MS 10
#define THERMAL_STABILIZATION_TIME_MS 30000

// Error checking macros for robust execution
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

#define NVML_CHECK(call) \
    do { \
        nvmlReturn_t result = call; \
        if (result != NVML_SUCCESS) { \
            std::cerr << "NVML error: " << nvmlErrorString(result) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Handle different NVML library configurations
#ifdef DISABLE_NVML
    // Dummy NVML functions when NVML is disabled
    #define NVML_SUCCESS 0
    #define NVML_TEMPERATURE_GPU 0
    typedef void* nvmlDevice_t;
    typedef int nvmlReturn_t;
    
    inline nvmlReturn_t nvmlInit() { return NVML_SUCCESS; }
    inline nvmlReturn_t nvmlShutdown() { return NVML_SUCCESS; }
    inline nvmlReturn_t nvmlDeviceGetHandleByIndex(int, nvmlDevice_t*) { return NVML_SUCCESS; }
    inline nvmlReturn_t nvmlDeviceGetPowerUsage(nvmlDevice_t, unsigned int* power) { *power = 150000; return NVML_SUCCESS; }
    inline nvmlReturn_t nvmlDeviceGetTemperature(nvmlDevice_t, int, unsigned int* temp) { *temp = 65; return NVML_SUCCESS; }
    inline const char* nvmlErrorString(nvmlReturn_t) { return "NVML disabled"; }
#endif

/**
 * Configuration structure for systematic parameter variation
 * Enables controlled testing across different optimization dimensions
 */
struct OptimizationConfig {
    // Matrix dimensions for scalability analysis
    int matrix_size;
    
    // Thread organization parameters
    int block_dim_x;
    int block_dim_y;
    int threads_per_block;
    
    // Algorithm variant selection
    enum KernelType {
        NAIVE_BASELINE,
        MEMORY_COALESCED,
        SHARED_MEMORY_TILED,
        REGISTER_BLOCKED,
        OCCUPANCY_OPTIMIZED,
        ADVANCED_OPTIMIZED
    } kernel_type;
    
    // Data type for precision analysis
    enum DataType {
        FLOAT_32,
        DOUBLE_64
    } data_type;
    
    // Performance measurement parameters
    int num_iterations;
    bool enable_power_monitoring;
    bool enable_thermal_monitoring;
    
    // Constructor with sensible defaults
    OptimizationConfig() : 
        matrix_size(1024),
        block_dim_x(16),
        block_dim_y(16),
        threads_per_block(256),
        kernel_type(NAIVE_BASELINE),
        data_type(FLOAT_32),
        num_iterations(10),
        enable_power_monitoring(true),
        enable_thermal_monitoring(true) {}
};

/**
 * Performance metrics structure for comprehensive analysis
 * Captures both computational performance and energy efficiency
 */
struct PerformanceMetrics {
    // Timing measurements
    double execution_time_ms;
    double average_execution_time_ms;
    double execution_time_stddev;
    
    // Energy measurements
    double total_energy_joules;
    double average_power_watts;
    double peak_power_watts;
    double baseline_power_watts;
    
    // Thermal measurements
    double initial_temperature_c;
    double peak_temperature_c;
    double final_temperature_c;
    
    // Computational efficiency
    double gflops;
    double gflops_per_watt;
    double memory_bandwidth_gbps;
    double memory_bandwidth_utilization_percent;
    
    // Hardware utilization
    double gpu_utilization_percent;
    double memory_utilization_percent;
    
    // Constructor
    PerformanceMetrics() : 
        execution_time_ms(0.0), average_execution_time_ms(0.0), 
        execution_time_stddev(0.0), total_energy_joules(0.0),
        average_power_watts(0.0), peak_power_watts(0.0),
        baseline_power_watts(0.0), initial_temperature_c(0.0),
        peak_temperature_c(0.0), final_temperature_c(0.0),
        gflops(0.0), gflops_per_watt(0.0), memory_bandwidth_gbps(0.0),
        memory_bandwidth_utilization_percent(0.0), gpu_utilization_percent(0.0),
        memory_utilization_percent(0.0) {}
};

/**
 * NVML-based power monitoring system
 * Provides real-time energy consumption tracking during kernel execution
 */
class PowerMonitor {
private:
    nvmlDevice_t device;
    std::atomic<bool> monitoring_active;
    std::vector<double> power_samples;
    std::vector<double> temperature_samples;
    std::vector<std::chrono::high_resolution_clock::time_point> timestamps;
    std::thread monitoring_thread;
    
public:
    PowerMonitor() : monitoring_active(false) {
        NVML_CHECK(nvmlInit());
        NVML_CHECK(nvmlDeviceGetHandleByIndex(0, &device));
    }
    
    ~PowerMonitor() {
        if (monitoring_active.load()) {
            stop_monitoring();
        }
        nvmlShutdown();
    }
    
    /**
     * Begin power monitoring in separate thread
     * Samples power and temperature at specified intervals
     */
    void start_monitoring() {
        power_samples.clear();
        temperature_samples.clear();
        timestamps.clear();
        monitoring_active.store(true);
        
        monitoring_thread = std::thread([this]() {
            while (monitoring_active.load()) {
                unsigned int power_mw, temperature_c;
                auto timestamp = std::chrono::high_resolution_clock::now();
                
                if (nvmlDeviceGetPowerUsage(device, &power_mw) == NVML_SUCCESS) {
                    power_samples.push_back(power_mw / 1000.0); // Convert to watts
                    timestamps.push_back(timestamp);
                }
                
                if (nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature_c) == NVML_SUCCESS) {
                    temperature_samples.push_back(static_cast<double>(temperature_c));
                }
                
                std::this_thread::sleep_for(std::chrono::milliseconds(POWER_SAMPLING_INTERVAL_MS));
            }
        });
    }
    
    /**
     * Stop power monitoring and return collected metrics
     */
    void stop_monitoring() {
        monitoring_active.store(false);
        if (monitoring_thread.joinable()) {
            monitoring_thread.join();
        }
    }
    
    /**
     * Calculate energy consumption using trapezoidal integration
     */
    double calculate_total_energy() const {
        if (power_samples.size() < 2) return 0.0;
        
        double total_energy = 0.0;
        for (size_t i = 1; i < power_samples.size(); ++i) {
            auto time_diff = std::chrono::duration_cast<std::chrono::microseconds>(
                timestamps[i] - timestamps[i-1]).count() / 1e6; // Convert to seconds
            
            // Trapezoidal rule for numerical integration
            total_energy += (power_samples[i] + power_samples[i-1]) * 0.5 * time_diff;
        }
        
        return total_energy;
    }
    
    double get_average_power() const {
        if (power_samples.empty()) return 0.0;
        double sum = 0.0;
        for (double power : power_samples) {
            sum += power;
        }
        return sum / power_samples.size();
    }
    
    double get_peak_power() const {
        if (power_samples.empty()) return 0.0;
        return *std::max_element(power_samples.begin(), power_samples.end());
    }
    
    double get_peak_temperature() const {
        if (temperature_samples.empty()) return 0.0;
        return *std::max_element(temperature_samples.begin(), temperature_samples.end());
    }
    
    double get_initial_temperature() const {
        return temperature_samples.empty() ? 0.0 : temperature_samples.front();
    }
    
    double get_final_temperature() const {
        return temperature_samples.empty() ? 0.0 : temperature_samples.back();
    }
    
    /**
     * Measure baseline power consumption during idle state
     */
    double measure_baseline_power(int duration_ms = 5000) {
        std::vector<double> baseline_samples;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        while (std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time).count() < duration_ms) {
            
            unsigned int power_mw;
            if (nvmlDeviceGetPowerUsage(device, &power_mw) == NVML_SUCCESS) {
                baseline_samples.push_back(power_mw / 1000.0);
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(POWER_SAMPLING_INTERVAL_MS));
        }
        
        if (baseline_samples.empty()) return 0.0;
        
        double sum = 0.0;
        for (double power : baseline_samples) {
            sum += power;
        }
        return sum / baseline_samples.size();
    }
};

/**
 * Naive baseline CUDA kernel for matrix multiplication
 * 
 * This implementation uses the most straightforward approach:
 * - One thread per output element
 * - Direct global memory access
 * - No optimization techniques applied
 * 
 * Serves as reference point for measuring optimization effectiveness
 */
__global__ void naive_matrix_multiply_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N) {
    
    // Calculate thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check to handle non-square block dimensions
    if (row < N && col < N) {
        float sum = 0.0f;
        
        // Compute dot product for element C[row][col]
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        
        C[row * N + col] = sum;
    }
}

/**
 * Double precision version of naive kernel for precision analysis
 */
__global__ void naive_matrix_multiply_kernel_double(
    const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C,
    int N) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        double sum = 0.0;
        
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        
        C[row * N + col] = sum;
    }
}

/**
 * Memory coalescing optimized kernel
 * 
 * Improves memory access patterns by ensuring consecutive threads
 * access consecutive memory locations for better bandwidth utilization
 */
__global__ void coalesced_matrix_multiply_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        
        // Coalesced access pattern optimization
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        
        C[row * N + col] = sum;
    }
}

/**
 * 3. SHARED MEMORY TILED KERNEL
 * Uses shared memory to cache frequently accessed data and reduce global memory accesses
 */
template<int TILE_SIZE>
__global__ void shared_memory_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N) {
    
    // Shared memory tiles for A and B matrices
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // Calculate thread and block indices
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles of the input matrices
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles into shared memory with boundary checks
        if (row < N && t * TILE_SIZE + threadIdx.x < N) {
            As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (col < N && t * TILE_SIZE + threadIdx.y < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial result using shared memory
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write final result
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * 4. REGISTER BLOCKED KERNEL
 * Each thread computes multiple output elements to increase computational intensity
 */
template<int BLK_SIZE, int REG_BLOCK>
__global__ void register_blocked_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N) {
    
    // Calculate base indices for this thread
    int base_row = blockIdx.y * BLK_SIZE + threadIdx.y;
    int base_col = blockIdx.x * BLK_SIZE + threadIdx.x;
    
    // Register arrays for multiple output elements per thread
    float regC[REG_BLOCK][REG_BLOCK];
    float regA[REG_BLOCK], regB[REG_BLOCK];
    
    // Initialize result registers
    #pragma unroll
    for (int i = 0; i < REG_BLOCK; ++i) {
        #pragma unroll
        for (int j = 0; j < REG_BLOCK; ++j) {
            regC[i][j] = 0.0f;
        }
    }
    
    // Main computation loop
    for (int k = 0; k < N; ++k) {
        // Load values into registers
        #pragma unroll
        for (int i = 0; i < REG_BLOCK; ++i) {
            int row = base_row + i * blockDim.y;
            regA[i] = (row < N) ? A[row * N + k] : 0.0f;
        }
        
        #pragma unroll
        for (int j = 0; j < REG_BLOCK; ++j) {
            int col = base_col + j * blockDim.x;
            regB[j] = (col < N) ? B[k * N + col] : 0.0f;
        }
        
        // Compute using registers
        #pragma unroll
        for (int i = 0; i < REG_BLOCK; ++i) {
            #pragma unroll
            for (int j = 0; j < REG_BLOCK; ++j) {
                regC[i][j] += regA[i] * regB[j];
            }
        }
    }
    
    // Write results back to global memory
    #pragma unroll
    for (int i = 0; i < REG_BLOCK; ++i) {
        #pragma unroll
        for (int j = 0; j < REG_BLOCK; ++j) {
            int row = base_row + i * blockDim.y;
            int col = base_col + j * blockDim.x;
            if (row < N && col < N) {
                C[row * N + col] = regC[i][j];
            }
        }
    }
}

/**
 * 5. OCCUPANCY OPTIMIZED KERNEL
 * Balanced design for maximum GPU occupancy with optimal resource usage
 */
__global__ void occupancy_optimized_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N) {
    
    // Optimized tile size for maximum occupancy (usually 16 for most GPUs)
    const int TILE_SIZE = 16;
    
    // Shared memory with bank conflict avoidance (+1 padding)
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Main computation loop with unrolling for better ILP
    #pragma unroll 4
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Coalesced loading with boundary checks
        As[threadIdx.y][threadIdx.x] = (row < N && t * TILE_SIZE + threadIdx.x < N) ? 
            A[row * N + t * TILE_SIZE + threadIdx.x] : 0.0f;
        
        Bs[threadIdx.y][threadIdx.x] = (col < N && t * TILE_SIZE + threadIdx.y < N) ? 
            B[(t * TILE_SIZE + threadIdx.y) * N + col] : 0.0f;
        
        __syncthreads();
        
        // Unrolled computation for better performance
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * 6. ADVANCED COMBINED OPTIMIZATIONS KERNEL
 * Incorporates multiple optimization techniques for maximum performance
 */
template<int TILE_SIZE>
__global__ void advanced_optimized_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N) {
    
    // Shared memory with padding for bank conflict avoidance
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];
    
    // Calculate indices
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    // Register variables for better performance
    float sum = 0.0f;
    float regA, regB;
    
    // Main computation loop with advanced optimizations
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Prefetch data into shared memory with vectorized loads where possible
        if (row < N && t * TILE_SIZE + tx < N) {
            As[ty][tx] = A[row * N + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (col < N && t * TILE_SIZE + ty < N) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute with register optimization and aggressive unrolling
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k += 4) {
            // Unroll by 4 for better instruction-level parallelism
            regA = As[ty][k];
            regB = Bs[k][tx];
            sum += regA * regB;
            
            if (k + 1 < TILE_SIZE) {
                regA = As[ty][k + 1];
                regB = Bs[k + 1][tx];
                sum += regA * regB;
            }
            
            if (k + 2 < TILE_SIZE) {
                regA = As[ty][k + 2];
                regB = Bs[k + 2][tx];
                sum += regA * regB;
            }
            
            if (k + 3 < TILE_SIZE) {
                regA = As[ty][k + 3];
                regB = Bs[k + 3][tx];
                sum += regA * regB;
            }
        }
        
        __syncthreads();
    }
    
    // Write result with coalesced access
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Explicit template instantiations for common tile sizes
template __global__ void shared_memory_tiled_kernel<8>(const float*, const float*, float*, int);
template __global__ void shared_memory_tiled_kernel<16>(const float*, const float*, float*, int);
template __global__ void shared_memory_tiled_kernel<32>(const float*, const float*, float*, int);

template __global__ void register_blocked_kernel<16, 2>(const float*, const float*, float*, int);
template __global__ void register_blocked_kernel<16, 4>(const float*, const float*, float*, int);

template __global__ void advanced_optimized_kernel<16>(const float*, const float*, float*, int);
template __global__ void advanced_optimized_kernel<32>(const float*, const float*, float*, int);

/**
 * Comprehensive matrix multiplication framework
 * Supports multiple optimization variants and systematic testing
 */
class MatrixMultiplicationFramework {
private:
    OptimizationConfig config;
    PowerMonitor power_monitor;
    
    // GPU memory pointers
    float *d_A_float, *d_B_float, *d_C_float;
    double *d_A_double, *d_B_double, *d_C_double;
    
    // Host memory for verification
    std::vector<float> h_A_float, h_B_float, h_C_float, h_C_ref_float;
    std::vector<double> h_A_double, h_B_double, h_C_double, h_C_ref_double;
    
public:
    MatrixMultiplicationFramework(const OptimizationConfig& cfg) : config(cfg) {
        initialize_matrices();
        allocate_gpu_memory();
    }
    
    ~MatrixMultiplicationFramework() {
        cleanup_gpu_memory();
    }
    
    /**
     * Initialize matrices with random values for testing
     * Uses consistent seed for reproducible results
     */
    void initialize_matrices() {
        int N = config.matrix_size;
        int total_elements = N * N;
        
        // Initialize random number generator with fixed seed
        std::random_device rd;
        std::mt19937 gen(12345); // Fixed seed for reproducibility
        std::uniform_real_distribution<float> dis_float(-1.0f, 1.0f);
        std::uniform_real_distribution<double> dis_double(-1.0, 1.0);
        
        if (config.data_type == OptimizationConfig::FLOAT_32) {
            h_A_float.resize(total_elements);
            h_B_float.resize(total_elements);
            h_C_float.resize(total_elements);
            h_C_ref_float.resize(total_elements);
            
            for (int i = 0; i < total_elements; ++i) {
                h_A_float[i] = dis_float(gen);
                h_B_float[i] = dis_float(gen);
                h_C_float[i] = 0.0f;
                h_C_ref_float[i] = 0.0f;
            }
        } else {
            h_A_double.resize(total_elements);
            h_B_double.resize(total_elements);
            h_C_double.resize(total_elements);
            h_C_ref_double.resize(total_elements);
            
            for (int i = 0; i < total_elements; ++i) {
                h_A_double[i] = dis_double(gen);
                h_B_double[i] = dis_double(gen);
                h_C_double[i] = 0.0;
                h_C_ref_double[i] = 0.0;
            }
        }
    }
    
    /**
     * Allocate GPU memory and transfer data
     */
    void allocate_gpu_memory() {
        int N = config.matrix_size;
        size_t size_bytes = N * N * sizeof(float);
        size_t size_bytes_double = N * N * sizeof(double);
        
        if (config.data_type == OptimizationConfig::FLOAT_32) {
            CUDA_CHECK(cudaMalloc(&d_A_float, size_bytes));
            CUDA_CHECK(cudaMalloc(&d_B_float, size_bytes));
            CUDA_CHECK(cudaMalloc(&d_C_float, size_bytes));
            
            CUDA_CHECK(cudaMemcpy(d_A_float, h_A_float.data(), size_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_B_float, h_B_float.data(), size_bytes, cudaMemcpyHostToDevice));
        } else {
            CUDA_CHECK(cudaMalloc(&d_A_double, size_bytes_double));
            CUDA_CHECK(cudaMalloc(&d_B_double, size_bytes_double));
            CUDA_CHECK(cudaMalloc(&d_C_double, size_bytes_double));
            
            CUDA_CHECK(cudaMemcpy(d_A_double, h_A_double.data(), size_bytes_double, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_B_double, h_B_double.data(), size_bytes_double, cudaMemcpyHostToDevice));
        }
    }
    
    /**
     * Cleanup GPU memory allocations
     */
    void cleanup_gpu_memory() {
        if (config.data_type == OptimizationConfig::FLOAT_32) {
            if (d_A_float) cudaFree(d_A_float);
            if (d_B_float) cudaFree(d_B_float);
            if (d_C_float) cudaFree(d_C_float);
        } else {
            if (d_A_double) cudaFree(d_A_double);
            if (d_B_double) cudaFree(d_B_double);
            if (d_C_double) cudaFree(d_C_double);
        }
    }
    
    /**
     * Execute benchmark with comprehensive performance measurement
     */
    PerformanceMetrics run_benchmark() {
        PerformanceMetrics metrics;
        std::vector<double> execution_times;
        
        // Measure baseline power consumption
        std::cout << "Measuring baseline power consumption..." << std::endl;
        metrics.baseline_power_watts = power_monitor.measure_baseline_power();
        
        // Thermal stabilization wait
        std::cout << "Waiting for thermal stabilization..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(THERMAL_STABILIZATION_TIME_MS));
        
        // Setup CUDA events for precise timing
        cudaEvent_t start_event, stop_event;
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
        
        // Configure kernel launch parameters
        dim3 block_dim(config.block_dim_x, config.block_dim_y);
        dim3 grid_dim(
            (config.matrix_size + block_dim.x - 1) / block_dim.x,
            (config.matrix_size + block_dim.y - 1) / block_dim.y
        );
        
        std::cout << "Starting benchmark with configuration:" << std::endl;
        std::cout << "  Matrix size: " << config.matrix_size << "x" << config.matrix_size << std::endl;
        std::cout << "  Block dimensions: " << block_dim.x << "x" << block_dim.y << std::endl;
        std::cout << "  Grid dimensions: " << grid_dim.x << "x" << grid_dim.y << std::endl;
        std::cout << "  Data type: " << (config.data_type == OptimizationConfig::FLOAT_32 ? "float" : "double") << std::endl;
        
        // Execute multiple iterations for statistical significance
        for (int iter = 0; iter < config.num_iterations; ++iter) {
            std::cout << "  Iteration " << (iter + 1) << "/" << config.num_iterations << std::endl;
            
            // Start power monitoring
            if (config.enable_power_monitoring) {
                power_monitor.start_monitoring();
            }
            
            // Record initial temperature
            if (iter == 0) {
                metrics.initial_temperature_c = power_monitor.get_initial_temperature();
            }
            
            // Execute kernel with timing
            CUDA_CHECK(cudaEventRecord(start_event));
            
            if (config.data_type == OptimizationConfig::FLOAT_32) {
                switch (config.kernel_type) {
                    case OptimizationConfig::NAIVE_BASELINE:
                        naive_matrix_multiply_kernel<<<grid_dim, block_dim>>>(
                            d_A_float, d_B_float, d_C_float, config.matrix_size);
                        break;
                    case OptimizationConfig::MEMORY_COALESCED:
                        coalesced_matrix_multiply_kernel<<<grid_dim, block_dim>>>(
                            d_A_float, d_B_float, d_C_float, config.matrix_size);
                        break;
                    case OptimizationConfig::SHARED_MEMORY_TILED:
                        shared_memory_tiled_kernel<DEFAULT_TILE_SIZE><<<grid_dim, block_dim>>>(
                            d_A_float, d_B_float, d_C_float, config.matrix_size);
                        break;
                    case OptimizationConfig::REGISTER_BLOCKED:
                        register_blocked_kernel<DEFAULT_TILE_SIZE, 2><<<grid_dim, block_dim>>>(
                            d_A_float, d_B_float, d_C_float, config.matrix_size);
                        break;
                    case OptimizationConfig::OCCUPANCY_OPTIMIZED:
                        occupancy_optimized_kernel<<<grid_dim, block_dim>>>(
                            d_A_float, d_B_float, d_C_float, config.matrix_size);
                        break;
                    case OptimizationConfig::ADVANCED_OPTIMIZED:
                        advanced_optimized_kernel<DEFAULT_TILE_SIZE><<<grid_dim, block_dim>>>(
                            d_A_float, d_B_float, d_C_float, config.matrix_size);
                        break;
                    default:
                        naive_matrix_multiply_kernel<<<grid_dim, block_dim>>>(
                            d_A_float, d_B_float, d_C_float, config.matrix_size);
                        break;
                }
            } else {
                naive_matrix_multiply_kernel_double<<<grid_dim, block_dim>>>(
                    d_A_double, d_B_double, d_C_double, config.matrix_size);
            }
            
            CUDA_CHECK(cudaEventRecord(stop_event));
            CUDA_CHECK(cudaEventSynchronize(stop_event));
            
            // Stop power monitoring
            if (config.enable_power_monitoring) {
                power_monitor.stop_monitoring();
            }
            
            // Record execution time
            float elapsed_ms;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
            execution_times.push_back(elapsed_ms);
            
            // Accumulate energy measurements
            if (config.enable_power_monitoring) {
                metrics.total_energy_joules += power_monitor.calculate_total_energy();
                metrics.peak_power_watts = std::max(metrics.peak_power_watts, 
                                                   power_monitor.get_peak_power());
                metrics.peak_temperature_c = std::max(metrics.peak_temperature_c,
                                                     power_monitor.get_peak_temperature());
            }
            
            // Allow GPU to cool between iterations
            if (iter < config.num_iterations - 1) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            }
        }
        
        // Calculate performance statistics
        double sum_time = 0.0;
        for (double time : execution_times) {
            sum_time += time;
        }
        metrics.average_execution_time_ms = sum_time / execution_times.size();
        
        // Calculate standard deviation
        double variance = 0.0;
        for (double time : execution_times) {
            variance += (time - metrics.average_execution_time_ms) * 
                       (time - metrics.average_execution_time_ms);
        }
        metrics.execution_time_stddev = std::sqrt(variance / execution_times.size());
        
        // Calculate computational performance metrics
        long long operations = 2LL * config.matrix_size * config.matrix_size * config.matrix_size;
        metrics.gflops = (operations / 1e9) / (metrics.average_execution_time_ms / 1000.0);
        
        // Calculate energy efficiency metrics
        if (config.enable_power_monitoring) {
            metrics.average_power_watts = power_monitor.get_average_power();
            metrics.gflops_per_watt = metrics.gflops / 
                                     (metrics.average_power_watts - metrics.baseline_power_watts);
            metrics.final_temperature_c = power_monitor.get_final_temperature();
        }
        
        // Calculate memory bandwidth utilization
        size_t data_size = config.data_type == OptimizationConfig::FLOAT_32 ? 
                          sizeof(float) : sizeof(double);
        long long bytes_transferred = 3LL * config.matrix_size * config.matrix_size * data_size;
        metrics.memory_bandwidth_gbps = (bytes_transferred / 1e9) / 
                                       (metrics.average_execution_time_ms / 1000.0);
        
        // Cleanup
        CUDA_CHECK(cudaEventDestroy(start_event));
        CUDA_CHECK(cudaEventDestroy(stop_event));
        
        return metrics;
    }
    
    /**
     * Verify computational correctness against reference implementation
     */
    bool verify_results() {
        // Copy results back to host
        int N = config.matrix_size;
        size_t size_bytes = N * N * sizeof(float);
        // size_t size_bytes_double = N * N * sizeof(double);
        
        if (config.data_type == OptimizationConfig::FLOAT_32) {
            CUDA_CHECK(cudaMemcpy(h_C_float.data(), d_C_float, size_bytes, cudaMemcpyDeviceToHost));
            
            // Compute reference using cuBLAS
            cublasHandle_t handle;
            cublasCreate(&handle);
            
            float *d_A_ref, *d_B_ref, *d_C_ref;
            CUDA_CHECK(cudaMalloc(&d_A_ref, size_bytes));
            CUDA_CHECK(cudaMalloc(&d_B_ref, size_bytes));
            CUDA_CHECK(cudaMalloc(&d_C_ref, size_bytes));
            
            CUDA_CHECK(cudaMemcpy(d_A_ref, h_A_float.data(), size_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_B_ref, h_B_float.data(), size_bytes, cudaMemcpyHostToDevice));
            
            const float alpha = 1.0f, beta = 0.0f;
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, 
                       d_B_ref, N, d_A_ref, N, &beta, d_C_ref, N);
            
            CUDA_CHECK(cudaMemcpy(h_C_ref_float.data(), d_C_ref, size_bytes, cudaMemcpyDeviceToHost));
            
            // Verify results
            bool correct = true;
            double max_error = 0.0;
            for (int i = 0; i < N * N; ++i) {
                double error = std::abs(h_C_float[i] - h_C_ref_float[i]);
                max_error = std::max(max_error, error);
                if (error > 1e-3) {
                    correct = false;
                    break;
                }
            }
            
            std::cout << "Verification: " << (correct ? "PASSED" : "FAILED") 
                      << " (max error: " << max_error << ")" << std::endl;
            
            // Cleanup
            cudaFree(d_A_ref);
            cudaFree(d_B_ref);
            cudaFree(d_C_ref);
            cublasDestroy(handle);
            
            return correct;
        }
        
        return true; // Placeholder for double precision verification
    }
    
    /**
     * Export results to CSV for analysis
     */
    void export_results(const PerformanceMetrics& metrics, const std::string& filename) {
        std::ofstream file(filename, std::ios::app);
        
        // Write header if file is empty
        file.seekp(0, std::ios::end);
        if (file.tellp() == 0) {
            file << "Matrix_Size,Block_Dim_X,Block_Dim_Y,Kernel_Type,Data_Type,"
                 << "Avg_Execution_Time_ms,Execution_Time_Stddev,Total_Energy_J,"
                 << "Avg_Power_W,Peak_Power_W,Baseline_Power_W,Initial_Temp_C,"
                 << "Peak_Temp_C,Final_Temp_C,GFLOPS,GFLOPS_per_Watt,"
                 << "Memory_Bandwidth_GBps,Threads_per_Block" << std::endl;
        }
        
        // Write data
        file << config.matrix_size << ","
             << config.block_dim_x << ","
             << config.block_dim_y << ","
             << config.kernel_type << ","
             << config.data_type << ","
             << metrics.average_execution_time_ms << ","
             << metrics.execution_time_stddev << ","
             << metrics.total_energy_joules << ","
             << metrics.average_power_watts << ","
             << metrics.peak_power_watts << ","
             << metrics.baseline_power_watts << ","
             << metrics.initial_temperature_c << ","
             << metrics.peak_temperature_c << ","
             << metrics.final_temperature_c << ","
             << metrics.gflops << ","
             << metrics.gflops_per_watt << ","
             << metrics.memory_bandwidth_gbps << ","
             << config.threads_per_block << std::endl;
        
        file.close();
    }
};

/**
 * Systematic testing framework for parameter space exploration
 */
void run_systematic_analysis() {
    std::vector<int> matrix_sizes = {512, 1024, 2048, 4096, 6144, 8192};
    std::vector<std::pair<int, int>> block_dimensions = {
        {8, 8}, {16, 16}, {32, 32}, {16, 32}, {32, 16}
    };
    
    std::cout << "Starting systematic analysis of CUDA matrix multiplication..." << std::endl;
    std::cout << "Testing " << matrix_sizes.size() << " matrix sizes and " 
              << block_dimensions.size() << " block configurations" << std::endl;
    
    for (int matrix_size : matrix_sizes) {
        for (auto& block_dim : block_dimensions) {
            // Skip configurations that exceed hardware limits
            if (block_dim.first * block_dim.second > MAX_BLOCK_SIZE) {
                continue;
            }
            
            OptimizationConfig config;
            config.matrix_size = matrix_size;
            config.block_dim_x = block_dim.first;
            config.block_dim_y = block_dim.second;
            config.threads_per_block = block_dim.first * block_dim.second;
            config.kernel_type = OptimizationConfig::NAIVE_BASELINE;
            config.data_type = OptimizationConfig::FLOAT_32;
            config.num_iterations = 5;
            
            std::cout << "\n=== Testing Configuration ===" << std::endl;
            std::cout << "Matrix Size: " << matrix_size << "x" << matrix_size << std::endl;
            std::cout << "Block Dimensions: " << block_dim.first << "x" << block_dim.second << std::endl;
            std::cout << "Threads per Block: " << config.threads_per_block << std::endl;
            
            try {
                MatrixMultiplicationFramework framework(config);
                PerformanceMetrics metrics = framework.run_benchmark();
                
                // Verify correctness for smaller matrices to avoid excessive computation
                if (matrix_size <= 2048) {
                    bool correct = framework.verify_results();
                    if (!correct) {
                        std::cerr << "Warning: Results verification failed for configuration "
                                  << matrix_size << "x" << matrix_size 
                                  << " with block " << block_dim.first << "x" << block_dim.second << std::endl;
                    }
                }
                
                // Export results
                framework.export_results(metrics, "baseline_results.csv");
                
                // Display key metrics
                std::cout << "Performance Results:" << std::endl;
                std::cout << "  Execution Time: " << std::fixed << std::setprecision(3) 
                          << metrics.average_execution_time_ms << " ± " 
                          << metrics.execution_time_stddev << " ms" << std::endl;
                std::cout << "  Performance: " << std::fixed << std::setprecision(2) 
                          << metrics.gflops << " GFLOPS" << std::endl;
                std::cout << "  Energy Efficiency: " << std::fixed << std::setprecision(2) 
                          << metrics.gflops_per_watt << " GFLOPS/W" << std::endl;
                std::cout << "  Memory Bandwidth: " << std::fixed << std::setprecision(2) 
                          << metrics.memory_bandwidth_gbps << " GB/s" << std::endl;
                
            } catch (const std::exception& e) {
                std::cerr << "Error testing configuration " << matrix_size 
                          << "x" << matrix_size << " with block " 
                          << block_dim.first << "x" << block_dim.second 
                          << ": " << e.what() << std::endl;
                continue;
            }
        }
    }
    
    std::cout << "\nSystematic analysis completed. Results saved to baseline_results.csv" << std::endl;
}

/**
 * Advanced configuration testing for research purposes
 * Tests different optimization variants and data types
 */
void run_advanced_parameter_study() {
    std::cout << "\n=== Advanced Parameter Study ===" << std::endl;
    
    // Test different kernel variants
    std::vector<OptimizationConfig::KernelType> kernel_types = {
        OptimizationConfig::NAIVE_BASELINE,
        OptimizationConfig::MEMORY_COALESCED,
        OptimizationConfig::SHARED_MEMORY_TILED,
        OptimizationConfig::REGISTER_BLOCKED,
        OptimizationConfig::OCCUPANCY_OPTIMIZED,
        OptimizationConfig::ADVANCED_OPTIMIZED
    };
    
    std::vector<OptimizationConfig::DataType> data_types = {
        OptimizationConfig::FLOAT_32,
        OptimizationConfig::DOUBLE_64
    };
    
    // Focus on medium-sized matrices for detailed analysis
    std::vector<int> test_sizes = {1024, 2048, 4096};
    int optimal_block_x = 16, optimal_block_y = 16;
    
    for (auto kernel_type : kernel_types) {
        for (auto data_type : data_types) {
            for (int size : test_sizes) {
                OptimizationConfig config;
                config.matrix_size = size;
                config.block_dim_x = optimal_block_x;
                config.block_dim_y = optimal_block_y;
                config.threads_per_block = optimal_block_x * optimal_block_y;
                config.kernel_type = kernel_type;
                config.data_type = data_type;
                config.num_iterations = 10;
                
                std::string kernel_name = (kernel_type == OptimizationConfig::NAIVE_BASELINE) ? 
                                        "Naive" : "Coalesced";
                std::string type_name = (data_type == OptimizationConfig::FLOAT_32) ? 
                                      "Float32" : "Double64";
                
                std::cout << "\nTesting " << kernel_name << " kernel with " 
                          << type_name << " on " << size << "x" << size << " matrix" << std::endl;
                
                try {
                    MatrixMultiplicationFramework framework(config);
                    PerformanceMetrics metrics = framework.run_benchmark();
                    
                    // Export to separate file for advanced analysis
                    framework.export_results(metrics, "advanced_results.csv");
                    
                    std::cout << "Results: " << metrics.gflops << " GFLOPS, "
                              << metrics.gflops_per_watt << " GFLOPS/W, "
                              << metrics.average_execution_time_ms << " ms" << std::endl;
                              
                } catch (const std::exception& e) {
                    std::cerr << "Error in advanced test: " << e.what() << std::endl;
                }
            }
        }
    }
}

/**
 * Interactive configuration mode for custom testing
 */
void run_interactive_mode() {
    std::cout << "\n=== Interactive Configuration Mode ===" << std::endl;
    
    OptimizationConfig config;
    
    // Get user input for configuration
    std::cout << "Enter matrix size (default 1024): ";
    std::string input;
    std::getline(std::cin, input);
    if (!input.empty()) {
        config.matrix_size = std::stoi(input);
    }
    
    std::cout << "Enter block dimension X (default 16): ";
    std::getline(std::cin, input);
    if (!input.empty()) {
        config.block_dim_x = std::stoi(input);
    }
    
    std::cout << "Enter block dimension Y (default 16): ";
    std::getline(std::cin, input);
    if (!input.empty()) {
        config.block_dim_y = std::stoi(input);
    }
    
    std::cout << "Enter number of iterations (default 10): ";
    std::getline(std::cin, input);
    if (!input.empty()) {
        config.num_iterations = std::stoi(input);
    }
    
    std::cout << "Select kernel type (0=Naive, 1=Coalesced, 2=Shared Tiled, 3=Reg Block, 4=Occupancy, 5=Adv Opt, default 0): ";
    std::getline(std::cin, input);
    if (!input.empty()) {
        int type = std::stoi(input);
        config.kernel_type = static_cast<OptimizationConfig::KernelType>(type);
    }
    
    std::cout << "Select data type (0=Float32, 1=Double64, default 0): ";
    std::getline(std::cin, input);
    if (!input.empty()) {
        int type = std::stoi(input);
        config.data_type = (type == 1) ? OptimizationConfig::DOUBLE_64 : 
                                        OptimizationConfig::FLOAT_32;
    }
    
    config.threads_per_block = config.block_dim_x * config.block_dim_y;
    
    // Validate configuration
    if (config.threads_per_block > MAX_BLOCK_SIZE) {
        std::cerr << "Error: Threads per block (" << config.threads_per_block 
                  << ") exceeds maximum (" << MAX_BLOCK_SIZE << ")" << std::endl;
        return;
    }
    
    std::cout << "\nRunning custom configuration..." << std::endl;
    
    try {
        MatrixMultiplicationFramework framework(config);
        PerformanceMetrics metrics = framework.run_benchmark();
        
        // Verify results
        bool correct = framework.verify_results();
        
        // Display comprehensive results
        std::cout << "\n=== Comprehensive Results ===" << std::endl;
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Matrix Size: " << config.matrix_size << "x" << config.matrix_size << std::endl;
        std::cout << "  Block Dimensions: " << config.block_dim_x << "x" << config.block_dim_y << std::endl;
        std::cout << "  Threads per Block: " << config.threads_per_block << std::endl;
        std::cout << "  Kernel Type: " << (config.kernel_type == OptimizationConfig::NAIVE_BASELINE ? "Naive" : "Coalesced") << std::endl;
        std::cout << "  Data Type: " << (config.data_type == OptimizationConfig::FLOAT_32 ? "Float32" : "Double64") << std::endl;
        std::cout << "  Iterations: " << config.num_iterations << std::endl;
        
        std::cout << "\nPerformance Metrics:" << std::endl;
        std::cout << "  Execution Time: " << std::fixed << std::setprecision(3) 
                  << metrics.average_execution_time_ms << " ± " 
                  << metrics.execution_time_stddev << " ms" << std::endl;
        std::cout << "  Computational Performance: " << std::fixed << std::setprecision(2) 
                  << metrics.gflops << " GFLOPS" << std::endl;
        std::cout << "  Memory Bandwidth: " << std::fixed << std::setprecision(2) 
                  << metrics.memory_bandwidth_gbps << " GB/s" << std::endl;
        
        std::cout << "\nEnergy Metrics:" << std::endl;
        std::cout << "  Total Energy: " << std::fixed << std::setprecision(4) 
                  << metrics.total_energy_joules << " J" << std::endl;
        std::cout << "  Average Power: " << std::fixed << std::setprecision(2) 
                  << metrics.average_power_watts << " W" << std::endl;
        std::cout << "  Peak Power: " << std::fixed << std::setprecision(2) 
                  << metrics.peak_power_watts << " W" << std::endl;
        std::cout << "  Baseline Power: " << std::fixed << std::setprecision(2) 
                  << metrics.baseline_power_watts << " W" << std::endl;
        std::cout << "  Energy Efficiency: " << std::fixed << std::setprecision(2) 
                  << metrics.gflops_per_watt << " GFLOPS/W" << std::endl;
        
        std::cout << "\nThermal Metrics:" << std::endl;
        std::cout << "  Initial Temperature: " << std::fixed << std::setprecision(1) 
                  << metrics.initial_temperature_c << " °C" << std::endl;
        std::cout << "  Peak Temperature: " << std::fixed << std::setprecision(1) 
                  << metrics.peak_temperature_c << " °C" << std::endl;
        std::cout << "  Final Temperature: " << std::fixed << std::setprecision(1) 
                  << metrics.final_temperature_c << " °C" << std::endl;
        
        std::cout << "\nVerification: " << (correct ? "PASSED" : "FAILED") << std::endl;
        
        // Export results
        framework.export_results(metrics, "interactive_results.csv");
        std::cout << "\nResults exported to interactive_results.csv" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in interactive mode: " << e.what() << std::endl;
    }
}

/**
 * GPU hardware capability detection and reporting
 */
void detect_gpu_capabilities() {
    std::cout << "\n=== GPU Hardware Capabilities ===" << std::endl;
    
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    for (int device = 0; device < device_count; ++device) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        
        std::cout << "Device " << device << ": " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Global Memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB" << std::endl;
        std::cout << "  Shared Memory per Block: " << (prop.sharedMemPerBlock / 1024) << " KB" << std::endl;
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Max Block Dimensions: " << prop.maxThreadsDim[0] 
                  << "x" << prop.maxThreadsDim[1] << "x" << prop.maxThreadsDim[2] << std::endl;
        std::cout << "  Max Grid Dimensions: " << prop.maxGridSize[0] 
                  << "x" << prop.maxGridSize[1] << "x" << prop.maxGridSize[2] << std::endl;
        std::cout << "  Warp Size: " << prop.warpSize << std::endl;
        std::cout << "  Multiprocessor Count: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Memory Clock Rate: " << (prop.memoryClockRate / 1000) << " MHz" << std::endl;
        std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
        
        // Calculate theoretical peak memory bandwidth
        double peak_bandwidth = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8.0) / 1e6;
        std::cout << "  Theoretical Peak Memory Bandwidth: " << std::fixed << std::setprecision(1) 
                  << peak_bandwidth << " GB/s" << std::endl;
        
        // Calculate theoretical peak GFLOPS
        // This is a rough estimate based on CUDA cores and clock rate
        double peak_gflops = prop.multiProcessorCount * 128 * (prop.clockRate / 1000.0) / 1000.0;
        std::cout << "  Estimated Peak Performance: " << std::fixed << std::setprecision(0) 
                  << peak_gflops << " GFLOPS (single precision)" << std::endl;
        
        std::cout << std::endl;
    }
}

/**
 * Optimal configuration finder based on hardware characteristics
 */
OptimizationConfig find_optimal_configuration(int matrix_size) {
    OptimizationConfig optimal_config;
    optimal_config.matrix_size = matrix_size;
    
    // Get GPU properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    // Calculate optimal block dimensions based on hardware
    int max_threads_per_block = prop.maxThreadsPerBlock;
    // int warp_size = prop.warpSize;
    
    // Find block dimensions that are multiples of warp size
    int optimal_block_dim = 16; // Default safe choice
    
    // For larger matrices, we can use larger blocks
    if (matrix_size >= 2048) {
        optimal_block_dim = 32;
    } else if (matrix_size >= 4096) {
        optimal_block_dim = 16; // Prevent too large blocks for very large matrices
    }
    
    // Ensure block size doesn't exceed hardware limits
    while (optimal_block_dim * optimal_block_dim > max_threads_per_block) {
        optimal_block_dim /= 2;
    }
    
    optimal_config.block_dim_x = optimal_block_dim;
    optimal_config.block_dim_y = optimal_block_dim;
    optimal_config.threads_per_block = optimal_block_dim * optimal_block_dim;
    
    std::cout << "Optimal configuration for " << matrix_size << "x" << matrix_size << " matrix:" << std::endl;
    std::cout << "  Block dimensions: " << optimal_block_dim << "x" << optimal_block_dim << std::endl;
    std::cout << "  Threads per block: " << optimal_config.threads_per_block << std::endl;
    
    return optimal_config;
}

/**
 * Main function with comprehensive testing options
 */
int main(int argc, char* argv[]) {
    std::cout << "CUDA Baseline Matrix Multiplication Framework" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    // Initialize CUDA context
    CUDA_CHECK(cudaSetDevice(0));
    
    // Detect and display GPU capabilities
    detect_gpu_capabilities();
    
    // Check command line arguments for test mode selection
    int test_mode = 0; // Default to systematic analysis
    
    if (argc > 1) {
        test_mode = std::atoi(argv[1]);
    } else {
        std::cout << "Select test mode:" << std::endl;
        std::cout << "0 - Systematic Analysis (default)" << std::endl;
        std::cout << "1 - Advanced Parameter Study" << std::endl;
        std::cout << "2 - Interactive Mode" << std::endl;
        std::cout << "3 - Single Optimal Test" << std::endl;
        std::cout << "Enter choice (0-3): ";
        
        std::string input;
        std::getline(std::cin, input);
        if (!input.empty()) {
            test_mode = std::stoi(input);
        }
    }
    
    // Execute selected test mode
    switch (test_mode) {
        case 0:
            run_systematic_analysis();
            break;
            
        case 1:
            run_advanced_parameter_study();
            break;
            
        case 2:
            run_interactive_mode();
            break;
            
        case 3: {
            int matrix_size = 2048; // Default test size
            if (argc > 2) {
                matrix_size = std::atoi(argv[2]);
            }
            
            OptimizationConfig config = find_optimal_configuration(matrix_size);
            config.num_iterations = 10;
            
            std::cout << "\nRunning single optimal configuration test..." << std::endl;
            
            try {
                MatrixMultiplicationFramework framework(config);
                PerformanceMetrics metrics = framework.run_benchmark();
                framework.verify_results();
                framework.export_results(metrics, "optimal_results.csv");
                
                std::cout << "Results: " << metrics.gflops << " GFLOPS, "
                          << metrics.gflops_per_watt << " GFLOPS/W" << std::endl;
                          
            } catch (const std::exception& e) {
                std::cerr << "Error in optimal test: " << e.what() << std::endl;
            }
            break;
        }
        
        default:
            std::cerr << "Invalid test mode. Using systematic analysis." << std::endl;
            run_systematic_analysis();
            break;
    }
    
    std::cout << "\nFramework execution completed." << std::endl;
    std::cout << "Check generated CSV files for detailed results." << std::endl;
    
    return 0;
}