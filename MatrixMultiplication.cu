#include <iostream>
#include <cuda.h>
#include <stdlib.h>
#include <ctime>
#include <cuda_runtime.h>
#include <chrono>

//based on the CUDA tutorial from CoffeBeforeArch. Big Matrix Multiplication

__global__ void MatrixMul (int *a, int *b, int *c, int n)
{
    //assign IDs to threads. Row is X and Col is Y
    //Block Dim would be constant = 256
    int row_id = blockIdx.x * blockDim.x + threadIdx.x;
    int col_id = blockIdx.y * blockDim.y + threadIdx.y;
    //Now each thread will know a value from 0 to 1024 in both row and column. 
    //But what were doing is giving X(horizontal) Threads tackle A and Y(vertical) ones tackle B

    //Again, protect against the threads that will be out of scope (of 1024*1024 i.e the remaining threads on GPU)
    if ((row_id < n) && (col_id < n))
    {   
        int sum = 0;
        for (size_t i = 0; i < n; i++)
        {
            //accumulate result of a single element. It's a 2D Matrix Math 
            sum += a[row_id * n + i] * b[i * n + col_id];
            //best way to understand is to take a,b and c as linear arrays. c is the the product of a and b on each index
            //to convert a 2D Matrx to 1D, we use -> Index = i*n + j
            //Here we're mapping all 3 1D arrays accordingly.
        }
        c[row_id * n + col_id] = sum;
    }
}

void init_matrices (int *a, int *b, int n)
{   
    srand(time(NULL));
    //random values I suppose
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            a[i * n + j] = rand() % 100;
            b[i * n + j] = rand() % 100;
        }
        //directly assume the last case -> when i = 1023 then index is will be 1023*1024 + [0 to 1023] which will end at 1048575th index
        //which would be ~ 1.05 million elements in a linear arrangement. Since they all decay to a linear arrangement due to how memory works
        //1.05 million is technically the total amount of threads we are launching for each element
        //it just so happens that we split it into a 1024*1024 elements grid and launch 4096*256 threads for it
    }
    
}
// Print a matrix (only prints a small part for clarity where you can pass on Name and Print Size)
void print_matrix(const char* name, int *mat, int n, int print_size) {
    std::cout << name << " (First " << print_size << "x" << print_size << " block):\n";
    for (int i = 0; i < print_size; i++) {
        for (int j = 0; j < print_size; j++) {
            std::cout << mat[i * n + j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main ()
{
    int n = 1 << 10;
    //or just write 1024 whatever

    //the size var to be used in memcpy and malloc. 1024 rows * 1024 cols * 4 bytes each
    size_t bytes = n * n * sizeof(int);

    //host ptrs. 2 for matrices 1 as resultant
    int *h_a, *h_b, *h_c;

    //allocate mem on host for their ptrs
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);

    //Device ptrs same way. 2 for matrices 1 as resultant
    int *d_a, *d_b, *d_c;

    //allocate mem on gpu for its ptrs
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    //Initialize the matrices on host
    init_matrices(h_a, h_b, n);

    //Measure total time (CPU + GPU) using std::chrono
    auto start_total = std::chrono::high_resolution_clock::now();

    //copy the 2 host ptrs onto the device ptrs
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    //CUDA events for GPU execution timing since nvprof wasn't working in my case
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //How many threads per block (kinda unnecessary)
    int BLOCK_SIZE = 16;
    //How many blocks in total (aka Grid Size)
    int GRID_SIZE = (n + BLOCK_SIZE - 1)/BLOCK_SIZE;       //safety division

    //calculating thread/block count as a dim3 object(s) to pass onto kernel
    //dim3 is scalable and can be 1D,2D or 3D. Here it becomes a 2D Grid/Matrix 
    dim3 grid(GRID_SIZE, GRID_SIZE);        //size is GRID_SIZE*GRID_SIZE -> which should be 64*64 = 4096
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);   //in essence 16*16 = 256 threads per block this means. 2D Grid so Threads will have both X and Y component
    
    //Start measuring the GPU time
    cudaEventRecord(start);

    //Basically -> whole Grid with 64 blocks in X and Y direction resp. and each block has 256 threads i.e 16 in X and Y direction resp.
    //So now, call the kernel with these two dim3 vars. Total threads are 4096*256 = ~ 1.05 million
    MatrixMul<<<grid,threads>>>(d_a, d_b, d_c, n);

    //cudaDeviceSynchronize();

    //Stop measuring the GPU time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    //Measure the elapsed time on the GPU
    float milliseconds = 0.f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    //copy only the resultant device ptr back to resultant host ptr
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    //Measure total time (CPU + GPU) using std::chrono
    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_total = end_total - start_total;

    //Print the timing results first
    std::cout << "CUDA Kernel Execution Time (GPU only): " << milliseconds << " ms" << std::endl;
    std::cout << "Total Execution Time (CPU + GPU): " << elapsed_total.count() << " seconds" << std::endl;

    // Print a small portion of all 3 matrices
    int print_size = 5;  // Print the first AxA block
    std::cout<<"\nPrinting some parts of all 3 Matrices ->\n\n";
    print_matrix("Matrix A", h_a, n, print_size);
    print_matrix("Matrix B", h_b, n, print_size);
    print_matrix("Matrix C (Resultant)", h_c, n, print_size);
    
    //Free mem on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    //on host as well please
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}