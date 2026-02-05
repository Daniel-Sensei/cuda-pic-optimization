#include <fftw3.h>
#include <math.h>
#include <string.h>
#include <cufft.h>

namespace depth_level_2
{

    // Normalizing the field
    __global__ void normalizeKernel(double* phi, int Nx, int Ny, int Nz) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int size = Nx * Ny * Nz;
        if (tid < size) {
            phi[tid] /= (double)(Nx * Ny * Nz);
        }
    }

    // Solving the Poisson equation in Fourier space
    __global__ void solvePoissonFourierKernel(cufftDoubleComplex* out, double Lx, double Ly, double Lz, int Nx, int Ny, int Nz) {
        
        int Nzh = Nz/2 + 1;
        int size = Nx * Ny * Nzh;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        if (tid >= size) return;

        int i = tid / (Ny * Nzh);
        int remainder = tid % (Ny * Nzh);
        int j = remainder / Nzh;
        int k = remainder % Nzh;

        int II = (2*i < Nx) ? i : Nx - i;
        int JJ = (2*j < Ny) ? j : Ny - j;
        double k1 = 2.0 * M_PI * II / Lx;
        double k2 = 2.0 * M_PI * JJ / Ly;
        double k3 = 2.0 * M_PI * k  / Lz;

        double fac = -(k1*k1 + k2*k2 + k3*k3);
        cufftDoubleComplex val = out[tid];
        
        if (fabs(fac) < 1e-14) {
            out[tid].x = 0.0;
            out[tid].y = 0.0;
        } else {
            out[tid].x = val.x / fac;
            out[tid].y = val.y / fac;
        }
    }


    // Solving the gradient in Fourier space
    __global__ void solveGradientFourierKernel(cufftDoubleComplex *out, cufftDoubleComplex *outX, 
                                           cufftDoubleComplex *outY, cufftDoubleComplex *outZ, 
                                           double Lx, double Ly, double Lz, int Nx, int Ny, int Nz, int sign) {
        int Nzh = Nz / 2 + 1;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= Nx * Ny * Nzh) return;

        int i = tid / (Ny * Nzh);
        int remainder = tid % (Ny * Nzh);
        int j = remainder / Nzh;
        int k = remainder % Nzh;

        int II = (2 * i < Nx) ? i : i - Nx;
        int JJ = (2 * j < Ny) ? j : j - Ny;
        double k1 = 2.0 * M_PI * II / Lx;
        double k2 = 2.0 * M_PI * JJ / Ly;
        double k3 = 2.0 * M_PI * k / Lz;

        cufftDoubleComplex value = out[tid];

        outX[tid].x = -value.y * k1 * sign;
        outX[tid].y =  value.x * k1 * sign;

        outY[tid].x = -value.y * k2 * sign;
        outY[tid].y =  value.x * k2 * sign;

        outZ[tid].x = -value.y * k3 * sign;
        outZ[tid].y =  value.x * k3 * sign;
    }


}

__global__ void computeRhoTotKernel(double* d_rho_tot, double* d_rhos, int ns, int nx, int ny, int nz) {
    
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int totalCells = nx * ny * nz;
    
    if (threadId >= totalCells) return;

    // Calculate the 3D index
    int i = threadId / (ny * nz);
    int j = (threadId / nz) % ny;
    int k = threadId % nz; 
    
    // Calculate linear index
    int idx = i * ny * nz + j * nz + k;

    // Compute the total charge density
    for (int is = 0; is < ns; is++) {
        int idx_rhos = is * nx * ny * nz + idx; 
        d_rho_tot[idx] += -4.0 * M_PI * d_rhos[idx_rhos];
    }
}


void poisson(double* d_phi, double* d_rho, double Lx, double Ly, double Lz, int Nx, int Ny, int Nz, int blockSize)
{
    int Nzh = Nz / 2 + 1;
    size_t size = Nx * Ny * Nz * sizeof(double);
    size_t complexSize = Nx * Ny * Nzh * sizeof(cufftDoubleComplex);

    double* d_in;
    cufftDoubleComplex* d_out;
    cufftHandle forwardPlan, inversePlan;

    // Create the CUFFT plans
    cufftPlan3d(&forwardPlan, Nx, Ny, Nz, CUFFT_D2Z);
    cufftPlan3d(&inversePlan, Nx, Ny, Nz, CUFFT_Z2D);

    // Allocate memory on the GPU
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, complexSize);

    // Copy the input data to the GPU
    cudaMemcpy(d_in, d_rho, size, cudaMemcpyDeviceToDevice);    
    cudaDeviceSynchronize();

    // Execute the forward FFT
    cufftExecD2Z(forwardPlan, d_in, d_out);

    // Solve the Poisson equation in Fourier space
    int totalThreads = Nx * Ny * Nzh;
    int gridSize = (totalThreads + blockSize - 1) / blockSize;
    depth_level_2::solvePoissonFourierKernel<<<gridSize, blockSize>>>(d_out, Lx, Ly, Lz, Nx, Ny, Nz);
    cudaDeviceSynchronize();

    // Execute the inverse FFT   
    cufftExecZ2D(inversePlan, d_out, d_in);
    cudaDeviceSynchronize();

    // Normalize the field
    int totalCells = Nx * Ny * Nz;
    gridSize = (totalCells + blockSize - 1) / blockSize;
    depth_level_2::normalizeKernel<<<gridSize, blockSize>>>(d_in, Nx, Ny, Nz);
    cudaDeviceSynchronize();

    // Copy the result back to phi
    cudaMemcpy(d_phi, d_in, size, cudaMemcpyDeviceToDevice);

    // Clean up
    cufftDestroy(forwardPlan);
    cufftDestroy(inversePlan);
    cudaFree(d_in);
    cudaFree(d_out);
}

void gradient(double *d_gradX, double *d_gradY, double *d_gradZ, double *d_phi, 
                   double Lx, double Ly, double Lz, int Nx, int Ny, int Nz, int sign, int blockSize) {
    
    // Memory allocation on the GPU
    int Nzh = Nz / 2 + 1;
    cufftDoubleComplex *d_out, *d_outX, *d_outY, *d_outZ;
    size_t complexSize = Nx * Ny * Nzh * sizeof(cufftDoubleComplex);
    cudaMalloc(&d_out,  complexSize);
    cudaMalloc(&d_outX, complexSize);
    cudaMalloc(&d_outY, complexSize);
    cudaMalloc(&d_outZ, complexSize);

    // CUFFT plans
    cufftHandle planForward, planInverse;
    cufftPlan3d(&planForward, Nx, Ny, Nz, CUFFT_D2Z);
    cufftPlan3d(&planInverse, Nx, Ny, Nz, CUFFT_Z2D);

    // FTT execution
    cufftExecD2Z(planForward, d_phi, d_out);
    cudaDeviceSynchronize();

    // Gradient computation in Fourier space
    int totalThreads = Nx * Ny * Nzh;
    int gridSize = (totalThreads + blockSize - 1) / blockSize;
    depth_level_2::solveGradientFourierKernel<<<gridSize, blockSize>>>(d_out, d_outX, d_outY, d_outZ, Lx, Ly, Lz, Nx, Ny, Nz, sign);
    cudaDeviceSynchronize();

    // Inverse FFT execution
    cufftExecZ2D(planInverse, d_outX, d_gradX);
    cufftExecZ2D(planInverse, d_outY, d_gradY);
    cufftExecZ2D(planInverse, d_outZ, d_gradZ);
    cudaDeviceSynchronize();

    // Normalization
    int totalCells = Nx * Ny * Nz;
    gridSize = (totalCells + blockSize - 1) / blockSize;
    depth_level_2::normalizeKernel<<<gridSize, blockSize>>>(d_gradX, Nx, Ny, Nz);
    depth_level_2::normalizeKernel<<<gridSize, blockSize>>>(d_gradY, Nx, Ny, Nz);
    depth_level_2::normalizeKernel<<<gridSize, blockSize>>>(d_gradZ, Nx, Ny, Nz);
    cudaDeviceSynchronize();

    // Clean up
    cufftDestroy(planForward);
    cufftDestroy(planInverse);
    cudaFree(d_out);
    cudaFree(d_outX);
    cudaFree(d_outY);
    cudaFree(d_outZ);
}
