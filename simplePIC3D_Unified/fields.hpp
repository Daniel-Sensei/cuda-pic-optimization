#include <fftw3.h>
#include <math.h>
#include <string.h>
#include <cufft.h>

namespace depth_level_2
{
    // Kernel to solve Poisson in Fourier space
    __global__ void solvePoissonFourierKernel(cufftDoubleComplex* out, double Lx, double Ly, double Lz, int Nx, int Ny, int Nz) {
        int Nzh = Nz/2 + 1;
        int size = Nx * Ny * Nzh;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= size) return;

        // Compute the 3D index
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


    // Kernel to solve the gradient in Fourier space
    __global__ void solveGradientFourierKernel(cufftDoubleComplex *out, cufftDoubleComplex *outX, 
                                            cufftDoubleComplex *outY, cufftDoubleComplex *outZ, 
                                            double Lx, double Ly, double Lz, int Nx, int Ny, int Nz, int sign) {
        int Nzh = Nz / 2 + 1;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= Nx * Ny * Nzh) return;

        // Compute the 3D index
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

        // Gradient computation
        outX[tid].x = -value.y * k1 * sign;
        outX[tid].y =  value.x * k1 * sign;

        outY[tid].x = -value.y * k2 * sign;
        outY[tid].y =  value.x * k2 * sign;

        outZ[tid].x = -value.y * k3 * sign;
        outZ[tid].y =  value.x * k3 * sign;
    }

    // Kernel to normalize the field
    __global__ void normalizeKernel(double* phi, int Nx, int Ny, int Nz) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int size = Nx * Ny * Nz;
        if (tid < size) {
            phi[tid] /= (double)(Nx * Ny * Nz);
        }
    }

} // end of depth_level_2 namespace


// Kernel to compute the total charge density
__global__ void computeRhoTotKernel(double* rho_tot, double* rhos, int ns, int nx, int ny, int nz) {
    

    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    // Total number of cells
    int totalCells = nx * ny * nz;

    if (threadId >= totalCells) return;

    // Index for 3D cell
    int i = threadId / (ny * nz); 
    int j = (threadId / nz) % ny;
    int k = threadId % nz; 
    
    // Linear index
    int idx = i * ny * nz + j * nz + k;

    // Compute the total charge density
    for (int is = 0; is < ns; is++) {
        int idx_rhos = is * nx * ny * nz + idx;
        rho_tot[idx] += -4.0 * M_PI * rhos[idx_rhos];
    }
}



void poisson(double* phi, double* rho, double Lx, double Ly, double Lz, int Nx, int Ny, int Nz, int blockSize) {
    
    double* in;
    int Nzh = Nz / 2 + 1;
    size_t realSize = Nx * Ny * Nz * sizeof(double);
    size_t complexSize = Nx * Ny * Nzh * sizeof(cufftDoubleComplex);

    // Create the CUFFT plans
    cufftDoubleComplex* out;
    cufftHandle forwardPlan, inversePlan;
    cufftPlan3d(&forwardPlan, Nx, Ny, Nz, CUFFT_D2Z);
    cufftPlan3d(&inversePlan, Nx, Ny, Nz, CUFFT_Z2D);
    cudaMallocManaged(&in, realSize);
    cudaMallocManaged(&out, complexSize);

    // Copy rho to in
    cudaMemcpy(in, rho, realSize, cudaMemcpyHostToDevice);

    // Execute forward FFT
    cufftExecD2Z(forwardPlan, (cufftDoubleReal*)in, (cufftDoubleComplex*)out);
    cudaDeviceSynchronize();

    // Block and grid size for the kernel
    int size = Nx * Ny * Nzh;
    int gridSize = (size + blockSize - 1) / blockSize;
    depth_level_2::solvePoissonFourierKernel<<<gridSize, blockSize>>>(out, Lx, Ly, Lz, Nx, Ny, Nz);
    cudaDeviceSynchronize();
    
    // Execute inverse FFT
    cufftExecZ2D(inversePlan, (cufftDoubleComplex*)out, (cufftDoubleReal*)in);
    cudaDeviceSynchronize();

    // Block and grid size for the kernel
    size = Nx * Ny * Nz;
    gridSize = (size + blockSize - 1) / blockSize;
    depth_level_2::normalizeKernel<<<gridSize, blockSize>>>(in, Nx, Ny, Nz);
    cudaDeviceSynchronize();
   
    // Copy in back to phi
    cudaMemcpy(phi, in, realSize, cudaMemcpyDeviceToDevice);

    // Clean up
    cufftDestroy(forwardPlan);
    cufftDestroy(inversePlan);
    cudaFree(in);
    cudaFree(out);
}

void gradient(double *gradX, double *gradY, double *gradZ, double *phi, 
              double Lx, double Ly, double Lz, int Nx, int Ny, int Nz, int sign, int blockSize) {
    
    // Memory allocation on the GPU
    int Nzh = Nz / 2 + 1;
    cufftDoubleComplex *out, *outX, *outY, *outZ;
    cudaMallocManaged(&out, Nx * Ny * Nzh * sizeof(cufftDoubleComplex)); 
    cudaMallocManaged(&outX, Nx * Ny * Nzh * sizeof(cufftDoubleComplex));
    cudaMallocManaged(&outY, Nx * Ny * Nzh * sizeof(cufftDoubleComplex)); 
    cudaMallocManaged(&outZ, Nx * Ny * Nzh * sizeof(cufftDoubleComplex));

    // Create the CUFFT plans
    cufftHandle forwardPlan, inversePlan;
    cufftPlan3d(&inversePlan, Nx, Ny, Nz, CUFFT_Z2D);
    cufftPlan3d(&forwardPlan, Nx, Ny, Nz, CUFFT_D2Z);
    cufftExecD2Z(forwardPlan, phi, out);
    cudaDeviceSynchronize();
    
    // Solve the gradient in Fourier space
    int size = Nx * Ny * Nzh;
    int gridSize = (size + blockSize - 1) / blockSize;
    depth_level_2::solveGradientFourierKernel<<<gridSize, blockSize>>>(out, outX, outY, outZ, 
                                                        Lx, Ly, Lz, Nx, Ny, Nz, sign);
    cudaDeviceSynchronize();

    // Execute the inverse FFT
    cufftExecZ2D(inversePlan, outX, gradX);
    cufftExecZ2D(inversePlan, outY, gradY);
    cufftExecZ2D(inversePlan, outZ, gradZ);

    // Normalize the field
    gridSize = (Nx * Ny * Nz + blockSize - 1) / blockSize;
    depth_level_2::normalizeKernel<<<gridSize, blockSize>>>(gradX, Nx, Ny, Nz);
    depth_level_2::normalizeKernel<<<gridSize, blockSize>>>(gradY, Nx, Ny, Nz);
    depth_level_2::normalizeKernel<<<gridSize, blockSize>>>(gradZ, Nx, Ny, Nz);
    cudaDeviceSynchronize();

    // Cleanup
    cudaFree(out);
    cudaFree(outX);
    cudaFree(outY);
    cudaFree(outZ);
    cufftDestroy(forwardPlan);
    cufftDestroy(inversePlan);
}