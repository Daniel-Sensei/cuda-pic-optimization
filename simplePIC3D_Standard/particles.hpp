#pragma once
#include "parameters.hpp"
#include "structures.hpp"
#include <cuda_runtime.h>


// GPU kernel for particle-to-grid charge density mapping with thread-level privatization.
// Calculates the charge contribution of each particle to nearby grid nodes using trilinear interpolation.
// Uses local arrays to reduce global memory access and improve performance.
__global__ void particles2GridAtomicFreePrivatization(
    double* d_rhos, double* d_nodeX, double* d_nodeY, double* d_nodeZ,
    double* d_rx, double* d_ry, double* d_rz, double* d_q,
    int nx, int ny, int nz, double dx, double dy, double dz, double invVOL,
    const int is, int np) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= np) return;

    // Indices of the cell to which the particle belongs
    int ix = int(d_rx[tid] / dx);
    int iy = int(d_ry[tid] / dy);
    int iz = int(d_rz[tid] / dz);

    // Local arrays for weights and indices
    double weights_local[8];
    int index_local[8];

    // Calculate distances for weight computation
    double xi[2], eta[2], zeta[2];
    xi[1]   = d_rx[tid] - d_nodeX[ix];
    eta[1]  = d_ry[tid] - d_nodeY[iy];
    zeta[1] = d_rz[tid] - d_nodeZ[iz];

    xi[0]   = d_nodeX[ix + 1] - d_rx[tid];
    eta[0]  = d_nodeY[iy + 1] - d_ry[tid];
    zeta[0] = d_nodeZ[iz + 1] - d_rz[tid];


    // Calculate weights
    double qp = d_q[tid];
    int idx = 0;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) {
                weights_local[idx] = xi[i] * eta[j] * zeta[k] * invVOL * qp * invVOL;
                int ixn = (ix + i) % nx;
                int iyn = (iy + j) % ny;
                int izn = (iz + k) % nz;
                index_local[idx] = is * nx * ny * nz + ixn * ny * nz + iyn * nz + izn;
                idx++;
            }

    // Update the grid with particle contributions  
    for (int i = 0; i < 8; i++) {
        atomicAdd(&d_rhos[index_local[i]], weights_local[i]);
    }
}


// GPU kernel for particle-to-grid charge density mapping with atomic-free coarsening
// This kernel processes multiple particles per thread to reduce the number of atomic operations needed, which is a form of coarsening. 
// Coarsening assigns a batch of particles to each thread instead of a single particle.
__global__ void particles2GridAtomicFreeCoarsening(
    double* d_rhos, double* d_nodeX, double* d_nodeY, double* d_nodeZ,
    double* d_rx, double* d_ry, double* d_rz, double* d_q,
    int nx, int ny, int nz, double dx, double dy, double dz, double invVOL,
    const int is, int np, int particlesNum) {

    // Global index
    int tid = (blockIdx.x * blockDim.x + threadIdx.x) * particlesNum;

    // Processo multiple particelle per thread
    for (int p = tid; p < min(tid + particlesNum, np); p++) {
        int ix = int(d_rx[p] / dx);
        int iy = int(d_ry[p] / dy);
        int iz = int(d_rz[p] / dz);

        // Distance calculation
        double xi[2], eta[2], zeta[2];
        xi[1]   = d_rx[p] - d_nodeX[ix];
        eta[1]  = d_ry[p] - d_nodeY[iy];
        zeta[1] = d_rz[p] - d_nodeZ[iz];

        xi[0]   = d_nodeX[ix + 1] - d_rx[p];
        eta[0]  = d_nodeY[iy + 1] - d_ry[p];
        zeta[0] = d_nodeZ[iz + 1] - d_rz[p];

        // Weight calculation
        double qp = d_q[p];
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 2; k++) {
                    double weight = xi[i] * eta[j] * zeta[k] * invVOL * qp * invVOL;
                    int ixn = (ix + i) % nx;
                    int iyn = (iy + j) % ny;
                    int izn = (iz + k) % nz;
                    int index = is * nx * ny * nz + ixn * ny * nz + iyn * nz + izn;
                    atomicAdd(&d_rhos[index], weight);
                }
    }
}

// GPU kernel for particle-to-grid charge density mapping with atomic-free aggregation.
// This kernel reduces atomic operation overhead by aggregating updates locally before committing them to global memory. 
// This approach minimizes the number of atomic updates on the grid.
__global__ void particles2GridAtomicFreeAggregation(
    double* d_rhos, const double* d_nodeX, const double* d_nodeY, const double* d_nodeZ,
    const double* d_rx, const double* d_ry, const double* d_rz, const double* d_q,
    int nx, int ny, int nz, double dx, double dy, double dz, double invVOL,
    int is, int np)
{
    // Global index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= np) return;

    // Indices of the cell to which the particle belongs
    int ix = int(d_rx[tid] / dx);
    int iy = int(d_ry[tid] / dy);
    int iz = int(d_rz[tid] / dz);

    // Calculate distances
    double xi[2], eta[2], zeta[2];
    xi[1]   = d_rx[tid] - d_nodeX[ix];
    eta[1]  = d_ry[tid] - d_nodeY[iy];
    zeta[1] = d_rz[tid] - d_nodeZ[iz];
    xi[0]   = d_nodeX[ix + 1] - d_rx[tid];
    eta[0]  = d_nodeY[iy + 1] - d_ry[tid];
    zeta[0] = d_nodeZ[iz + 1] - d_rz[tid];

    double qp = d_q[tid];
    double count = 0.0;
    int prevIdx = -1;

    // Calculate weights and update grid
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                int ixn = (ix + i == nx) ? ix - (nx - 1) - 1 : ix;
                int iyn = (iy + j == ny) ? iy - (ny - 1) - 1 : iy;
                int izn = (iz + k == nz) ? iz - (nz - 1) - 1 : iz;

                if (nx == 1) ix = -i;
                if (ny == 1) iy = -j;
                if (nz == 1) iz = -k;

                // Global index
                int idx = is * nx * ny * nz + (ixn + i) * ny * nz + (iyn + j) * nz + (izn + k);
                // Weight calculation
                double weight = xi[i] * eta[j] * zeta[k] * invVOL * qp;
                // Update grid
                double value = weight * invVOL;

                if (idx == prevIdx) {
                    count += value;
                } else {
                    // Commit the accumulated value
                    if (prevIdx != -1 && count != 0.0) {
                        atomicAdd(&d_rhos[prevIdx], count);
                    }
                    // Start a new accumulation
                    prevIdx = idx;
                    count = value;
                }
            }
        }
    }

    // Commit the last accumulated value
    if (prevIdx != -1 && count != 0.0) {
        atomicAdd(&d_rhos[prevIdx], count);
    }
}


// GPU kernel for particle-to-grid charge density mapping.
__global__ void particles2GridKernel(
    double* d_rhos, double* d_nodeX, double* d_nodeY, double* d_nodeZ,
    double* d_rx, double* d_ry, double* d_rz, double* d_q,
    int nx, int ny, int nz, double dx, double dy, double dz, double invVOL,
    const int is, int np) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= np) return;

    // Indices of the cell to which the particle belongs
    int ix = int(d_rx[tid] / dx);
    int iy = int(d_ry[tid] / dy);
    int iz = int(d_rz[tid] / dz);

    double xi[2], eta[2], zeta[2];
    double weights[2][2][2];

    // Calculate distances for weight computation
    xi[1]   = d_rx[tid] - d_nodeX[ix];
    eta[1]  = d_ry[tid] - d_nodeY[iy];
    zeta[1] = d_rz[tid] - d_nodeZ[iz];

    xi[0]   = d_nodeX[ix + 1] - d_rx[tid];
    eta[0]  = d_nodeY[iy + 1] - d_ry[tid];
    zeta[0] = d_nodeZ[iz + 1] - d_rz[tid];

    // Calculate weights
    double qp = d_q[tid];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                weights[i][j][k] = xi[i] * eta[j] * zeta[k] * invVOL * qp;

    // Update the grid with particle contributions
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                int ixn = (ix + i == nx) ? ix - (nx - 1) - 1 : ix;
                int iyn = (iy + j == ny) ? iy - (ny - 1) - 1 : iy;
                int izn = (iz + k == nz) ? iz - (nz - 1) - 1 : iz;

                if (nx == 1) ix = -i;
                if (ny == 1) iy = -j;
                if (nz == 1) iz = -k;

                int index = is * nx * ny * nz + (ixn + i) * ny * nz + (iyn + j) * nz + (izn + k);
                atomicAdd(&d_rhos[index], weights[i][j][k] * invVOL);
            }
        }
    }
}


// GPU kernel for updating particle velocities using the interpolated electric field.
__global__ void updateParticlePositionKernel(double* rx, double* ry, double* rz,
                                             double* vx, double* vy, double* vz,
                                             const int np, double Lx, double Ly, double Lz, double dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < np)
    {
        // Update particle position
        rx[idx] += vx[idx] * dt;
        ry[idx] += vy[idx] * dt;
        rz[idx] += vz[idx] * dt;

        // Periodic boundary conditions
        if (rx[idx] >= Lx) rx[idx] = fmod(rx[idx], Lx);
        if (rx[idx] < 0)   rx[idx] = fmod(rx[idx] + Lx, Lx);

        if (ry[idx] >= Ly) ry[idx] = fmod(ry[idx], Ly);
        if (ry[idx] < 0)   ry[idx] = fmod(ry[idx] + Ly, Ly);

        if (rz[idx] >= Lz) rz[idx] = fmod(rz[idx], Lz);
        if (rz[idx] < 0)   rz[idx] = fmod(rz[idx] + Lz, Lz);

        // Adjust for boundary values
        if (rx[idx] == Lx) rx[idx] = 0;
        if (ry[idx] == Ly) ry[idx] = 0;
        if (rz[idx] == Lz) rz[idx] = 0;
    }
}

// GPU kernel for updating particle velocities using the interpolated electric field.
__global__ void updateParticleVelocityKernel(double* rx, double* ry, double* rz,
                                  double* vx, double* vy, double* vz,
                                  double* Exn, double* Eyn, double* Ezn,
                                  double* nodeX, double* nodeY, double* nodeZ,
                                  int nx, int ny, int nz, 
                                  double dx, double dy, double dz, double invVOL,
                                  double qom, double dt, int np) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= np) return;
 
    // Cell indices
    int ix = int(rx[i] / dx);
    int iy = int(ry[i] / dy);
    int iz = int(rz[i] / dz);

    // Weights
    double weights[2][2][2] = {0};

    // Calculate distances for weight computation
    double xi[2], eta[2], zeta[2];
    xi[1]   = rx[i] - nodeX[ix];
    eta[1]  = ry[i] - nodeY[iy];
    zeta[1] = rz[i] - nodeZ[iz];
    xi[0]   = nodeX[ix + 1] - rx[i];
    eta[0]  = nodeY[iy + 1] - ry[i];
    zeta[0] = nodeZ[iz + 1] - rz[i];

    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                weights[ii][jj][kk] = fabs(xi[ii] * eta[jj] * zeta[kk]) * invVOL;

    // Electric field at particle position
    double Ep[3] = {0.0, 0.0, 0.0};
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                int ixn = (ix + ii) % nx;
                int iyn = (iy + jj) % ny;
                int izn = (iz + kk) % nz;

                int index = ixn * ny * nz + iyn * nz + izn;
                Ep[0] += weights[ii][jj][kk] * Exn[index];
                Ep[1] += weights[ii][jj][kk] * Eyn[index];
                Ep[2] += weights[ii][jj][kk] * Ezn[index];
            }

    // Update particle velocity
    vx[i] += qom * Ep[0] * dt;
    vy[i] += qom * Ep[1] * dt;
    vz[i] += qom * Ep[2] * dt;
}