#pragma once

#include "parameters.hpp"
#include "structures.hpp"

// Kernel for updating the particle position
__global__ void updateParticlePositionKernel(double* rx, double* ry, double* rz,
                                             double* vx, double* vy, double* vz,
                                             const int np, double Lx, double Ly, double Lz, double dt)
{
    // Global index
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx < np)
    {
        // Update the position
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

        // Check for particles at the boundary
        if (rx[idx] == Lx) rx[idx] = 0;
        if (ry[idx] == Ly) ry[idx] = 0;
        if (rz[idx] == Lz) rz[idx] = 0;
    }
}

__device__ void calculateWeightsDevice(double weights[2][2][2], double x, double y, double z,
                                     double q, int ix, int iy, int iz,
                                     simu_grid* grid, double invVOL) {
                                
    double xi[2], eta[2], zeta[2];
    
    // Calculate distances for weight computation
    xi[0]   = (grid->nodeX[ix] - x);
    eta[0]  = (grid->nodeY[iy] - y); 
    zeta[0] = (grid->nodeZ[iz] - z);
    xi[1]   = (x - grid->nodeX[ix]);
    eta[1]  = (y - grid->nodeY[iy]);
    zeta[1] = (z - grid->nodeZ[iz]);
    
    // Calculate weights
    for (int i=0; i < 2; i++)
        for (int j=0; j < 2; j++)
            for(int k=0; k < 2; k++)
                weights[i][j][k] = q * xi[i] * eta[j] * zeta[k] * invVOL;
}


// Kernel for updating the grid with particle contributions
__global__ void updateParticlePositionKernel(simu_particles** part, const unsigned int is, int np[], double Lx, double Ly, double Lz, double dt)
{
    // Global index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < np[is])
    {
        // Update the position
        part[is]->rx[i] += part[is]->vx[i] * dt;
        part[is]->ry[i] += part[is]->vy[i] * dt;
        part[is]->rz[i] += part[is]->vz[i] * dt;

        // Periodic boundary conditions
        if (part[is]->rx[i] >= Lx) part[is]->rx[i] = (part[is]->rx[i]/Lx - int(     part[is]->rx[i]/Lx)) * Lx;
        if (part[is]->rx[i] <  0)  part[is]->rx[i] = (part[is]->rx[i]/Lx + int(fabs(part[is]->rx[i]/Lx)) + 1) * Lx;
        if (part[is]->ry[i] >= Ly) part[is]->ry[i] = (part[is]->ry[i]/Ly - int(     part[is]->ry[i]/Ly)) * Ly;
        if (part[is]->ry[i] <  0)  part[is]->ry[i] = (part[is]->ry[i]/Ly + int(fabs(part[is]->ry[i]/Ly)) + 1) * Ly;
        if (part[is]->rz[i] >= Lz) part[is]->rz[i] = (part[is]->rz[i]/Lz - int(     part[is]->rz[i]/Lz)) * Lz;
        if (part[is]->rz[i] <  0)  part[is]->rz[i] = (part[is]->rz[i]/Lz + int(fabs(part[is]->rz[i]/Lz)) + 1) * Lz;

        // Check for particles at the boundary
        if (part[is]->rx[i] == Lx) part[is]->rx[i] = 0;
        if (part[is]->ry[i] == Ly) part[is]->ry[i] = 0;
        if (part[is]->rz[i] == Lz) part[is]->rz[i] = 0;
    }
}

// GPU kernel for particle-to-grid charge density mapping with atomic-free coarsening
// This kernel processes multiple particles per thread to reduce the number of atomic operations needed, which is a form of coarsening. 
// Coarsening assigns a batch of particles to each thread instead of a single particle.
__global__ void particles2GridAtomicFreeCoarsening(
    simu_fields* fields, simu_grid* grid, simu_particles** part,
    int nx, int ny, int nz, double dx, double dy, double dz, double invVOL,
    const int is, int np, int particlesNum) {

    // Global index
    int tid = (blockIdx.x * blockDim.x + threadIdx.x) * particlesNum;

    // Processo multiple particelle per thread
    for (int p = tid; p < min(tid + particlesNum, np); p++) {
        int ix = int(part[is]->rx[p] / dx);
        int iy = int(part[is]->ry[p] / dy);
        int iz = int(part[is]->rz[p] / dz);

        // Distance calculation
        double xi[2], eta[2], zeta[2];
        xi[1]   = part[is]->rx[p] - grid->nodeX[ix];
        eta[1]  = part[is]->ry[p] - grid->nodeY[iy];
        zeta[1] = part[is]->rz[p] - grid->nodeZ[iz];

        xi[0]   = grid->nodeX[ix + 1] - part[is]->rx[p];
        eta[0]  = grid->nodeY[iy + 1] - part[is]->ry[p];
        zeta[0] = grid->nodeZ[iz + 1] - part[is]->rz[p];

        // Weight calculation
        double qp = part[is]->q[p];
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 2; k++) {
                    double weight = xi[i] * eta[j] * zeta[k] * invVOL * qp * invVOL;
                    int ixn = (ix + i) % nx;
                    int iyn = (iy + j) % ny;
                    int izn = (iz + k) % nz;
                    int index = is * nx * ny * nz + ixn * ny * nz + iyn * nz + izn;
                    atomicAdd(&fields->rhos[index], weight);
                }
    }
}


// Kernel for assigning particles to the grid
__global__ void particles2GridKernel(simu_fields* fields,
                                 simu_grid* grid, int nx, int ny, int nz, 
                                 double dx, double dy, double dz, double invVOL,
                                 simu_particles** part, const int is, int np) {
    // Global index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= np) return;

    // Indices of the cell to which the particle belongs
    int ix = int(part[is]->rx[tid] / dx);
    int iy = int(part[is]->ry[tid] / dy);
    int iz = int(part[is]->rz[tid] / dz);

    // Weight calculation inline
    double xi[2], eta[2], zeta[2];
    double weights[2][2][2];

    // Calculate distances for weight computation
    xi[1]   = part[is]->rx[tid] - grid->nodeX[ix];
    eta[1]  = part[is]->ry[tid] - grid->nodeY[iy];
    zeta[1] = part[is]->rz[tid] - grid->nodeZ[iz];

    xi[0]   = grid->nodeX[ix+1] - part[is]->rx[tid];
    eta[0]  = grid->nodeY[iy+1] - part[is]->ry[tid];
    zeta[0] = grid->nodeZ[iz+1] - part[is]->rz[tid];

    // Calculate weights
    double qp = part[is]->q[tid];
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

                int index = is*nx*ny*nz + (ixn + i)*ny*nz + (iyn + j)*nz + (izn + k);
                atomicAdd(&fields->rhos[index], weights[i][j][k] * invVOL);
            }
        }
    }
}

// Kernel for updating the particle velocity
__global__ void updateParticleVelocityKernel(double* rx, double* ry, double* rz,
                                  double* vx, double* vy, double* vz,
                                  double* Exn, double* Eyn, double* Ezn,
                                  double* nodeX, double* nodeY, double* nodeZ,
                                  int nx, int ny, int nz, 
                                  double dx, double dy, double dz, double invVOL,
                                  double qom, double dt, int np) {
    // Global index
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

    // Calculate the electric field at the particle position
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

    // Update the particle velocity
    vx[i] += qom * Ep[0] * dt;
    vy[i] += qom * Ep[1] * dt;
    vz[i] += qom * Ep[2] * dt;
}

