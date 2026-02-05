#ifndef INIT_HPP
#define INIT_HPP

#include <math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "parameters.hpp"
#include "structures.hpp"

// Kernel for initializing the particles with a maxwellian distribution
__global__ void maxwellianKernel(simu_particles* part, simu_grid* grid, simu_fields* EMf,
                                       int npcx, int npcy, int npcz,
                                       int nx, int ny, int nz,
                                       double dx, double dy, double dz,
                                       double qom, double u0, double v0, double w0,
                                       double uth0, double vth0, double wth0, int npc,
                                       double* harvest_array, double* theta_array) {
    // Global thread index
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    // Total number of particles
    int totalParticles = nx * ny * nz * npcx * npcy * npcz;

    if (threadId >= totalParticles) return;

    // Number of particles per cell
    int particlesPerCell = npcx * npcy * npcz;

    // Determine the cell and particle based on threadId
    int cellIndex = threadId / particlesPerCell;
    int particleInCell = threadId % particlesPerCell;
    
    // Convert the cell index to 3D coordinates
    int i = cellIndex / (ny * nz);
    int j = (cellIndex / nz) % ny;
    int k = cellIndex % nz; //fastest varying index

    // Convert the particle index to local coordinates
    int ii = particleInCell / (npcy * npcz);
    int jj = (particleInCell / npcz) % npcy;
    int kk = particleInCell % npcz;

    // Calculate the global particle counter
    int counter = threadId;

    // Calculate the index for the randomization arrays
    int random_index = counter * 2;

    // Position of the particle
    part->rx[counter] = (ii + 0.5) * (dx / npcx) + grid->nodeX[i];
    part->ry[counter] = (jj + 0.5) * (dy / npcy) + grid->nodeY[j];
    part->rz[counter] = (kk + 0.5) * (dz / npcz) + grid->nodeZ[k];

    // Charge
    part->q[counter] = (qom / fabs(qom)) * 
                       (fabs(EMf->rhos[cellIndex] / npc) * (dx * dy * dz));

    // Velocity vx and vy
    double harvest = harvest_array[random_index];
    double prob = sqrt(-2.0 * log(1.0 - 0.999999 * harvest));
    double theta = theta_array[random_index];
    part->vx[counter] = u0 + uth0 * prob * cos(theta);
    part->vy[counter] = v0 + vth0 * prob * sin(theta);

    // Velocity vz
    harvest = harvest_array[random_index + 1];
    prob = sqrt(-2.0 * log(1.0 - 0.999999 * harvest));
    theta = theta_array[random_index + 1];
    part->vz[counter] = w0 + wth0 * prob * cos(theta);

    // ID assignment
    part->ID[counter] = counter;
}

// Function to initialize the arrays with random numbers
void initRandomNumbers(double* harvest_array, double* theta_array, int is, int npc[], 
                      int npcx[], int npcy[], int npcz[], int nx, int ny, int nz){
 
  int random_index = 0;
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)     
      for (int k = 0; k < nz; k++)
        for (int ii = 0; ii < npcx[is]; ii++)
          for (int jj = 0; jj < npcy[is]; jj++)
            for (int kk = 0; kk < npcz[is]; kk++){
              harvest_array[random_index] = rand() / (double)RAND_MAX;
              theta_array[random_index] = 2.0 * M_PI * (rand() / (double)RAND_MAX);
              harvest_array[random_index + 1] = rand() / (double)RAND_MAX;
              theta_array[random_index + 1] = 2.0 * M_PI * (rand() / (double)RAND_MAX);
              random_index += 2;
            }       
}

// Function to initialize the particles with the two-streams distribution
__global__ void initPartTwostreamsKernel(simu_particles* part, simu_grid* grid, simu_fields* EMf,
                                         int npcx, int npcy, int npcz,
                                         int nx, int ny, int nz,
                                         double dx, double dy, double dz,
                                         double qom, double u0, double v0, double w0,
                                         double uth0, double vth0, double wth0, int npc,
                                         double* harvest_array, double* theta_array)
{
    // Global thread index
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    // Total number of particles
    int totalParticles = nx * ny * nz * npcx * npcy * npcz; 
    
    if (threadId >= totalParticles) return; 
    
    // Number of particles per cell
    int particlesPerCell = npcx * npcy * npcz;

    // Determine the cell and particle based on threadId
    int cellIndex = threadId / particlesPerCell;
    int particleInCell = threadId % particlesPerCell;

    // Convert the cell index to 3D coordinates
    int i = cellIndex / (ny * nz);
    int j = (cellIndex / nz) % ny;
    int k = cellIndex % nz;   
    
    // Convert the particle index to local coordinates
    int ii = particleInCell / (npcy * npcz);
    int jj = (particleInCell / npcz) % npcy;
    int kk = particleInCell % npcz;

    //  Calculate the global particle counter
    int counter = threadId;

    // Calculate the index for the randomization arrays
    int random_index = counter * 2;

    // Sign for the velocity
    double sign = (counter % 2 == 0) ? -1.0 : 1.0;

    // Position
    part->rx[counter] = (ii + 0.5) * (dx / npcx) + grid->nodeX[i];
    part->ry[counter] = (jj + 0.5) * (dy / npcy) + grid->nodeY[j];
    part->rz[counter] = (kk + 0.5) * (dz / npcz) + grid->nodeZ[k];

    // Charge
    part->q[counter] = (qom / fabs(qom)) *
                       (fabs(EMf->rhos[cellIndex] / npc) * (dx * dy * dz));

    // Vx velocity
    double harvest = harvest_array[random_index];
    double prob = sqrt(-2.0 * log(1.0 - 0.999999 * harvest));
    double theta = theta_array[random_index];
    part->vx[counter] = u0 * sign + uth0 * prob * cos(theta);

    // Vy velocity
    part->vy[counter] = v0 + vth0 * prob * sin(theta);

    // Vz velocity
    harvest = harvest_array[random_index + 1];
    prob = sqrt(-2.0 * log(1.0 - 0.999999 * harvest));
    theta = theta_array[random_index + 1];
    part->vz[counter] = w0 + wth0 * prob * cos(theta);

    // ID
    part->ID[counter] = counter;
}

// Function to initialize the grid
void initGrid(simu_grid * grid, double Lx, double Ly, double Lz, int nx, int ny, int nz)
{
  grid->minX = 0; grid->maxX = Lx;
  grid->minY = 0; grid->maxY = Ly;
  grid->minZ = 0; grid->maxZ = Lz;

  double dx = Lx / nx;
  double dy = Ly / ny;
  double dz = Lz / nz;

  for (int i = 0; i < nx+1; i++)
    grid->nodeX[i] = i * dx;

  for (int i = 0; i < ny+1; i++)
    grid->nodeY[i] = i * dy;
  
  for (int i = 0; i < nz+1; i++)
    grid->nodeZ[i] = i * dz;
}

// Initialize Magnetic and Electric Field with initial configuration for maxwellian distribution
void initEMfields(simu_fields * EMf, double B0x, double B0y, double B0z, double Amp, double qom[], double Lx, int ns, int nx, int ny, int nz, double dx)
{
  for (int i = 0; i < nx * ny * nz; i++)
  {
    for (int is = 0; is < ns; is++)
      EMf->rhos[is * nx * ny * nz + i] = (qom[is]/fabs(qom[is])) * (1 / (4 * M_PI));

    EMf->Bxn[i] = B0x;
    EMf->Byn[i] = B0y;
    EMf->Bzn[i] = B0z;
  }
 
  if (ny == 1 && nz == 1)
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            for (int k = 0; k < nz; k++)
            {
                EMf->Exn[i * nz * ny + j * nz + k] = Amp * (2 * M_PI / Lx) * cos(i * dx * 2 * M_PI / Lx);
                EMf->Eyn[i * nz * ny + j * nz + k] = 0.0;
                EMf->Ezn[i * nz * ny + j * nz + k] = 0.0;
            }
  else
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            for (int k = 0; k < nz; k++)
            {
                EMf->Exn[i * nz * ny + j * nz + k] = 0.0;
                EMf->Eyn[i * nz * ny + j * nz + k] = 0.0;
                EMf->Ezn[i * nz * ny + j * nz + k] = 0.0;
            }
}

// Initialize Magnetic and Electric Field with initial configuration for two-streams distribution
void initEMFieldsTwostreams(simu_fields * EMf, double B0x, double B0y, double B0z, double qom[], int ns, int nCells)
{
  for (int i = 0; i < nCells; i++)
  {
    for (int is = 0; is < ns; is++)
      EMf->rhos[is * nCells + i] = (qom[is]/fabs(qom[is])) * (1 / (4 * M_PI));

    EMf->Bxn[i] = B0x;
    EMf->Byn[i] = B0y;
    EMf->Bzn[i] = B0z;
    EMf->Exn[i] = 0.0;
    EMf->Eyn[i] = 0.0;
    EMf->Ezn[i] = 0.0;
  }
}

#endif
