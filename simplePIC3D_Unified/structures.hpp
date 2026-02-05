#ifndef STRUCTURES_HPP
#define STRUCTURES_HPP

#include <new>
using std::nothrow;
#include <iostream>
#include <cuda_runtime.h>
using std::cout;

/*! \brief Contains the simulation grid with additional informations
 */
struct simu_grid
{
  double* nodeX;            /*!< [nx+1]; Linearized x position of grid node. */
  double* nodeY;            /*!< [ny+1]; Linearized y position of grid node. */
  double* nodeZ;            /*!< [nz+1]; Linearized z position of grid node. */
  double minX, maxX;        /*!< min and MAX positions for particles along x. */
  double minY, maxY;        /*!< min and MAX positions for particles along y. */
  double minZ, maxZ;        /*!< min and MAX positions for particles along z. */
};

/*! \brief Contains all field values used in the solver
 */
struct simu_fields
{
  double* phi;       /*!< [nx*ny*nz]; Linearized electric potential on nodes. */
  double* rho_tot;   /*!< [nx*ny*nz]; Linearized charge density on nodes. */

  double* Exn;       /*!< [nx*ny*nz]; Linearized electric field on nodes along x. */
  double* Eyn;       /*!< [nx*ny*nz]; Linearized electric field on nodes along y. */
  double* Ezn;       /*!< [nx*ny*nz]; Linearized electric field on nodes along z. */
                         
  double* Bxn;       /*!< [nx*ny*nz]; Linearized magnetic field on nodes along x. */
  double* Byn;       /*!< [nx*ny*nz]; Linearized magnetic field on nodes along y. */
  double* Bzn;       /*!< [nx*ny*nz]; Linearized magnetic field on nodes along z. */
                         
  double* Bxc;       /*!< [nx*ny*nz]; Linearized magnetic field on cells along x. */
  double* Byc;       /*!< [nx*ny*nz]; Linearized magnetic field on cells along y. */
  double* Bzc;       /*!< [nx*ny*nz]; Linearized magnetic field on cells along z. */
                         
  double* rhos;      /*!< [ns*nx*ny*nz]; Linearized charge density per species. */
                          
  double* Jxs;       /*!< [ns*nx*ny*nz]; Linearized current per species along x. */
  double* Jys;       /*!< [ns*nx*ny*nz]; Linearized current per species along y. */
  double* Jzs;       /*!< [ns*nx*ny*nz]; Linearized current per species along z. */
};

/*! \brief Contains all particles.
 */
struct simu_particles
{
  double* rx;  /*!< [np]; x position. */
  double* ry;  /*!< [np]; y position. */
  double* rz;  /*!< [np]; z position. */

  double* vx;  /*!< [np]; Speed along x. */
  double* vy;  /*!< [np]; Speed along y. */
  double* vz;  /*!< [np]; Speed along z. */

  double* q;   /*!< [np]; Charge. */

  int* ID;     /*!< [np]; Particle ID. */
};

// Unified Memory Allocations

void allocateGridUnified(simu_grid** p_grid, int nx, int ny, int nz)
{
  cudaError_t err;
  // Allocate grid
  cudaMallocManaged(p_grid, sizeof(simu_grid));
  cudaMallocManaged(&(*p_grid)->nodeX, (nx + 1) * sizeof(double));
  cudaMallocManaged(&(*p_grid)->nodeY, (ny + 1) * sizeof(double));
  cudaMallocManaged(&(*p_grid)->nodeZ, (nz + 1) * sizeof(double));
  cudaMallocManaged(&(*p_grid)->nodeX, (nx + 1) * sizeof(double));
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "ERROR: cudaMallocManaged failed for grid.nodeX: " 
              << cudaGetErrorString(err) << "\n";
    exit(EXIT_FAILURE);
  }
}

void allocateFieldsUnified(simu_fields** p_fields, int ns, int nCells, int nNodes)
{
  // Allocate fields
  cudaMallocManaged(p_fields, sizeof(simu_fields));
  cudaMallocManaged(&(*p_fields)->phi, nNodes * sizeof(double));
  cudaMallocManaged(&(*p_fields)->rho_tot, nNodes * sizeof(double));
  cudaMallocManaged(&(*p_fields)->Exn, nNodes * sizeof(double));
  cudaMallocManaged(&(*p_fields)->Eyn, nNodes * sizeof(double));
  cudaMallocManaged(&(*p_fields)->Ezn, nNodes * sizeof(double));
  cudaMallocManaged(&(*p_fields)->Bxn, nNodes * sizeof(double));
  cudaMallocManaged(&(*p_fields)->Byn, nNodes * sizeof(double));
  cudaMallocManaged(&(*p_fields)->Bzn, nNodes * sizeof(double));
  cudaMallocManaged(&(*p_fields)->Bxc, nCells * sizeof(double));
  cudaMallocManaged(&(*p_fields)->Byc, nCells * sizeof(double));
  cudaMallocManaged(&(*p_fields)->Bzc, nCells * sizeof(double));
  cudaMallocManaged(&(*p_fields)->rhos, ns * nCells * sizeof(double));
  cudaMallocManaged(&(*p_fields)->Jxs, ns * nCells * sizeof(double));
  cudaMallocManaged(&(*p_fields)->Jys, ns * nCells * sizeof(double));
  cudaMallocManaged(&(*p_fields)->Jzs, ns * nCells * sizeof(double));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "ERROR: cudaMallocManaged failed for fields: " 
              << cudaGetErrorString(err) << "\n";
    exit(EXIT_FAILURE);
  }
}

void allocateParticlesUnified(simu_particles*** p_part, int ns, int np[])
{
  // Allocate particles
  cudaMallocManaged(p_part, ns * sizeof(simu_particles*));
  for (int is = 0; is < ns; ++is) {
    cudaMallocManaged(&(*p_part)[is], sizeof(simu_particles));
    cudaMallocManaged(&(*p_part)[is]->rx, np[is] * sizeof(double));
    cudaMallocManaged(&(*p_part)[is]->ry, np[is] * sizeof(double));
    cudaMallocManaged(&(*p_part)[is]->rz, np[is] * sizeof(double));
    cudaMallocManaged(&(*p_part)[is]->vx, np[is] * sizeof(double));
    cudaMallocManaged(&(*p_part)[is]->vy, np[is] * sizeof(double));
    cudaMallocManaged(&(*p_part)[is]->vz, np[is] * sizeof(double));
    cudaMallocManaged(&(*p_part)[is]->q, np[is] * sizeof(double));
    cudaMallocManaged(&(*p_part)[is]->ID, np[is] * sizeof(int));
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "ERROR: cudaMallocManaged failed for parts: " 
              << cudaGetErrorString(err) << "\n";
    exit(EXIT_FAILURE);
  }
}

// Unified Memory Free

void freeGridUnified(simu_grid** p_grid)
{
  // Free grid
  cudaFree((*p_grid)->nodeX);
  cudaFree((*p_grid)->nodeY);
  cudaFree((*p_grid)->nodeZ);
  cudaFree(*p_grid);
}

void freeFieldsUnified(simu_fields** p_fields)
{
  // Free fields
  cudaFree((*p_fields)->phi);
  cudaFree((*p_fields)->rho_tot);
  cudaFree((*p_fields)->Exn);
  cudaFree((*p_fields)->Eyn);
  cudaFree((*p_fields)->Ezn);
  cudaFree((*p_fields)->Bxn);
  cudaFree((*p_fields)->Byn);
  cudaFree((*p_fields)->Bzn);
  cudaFree((*p_fields)->Bxc);
  cudaFree((*p_fields)->Byc);
  cudaFree((*p_fields)->Bzc);
  cudaFree((*p_fields)->rhos);
  cudaFree((*p_fields)->Jxs);
  cudaFree((*p_fields)->Jys);
  cudaFree((*p_fields)->Jzs);
  cudaFree(*p_fields);
}

void freeParticlesUnified(simu_particles*** p_part, int ns)
{

  // Free particles
  for (int is = 0; is < ns; ++is) {
    cudaFree((*p_part)[is]->rx);
    cudaFree((*p_part)[is]->ry);
    cudaFree((*p_part)[is]->rz);
    cudaFree((*p_part)[is]->vx);
    cudaFree((*p_part)[is]->vy);
    cudaFree((*p_part)[is]->vz);
    cudaFree((*p_part)[is]->q);
    cudaFree((*p_part)[is]->ID);
    cudaFree((*p_part)[is]);
  }
  cudaFree(*p_part);
}


void allocateRandomArrays(double**& harvest_array, double**& theta_array, int ns, int nx, int ny, int nz, int npcx[], int npcy[], int npcz[]){
    cudaMallocManaged(&harvest_array, ns * sizeof(double*));
    cudaMallocManaged(&theta_array, ns * sizeof(double*));

    // Allocate memory for each species
    for (int is = 0; is < ns; ++is) {
        size_t totalSize = 2 * npcx[is] * npcy[is] * npcz[is] * nx * ny * nz;
        cudaMallocManaged(&harvest_array[is], totalSize * sizeof(double));
        cudaMallocManaged(&theta_array[is], totalSize * sizeof(double));
    }
}

// Free Random Arrays
void freeRandomArrays(double** harvest_array, double** theta_array, const Parameters& p){
    for (int is = 0; is < p.ns; ++is) {
        cudaFree(harvest_array[is]);
        cudaFree(theta_array[is]);
    }
    cudaFree(harvest_array);
    cudaFree(theta_array);
}

#endif