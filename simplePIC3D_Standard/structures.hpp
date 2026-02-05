#ifndef STRUCTURES_HPP
#define STRUCTURES_HPP

#include<iostream>
using std::cout;

#include <new>
using std::nothrow;

/*! \brief Contains the simulation grid with additional informations
 */
struct simu_grid
{
  double* nodeX;            /*!< [nx+1]; Linearized x position of grid node. */
  double* nodeY;            /*!< [ny+1]; Linearized y position of grid node. */
  double* nodeZ;            /*!< [nz+1]; Linearized z position of grid node. */
  double minX; double maxX; /*!< min and MAX positions for particles along x. */
  double minY; double maxY; /*!< min and MAX positions for particles along y. */
  double minZ; double maxZ; /*!< min and MAX positions for particles along z. */
};

/*! \brief Contains all field values used in the solver
 */
struct simu_fields
{
  double* phi;       /*!<    [nx*ny*nz]; Linearized electric potential on nodes. */
  double* rho_tot;   /*!<    [nx*ny*nz]; Linearized charge density on nodes. */

  double* Exn;       /*!<    [nx*ny*nz]; Linearized electric field on nodes along x. */
  double* Eyn;       /*!<    [nx*ny*nz]; Linearized electric field on nodes along y. */
  double* Ezn;       /*!<    [nx*ny*nz]; Linearized electric field on nodes along z. */
                         
  double* Bxn;       /*!<    [nx*ny*nz]; Linearized magnetic field on nodes along x. */
  double* Byn;       /*!<    [nx*ny*nz]; Linearized magnetic field on nodes along y. */
  double* Bzn;       /*!<    [nx*ny*nz]; Linearized magnetic field on nodes along z. */
                         
  double* Bxc;       /*!<    [nx*ny*nz]; Linearized magnetic field on cells along x. */
  double* Byc;       /*!<    [nx*ny*nz]; Linearized magnetic field on cells along y. */
  double* Bzc;       /*!<    [nx*ny*nz]; Linearized magnetic field on cells along z. */
                         
  double* rhos;      /*!< [ns*nx*ny*nz]; Linearized charge density per species. */
                          
  double* Jxs;       /*!< [ns*nx*ny*nz]; Linearized current per species along x. */
  double* Jys;       /*!< [ns*nx*ny*nz]; Linearized current per species along y. */
  double* Jzs;       /*!< [ns*nx*ny*nz]; Linearized current per species along z. */
};


/*! \brief Contains all particles.
 *
 * This is (nominally) by far the largest data structure, containing active as well as
 * free particles, with the least amount of data for each particle: position,
 * velocity, and species, and whether this particle is free or not, i.e. active in the
 * domain or available for representing a new particle to be injected in the domain.
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

  int*  ID;   /*!< [np]; Particle ID. */
};

void allocateRandomArrays(double**& harvest_array, double**& theta_array, 
                          double*& d_harvest_array, double*& d_theta_array, 
                          int ns, int nx, int ny, int nz, 
                          int npcx[], int npcy[], int npcz[]) {

    
    // Memory allocation on the host (CPU) for the arrays
    harvest_array = new (nothrow) double*[ns];
    theta_array = new (nothrow) double*[ns];
    if (!harvest_array || !theta_array) {
        cout << "ERROR: Memory allocation failed for harvest_array or theta_array\n";
        exit(EXIT_FAILURE);
    }
    int size=0;
    for (int is = 0; is < ns; ++is) {
        size_t totalSize = 2 * npcx[is] * npcy[is] * npcz[is] * nx * ny * nz;
        size+=totalSize;
        harvest_array[is] = new (nothrow) double[totalSize];
        theta_array[is] = new (nothrow) double[totalSize];

        if (!harvest_array[is] || !theta_array[is]) {
            cout << "ERROR: Memory allocation failed for harvest_array[" << is << "] or theta_array[" << is << "]\n";
            exit(EXIT_FAILURE);
        }
    }

    // Memory allocation on the device (GPU) for the arrays
    cudaMalloc(&d_harvest_array, size * sizeof(double));
    cudaMalloc(&d_theta_array, size * sizeof(double));
    if (cudaGetLastError() != cudaSuccess) {
        cout << "ERROR: CUDA memory allocation failed for harvest_array or theta_array\n";
        exit(EXIT_FAILURE);
    }
}


void allocateGrid(simu_grid* * p_grid, double** d_nodeX, double** d_nodeY, double** d_nodeZ, int nx, int ny, int nz)
{
    // Memory allocation on the host (CPU) for the grid structure
    if ((*p_grid = new (nothrow) simu_grid) == nullptr) {cout << "ERROR: Memory allocation failed for grid\n"; exit(EXIT_FAILURE);}
    if (( (*p_grid)->nodeX = new (nothrow) double[nx + 1]) == nullptr) {cout << "ERROR: Memory allocation failed for grid.nodeX\n"; exit(EXIT_FAILURE);}
    if (( (*p_grid)->nodeY = new (nothrow) double[ny + 1]) == nullptr) {cout << "ERROR: Memory allocation failed for grid.nodeY\n"; exit(EXIT_FAILURE);}
    if (( (*p_grid)->nodeZ = new (nothrow) double[nz + 1]) == nullptr) {cout << "ERROR: Memory allocation failed for grid.nodeZ\n"; exit(EXIT_FAILURE);}  
    
    // Memory allocation on the device (GPU) for the corresponding arrays
    if (cudaMalloc(d_nodeX, (nx + 1) * sizeof(double)) != cudaSuccess) {
            std::cerr << "ERROR: Memory allocation failed for grid.nodeX\n";
            exit(EXIT_FAILURE);
        }
    if (cudaMalloc(d_nodeY, (ny + 1) * sizeof(double)) != cudaSuccess) {
        cout << "ERROR: CUDA memory allocation failed for d_nodeY\n";
        exit(EXIT_FAILURE);
    }
    if (cudaMalloc(d_nodeZ, (nz + 1) * sizeof(double)) != cudaSuccess) {
        cout << "ERROR: CUDA memory allocation failed for d_nodeZ\n";
        exit(EXIT_FAILURE);
    }
}

void allocateFields(simu_fields** p_fields, double** d_phi, double** d_rho_tot, double** d_rho, double** d_Exn, double** d_Eyn, double** d_Ezn, 
                    double** d_rhos, double** d_gradX, double** d_gradY, double** d_gradZ, int ns, int nCells, int nNodes){
    // Memory allocation on the host (CPU) for the fields structure
    if ((*p_fields = new (nothrow) simu_fields) == nullptr) {cout << "ERROR: Memory allocation failed for fields\n"; exit(EXIT_FAILURE);}
    if (( (*p_fields)->phi     = new (nothrow) double[nNodes]) == nullptr)    {cout << "ERROR: Memory allocation failed for fields.phi\n";     exit(EXIT_FAILURE);}
    if (( (*p_fields)->rho_tot = new (nothrow) double[nNodes]) == nullptr)    {cout << "ERROR: Memory allocation failed for fields.rho_tot\n"; exit(EXIT_FAILURE);}
    if (( (*p_fields)->Exn     = new (nothrow) double[nNodes]) == nullptr)    {cout << "ERROR: Memory allocation failed for fields.Exn\n";     exit(EXIT_FAILURE);}
    if (( (*p_fields)->Eyn     = new (nothrow) double[nNodes]) == nullptr)    {cout << "ERROR: Memory allocation failed for fields.Eyn\n";     exit(EXIT_FAILURE);}
    if (( (*p_fields)->Ezn     = new (nothrow) double[nNodes]) == nullptr)    {cout << "ERROR: Memory allocation failed for fields.Ezn\n";     exit(EXIT_FAILURE);}
    if (( (*p_fields)->Bxn     = new (nothrow) double[nNodes]) == nullptr)    {cout << "ERROR: Memory allocation failed for fields.Bxn\n";     exit(EXIT_FAILURE);}
    if (( (*p_fields)->Byn     = new (nothrow) double[nNodes]) == nullptr)    {cout << "ERROR: Memory allocation failed for fields.Byn\n";     exit(EXIT_FAILURE);}
    if (( (*p_fields)->Bzn     = new (nothrow) double[nNodes]) == nullptr)    {cout << "ERROR: Memory allocation failed for fields.Bzn\n";     exit(EXIT_FAILURE);}
    if (( (*p_fields)->Bxc     = new (nothrow) double[nCells]) == nullptr)    {cout << "ERROR: Memory allocation failed for fields.Bxc\n";     exit(EXIT_FAILURE);}
    if (( (*p_fields)->Byc     = new (nothrow) double[nCells]) == nullptr)    {cout << "ERROR: Memory allocation failed for fields.Byc\n";     exit(EXIT_FAILURE);}
    if (( (*p_fields)->Bzc     = new (nothrow) double[nCells]) == nullptr)    {cout << "ERROR: Memory allocation failed for fields.Bzc\n";     exit(EXIT_FAILURE);}
    if (( (*p_fields)->rhos    = new (nothrow) double[ns*nCells]) == nullptr) {cout << "ERROR: Memory allocation failed for fields.rhos\n";    exit(EXIT_FAILURE);}
    if (( (*p_fields)->Jxs     = new (nothrow) double[ns*nCells]) == nullptr) {cout << "ERROR: Memory allocation failed for fields.Jxs\n";     exit(EXIT_FAILURE);}
    if (( (*p_fields)->Jys     = new (nothrow) double[ns*nCells]) == nullptr) {cout << "ERROR: Memory allocation failed for fields.Jys\n";     exit(EXIT_FAILURE);}
    if (( (*p_fields)->Jzs     = new (nothrow) double[ns*nCells]) == nullptr) {cout << "ERROR: Memory allocation failed for fields.Jzs\n";     exit(EXIT_FAILURE);}
   
    // Memory allocation on the device (GPU) for the corresponding arrays
    cudaMalloc(d_phi,     nNodes * sizeof(double));
    cudaMalloc(d_rho_tot, nNodes * sizeof(double));
    cudaMalloc(d_Exn,     nNodes * sizeof(double));
    cudaMalloc(d_Eyn,     nNodes * sizeof(double));
    cudaMalloc(d_Ezn,     nNodes * sizeof(double));
    cudaMalloc(d_rhos,    ns * nCells * sizeof(double));
    cudaMalloc(d_gradX, sizeof(double) * nCells);
    cudaMalloc(d_gradY, sizeof(double) * nCells);
    cudaMalloc(d_gradZ, sizeof(double) * nCells);
    cudaMalloc(d_rho, sizeof(double) * nCells);
    
    // Check for errors
    if (cudaGetLastError() != cudaSuccess) {
        cout << "ERROR: CUDA memory allocation failed for one or more fields\n";
        exit(EXIT_FAILURE);
    }
}

void allocateParticles(simu_particles** * p_part, double** d_rx, double** d_ry, double** d_rz,
                       double** d_vx, double** d_vy, double** d_vz, double** d_q, int** d_ID,
                       int ns, int np[])
    {
    // Memory allocation on the host (CPU) for the particles structure
    if ((*p_part = new (nothrow) simu_particles*[ns]) == nullptr) {cout << "ERROR: Memory allocation failed for part\n"; exit(EXIT_FAILURE);}
    int totalParticles = 0;
    for (int is = 0; is < ns; is++) {
        totalParticles += np[is];
        if (( (*p_part)[is] = new (nothrow) simu_particles) == nullptr) {cout << "ERROR: Memory allocation failed for part[" << is << "]\n"; exit(EXIT_FAILURE);} 
        if (( (*p_part)[is]->rx = new (nothrow) double[np[is]]) == nullptr) {cout << "ERROR: Memory allocation failed for part[" << is << "].rx\n"; exit(EXIT_FAILURE);}
        if (( (*p_part)[is]->ry = new (nothrow) double[np[is]]) == nullptr) {cout << "ERROR: Memory allocation failed for part[" << is << "].ry\n"; exit(EXIT_FAILURE);}
        if (( (*p_part)[is]->rz = new (nothrow) double[np[is]]) == nullptr) {cout << "ERROR: Memory allocation failed for part[" << is << "].rz\n"; exit(EXIT_FAILURE);}
        if (( (*p_part)[is]->vx = new (nothrow) double[np[is]]) == nullptr) {cout << "ERROR: Memory allocation failed for part[" << is << "].vx\n"; exit(EXIT_FAILURE);}
        if (( (*p_part)[is]->vy = new (nothrow) double[np[is]]) == nullptr) {cout << "ERROR: Memory allocation failed for part[" << is << "].vy\n"; exit(EXIT_FAILURE);}
        if (( (*p_part)[is]->vz = new (nothrow) double[np[is]]) == nullptr) {cout << "ERROR: Memory allocation failed for part[" << is << "].vz\n"; exit(EXIT_FAILURE);}
        if (( (*p_part)[is]->q  = new (nothrow) double[np[is]]) == nullptr) {cout << "ERROR: Memory allocation failed for part[" << is << "].q\n" ; exit(EXIT_FAILURE);}
        if (( (*p_part)[is]->ID = new (nothrow)    int[np[is]]) == nullptr) {cout << "ERROR: Memory allocation failed for part[" << is << "].ID\n"; exit(EXIT_FAILURE);}
    }

    // Memory allocation on the device (GPU) for the corresponding arrays
    cudaMalloc(d_rx, totalParticles * sizeof(double));
    cudaMalloc(d_ry, totalParticles * sizeof(double));
    cudaMalloc(d_rz, totalParticles * sizeof(double));
    cudaMalloc(d_vx, totalParticles * sizeof(double));
    cudaMalloc(d_vy, totalParticles * sizeof(double));
    cudaMalloc(d_vz, totalParticles * sizeof(double));
    cudaMalloc(d_q,  totalParticles * sizeof(double));
    cudaMalloc(d_ID, totalParticles * sizeof(int));

    // Check for errors
    if (cudaGetLastError() != cudaSuccess) {
        cout << "ERROR: CUDA memory allocation failed for one or more particle arrays\n";
        exit(EXIT_FAILURE);
    }
}

void freeGrid(simu_grid** p_grid, double** d_nodeX, double** d_nodeY, double** d_nodeZ)
{
    // Free memory on the host (CPU)
    delete[] (*p_grid)->nodeX; (*p_grid)->nodeX = nullptr;
    delete[] (*p_grid)->nodeY; (*p_grid)->nodeY = nullptr;
    delete[] (*p_grid)->nodeZ; (*p_grid)->nodeZ = nullptr;

    // Free memory on the device (GPU)
    cudaFree(d_nodeX);
    cudaFree(d_nodeY);
    cudaFree(d_nodeZ);
    delete *p_grid;
}

void freeFields(simu_fields** p_fields, double** d_phi, double** d_rho_tot, double** d_rho, double** d_gradX, double** d_gradY, double** d_gradZ,
    double** d_Exn, double** d_Eyn, double** d_Ezn, double** d_rhos, double** d_phi_poisson, double** d_phi_gradient)
{
    // Free memory on the host (CPU)
    delete[] (*p_fields)->phi; (*p_fields)->phi = nullptr;
    delete[] (*p_fields)->rho_tot; (*p_fields)->rho_tot = nullptr;
    delete[] (*p_fields)->Exn; (*p_fields)->Exn = nullptr;
    delete[] (*p_fields)->Eyn; (*p_fields)->Eyn = nullptr;
    delete[] (*p_fields)->Ezn; (*p_fields)->Ezn = nullptr;
    delete[] (*p_fields)->Bxn; (*p_fields)->Bxn = nullptr;
    delete[] (*p_fields)->Byn; (*p_fields)->Byn = nullptr;
    delete[] (*p_fields)->Bzn; (*p_fields)->Bzn = nullptr;
    delete[] (*p_fields)->Bxc; (*p_fields)->Bxc = nullptr;
    delete[] (*p_fields)->Byc; (*p_fields)->Byc = nullptr;
    delete[] (*p_fields)->Bzc; (*p_fields)->Bzc = nullptr;
    delete[] (*p_fields)->rhos; (*p_fields)->rhos = nullptr;
    delete[] (*p_fields)->Jxs; (*p_fields)->Jxs = nullptr;
    delete[] (*p_fields)->Jys; (*p_fields)->Jys = nullptr;
    delete[] (*p_fields)->Jzs; (*p_fields)->Jzs = nullptr;
    delete *p_fields;

    // Free memory on the device (GPU)
    cudaFree(d_phi); 
    cudaFree(d_rho_tot);
    cudaFree(d_Exn); 
    cudaFree(d_Eyn);
    cudaFree(d_Ezn);
    cudaFree(d_rhos);
    cudaFree(d_gradX);
    cudaFree(d_gradY);
    cudaFree(d_gradZ);
    cudaFree(d_rho);
    cudaFree(d_phi_poisson);
    cudaFree(d_phi_gradient);
}


void freeParticles(simu_particles** *p_part, int ns, double** d_rx, double** d_ry, double** d_rz, 
                   double** d_vx, double** d_vy, double** d_vz, double** d_q, int** d_ID)
{
    // Free memory on the host (CPU)
    for (int is = 0; is < ns; is++) {
        delete[] (*p_part)[is]->rx; (*p_part)[is]->rx = nullptr;
        delete[] (*p_part)[is]->ry; (*p_part)[is]->ry = nullptr;
        delete[] (*p_part)[is]->rz; (*p_part)[is]->rz = nullptr;
        delete[] (*p_part)[is]->vx; (*p_part)[is]->vx = nullptr;
        delete[] (*p_part)[is]->vy; (*p_part)[is]->vy = nullptr;
        delete[] (*p_part)[is]->vz; (*p_part)[is]->vz = nullptr;
        delete[] (*p_part)[is]->q;  (*p_part)[is]->q  = nullptr;
        delete[] (*p_part)[is]->ID; (*p_part)[is]->ID = nullptr;
    }
    delete[] *p_part;
    *p_part = nullptr;

    // Free memory on the device (GPU)
    cudaFree(d_rx);
    cudaFree(d_ry);
    cudaFree(d_rz);
    cudaFree(d_vx);
    cudaFree(d_vy);
    cudaFree(d_vz);
    cudaFree(d_q);
    cudaFree(d_ID);
}

void freeRandomArrays(double** harvest_array, double** theta_array, double* d_harvest_array, double* d_theta_array, int ns)
{
    // Free memory on the host (CPU)
    for (int is = 0; is < ns; ++is) {
        delete[] harvest_array[is];
        delete[] theta_array[is];
    }
    delete[] harvest_array;
    delete[] theta_array;

    // Free memory on the device (GPU)
    cudaFree(d_harvest_array);
    cudaFree(d_theta_array);
}


#endif