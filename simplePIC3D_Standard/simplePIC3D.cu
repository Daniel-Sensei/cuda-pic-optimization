#include <cassert>
#include <cstdlib>
//#include <filesystem>

#include "fields.hpp"
#include "init.hpp"
#include "io.hpp"
#include "parameters.hpp"
#include "particles.hpp"
#include "structures.hpp"
#include "util.hpp"

int main()
{
  // loading parameters from the config.cfg file
  Parameters p;
  loadParameters("config.cfg", &p);
  printParameters(&p);

  // declaring structures of arrays for grid, fields and particles    
  simu_grid* grid;
  simu_fields* fields;
  simu_particles** part; // ~ part[ns];
  
  // Declare device (GPU) memory pointers
  // These will hold copies of host memory for CUDA kernel operations
  double* d_nodeX, * d_nodeY, * d_nodeZ;
  double* d_Exn, * d_Eyn, * d_Ezn;
  double* d_rho_tot, * d_rhos, * d_rho;
  double* d_phi;
  double* d_rx, * d_ry, * d_rz;
  double* d_vx, * d_vy, * d_vz;
  double* d_q;
  int* d_ID;
  double* d_phi_poisson,* d_phi_gradient;
  double* d_gradX, * d_gradY,* d_gradZ;

  // Random number generation arrays for particle initialization
  double **harvest_array, **theta_array;
  double* d_harvest_array, * d_theta_array;

  // Memory allocation
  allocateGrid(&grid, &d_nodeX, &d_nodeY, &d_nodeZ, p.nx, p.ny, p.nz);
  allocateFields(&fields, &d_phi, &d_rho_tot, &d_rho, &d_Exn, &d_Eyn, &d_Ezn, &d_rhos, &d_gradX, &d_gradY, &d_gradZ, p.ns, p.nCells, p.nNodes);
  allocateParticles(&part, &d_rx, &d_ry, &d_rz, &d_vx, &d_vy, &d_vz, &d_q, &d_ID, p.ns, p.np);
  allocateRandomArrays(harvest_array, theta_array, d_harvest_array, d_theta_array, p.ns, p.nx, p.ny, p.nz, p.npcx, p.npcy, p.npcz);
  
  // Initialization
  printf("Initializing simulation\n");
  initGrid(grid, p.Lx, p.Ly, p.Lz, p.nx, p.ny, p.nz);

  // Configure CUDA grid and block sizes for parallel computation
  int totalCells = p.nx * p.ny * p.nz; 
  int blockSize=128;
  std::cout << "Block total size: " << blockSize << std::endl;
  dim3 gridSize = ((totalCells + blockSize - 1) / blockSize);

  cudaMemcpy(d_nodeX, grid->nodeX, (p.nx+1) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_nodeY, grid->nodeY, (p.ny+1) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_nodeZ, grid->nodeZ, (p.nz+1) * sizeof(double), cudaMemcpyHostToDevice);
  

  for(int is=0; is<p.ns; is++){
    int offsetParticles = is * p.np[is];
    int offsetFields = 2 * p.npcx[is] * p.npcy[is] * p.npcz[is] * p.nx * p.ny * p.nz;
    
    cudaMemcpy(d_q + offsetParticles, part[is]->q, p.np[is] * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx + offsetParticles, part[is]->vx, p.np[is]  * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy + offsetParticles, part[is]->vy, p.np[is]  * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz + offsetParticles, part[is]->vz, p.np[is]  * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rx + offsetParticles, part[is]->rx, p.np[is] * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ry + offsetParticles, part[is]->ry, p.np[is] * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rz + offsetParticles, part[is]->rz, p.np[is] * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ID + offsetParticles, part[is]->ID, p.np[is] * sizeof(int), cudaMemcpyHostToDevice);
    
    // Generating random number arrays
    initRandomNumbers(harvest_array[is], theta_array[is], is, p.npc, p.npcx, p.npcy, p.npcz, p.nx, p.ny, p.nz);
      
    // Copying data to device memory
    cudaMemcpy(d_harvest_array + offsetFields*is, harvest_array[is], offsetFields * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_theta_array + offsetFields*is, theta_array[is], offsetFields * sizeof(double), cudaMemcpyHostToDevice);   
  }
 
  // Initialize simulation based on configuration type
  if (string(p.init_case) == "twostreams") 
  {  
    // Two-stream instability initialization
    // Sets up electromagnetic fields and initializes particles with specific velocity distributions
    initEMFieldsTwostreams(fields, p.B0x, p.B0y, p.B0z, p.qom, p.ns, p.nCells);
    printf("Initializing particles for two stream instability\n");
    
    // Copying initial data to device memory
    cudaMemcpy(d_rhos, fields->rhos, p.ns * p.nCells * sizeof(double), cudaMemcpyHostToDevice);

    // loop over species
    for (int is = 0; is < p.ns; is++){

      // Total number of cells
      int totalCells = p.nx * p.ny * p.nz * p.npcx[is] * p.npcy[is] * p.npcx[is];

      // Offset for particle arrays
      int offsetParticles = is * p.np[is];

      // Offset for random number arrays
      int offsetFields = 2 * p.npcx[is] * p.npcy[is] * p.npcz[is] * p.nx * p.ny * p.nz;

      // Grid and block sizes for parallel computation
      dim3 gridSize = ((totalCells + blockSize - 1) / blockSize);    
      
      // Initialize particles following two-stream instability distribution
      initPartTwostreamsKernel<<<gridSize, blockSize>>>(d_rx + offsetParticles,
        d_ry + offsetParticles, d_rz + offsetParticles, d_q + offsetParticles, d_vx + offsetParticles,
        d_vy + offsetParticles, d_vz + offsetParticles, d_ID + offsetParticles, d_nodeX, d_nodeY, d_nodeZ,
        d_rhos + is * p.nCells, p.npcx[is], p.npcy[is], p.npcz[is], p.nx, p.ny, p.nz, p.dx, p.dy, p.dz,
        p.qom[is], p.u0[is], p.v0[is], p.w0[is], p.uth0[is], p.vth0[is], p.wth0[is], p.npc[is],
        d_harvest_array + offsetFields*is, d_theta_array + offsetFields*is
      );
  }
  }else if (string(p.init_case) == "randomInit")
  {
    // Random initialization of particles following Maxwellian distribution
    initEMfields(fields, p.B0x, p.B0y, p.B0z, p.Amp, p.qom, p.Lx, p.ns, p.nx, p.ny, p.nz, p.dx);
    cudaMemcpy(d_rhos, fields->rhos, p.ns * p.nCells * sizeof(double), cudaMemcpyHostToDevice);
   
    for (int is = 0; is < p.ns; is++)
    {
      // Total number of cells
      int totalCells = p.nx * p.ny * p.nz * p.npcx[is] * p.npcy[is] * p.npcx[is];

      // Offset for particle arrays
      int offsetParticles = is * p.np[is];
      // Offset for random number arrays
      int offsetFields = 2 * p.npcx[is] * p.npcy[is] * p.npcz[is] * p.nx * p.ny * p.nz;
      
      // Grid and block sizes for parallel computation
      dim3 gridSize = ((totalCells + blockSize - 1) / blockSize);   
      
      // Initialize particles following Maxwellian distribution
      maxwellianKernel<<<gridSize, blockSize>>>(d_rx + offsetParticles,
        d_ry + offsetParticles, d_rz + offsetParticles, d_q + offsetParticles, d_vx + offsetParticles,
        d_vy + offsetParticles, d_vz + offsetParticles, d_ID + offsetParticles, d_nodeX, d_nodeY, d_nodeZ,
                                                    d_rhos, p.npcx[is], p.npcy[is], p.npcz[is],  p.nx, p.ny, p.nz, p.dx, p.dy, p.dz,
      p.qom[is], p.u0[is], p.v0[is], p.w0[is], p.uth0[is], p.vth0[is], p.wth0[is], p.npc[is], d_harvest_array+ offsetFields*is, d_theta_array+ offsetFields*is);
    }
  }
  else
  {
    printf("Unable to find a valid configuration. Aborting...\n");
    exit(EXIT_FAILURE);
  }

  cudaMemcpy(fields->rho_tot, d_rho_tot, p.nCells * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(fields->rhos, d_rhos, p.ns * p.nCells * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(fields->Exn, d_Exn, p.nCells * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(fields->Eyn, d_Eyn, p.nCells * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(fields->Ezn, d_Ezn, p.nCells * sizeof(double), cudaMemcpyDeviceToHost);
  for(int is=0; is<p.ns; is++){
    int offsetParticles = is * p.np[is];
    cudaMemcpy(part[is]->rx, d_rx + offsetParticles, p.np[is]  * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(part[is]->ry, d_ry + offsetParticles, p.np[is] * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(part[is]->rz, d_rz + offsetParticles, p.np[is] * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(part[is]->vx, d_vx + offsetParticles, p.np[is] * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(part[is]->vy, d_vy + offsetParticles, p.np[is] * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(part[is]->vz, d_vz + offsetParticles, p.np[is] * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(part[is]->q, d_q + offsetParticles, p.np[is] * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(part[is]->ID, d_ID + offsetParticles, p.np[is] * sizeof(int), cudaMemcpyDeviceToHost);
  }
  // output
  string out_dir_path = "out/";
  system("mkdir -p out");
  saveGlobalQuantities(fields, part, p.ns, p.np, p.qom, p.nCells, p.dx, p.dy, p.dz, 0, out_dir_path.c_str());
  saveFields(fields, 0, p.ns, p.nCells, out_dir_path.c_str()); 
  cudaMalloc(&d_phi_poisson, sizeof(double) * p.nCells);
  cudaMalloc(&d_phi_gradient, sizeof(double) * p.nCells);
 
  // main loop
  util::Timer clTimer;
  for (int it = 1; it <= p.nsteps; it++)
  {   
    printf("\rRunning simulation - step %d", it); fflush(stdout);
   
    // Reset total charge density
    cudaMemset(d_rhos, 0, sizeof(double) * p.ns * p.nCells);
    
    for (int is = 0; is < p.ns; is++) 
    {
      // Block size for parallel computation
      int gridSize = (p.np[is] + blockSize - 1) / blockSize;
      int offsetParticles = is * p.np[is];
      
      // Update particle positions
      updateParticlePositionKernel<<<gridSize, blockSize>>>(d_rx + offsetParticles,
        d_ry + offsetParticles, d_rz + offsetParticles, d_vx + offsetParticles,
        d_vy + offsetParticles, d_vz + offsetParticles, p.np[is], p.Lx, p.Ly, p.Lz, p.dt);
  
      /*// Update particle positions in host memory
      particles2GridKernel<<<gridSize, blockSize>>>(d_rhos, d_nodeX, d_nodeY, d_nodeZ,
          d_rx+offsetParticles, d_ry+offsetParticles, d_rz+offsetParticles, d_q+offsetParticles,  
          p.nx, p.ny, p.nz, 
          p.dx, p.dy, p.dz, p.invVOL,
          is, p.np[is]);*/

     /* //AtomicFreePrivatization version
      particles2GridAtomicFreePrivatization<<<gridSize, blockSize>>>(d_rhos, d_nodeX, d_nodeY, d_nodeZ,
          d_rx+offsetParticles, d_ry+offsetParticles, d_rz+offsetParticles, d_q+offsetParticles,  
          p.nx, p.ny, p.nz, 
          p.dx, p.dy, p.dz, p.invVOL,
          is, p.np[is]);*/

       //AtomicFreeCoarsening version
       
      int particlesPerThread = 32; //Best configuration among all the tested numbers
      int numThreads = (p.np[is] + particlesPerThread - 1) / particlesPerThread;
      int gridSize2 = (numThreads + blockSize - 1) / blockSize; 
      particles2GridAtomicFreeCoarsening<<<gridSize2, blockSize>>>(d_rhos, d_nodeX, d_nodeY, d_nodeZ,
          d_rx+offsetParticles, d_ry+offsetParticles, d_rz+offsetParticles, d_q+offsetParticles,  
          p.nx, p.ny, p.nz, 
          p.dx, p.dy, p.dz, p.invVOL,
          is, p.np[is], particlesPerThread);

      /* //AtomicFreeAggregation version
      particles2GridAtomicFreeAggregation<<<gridSize, blockSize>>>(d_rhos, d_nodeX, d_nodeY, d_nodeZ,
          d_rx+offsetParticles, d_ry+offsetParticles, d_rz+offsetParticles, d_q+offsetParticles,  
          p.nx, p.ny, p.nz, 
          p.dx, p.dy, p.dz, p.invVOL,
          is, p.np[is]);*/
    }
    // field solver begin
    cudaMemset(d_rho_tot, 0, sizeof(double) * p.nCells);
    
    // Compute total charge density
    computeRhoTotKernel<<<gridSize, blockSize>>>(d_rho_tot, d_rhos, p.ns, p.nx, p.ny, p.nz);
    
    
    cudaMemcpy(d_rho, d_rho_tot, p.nCells * sizeof(double), cudaMemcpyDeviceToDevice);

    // Solve Poisson equation
    poisson(d_phi_poisson, d_rho, p.Lx, p.Ly, p.Lz, p.nx, p.ny, p.nz, blockSize);

    // Copy data to device memory
    cudaMemcpy(d_phi_gradient, d_phi_poisson, p.nCells * sizeof(double), cudaMemcpyDeviceToDevice);
    
    // Compute electric field
    gradient(d_gradX, d_gradY, d_gradZ, d_phi_gradient, p.Lx, p.Ly, p.Lz, p.nx, p.ny, p.nz, -1, blockSize);

    cudaMemcpy(d_Exn, d_gradX, p.nCells * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_Eyn, d_gradY, p.nCells * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_Ezn, d_gradZ, p.nCells * sizeof(double), cudaMemcpyDeviceToDevice);

    for (int is = 0; is < p.ns; is++){
      gridSize = (p.np[is] + blockSize - 1) / blockSize;
      int offsetParticles = is * p.np[is];
      
      // Update particle velocities
      updateParticleVelocityKernel<<<gridSize, blockSize>>>(
        d_rx + offsetParticles,
        d_ry + offsetParticles, d_rz + offsetParticles, d_vx + offsetParticles,
        d_vy + offsetParticles, d_vz + offsetParticles,
        d_Exn, d_Eyn, d_Ezn,
        d_nodeX, d_nodeY, d_nodeZ,
        p.nx, p.ny, p.nz,
        p.dx, p.dy, p.dz, p.invVOL,
        p.qom[is], p.dt, p.np[is]
        );
   }
   
    // output to file
    if (it % p.global_save_freq == 0 || it % p.fields_save_freq == 0 || it % p.part_save_freq == 0) {
      cudaMemcpy(fields->rho_tot, d_rho_tot, p.nCells * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(fields->rhos, d_rhos, p.ns * p.nCells * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(fields->Exn, d_Exn, p.nCells * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(fields->Eyn, d_Eyn, p.nCells * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(fields->Ezn, d_Ezn, p.nCells * sizeof(double), cudaMemcpyDeviceToHost);

      for (int is = 0; is < p.ns; is++){
        int offsetParticles = is * p.np[is];
        cudaMemcpy(part[is]->rx, d_rx + offsetParticles, p.np[is] * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(part[is]->ry, d_ry + offsetParticles, p.np[is] * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(part[is]->rz, d_rz + offsetParticles, p.np[is] * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(part[is]->vx, d_vx + offsetParticles, p.np[is] * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(part[is]->vy, d_vy + offsetParticles, p.np[is] * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(part[is]->vz, d_vz + offsetParticles, p.np[is] * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(part[is]->q, d_q + offsetParticles, p.np[is] * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(part[is]->ID, d_ID + offsetParticles, p.np[is] * sizeof(int), cudaMemcpyDeviceToHost);
      }
      if (it % p.global_save_freq == 0) 
        saveGlobalQuantities(fields, part, p.ns, p.np, p.qom, p.nCells, p.dx, p.dy, p.dz, it, out_dir_path.c_str());
      if (it % p.fields_save_freq == 0)
        saveFields(fields, it, p.ns, p.nCells, out_dir_path.c_str());
      if (it % p.part_save_freq == 0)
        for (int is = 0; is < p.ns; is++) saveParticles(part, is, p.np, it, out_dir_path.c_str());
    }
    
  }  // main loop end

  

  double elapsedTime = static_cast<double>(clTimer.getTimeMilliseconds());
  std::cout << std::endl << "Simulation terminated." << std::endl;
  std::cout << "Simulation loop elapsed time: " << elapsedTime << " ms (corresponding to " << (elapsedTime / 1000.0) << " s)" << std::endl;

  // memory de-allocation
  freeGrid(&grid, &d_nodeX, &d_nodeY, &d_nodeZ);
  freeFields(&fields, &d_phi, &d_rho_tot, &d_rho, &d_gradX, &d_gradY, &d_gradZ, &d_Exn, &d_Eyn, &d_Ezn, &d_rhos, &d_phi_poisson, &d_phi_gradient);
  freeParticles(&part, p.ns, &d_rx, &d_ry, &d_rz, &d_vx, &d_vy, &d_vz, &d_q, &d_ID);
  freeRandomArrays(harvest_array, theta_array, d_harvest_array, d_theta_array, p.ns);  

  return 0;
}
