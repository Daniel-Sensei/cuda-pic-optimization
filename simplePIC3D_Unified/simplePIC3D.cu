#include <cassert>
#include <cstdlib>
#include "fields.hpp"
#include "init.hpp"
#include "io.hpp"
#include "parameters.hpp"
#include "particles.hpp"
#include "structures.hpp"
#include "util.hpp"
#include <cuda_runtime.h>

int main(int argc, char *argv[])
{
  // Block configuration
  int blockSizeX = 128, blockSizeY = 1, blockSizeZ = 1;

  // Print the block size
  std::cout << "Block size: " << blockSizeX << " " << blockSizeY << " " << blockSizeZ << std::endl;

  // Loading parameters from the config.cfg file
  Parameters p;
  loadParameters("config.cfg", &p);
  printParameters(&p);

  // Declaring structures of arrays for grid, fields and particles
  simu_grid *grid;
  simu_fields *field;
  simu_particles **part; // ~ part[ns];
  double **harvest_array = nullptr;
  double **theta_array = nullptr;

  // Memory allocation
  allocateGridUnified(&grid, p.nx, p.ny, p.nz);
  allocateFieldsUnified(&field, p.ns, p.nCells, p.nNodes);
  allocateParticlesUnified(&part, p.ns, p.np);
  allocateRandomArrays(harvest_array, theta_array, p.ns, p.nx, p.ny, p.nz, p.npcx, p.npcy, p.npcz);

  // Initialization
  printf("Initializing simulation\n");
  initGrid(grid, p.Lx, p.Ly, p.Lz, p.nx, p.ny, p.nz);

  // Block size and grid size
  int totalCells = p.nx * p.ny * p.nz;
  int blockSize = blockSizeX * blockSizeY * blockSizeZ;
  int gridSize = ((totalCells + blockSize - 1) / blockSize); 

  // Print the blockSize
  std::cout << "Block total size: " << blockSize << std::endl;
  
  // Initialization of the particles
  if (string(p.init_case) == "twostreams")
  {
    // Initialize the EM fields
    initEMFieldsTwostreams(field, p.B0x, p.B0y, p.B0z, p.qom, p.ns, p.nCells);
    printf("Initializing particles for two stream instability\n");
    for (int is = 0; is < p.ns; is++)
    {
      // Initialize the arrays containing the random numbers
      initRandomNumbers(harvest_array[is], theta_array[is], is, p.npc, p.npcx, p.npcy, p.npcz, p.nx, p.ny, p.nz);
       
      // Grid size
      int totalCells = p.nx * p.ny * p.nz * p.npcx[is] * p.npcy[is] * p.npcx[is]; 
      dim3 gridSize = ((totalCells + blockSize - 1) / blockSize);

      // Initialize the particles
      initPartTwostreamsKernel<<<gridSize, blockSize>>>(part[is], grid, field,
                                                        p.npcx[is], p.npcy[is], p.npcz[is],
                                                        p.nx, p.ny, p.nz,
                                                        p.dx, p.dy, p.dz,
                                                        p.qom[is], p.u0[is], p.v0[is], p.w0[is],
                                                        p.uth0[is], p.vth0[is], p.wth0[is],
                                                        p.npc[is], harvest_array[is], theta_array[is]);
    }
  }
  else if (string(p.init_case) == "randomInit")
  {
    // Initialize the EM fields
    initEMfields(field, p.B0x, p.B0y, p.B0z, p.Amp, p.qom, p.Lx, p.ns, p.nx, p.ny, p.nz, p.dx);
    for (int is = 0; is < p.ns; is++)
    {
      // Initialize the arrays containing the random numbers
      initRandomNumbers(harvest_array[is], theta_array[is], is, p.npc, p.npcx, p.npcy, p.npcz, p.nx, p.ny, p.nz);
      
      // Grid size
      int totalCells = p.nx * p.ny * p.nz * p.npcx[is] * p.npcy[is] * p.npcx[is]; // Numero totale di celle
      dim3 gridSize = ((totalCells + blockSize - 1) / blockSize);
      
      // Initialize the particles with a maxwellian distribution
      maxwellianKernel<<<gridSize, blockSize>>>(part[is], grid, field,
                                                p.npcx[is], p.npcy[is], p.npcz[is],
                                                p.nx, p.ny, p.nz,
                                                p.dx, p.dy, p.dz,
                                                p.qom[is], p.u0[is], p.v0[is], p.w0[is],
                                                p.uth0[is], p.vth0[is], p.wth0[is],
                                                p.npc[is], harvest_array[is], theta_array[is]);
    }
  }
  else
  {
    printf("Unable to find a valid configuration. Aborting...\n");
    exit(EXIT_FAILURE);
  }

  // output
  string out_dir_path = "out/";
  // std::filesystem::create_directory(out_dir_path);
  system("mkdir -p out");
  saveGlobalQuantities(field, part, p.ns, p.np, p.qom, p.nCells, p.dx, p.dy, p.dz, 0, out_dir_path.c_str());
  saveFields(field, 0, p.ns, p.nCells, out_dir_path.c_str());

  // main loop
  util::Timer clTimer;
  for (int it = 1; it <= p.nsteps; it++)
  {
    printf("\rRunning simulation - step %d", it);
    fflush(stdout);

    // Reset the rho field
    cudaMemset(field->rhos, 0, sizeof(double) * p.ns * p.nCells);
    for (int is = 0; is < p.ns; is++)
    {
      // Grid size
      gridSize = (p.np[is] + blockSize - 1) / blockSize;

      // Update the particle position
      updateParticlePositionKernel<<<gridSize, blockSize>>>(part[is]->rx, part[is]->ry, part[is]->rz,
                                                                 part[is]->vx, part[is]->vy, part[is]->vz,
                                                                 p.np[is], p.Lx, p.Ly, p.Lz, p.dt);
      int particlesPerThread = 32; //Best configuration among all the tested numbers
      int numThreads = (p.np[is] + particlesPerThread - 1) / particlesPerThread;
      int gridSize2 = (numThreads + blockSize - 1) / blockSize; 
      particles2GridAtomicFreeCoarsening<<<gridSize2, blockSize>>>(field, grid, part,
          p.nx, p.ny, p.nz, 
          p.dx, p.dy, p.dz, p.invVOL,
          is, p.np[is], particlesPerThread);
      /*// Assign the particles to the grid
      particles2GridKernel<<<gridSize, blockSize>>>(
          field, grid, p.nx, p.ny, p.nz,
          p.dx, p.dy, p.dz, p.invVOL,
          part, is, p.np[is]);*/
    }

    // field solver begin
    cudaMemset(field->rho_tot, 0.0, sizeof(double) * p.nCells);

    // Compute the total charge density
    computeRhoTotKernel<<<gridSize, blockSize>>>(field->rho_tot, field->rhos, p.ns, p.nx, p.ny, p.nz);

    poisson(field->phi, field->rho_tot, p.Lx, p.Ly, p.Lz, p.nx, p.ny, p.nz, blockSize); 
    gradient(field->Exn, field->Eyn, field->Ezn, field->phi, p.Lx, p.Ly, p.Lz, p.nx, p.ny, p.nz, -1, blockSize);
    for (int is = 0; is < p.ns; is++)
    {
      // Grid size
      gridSize = (p.np[is] + blockSize - 1) / blockSize;

      // Update the particle velocity
      updateParticleVelocityKernel<<<gridSize, blockSize>>>(
          part[is]->rx, part[is]->ry, part[is]->rz,
          part[is]->vx, part[is]->vy, part[is]->vz,
          field->Exn, field->Eyn, field->Ezn,
          grid->nodeX, grid->nodeY, grid->nodeZ,
          p.nx, p.ny, p.nz,
          p.dx, p.dy, p.dz, p.invVOL,
          p.qom[is], p.dt, p.np[is]);
    }

    // Output to file
    if (it % p.global_save_freq == 0)
      saveGlobalQuantities(field, part, p.ns, p.np, p.qom, p.nCells, p.dx, p.dy, p.dz, it, out_dir_path.c_str());
    if (it % p.fields_save_freq == 0)
      saveFields(field, it, p.ns, p.nCells, out_dir_path.c_str());
    if (it % p.part_save_freq == 0)
      for (int is = 0; is < p.ns; is++)
        saveParticles(part, is, p.np, it, out_dir_path.c_str());
  } // main loop end

  double elapsedTime = static_cast<double>(clTimer.getTimeMilliseconds());
  std::cout << std::endl
            << "Simulation terminated." << std::endl;
  std::cout << "Simulation loop elapsed time: " << elapsedTime << " ms (corresponding to " << (elapsedTime / 1000.0) << " s)" << std::endl;

  // Memory de-allocation
  freeRandomArrays(harvest_array, theta_array, p);
  freeGridUnified(&grid);
  freeFieldsUnified(&field);
  freeParticlesUnified(&part, p.ns);

  return 0;
}
