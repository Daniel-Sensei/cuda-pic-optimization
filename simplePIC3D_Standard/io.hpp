#pragma once

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
using std::string;
#include "parameters.hpp"
#include "structures.hpp"

void saveGlobalQuantities(simu_fields * EMf, simu_particles** part, int ns, int np[], double qom[], int nCells, double dx, double dy, double dz, const int step, const char *dir_path)
{
  std::string file_path = std::string(dir_path) + "global_quantities.txt";

  FILE *f;
  if (step == 0)
    f = fopen(file_path.c_str(), "w");
  else
    f = fopen(file_path.c_str(), "a");

  if (!f)
  {
    printf("Unable to oper %s file\n", file_path.c_str());
    exit(EXIT_FAILURE);
  }

  double totalEnergy = 0.0, magneticEnergy = 0.0, electricEnergy = 0.0, particleTotalEnergy = 0.0, particleEnergy[ns];

  for (int i = 0; i < nCells; i++)
  {
    magneticEnergy += EMf->Bxc[i]*EMf->Bxc[i] + EMf->Byc[i]*EMf->Byc[i] + EMf->Bzc[i]*EMf->Bzc[i];
    electricEnergy += EMf->Exn[i]*EMf->Exn[i] + EMf->Eyn[i]*EMf->Eyn[i] + EMf->Ezn[i]*EMf->Ezn[i];
  }
  magneticEnergy = magneticEnergy * dx * dy * dz / (8 * M_PI);
  electricEnergy = electricEnergy * dx * dy * dz / (8 * M_PI);


  for (int is = 0; is < ns; is++)
  {
    particleEnergy[is] = 0;
    for (int i = 0; i < np[is]; i++)
      particleEnergy[is] += (0.5/fabs(qom[0])) * (part[is]->vx[i]*part[is]->vx[i] + part[is]->vy[i]*part[is]->vy[i] + part[is]->vz[i]*part[is]->vz[i]);
    
    particleTotalEnergy += particleEnergy[is];
  }

  totalEnergy = magneticEnergy + electricEnergy + particleTotalEnergy;


  double rho_avg[ns];
  for (int is = 0; is < ns; is++)
  {
    rho_avg[is] = 0;
    for(int i = is*nCells; i < is*nCells + nCells; i++)
      rho_avg[is] += EMf->rhos[i];
    rho_avg[is] /= nCells;
  }

  char str[32];
  if (step == 0)
  {
    fprintf(f, "%10s%18s%18s%28s%24s", "step", "Total_Energy", "Magnetic_Energy", "Electric_Energy", "Particle_Total_Energy");
    for (int is = 0; is < ns; is++)
    {
      std::string label = "Particle_Energy[" + std::to_string(is) + "]";
      fprintf(f, "%24s", label.c_str());
    }
    for (int is = 0; is < ns; is++)
    {
      std::string label = "rho_avg[" + std::to_string(is) + "]";
      fprintf(f, "%24s", label.c_str());
    }

    std::string label;
    for (int is = 0; is < ns; is++)
    {
      label = "vx_avg[" + std::to_string(is) + "]";
      fprintf(f, "%24s", label.c_str());
      label = "vy_avg[" + std::to_string(is) + "]";
      fprintf(f, "%24s", label.c_str());
      label = "vz_avg[" + std::to_string(is) + "]";
      fprintf(f, "%24s", label.c_str());
    }

    fprintf(f, "\n");

  }
  
  sprintf(str, "%10d ", step);
  fprintf(f, "%s ", str);
  sprintf(str, "%16.6f ", totalEnergy);
  fprintf(f, "%s ", str);
  sprintf(str, "%16.6f ", magneticEnergy);
  fprintf(f, "%s ", str);
  //sprintf(str, "%16.6f ", electricEnergy);
  sprintf(str, "%26.16f ", electricEnergy);
  fprintf(f, "%s ", str);
  sprintf(str, "%22.6f ", particleTotalEnergy);
  fprintf(f, "%s ", str);
  for (int is = 0; is < ns; is++)
  {
    sprintf(str, "%22.6f ", particleEnergy[is]);
    fprintf(f, "%s ", str);
  }
  for (int is = 0; is < ns; is++)
  {
    sprintf(str, "%22.6f ", rho_avg[is]);
    fprintf(f, "%s ", str);
  }

  double vx_tot = 0;
  double vy_tot = 0;
  double vz_tot = 0;
  for (int is = 0; is < ns; is++)
  {
    vx_tot = 0;
    vy_tot = 0;
    vz_tot = 0;
    
    for (int i = 0; i < np[is]; i++)
    {
      vx_tot += part[is]->vx[i];
      vy_tot += part[is]->vy[i];
      vz_tot += part[is]->vz[i];
    }

    sprintf(str, "%22.6f ", vx_tot/np[is]); fprintf(f, "%s ", str);
    sprintf(str, "%22.6f ", vy_tot/np[is]); fprintf(f, "%s ", str);
    sprintf(str, "%22.6f ", vz_tot/np[is]); fprintf(f, "%s ", str);
  }

  fprintf(f, "\n");

  fclose(f);
}

void saveParticles(simu_particles** part, const unsigned int is, int np[], const unsigned int it, const char *dir_path)
{
  std::string file_path = std::string(dir_path) + "part" + std::to_string(is) + "_" + std::to_string(it) + ".txt";

  FILE *f;
  f = fopen(file_path.c_str(), "w");

  if (!f)
  {
    printf("Unable to oper %s file\n", file_path.c_str());
    exit(EXIT_FAILURE);
  }

  char str[32];
  fprintf(f, "%10s%12s%12s%12s%12s%12s%12s%12s\n", "ID", "rx", "ry", "rz", "q", "vx", "vy", "vz");
  for (int i = 0; i < np[is]; i++)
  {
    sprintf(str, "%10d ", part[is]->ID[i]);
    fprintf(f, "%s ", str);
    sprintf(str, "%10.6f ", part[is]->rx[i]);
    fprintf(f, "%s ", str);
    sprintf(str, "%10.6f ", part[is]->ry[i]);
    fprintf(f, "%s ", str);
    sprintf(str, "%10.6f ", part[is]->rz[i]);
    fprintf(f, "%s ", str);
    sprintf(str, "%10.6f ", part[is]->q[i]);
    fprintf(f, "%s ", str);
    sprintf(str, "%10.6f ", part[is]->vx[i]);
    fprintf(f, "%s ", str);
    sprintf(str, "%10.6f ", part[is]->vy[i]);
    fprintf(f, "%s ", str);
    sprintf(str, "%10.6f ", part[is]->vz[i]);
    fprintf(f, "%s ", str);

    fprintf(f, "\n");
  }

  fclose(f);
}

void saveFields(simu_fields * EMf, const unsigned int it, const int ns, const int nCells, const char * path)
{
  FILE *f;
  char padded_it[6];
  sprintf(padded_it, "%06d", it);
  
  //Exn
  std::string fname = path;
  fname += "Exn_"+string(padded_it)+".dat";	  
  f = fopen(fname.c_str(), "w");
  if (!f)
  {
    printf("Unable to open %s file\n", path);
    exit(EXIT_FAILURE);
  }
  fwrite(EMf->Exn, sizeof(EMf->Exn[0]), nCells, f);
  fclose(f);
  //Eyn
  fname = path ;
  fname += "Eyn_"+string(padded_it)+".dat";	  
  f = fopen(fname.c_str(), "w");
  fwrite(EMf->Eyn, sizeof(EMf->Eyn[0]), nCells, f);
  fclose(f);
  //Ezn
  fname = path ;
  fname += "Ezn_"+string(padded_it)+".dat";	  
  f = fopen(fname.c_str(), "w");
  fwrite(EMf->Ezn, sizeof(EMf->Ezn[0]), nCells, f);
  fclose(f);
  //rho
  fname = path ;
  fname += "rhos_"+string(padded_it)+".dat";	  
  f = fopen(fname.c_str(), "w");
  fwrite(EMf->rhos, sizeof(EMf->rhos[0]), ns*nCells, f);
  fclose(f);
}




void loadParticles(const simu_particles &part, const unsigned int is, int np[], const char *path)
{
  FILE *f = fopen(path, "r");

  if (!f)
  {
    printf("File %s not found\n", path);
    exit(EXIT_FAILURE);
  }

  char str[32];
  for (int head = 0; head < 8; head++)
    fscanf(f, "%s", str);
  for (int i = 0; i < np[is]; i++)
  {
    fscanf(f, "%s", str);
    part.ID[i] = atoi(str);
    fscanf(f, "%s", str);
    part.rx[i] = atof(str);
    fscanf(f, "%s", str);
    part.ry[i] = atof(str);
    fscanf(f, "%s", str);
    part.rz[i] = atof(str);
    fscanf(f, "%s", str);
    part.q[i] = atof(str);
    fscanf(f, "%s", str);
    part.vx[i] = atof(str);
    fscanf(f, "%s", str);
    part.vy[i] = atof(str);
    fscanf(f, "%s", str);
    part.vz[i] = atof(str);
  }

  fclose(f);
}
