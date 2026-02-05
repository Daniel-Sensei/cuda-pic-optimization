#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

#include <libconfig.h>
#include <stdio.h>

// Species
#define NS_MAX 8
//#define ELECTRONS 0
//#define IONS      1

struct Parameters
{
  // Domain: to be load
  double Lx;                             /*!< Grid size along x */
  double Ly;                             /*!< Grid size along y */
  double Lz;                             /*!< Grid size along z */
  int nx;                                /*!< Grid cells along x */
  int ny;                                /*!< Grid cells along y */
  int nz;                                /*!< Grid cells along z */

  // Domain: to be compute
  int nCells;                            /*!< Total number of cells */
  int nNodes;                            /*!< Total number of nodes */
  double dx;                             /*!< Cell size along x */
  double dy;                             /*!< Cell size along y */
  double dz;                             /*!< Cell size along z */
  double invVOL;

  // Species of particles
  int ns;                                /*!< Number of species */
  double qom[NS_MAX];                    /*!<  */

  double u0[NS_MAX];                     /*!<  */
  double v0[NS_MAX];                     /*!<  */
  double w0[NS_MAX];                     /*!<  */

  double uth0[NS_MAX];                   /*!<  */
  double vth0[NS_MAX];                   /*!<  */
  double wth0[NS_MAX];                   /*!<  */

  double B0x;                            /*!<  */
  double B0y;                            /*!<  */
  double B0z;                            /*!<  */

  int npcx[NS_MAX];                      /*!< Per species initial number of particles per cell along x */
  int npcy[NS_MAX];                      /*!< Per species initial number of particles per cell along y */
  int npcz[NS_MAX];                      /*!< Per species initial number of particles per cell along z */

  int npc[NS_MAX];                       /*!< Per species number of particles per cell */
  int np[NS_MAX];                        /*!< Per species number of particles */

  const char* init_case = "twostreams";  /*!< Case defining initial conditions> */
  
  // Time parameters
  int nsteps;
  int fields_save_freq;           
  int part_save_freq;
  int global_save_freq;

  double dt;
  double Amp;
  int random_seed;
};



void loadParameters(const char* filename, Parameters * p)
{
   config_t cfg;
   config_init(&cfg);
   
   // Load config file
   if (!config_read_file(&cfg, filename)) {
       fprintf(stderr, "Errore nel file %s:%d - %s\n", config_error_file(&cfg), config_error_line(&cfg), config_error_text(&cfg));
       config_destroy(&cfg);
       exit (EXIT_FAILURE);
   }

   if (!config_lookup_int(&cfg, "nx", &p->nx)) {fprintf(stderr, "Parameter 'nx' not found!\n"); exit(EXIT_FAILURE);}
   if (!config_lookup_int(&cfg, "ny", &p->ny)) {fprintf(stderr, "Parameter 'ny' not found!\n"); exit(EXIT_FAILURE);}
   if (!config_lookup_int(&cfg, "nz", &p->nz)) {fprintf(stderr, "Parameter 'nz' not found!\n"); exit(EXIT_FAILURE);}
   if (!config_lookup_float(&cfg, "Lx", &p->Lx)) {fprintf(stderr, "Parameter 'Lx' not found!\n"); exit(EXIT_FAILURE);}
   if (!config_lookup_float(&cfg, "Ly", &p->Ly)) {fprintf(stderr, "Parameter 'Ly' not found!\n"); exit(EXIT_FAILURE);}
   if (!config_lookup_float(&cfg, "Lz", &p->Lz)) {fprintf(stderr, "Parameter 'Lz' not found!\n"); exit(EXIT_FAILURE);}
   if (!config_lookup_int(&cfg, "ns", &p->ns)) {fprintf(stderr, "Parameter 'ns' not found!\n"); exit(EXIT_FAILURE);}

   config_setting_t *array_setting = config_lookup(&cfg, "qom");
   if (!array_setting) fprintf(stderr, "Parameter 'qom' not found!\n"); else
      for (int i = 0; i < config_setting_length(array_setting); i++) p->qom[i] = config_setting_get_float_elem(array_setting, i);
   
   array_setting = config_lookup(&cfg, "u0");
   if (!array_setting) fprintf(stderr, "Parameter 'u0' not found!\n"); else
      for (int i = 0; i < config_setting_length(array_setting); i++) p->u0[i] = config_setting_get_float_elem(array_setting, i);

   array_setting = config_lookup(&cfg, "v0");
   if (!array_setting) fprintf(stderr, "Parameter 'v0' not found!\n"); else
      for (int i = 0; i < config_setting_length(array_setting); i++) p->v0[i] = config_setting_get_float_elem(array_setting, i);

   array_setting = config_lookup(&cfg, "w0");
   if (!array_setting) fprintf(stderr, "Parameter 'w0' not found!\n"); else
      for (int i = 0; i < config_setting_length(array_setting); i++) p->w0[i] = config_setting_get_float_elem(array_setting, i);

   array_setting = config_lookup(&cfg, "uth0");
   if (!array_setting) fprintf(stderr, "Parameter 'uth0' not found!\n"); else
      for (int i = 0; i < config_setting_length(array_setting); i++) p->uth0[i] = config_setting_get_float_elem(array_setting, i);

   array_setting = config_lookup(&cfg, "vth0");
   if (!array_setting) fprintf(stderr, "Parameter 'vth0' not found!\n"); else
      for (int i = 0; i < config_setting_length(array_setting); i++) p->vth0[i] = config_setting_get_float_elem(array_setting, i);

   array_setting = config_lookup(&cfg, "wth0");
   if (!array_setting) fprintf(stderr, "Parameter 'wth0' not found!\n"); else
      for (int i = 0; i < config_setting_length(array_setting); i++) p->wth0[i] = config_setting_get_float_elem(array_setting, i);

   if (!config_lookup_float(&cfg, "B0x", &p->B0x)) {fprintf(stderr, "Parameter 'B0x' not found!\n"); exit(EXIT_FAILURE);}
   if (!config_lookup_float(&cfg, "B0y", &p->B0y)) {fprintf(stderr, "Parameter 'B0y' not found!\n"); exit(EXIT_FAILURE);}
   if (!config_lookup_float(&cfg, "B0z", &p->B0z)) {fprintf(stderr, "Parameter 'B0z' not found!\n"); exit(EXIT_FAILURE);}

   array_setting = config_lookup(&cfg, "npcx");
   if (!array_setting) fprintf(stderr, "Parameter 'npcx' not found!\n"); else
      for (int i = 0; i < config_setting_length(array_setting); i++) p->npcx[i] = config_setting_get_int_elem(array_setting, i);

   array_setting = config_lookup(&cfg, "npcy");
   if (!array_setting) fprintf(stderr, "Parameter 'npcy' not found!\n"); else
      for (int i = 0; i < config_setting_length(array_setting); i++) p->npcy[i] = config_setting_get_int_elem(array_setting, i);

   array_setting = config_lookup(&cfg, "npcz");
   if (!array_setting) fprintf(stderr, "Parameter 'npcz' not found!\n"); else
      for (int i = 0; i < config_setting_length(array_setting); i++) p->npcz[i] = config_setting_get_int_elem(array_setting, i);

   if (!config_lookup_string(&cfg, "init_case", &p->init_case)) {fprintf(stderr, "Parameter 'init_case' not found!\n"); exit(EXIT_FAILURE);}
   if (!config_lookup_int(&cfg, "random_seed", &p->random_seed)) {fprintf(stderr, "Parameter 'random_seed' not found!\n"); exit(EXIT_FAILURE);}
   
   if (!config_lookup_int(&cfg, "nsteps", &p->nsteps)) {fprintf(stderr, "Parameter 'nsteps' not found!\n"); exit(EXIT_FAILURE);}
   if (!config_lookup_int(&cfg, "fields_save_freq", &p->fields_save_freq)) {fprintf(stderr, "Parameter 'fields_save_freq' not found!\n"); exit(EXIT_FAILURE);}
   if (!config_lookup_int(&cfg, "part_save_freq", &p->part_save_freq)) {fprintf(stderr, "Parameter 'part_save_freq' not found!\n"); exit(EXIT_FAILURE);}
   if (!config_lookup_int(&cfg, "global_save_freq", &p->global_save_freq)) {fprintf(stderr, "Parameter 'global_save_freq' not found!\n"); exit(EXIT_FAILURE);}

   if (!config_lookup_float(&cfg, "dt", &p->dt)) {fprintf(stderr, "Parameter 'dt' not found!\n"); exit(EXIT_FAILURE);}
   if (!config_lookup_float(&cfg, "Amp", &p->Amp)) {fprintf(stderr, "Parameter 'Amp' not found!\n"); exit(EXIT_FAILURE);}


   // setting remaining parameters
   p->nCells = p->nx * p->ny * p->nz;
   p->nNodes = (p->nx + 1) * (p->ny + 1) * (p->nz + 1);
   for (int is = 0; is < p->ns; is++) {
      p->npc[is] = p->npcx[is] * p->npcy[is] * p->npcz[is];
      p->np[is] = p->npc[is] * p->nCells;
   }
   for (int is = p->ns; is < NS_MAX; is++) {
      p->npc[is] = 0;
      p->np[is] = 0;
      p->qom[is] = 0;
      p->u0[is] = p->v0[is] = p->w0[is] = 0;
      p->uth0[is] = p->vth0[is] = p->wth0[is] = 0;
      p->npcx[is] = p->npcy[is] = p->npcz[is] = 0;
   }
   p->dx = p->Lx / p->nx;
   p->dy = p->Ly / p->ny;
   p->dz = p->Lz / p->nz;
   p->invVOL = 1.0 / (p->dx *p->dy * p->dz);

   srand(p->random_seed);
}

void printParameters(Parameters * p)
{
   printf("Configuration parameters:\n");
   printf("%18s = ", "nx"); printf("%d\n", p->nx);
   printf("%18s = ", "ny"); printf("%d\n", p->ny);
   printf("%18s = ", "nz"); printf("%d\n", p->nz);
   printf("%18s = ", "Lx"); printf("%.6f\n", p->Lx);
   printf("%18s = ", "Ly"); printf("%.6f\n", p->Ly);
   printf("%18s = ", "Lz"); printf("%.6f\n", p->Lz);
   printf("%18s = ", "ns"); printf("%d\n", p->ns);
   printf("%18s = [", "qom"); for (int i = 0; i < p->ns-1; i++) printf("%f, ", p->qom[i]); printf("%f]\n", p->qom[p->ns-1]);
   printf("%18s = [", "u0");  for (int i = 0; i < p->ns-1; i++) printf("%f, ", p->u0[i] ); printf("%f]\n", p->u0[p->ns-1]);
   printf("%18s = [", "v0"); for (int i = 0; i < p->ns-1; i++) printf("%f, ", p->v0[i] ); printf("%f]\n", p->v0[p->ns-1]);
   printf("%18s = [", "w0"); for (int i = 0; i < p->ns-1; i++) printf("%f, ", p->w0[i] ); printf("%f]\n", p->w0[p->ns-1]);
   printf("%18s = [", "uth0"); for (int i = 0; i < p->ns-1; i++) printf("%f, ", p->uth0[i] ); printf("%f]\n", p->uth0[p->ns-1]);
   printf("%18s = [", "vth0"); for (int i = 0; i < p->ns-1; i++) printf("%f, ", p->vth0[i] ); printf("%f]\n", p->vth0[p->ns-1]);
   printf("%18s = [", "wth0"); for (int i = 0; i < p->ns-1; i++) printf("%f, ", p->wth0[i] ); printf("%f]\n", p->wth0[p->ns-1]);
   printf("%18s = ", "B0x"); printf("%.6f\n", p->B0x);
   printf("%18s = ", "B0y"); printf("%.6f\n", p->B0y);
   printf("%18s = ", "B0z"); printf("%.6f\n", p->B0z);
   printf("%18s = [", "npcx"); for (int i = 0; i < p->ns-1; i++) printf("%d, ", p->npcx[i] ); printf("%d]\n", p->npcx[p->ns-1]);
   printf("%18s = [", "npcy"); for (int i = 0; i < p->ns-1; i++) printf("%d, ", p->npcy[i] ); printf("%d]\n", p->npcy[p->ns-1]);
   printf("%18s = [", "npcz"); for (int i = 0; i < p->ns-1; i++) printf("%d, ", p->npcz[i] ); printf("%d]\n", p->npcz[p->ns-1]);
   printf("%18s = ", "init_case"); printf("%s\n", p->init_case);
   printf("%18s = %d\n", "random_seed", p->random_seed);
   printf("%18s = %d\n", "nsteps", p->nsteps);
   printf("%18s = %d\n", "fields_save_freq", p->fields_save_freq);
   printf("%18s = %d\n", "part_save_freq", p->part_save_freq);
   printf("%18s = %d\n", "global_save_freq", p->global_save_freq);
   printf("%18s = %f\n", "dt", p->dt);
   printf("%18s = %f\n", "Amp", p->Amp);
   printf("%18s = [", "npc"); for (int i = 0; i < p->ns-1; i++) printf("%d, ", p->npc[i]); printf("%f]\n", p->npc[p->ns-1]);
   printf("%18s = [", "np"); for (int i = 0; i < p->ns-1; i++) printf("%d, ", p->np[i]); printf("%f]\n", p->np[p->ns-1]);
   printf("%18s = %d\n", "nCells", p->nCells);
   printf("%18s = %d\n", "nNodes", p->nNodes);
   printf("%18s = %f\n", "dx", p->dx);
   printf("%18s = %f\n", "dy", p->dy);
   printf("%18s = %f\n", "dz", p->dz);
   printf("%18s = %f\n", "invVOL", p->invVOL);
}

#endif
