# ⚡ SimplePIC3D CUDA Parallelization

![CUDA](https://img.shields.io/badge/CUDA-Parallel%20Computing-76B900?logo=nvidia)
![C++](https://img.shields.io/badge/C++-Performance-00599C?logo=cplusplus)
![GPU](https://img.shields.io/badge/GPU-NVIDIA%20V100-green)
![License](https://img.shields.io/badge/License-Academic-yellow)

High-performance GPU parallelization of the **Particle-in-Cell (PIC)** method for plasma physics simulations. Achieves **~10x speedup** over serial implementation through CUDA optimization techniques.

---

## 📖 Overview

This project implements CUDA parallelization of the **simplePIC3D** electrostatic plasma simulation code. The Particle-in-Cell method is a fundamental technique in computational physics for simulating plasma dynamics and particle systems.

**Key Achievement:** Reduced execution time from **11.02 seconds** (serial) to **1.15 seconds** (CUDA Standard) for 10 simulation steps with 400 particles.

### The PIC Method

The simulation models plasma dynamics through the interaction between electromagnetic fields and charged particles (ions and electrons). The algorithm follows a main loop:

1. **Particles → Grid:** Interpolate particle charge density to grid
2. **Field Solver:** Compute electric field using Poisson's equation (FFT-based)
3. **Grid → Particles:** Interpolate electric field back to particle positions
4. **Particle Mover:** Update particle velocities and positions using leap-frog integration

---

## 🏗️ Implementation Versions

### Core Implementations

| Version | Description | Memory Model | Performance |
|---------|-------------|--------------|-------------|
| **Unified** | Straightforward CUDA parallelization using Unified Memory | Unified Memory | 1.21s (128x1x1) |
| **Standard** | Explicit host/device memory management | Standard CUDA | 1.15s (128x1x1) |

### Advanced Optimizations

| Version | Technique | Key Benefit |
|---------|-----------|-------------|
| **Atomic-Free Privatization** | Local memory for particle contributions | Reduced atomic operations, higher arithmetic intensity (14.01 FLOP/byte) |
| **Atomic-Free Coarsening** | Process 32 particles per thread | Massive speedup: **0.75ms** vs 13.13ms baseline |
| **Atomic-Free Aggregation** | Batch updates to global memory | Moderate arithmetic intensity (8.09 FLOP/byte) |

---

## 🚀 Performance Results

### Execution Time Comparison

```
Serial (Intel Xeon Gold 5118):  11.02 seconds
CUDA Unified (128x1x1):          1.21 seconds  (~9x speedup)
CUDA Standard (128x1x1):         1.15 seconds  (~10x speedup)
```

### Key Kernel Performance (Standard 512x1x1)

| Kernel | Time (ms) | AI (FLOP/byte) | Performance (GFLOP/s) | Warp Occ. (%) |
|--------|-----------|----------------|----------------------|---------------|
| **particles2Grid (Coarsening)** | 0.75 | 0.40 | 204.08 | 31.74 |
| **updateParticleVelocity** | 154.37 | 1.79 | 1358.54 | 63.03 |
| **updateParticlePosition** | 0.144 | 0.08 | 68.34 | 87.66 |
| **solvePoissonFourier** | 6.27 | 5.79 | 489.89 | 23.82 |

### Scalability Insights

- **Small datasets (50-100 particles):** Standard memory performs better
- **Large datasets (200-400 particles):** Unified memory shows performance advantages
- **Optimal block size:** Generally **512 threads** for best balance of occupancy and execution time
- **Coarsening optimization:** Best overall performance improvement (~17x faster than baseline)

---

## 💻 Technical Implementation

### CUDA Kernels

#### 1. Initialization Kernels
- **maxwellianKernel:** Initialize particle velocities following Maxwell-Boltzmann distribution
- **initPartTwostreamsKernel:** Setup two-stream instability simulation

#### 2. Main Loop Kernels
- **updateParticlePositionKernel:** Update particle positions based on velocity
- **particles2GridKernel:** Map particle charge density to grid (5 optimization variants)
- **computeRhoTotKernel:** Calculate total charge density across grid
- **poisson (cuFFT):** Solve Poisson's equation in Fourier space
- **gradient:** Compute electric field gradient
- **updateParticleVelocityKernel:** Update velocities using interpolated electric field

### Optimization Techniques

1. **Atomic Operations Management:** Replaced global atomic adds with privatization
2. **Thread Coarsening:** Process multiple particles per thread (32:1 ratio)
3. **Memory Access Patterns:** Optimized grid interpolation and boundary conditions
4. **FFT Acceleration:** Leveraged cuFFT library for spectral solver

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **CUDA** | GPU parallel computing framework |
| **cuFFT** | Fast Fourier Transform on GPU |
| **C++** | Core implementation language |
| **NVIDIA Nsight Compute** | Kernel profiling and Roofline analysis |
| **NVIDIA V100 GPU** | Target hardware for optimization |

---

## 📊 Roofline Analysis

The project includes comprehensive Roofline analysis for key kernels:

- **Privatization:** High arithmetic intensity (16.15 FLOP/byte) but limited by warp occupancy
- **Coarsening:** Memory-bound regime with excellent throughput
- **UpdateParticleVelocity:** Memory-bound due to interpolation-heavy workload

---

## 🚀 Getting Started

### Prerequisites

- CUDA Toolkit 11.0 or higher
- NVIDIA GPU with Compute Capability 7.0+ (tested on V100)
- C++ compiler with C++11 support
- cuFFT library

### Building the Project

```bash
# Clone the repository
git clone https://github.com/yourusername/simplePIC3D-cuda.git
cd simplePIC3D-cuda

# Build (standard version)
make

# Run simulation
make run

# Verify correctness (checksum validation)
make md5sum
# Expected: 1661f429b3f4d53cd06351432c9e1ffb
```

### Profiling with Nsight Compute

```bash
# Profile specific kernel
ncu --set full -f --import-source yes -o profile_output ./simplePIC3D

# View results in GUI
ncu-ui profile_output.ncu-rep
```

### Visualization with ParaView

```bash
# Generate visualization files and open ParaView
make dat2xfm
```
---

## 👥 Authors

- **Daniel Curcio**
- **Ilaria Raffaela Vasile**

*University of Calabria, Department of Mathematics and Computer Science*

---

## 🎓 Academic Context

This project was developed for the **GPU Computing** course at the University of Calabria. It demonstrates advanced CUDA optimization techniques including atomic-free algorithms, thread coarsening, and performance analysis using the Roofline model.

---

## 📄 License

This project is for academic purposes only.

---

## 🙏 Acknowledgments

- **Prof. Francesco Pucci** (ISTP | CNR) - Project guidance
- **Prof. Donato D'Ambrosio** (University of Calabria) - Course instruction
- NVIDIA for V100 GPU access and profiling tools
- Original simplePIC3D reference implementation
