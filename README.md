# riscv-matmul-vec

Official repository of the software project developed for the Master's Thesis in Computer Science at the University of Salerno: **"Explicit Vectorization of Matrix Multiplication on RISC-V with RVV"**.

## 📖 Project Description

This project explores optimization techniques for the General Matrix Multiply (GEMM) operation on **RISC-V** architectures. The main focus is on single-core optimization leveraging the **explicit vectorization** provided by the **RVV** (RISC-V Vector Extension).

The code compares classic scalar implementations with progressively optimized approaches:

- Standard vectorization.
- **Tiling** techniques to improve cache locality (L1/L2).
- Loop reordering and formulations based on the "outer product".
- Dynamic management of vector registers and the LMUL parameter at runtime.

## 📂 Repository Structure

The main C source code and automation scripts are located in the root directory and support folders:

```text
riscv-matmul-vec/
├── benchsuite/         # Benchmark suite and performance tests
├── report/             # Data and reports generated from executions
├── script/             # Python scripts and analysis utilities
├── test/               # Test files for correctness validation
├── baseline.c          # Basic scalar implementation
├── tiling*.c           # Various Tiling implementation versions (v2, v3, etc.)
├── reordered_tiling.c  # Tiling with advanced loop reordering
├── utils.c / .h        # Utility functions for matrices, time measurement, etc.
├── benchmark.sh        # Script for automated benchmark execution
├── emu.sh              # Script for execution via emulator (QEMU/Spike)
├── stat.sh             # Script for statistics collection (e.g., cache misses)
└── makefile            # Compiles the different versions and generates reports/graphs
```

## ⚙️ Requirements

To compile and run this project, you need:

- **Hardware/Emulator**: A RISC-V board equipped with the RVV extension 1.0 (e.g., **SpacemiT X60**, **Banana Pi BPI-F3**) or a compatible emulator (QEMU/Spike).
- **Toolchain**: A RISC-V compiler with support for RVV intrinsics (e.g., a recent 14.x `riscv64-unknown-linux-gnu-gcc`).

## 🚀 Build and Execution

To compile the project, you can simply use `make`. The `Makefile` is configured to handle the build process of the different implementations automatically.

```bash
# Compile a specific version 
make <version>

# Compile all main versions
make all
```

### Benchmark Execution

To automate performance measurement for the different code versions (baseline, tiling, reordered, etc.), use the dedicated script:

```bash
./benchmark.sh
```

To run the programs in an emulated environment:

```bash
./emu.sh [optional_arguments]
```

## 📊 Performance Analysis

The goal of this codebase is to generate the performance data (GFLOPS, memory bandwidth, cache miss rate) discussed in the thesis.

The `Makefile` itself includes targets to seamlessly generate **graphs and tables** directly from the raw execution results. Utility scripts (`stat.sh` and those in the `script/` folder) help extrapolate execution profiles and compare how the memory architecture (e.g., on the Banana Pi board or the X60) responds to vector instructions compared to scalar ones.

## 👤 Author

**Sergio Guastaferro** *Candidate, University of Salerno*

**Advisor:** Prof. Biagio Cosenza

**Tutor:** Dott. Lorenzo Carpentieri

---

*Project released under the Apache 2.0 license. If you use this code for academic purposes, please cite the thesis work.*
