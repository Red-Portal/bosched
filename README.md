
# A Probabilistic Machine Learning Approach to Scheduling Parallel Loops with Bayesian Optimization

This repository contains the code of our paper ["A Probabilistic Machine Learning Approach to Scheduling Parallel Loops with Bayesian Optimization"](https://www.researchgate.net/publication/341869288_A_Probabilistic_Machine_Learning_Approach_to_Scheduling_Parallel_Loops_with_Bayesian_Optimization).

## Installing
We have to build two things: the custom GCC compiler and the *bosched* runtime

### Building Custom GCC
```shell
cd gcc
./contrib/download_prerequisites
mkdir build && cd build
../configure --prefix=$PWD --disable-multilib --enable-languages=c,c++,fortran
make -j ${CPUs on your machine} && make install
```
This should install our customized GCC in the build directory.
The compiler binaries are in `build/bin/*` and the runtime libraries are in `build/lib64/*`.

### Building `bosched` Runtime
```shell
mkdir build && cd build
cmake -G "Unix Makefiles" ../
make -j ${CPUs on your machine} && make install
```
This should install our bosched runtime (`libbosched.so`) in `build`.

## Usage
To use bosched, you need to build the target binary using our custom GCC compiler and link some dynamic libraries.
For example, you have to compile a target binary such as below.
`BOSCHED_ROOT` denotes a variable containing the root directory of bosched.

```shell 
$BOSCHED_ROOT/build/bin/gcc target.c -L$BOSCHED_ROOT/build -L$BOSCHED_ROOT/gcc/build/lib64 -lbosched -lgomp
```
And execute after adding the following dynamic link paths.
```shell 
export LD_LIBRARY_PATH=$BOSCHED_ROOT/build:$BOSCHED_ROOT/gcc/build/lib64
```

After executing the target binary multiple times, a file of the following format would have appeared.
```shell
.boststate.target.json 
```
This file contains all the state and informations used in BO FSS.

You can define the following environment variables.

* `EVAL`: Enable evaluation mode. Uses the best parameter found so far.
* `DEBUG`: Enable debug mode
* `OMP_SCHEDULE`: Select OpenMP schedule. For BO FSS, select BO_FSS.

The offline bosched tuner is invoked by executing the julia script `bosched.jl`.

```shell
julia bosched --mode bosched --time_samples 4 target
```
`time_samples` is the number of samples to use for our locality-aware version.
Executing the above script will update the `.bostate` file, with the optimization results.
Executing the target binary after this will automatically use the results.

## Project Structure
* All of the `bosched` runtime is in the root folder.
* Our modifications to GCC are done in `gcc/gcc/omp-expand.c`
* OpenMP runtime related modification are in `gcc/libgomp/*`
* BO FSS offline tuning related code is in `bosched.jl` and `LABO/*`. `LABO` exclusively contains BO relted code.

## Dependencies
* Linux
* C++17
* Julia 1.3.1
* CMake 3.8
* [Boost](https://www.boost.org/)
* [HighFive](https://github.com/BlueBrain/HighFive)
* BLAS, LAPACK

### Julia Packages
* ArgParse
* JSON
* HDF5
* Plots
* Statistics
* TerminalMenus
* UnicodePlots
* AdvancedHMC
* Distributions
* GaussianProcesses
* LineSearches
* LinearAlgebra
* MCMCDiagnostics
* NLopt
* Optim
* ProgressMeter
* Random
* Roots
* SpecialFunctions
* Statistics
* StatsBase
