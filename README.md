# SKA-DFT

Merge
C_DFT(https://github.com/ska-telescope/C_DFT) 
and 
CUDA_DFT (https://github.com/ska-telescope/CUDA_DFT) 

CPU version:

` ./dft cpu`

CUDA version:

  ` ./dft cuda`

StarPU version:

`./dft starpu`

or 

`STARPU_SCHED=ws STARPU_WORKER_STATS=1 ./dft starpu`


How to install StarPU:

1. Install dependence library:

    Install CUDA and set CUDA environment
    Install ATLAS for using ATLAS BLAS library (--need it when we test Cholesky)

    sudo add-apt-repository universe
    sudo add-apt-repository main
    sudo apt-get update 
    sudo apt-get install libatlas-base-dev liblapack-dev libblas-dev


    Set environment: (Add the following into the last line of the file home/.bashrc)

    export C_INCLUDE_PATH=/usr/include/atlas:$C_INCLUDE_PATH

    Install FFTW (we install fftw-3.3.8) (--need it when we test Starpufft)

Installing FFTW in both single and double precision：

./configure --enable-shared [ other options ]
sudo make CFLAGS=-fPIC
sudo make install

make clean
./configure --enable-shared --enable-float [ other options ]
sudo make CFLAGS=-fPIC
sudo make install


2. Install StarPU

    Getting StarPU Source code (insall 1.3.2)
    Configuring StarPU:

    ./autogen.sh
    mkdir build
    cd build
    ../configure --prefix=$HOME/starpu --enable-openmp --disable-opencl --enable-blas-lib=atlas --enable-cuda --enable-starpufft-examples
    make
    make check
    make install

    Set the environment

    export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$STARPU_PATH/lib/pkgconfig
    export C_INCLUDE_PATH=$C_INCLUDE_PATH:$STARPU_PATH/include/starpu/1.3
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$STARPU_PATH/lib


