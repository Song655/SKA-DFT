
// Copyright 2019 Adam Campbell, Seth Hall, Andrew Ensor
// Copyright 2019 High Performance Computing Research Laboratory, Auckland University of Technology (AUT)

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.

// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.

// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math_constants.h>
#include <device_launch_parameters.h>
#include <numeric>

#include "dft.h"

__global__ void direct_fourier_transform(const __restrict__ PRECISION3 *visibility, PRECISION2 *vis_intensity, const int vis_count, const PRECISION3 *sources, const int source_count)
{
	const int vis_indx = blockIdx.x * blockDim.x + threadIdx.x;

	if(vis_indx >= vis_count)
		return;

	PRECISION2 source_sum = MAKE_PRECISION2(0.0, 0.0);
	PRECISION term = 0.0;
	PRECISION w_correction = 0.0;
	PRECISION image_correction = 0.0;
	PRECISION theta = 0.0;
	PRECISION src_correction = 0.0;

	const PRECISION3 vis = visibility[vis_indx];
	PRECISION3 src;
	PRECISION2 theta_complex = MAKE_PRECISION2(0.0, 0.0);

	const double two_PI = CUDART_PI + CUDART_PI;
	// For all sources
	for(int src_indx = 0; src_indx < source_count; ++src_indx)
	{	
		src = sources[src_indx];
		
		// formula sqrt
		// term = sqrt(1.0 - (src.x * src.x) - (src.y * src.y));
		// image_correction = term;
		// w_correction = term - 1.0; 

		// approximation formula (unit test fails as less accurate)
		term = 0.5 * ((src.x * src.x) + (src.y * src.y));
		w_correction = -term;
		image_correction = 1.0 - term;

		src_correction = src.z / image_correction;

		theta = (vis.x * src.x + vis.y * src.y + vis.z * w_correction) * two_PI;
		sincos(theta, &(theta_complex.y), &(theta_complex.x));
		source_sum.x += theta_complex.x * src_correction;
		source_sum.y += -theta_complex.y * src_correction;
	}

	vis_intensity[vis_indx] = MAKE_PRECISION2(source_sum.x, source_sum.y);
}


/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
void check_cuda_error_aux(const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;

	printf(">>> CUDA ERROR: %s returned %s at %s : %u ",statement, file, cudaGetErrorString(err), line);
	exit(EXIT_FAILURE);
}

void extract_visibilities_cuda(Config *config, Source *sources, Visibility *visibilities,
	Complex *vis_intensity, int num_visibilities)
{
	//Allocating GPU memory for visibility intensity
	PRECISION3 *device_sources;
	PRECISION3 *device_visibilities;
	PRECISION2 *device_intensities;

	if(config->enable_messages)
		printf(">>> UPDATE: Allocating GPU memory...\n\n");

	//copy the sources to the GPU.
	CUDA_CHECK_RETURN(cudaMalloc(&device_sources,  sizeof(PRECISION3) * config->num_sources));
	CUDA_CHECK_RETURN(cudaMemcpy(device_sources, sources, 
		config->num_sources * sizeof(PRECISION3), cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	//copy the visibilities to the GPU
	CUDA_CHECK_RETURN(cudaMalloc(&device_visibilities,  sizeof(PRECISION3) * num_visibilities));
	CUDA_CHECK_RETURN(cudaMemcpy(device_visibilities, visibilities, 
		num_visibilities * sizeof(PRECISION3), cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	// Allocate memory on GPU for storing extracted visibility intensities
	CUDA_CHECK_RETURN(cudaMalloc(&device_intensities,  sizeof(PRECISION2) * num_visibilities));
	cudaDeviceSynchronize();

	// Define number of blocks and threads per block on GPU
        int threads_per_block = min(config->gpu_num_threads_per_block, num_visibilities);
        int num_blocks = ceil((double)num_visibilities / threads_per_block);

	dim3 kernel_threads(threads_per_block, 1, 1);
	dim3 kernel_blocks(num_blocks, 1, 1);

	if(config->enable_messages)
		printf(">>> UPDATE: Calling DFT GPU Kernel to create %d visibilities...\n\n", num_visibilities);

	//record events for timing kernel execution
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	direct_fourier_transform<<<kernel_blocks,kernel_threads>>>(device_visibilities,
		device_intensities, num_visibilities, device_sources, config->num_sources);
	cudaDeviceSynchronize();

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	if(config->enable_messages)
		printf(">>> UPDATE: DFT GPU Kernel Completed, Time taken %f mS...\n\n",milliseconds);

	CUDA_CHECK_RETURN(cudaMemcpy(vis_intensity, device_intensities, 
		num_visibilities * sizeof(PRECISION2), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	if(config->enable_messages)
		printf(">>> UPDATE: Copied Visibility Data back to Host - Completed...\n\n");

	// Clean up
	CUDA_CHECK_RETURN(cudaFree(device_intensities));
	CUDA_CHECK_RETURN(cudaFree(device_sources));
	CUDA_CHECK_RETURN(cudaFree(device_visibilities));
	CUDA_CHECK_RETURN(cudaDeviceReset());
}



