
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

#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <sys/time.h>
#include <cstring>
#include <time.h>
#include <sys/time.h>
#include <starpu.h>
#include "dft.h"

float timedifference_msec(struct timeval t0, struct timeval t1)
{
    return (t1.tv_sec - t0.tv_sec) * 1000.0f + (t1.tv_usec - t0.tv_usec) / 1000.0f;
}

int main(int argc, char **argv) {
    printf("==============================================================\n");
    printf(">>> AUT HPC Research Laboratory - Direct Fourier Transform <<<\n");
    printf("==============================================================\n\n");

    if (argv[1] == NULL || !((strcmp(argv[1], "cuda") == 0) || (strcmp(argv[1], "cpu") == 0) || (strcmp(argv[1], "starpu") == 0))) {
        printf(">>> Usage: \n ./dft cuda \n or: \n ./dft cpu \n or: \n ./dft starpu\n");
        return EXIT_FAILURE;
    }
    // Seed random from time
    srand(time(NULL));

    // Initialise the DFT configuration
    Config config;
    init_config(&config);

    // Obtain Sources from file, or synthesize
    Source *sources = NULL;
    load_sources(&config, &sources);
    // Something went wrong during loading of sources
    if (sources == NULL) {
        printf(">>> ERROR: Source memory was unable to be allocated...\n\n");
        return EXIT_FAILURE;
    }


    // Obtain Visibilities from file, or synthesize
    Visibility *visibilities = NULL;
    Complex *vis_intensity = NULL;
    load_visibilities(&config, &visibilities, &vis_intensity);

    // Something went wrong during loading of visibilities
    if (visibilities == NULL || vis_intensity == NULL) {
        printf(">>> ERROR: Visibility memory was unable to be allocated...\n\n");
        if (sources) free(sources);
        if (visibilities) free(visibilities);
        if (vis_intensity) free(vis_intensity);
        return EXIT_FAILURE;
    }

    printf(">>> UPDATE: Performing extraction of visibilities from sources...\n\n");
    
    struct timeval timeStart, timeEnd;
    gettimeofday(&timeStart,0);

    if (strcmp(argv[1], "cuda") == 0) {
        printf(">>> ======CUDA Version===== \n\n");
        extract_visibilities_cuda(&config, sources, visibilities, vis_intensity, config.num_visibilities);
    } else if (strcmp(argv[1], "cpu") == 0) {
        printf(">>> ======CPU Version===== \n\n");
        extract_visibilities_cpu(sources, visibilities, vis_intensity,config.num_sources, config.num_visibilities);
    } else if (strcmp(argv[1], "starpu") == 0){
        printf(">>> ======STARPU Version===== \n\n");
        starpu_launch(&config, sources, visibilities, vis_intensity);
    }
    gettimeofday(&timeEnd,0);
    float timeDiff = timedifference_msec(timeStart,timeEnd);
    printf(">>> TOOK %f SECONDS to Extract...\n\n", timeDiff/1000.0);

    printf(">>> UPDATE: Visibility extraction complete...\n\n");

    // Save visibilities to file
    save_visibilities(&config, visibilities, vis_intensity);

    // Clean up
    if (visibilities) free(visibilities);
    if (sources) free(sources);
    if (vis_intensity) free(vis_intensity);

    printf(">>> UPDATE: Direct Fourier Transform operations complete, exiting...\n\n");

    return EXIT_SUCCESS;
}
