#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <float.h>

#include "dft.h"



// Performs the direct fourier transformation to obtain the complex brightness
// of each visibility from each identified source. This is the meat of the algorithm.
void extract_visibilities_cpu(Config *config, Source *sources, Visibility *visibilities,Complex *vis_intensity, int num_visibilities)
{
	for(int vis_indx = 0; vis_indx < num_visibilities; ++vis_indx)
	{
		Visibility *vis = &visibilities[vis_indx];
		Complex *source_sum = &vis_intensity[vis_indx];
        *source_sum = (Complex) {.real = 0.0, .imaginary = 0.0};

		// For all sources, obtain some portion of brightness
		for(int src_indx = 0; src_indx < config->num_sources; ++src_indx)
		{
			Source *src             = &sources[src_indx];
			double image_correction = sqrt(1.0 - pow(src->l, 2.0) - pow(src->m, 2.0));
			double theta            = vis->u * src->l + vis->v * src->m + vis->w * (image_correction - 1.0);
			
			Complex theta_complex = (Complex) {
				.real =  cos(2.0 * M_PI * theta),
				.imaginary = -sin(2.0 * M_PI * theta)
			};

			double normalized_intensity     = src->intensity / image_correction;
			theta_complex.real             *= normalized_intensity;
			theta_complex.imaginary        *= normalized_intensity;
			source_sum->real                += theta_complex.real;
			source_sum->imaginary           += theta_complex.imaginary;
		}
	}
}
