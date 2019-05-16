#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <float.h>

#include "dft.h"

// Initializes the configuration of the algorithm
void init_config(Config *config)
{
	// Number of sources to synthesize if no file provided.
	config->num_sources = 1;

    // Number of visibilities per source to synthesize
    // if no file provided.
    config->num_visibilities = 10000;

    // Disregard visibility w coordinate during transformation
    config->force_zero_w_term = false;

    // Use fixed sources (not from file)
    config->synthetic_sources = false;

    // Use fixed visibilities (not from file)
    config->synthetic_visibilities = false;

    // if using synthetic visibility creation, set this flag to
    // Gaussian distribute random visibility positions
    config->gaussian_distribution_sources = false;

	// Cache File for Sources
	// File format : first row = number of sources in file
	// subsequent rows = each unique source in the form:
	// l, m, intensity
	// note: data can be either single or double precision
	config->source_file = "../data/example_sources.txt";

	// Cache File for Visibilities
	// File format : first row = number of visibilities in file
	// subsequent rows = each unique visibility in the form:
	// u, v, w, brightness (real), brightness (imag), intensity
	// note: data can be either single or double precision
	config->vis_src_file    = "../data/example_visibilities.txt";

    // Destination for processed visibilities
    config->vis_dest_file 	= "../data/vis_output_test.csv";

	// Dimension of Fourier domain grid
	config->grid_size = 1024;

	// Fourier domain grid cell size in radians
	config->cell_size = 4.848136811095360e-06;

	// Frequency of visibility uvw terms
	config->frequency_hz = 300e6;

	// Scalar for visibility coordinates
	config->uv_scale = config->grid_size * config->cell_size;

	// Range for visibility u coordinates for synthesizing data
	config->min_u = -(config->grid_size / 2.0);
	config->max_u = config->grid_size / 2.0;

	// Range for visibility v coordinates for synthesizing data
	config->min_v = -(config->grid_size / 2.0);
	config->max_v = config->grid_size / 2.0;

	// Range for visibility w coordinates for synthesizing data
	config->min_w = config->min_v / 10; // fixed to v coordinates for now
	config->max_w = config->max_v / 10; // fixed to v coordinates for now

    // Number of CUDA blocks (gpu specific)
    config->gpu_num_blocks = 32;

    // Number of CUDA threads per block (updated when reading vis from file)
    config->gpu_num_threads = config->num_visibilities / config->gpu_num_blocks;

    // Enables/disables the printing of information during DFT
    config->enable_messages = true;

}

// Loads sources into memory from some source file, or generates
// synthetic sources for testing purposes.
void load_sources(Config *config, Source **sources)
{
	if(config->synthetic_sources)
	{
        if(config->enable_messages)
            printf(">>> UPDATE: Using synthetic Sources...\n\n");

        *sources = (Source*) calloc(config->num_sources, sizeof(Source));
        if(*sources == NULL)
		{
			printf(">>> ERROR: Unable to allocate memory for sources...\n\n");
			return;
		}

		// synthesize n sources
		for(int src_indx = 0; src_indx < config->num_sources; ++src_indx)
		{
			(*sources)[src_indx] = (Source) {
				.l = random_in_range(config->min_u,config->max_u) * config->cell_size,
				.m = random_in_range(config->min_v,config->max_v) * config->cell_size,
				.intensity = 1.0  // fixed intensity for testing purposes
			};
		}

        if(config->enable_messages)
            printf(">>> UPDATE: Successfully loaded %d synthetic sources..\n\n",config->num_sources);
	}
	else // Using sources from file
	{
        if(config->enable_messages)
            printf(">>> UPDATE: Using Sources from file...\n\n");

		FILE *file = fopen(config->source_file, "r");
		if(file == NULL)
		{
			printf(">>> ERROR: Unable to locate sources file...\n\n");
			return;
		}

		// Reading in the counter for number of sources
		fscanf(file, "%d\n", &(config->num_sources));
        *sources = (Source*) calloc(config->num_sources, sizeof(Source));

		if(*sources == NULL) 
		{
			printf(">>> ERROR: Unable to allocate memory for sources...\n\n");
			fclose(file);
			return;
		}

        PRECISION temp_l = 0.0;
        PRECISION temp_m = 0.0;
        PRECISION temp_intensity = 0.0;

		// Read in n number of sources
		for(int src_indx = 0; src_indx < config->num_sources; ++src_indx)
		{
			// Read in provided source attributes 
            fscanf(file, "%lf %lf %lf\n", &temp_l, &temp_m, &temp_intensity);

            (*sources)[src_indx] = (Source) {
                    .l = temp_l * config->cell_size,
                    .m = temp_m * config->cell_size,
                    .intensity = temp_intensity
            };
		}

		// Clean up
		fclose(file);
        if(config->enable_messages)
            printf(">>> UPDATE: Successfully loaded %d sources from file..\n\n",config->num_sources);	}
}

// Loads visibilities into memory from some source file, or generates
// synthetic visibilities for testing purposes. Gaussian distribution
// is optional.
void load_visibilities(Config *config, Visibility **visibilities, Complex **vis_intensity)
{
    if(config->synthetic_visibilities)
    {
        if(config->enable_messages)
            printf(">>> UPDATE: Using synthetic Visibilities...\n\n");

        *visibilities =  (Visibility*) calloc(config->num_visibilities, sizeof(Visibility));
        if(*visibilities == NULL)  return;

        *vis_intensity =  (Complex*) calloc(config->num_visibilities, sizeof(Complex));
        if(*vis_intensity == NULL)
        {
            if(*visibilities) free(*visibilities);
            return;
        }

        PRECISION gaussian_u = 1.0;
        PRECISION gaussian_v = 1.0;
        PRECISION gaussian_w = 1.0;

        //try randomize visibilities in the center of the grid
        for(int vis_indx = 0; vis_indx < config->num_visibilities; ++vis_indx)
        {
            if(config->gaussian_distribution_sources)
            {
                gaussian_u = generate_sample_normal();
                gaussian_v = generate_sample_normal();
                gaussian_w = generate_sample_normal();
            }

            PRECISION u = random_in_range(config->min_u, config->max_u) * gaussian_u;
            PRECISION v = random_in_range(config->min_v, config->max_v) * gaussian_v;
            PRECISION w = random_in_range(config->min_w, config->max_w) * gaussian_w;

            (*visibilities)[vis_indx] = (Visibility) {
                    .u = u / config->uv_scale,
                    .v = v / config->uv_scale,
                    .w = (config->force_zero_w_term) ? 0.0 : w / config->uv_scale
            };
        }
    }
    else // Reading visibilities from file
    {
        if(config->enable_messages)
            printf(">>> UPDATE: Using Visibilities from file...\n\n");

        FILE *file = fopen(config->vis_src_file, "r");
        if(file == NULL)
        {
            printf(">>> ERROR: Unable to locate visibilities file...\n\n");
            return;
        }

        // Reading in the counter for number of visibilities
        fscanf(file, "%d\n", &(config->num_visibilities));
        // Update gpu threads based on new number of visibilities (non-default)
        config->gpu_num_threads = config->num_visibilities / config->gpu_num_blocks;

        *visibilities = (Visibility*) calloc(config->num_visibilities, sizeof(Visibility));
        *vis_intensity =  (Complex*) calloc(config->num_visibilities, sizeof(Complex));

        // File found, but was memory allocated?
        if(*visibilities == NULL || *vis_intensity == NULL)
        {
            printf(">>> ERROR: Unable to allocate memory for visibilities...\n\n");
            if(file) fclose(file);
            if(*visibilities) free(*visibilities);
            if(*vis_intensity) free(*vis_intensity);
            return;
        }

        double u = 0.0;
        double v = 0.0;
        double w = 0.0;
        Complex brightness;
        double intensity = 0.0;

        // Used to scale visibility coordinates from wavelengths
        // to meters
        double wavelength_to_meters = config->frequency_hz / C;

        // Read in n number of visibilities
        for(int vis_indx = 0; vis_indx < config->num_visibilities; ++vis_indx)
        {
            // Read in provided visibility attributes
            // u, v, w, brightness (real), brightness (imag), intensity
            fscanf(file, "%lf %lf %lf %lf %lf %lf\n", &u, &v, &w,
                   &(brightness.real), &(brightness.imaginary), &intensity);

            (*visibilities)[vis_indx] = (Visibility) {
                    .u = u * wavelength_to_meters,
                    .v = v * wavelength_to_meters,
                    .w = (config->force_zero_w_term) ? 0.0 : w * wavelength_to_meters
            };
        }

        // Clean up
        fclose(file);
        if(config->enable_messages)
            printf(">>> UPDATE: Successfully loaded %d visibilities from file...\n\n",config->num_visibilities);
    }
}


// Saves the extracted visibility data to file
// note: file format is first row is the number of visibilities
// every subsequent row represents a unique visibility in the
// form (u, v, w, brightness (real), brightness (imag), intensity)
void save_visibilities(Config *config, Visibility *visibilities, Complex *vis_intensity)
{
    // Save visibilities to file
    FILE *file = fopen(config->vis_dest_file, "w");
    // Unable to open file
    if(file == NULL)
    {
        printf(">>> ERROR: Unable to save visibilities to file...\n\n");
        return;
    }

    if(config->enable_messages)
        printf(">>> UPDATE: Writing visibilities to file...\n\n");

    // Record number of visibilities
    fprintf(file, "%d\n", config->num_visibilities);

    // Used to scale visibility coordinates from meters to
    // wavelengths (useful for gridding, inverse DFT etc.)
    double meters_to_wavelengths = config->frequency_hz / C;

    // Record individual visibilities
    for(int vis_indx = 0; vis_indx < config->num_visibilities; ++vis_indx)
    {
        // u, v, w, real, imag, weight (intensity)
        fprintf(file, "%f %f %f %f %f %f\n",
                visibilities[vis_indx].u / meters_to_wavelengths,
                visibilities[vis_indx].v / meters_to_wavelengths,
                visibilities[vis_indx].w / meters_to_wavelengths,
                vis_intensity[vis_indx].real,
                vis_intensity[vis_indx].imaginary,
                1.0); // static intensity (for now)
    }

    // Clean up
    fclose(file);
    if(config->enable_messages)
        printf(">>> UPDATE: Completed writing of visibilities to file...\n\n");
}

// Generate a random double in range [min, max)
PRECISION random_in_range(PRECISION min, PRECISION max)
{
    PRECISION range = (max - min);
    PRECISION div = RAND_MAX / range;
    return min + (rand() / div);
}

// Generate a gaussian distribution multiplier
// note: used to better "sample" synthetic visibilities
// u, v coordinates instead of randomly distributed
PRECISION generate_sample_normal()
{
    PRECISION u = ((PRECISION) rand() / RAND_MAX) * 2.0 - 1.0;
    PRECISION v = ((PRECISION) rand() / RAND_MAX) * 2.0 - 1.0;
    PRECISION r = u * u + v * v;
    if(r <= 0.0 || r > 1.0)
        return generate_sample_normal();
    return u * sqrt(-2.0 * log(r) / r);
}
