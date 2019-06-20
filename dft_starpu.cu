#include <starpu.h>
#include "dft.h"

static int num_threads_per_block;

void cpu_extract_vis_codelet(void *descr[], STARPU_ATTRIBUTE_UNUSED void *arg){

    Source *sources = (Source*) STARPU_VECTOR_GET_PTR(descr[0]);
    Visibility *vis  = (Visibility*) STARPU_VECTOR_GET_PTR(descr[1]);
    Complex *vis_intensity = (Complex*) STARPU_VECTOR_GET_PTR(descr[2]);

    int num_sources = STARPU_VECTOR_GET_NX(descr[0]);
    int num_visibilities = STARPU_VECTOR_GET_NX(descr[1]);

    extract_visibilities_cpu(sources,vis,vis_intensity,num_sources,num_visibilities);
}

void cuda_extract_vis_codelet(void *descr[], STARPU_ATTRIBUTE_UNUSED void *arg){

    PRECISION3 *sources = (PRECISION3*) STARPU_VECTOR_GET_PTR(descr[0]);
    PRECISION3 *vis  = (PRECISION3*) STARPU_VECTOR_GET_PTR(descr[1]);
    PRECISION2 *vis_intensity = (PRECISION2*) STARPU_VECTOR_GET_PTR(descr[2]);

    int num_sources = STARPU_VECTOR_GET_NX(descr[0]);
    int num_visibilities = STARPU_VECTOR_GET_NX(descr[1]);

    dim3 kernel_blocks(num_threads_per_block, 1, 1);
    dim3 kernel_threads((num_visibilities + num_threads_per_block -1)/num_threads_per_block, 1, 1);

    direct_fourier_transform<<<kernel_threads, kernel_blocks,0,starpu_cuda_get_local_stream()>>>(vis, vis_intensity, num_visibilities, sources, num_sources);
    cudaStreamSynchronize(starpu_cuda_get_local_stream());
} 

int starpu_launch(Config *config, Source *sources, Visibility *visibilities,Complex *vis_intensity){
    
    num_threads_per_block = config-> gpu_num_threads_per_block;
    /* start the runtime */
    int ret;
    ret = starpu_init(NULL);
    if (ret == -ENODEV)
        return 77;
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

    struct starpu_perfmodel extract_perf_model;
    struct starpu_codelet cl;
    starpu_data_handle_t source_handle, vis_handle, intensity_handle;

    ret = starpu_memory_pin(sources, config->num_sources*sizeof(sources[0]));
    ret += starpu_memory_pin(visibilities, config->num_visibilities*sizeof(visibilities[0]));
    ret += starpu_memory_pin(vis_intensity,config->num_visibilities*sizeof(vis_intensity[0]));
    if (ret !=0 ) return 77;

    //define perfmodel
    starpu_perfmodel_init(&extract_perf_model);
    extract_perf_model.type = STARPU_HISTORY_BASED;
    extract_perf_model.symbol = "extract_perf_model";

    //define codelet
    starpu_codelet_init(&cl);
    cl.cpu_funcs[0] = cpu_extract_vis_codelet;
    cl.cpu_funcs_name[0] = "cpu_extract_vis";
    cl.cuda_funcs[0] = cuda_extract_vis_codelet;   //cuda codelet
    cl.nbuffers = 3;
    cl.modes[0] = STARPU_R;
    cl.modes[1] = STARPU_R;
    cl.modes[2] = STARPU_W;
    //cl.model = &extract_perf_model;

    //register data
    starpu_vector_data_register(&source_handle, STARPU_MAIN_RAM, (uintptr_t)sources, config->num_sources, sizeof(sources[0]));
    starpu_vector_data_register(&vis_handle, STARPU_MAIN_RAM, (uintptr_t)visibilities, config->num_visibilities, sizeof(visibilities[0]));
    starpu_vector_data_register(&intensity_handle, STARPU_MAIN_RAM, (uintptr_t)vis_intensity, config->num_visibilities, sizeof(vis_intensity[0]));

    struct starpu_data_filter horiz;
    memset(&horiz, 0, sizeof(horiz));
    horiz.filter_func = starpu_vector_filter_block;
    horiz.nchildren = config->num_task_slices;

    starpu_data_partition(vis_handle, &horiz);
    starpu_data_partition(intensity_handle, &horiz);

    for (int i = 0; i < horiz.nchildren; i++)
    {
        struct starpu_task *task = starpu_task_create();
        task->synchronous = 0;
        task->cl = &cl;

        task->handles[0] = source_handle;
        task->handles[1] = starpu_data_get_sub_data(vis_handle, 1, i);
        task->handles[2] = starpu_data_get_sub_data(intensity_handle,1,i);

        ret = starpu_task_submit(task);
        if (ret == -ENODEV)
        {
            ret = 77;
        }
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
        starpu_data_wont_use(starpu_data_get_sub_data(intensity_handle,1,i));
    }

    ret = starpu_task_wait_for_all(); 
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");

    starpu_data_unpartition(vis_handle, STARPU_MAIN_RAM);
    starpu_data_unpartition(intensity_handle, STARPU_MAIN_RAM);

    ret = starpu_memory_unpin(sources, config->num_sources*sizeof(sources[0]));
    ret += starpu_memory_unpin(visibilities, config->num_visibilities*sizeof(visibilities[0]));
    ret += starpu_memory_unpin(vis_intensity,config->num_visibilities*sizeof(vis_intensity[0]));
    if (ret !=0 ) return 77;

    starpu_data_unregister(source_handle);
    starpu_data_unregister(vis_handle);
    starpu_data_unregister(intensity_handle);

    starpu_shutdown();
}

