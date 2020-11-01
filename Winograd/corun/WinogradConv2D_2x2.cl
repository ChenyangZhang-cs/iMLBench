#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

__kernel void WinogradConv2D_2x2_kernel (__global DATA_TYPE *input, 
                                        __global DATA_TYPE *output, 
                                        __global DATA_TYPE *transformed_filter, 
                                        int N, int out_map_size) {
    // int tile_j = get_global_id(0);
	// int tile_i = get_global_id(1);
    int tile_j = get_global_id(1);
	int tile_i = get_global_id(0);

    // input transformation

    DATA_TYPE input_tile[4][4], tmp_tile[4][4], transformed_tile[4][4];
    for (int i = 0; i < 4; i ++) {
        for (int j = 0; j < 4; j ++) { 
            int x = 2 * tile_i + i;
            int y = 2 * tile_j + j;
            if (x >= N || y >= N) {
                input_tile[i][j] = 0;
                continue;
            }
            input_tile[i][j] = input[x * N + y];
        }
    } 

    // const float Bt[4][4] = {
    //     {1.0f, 0.0f, -1.0f, 0.0f},
    //     {0.0f, 1.0f, 1.0f, 0.0f},
    //     {0.0f, -1.0f, 1.0f, 0.0f},
    //     {0.0f, 1.0f, 0.0f, -1.0f}
    // }

    // Bt * d
    for (int j = 0; j < 4; j ++) {
        tmp_tile[0][j] = input_tile[0][j] - input_tile[2][j];
        tmp_tile[1][j] = input_tile[1][j] + input_tile[2][j];
        tmp_tile[2][j] = -input_tile[1][j] + input_tile[2][j];
        tmp_tile[3][j] = input_tile[1][j] - input_tile[3][j];
    }
    // d * B
    for (int i = 0; i < 4; i ++) {
        transformed_tile[i][0] = tmp_tile[i][0] - tmp_tile[i][2];
        transformed_tile[i][1] = tmp_tile[i][1] + tmp_tile[i][2];
        transformed_tile[i][2] = -tmp_tile[i][1] + tmp_tile[i][2];
        transformed_tile[i][3] = tmp_tile[i][1] - tmp_tile[i][3];
    }

    // element-wise multiplication

    DATA_TYPE multiplied_tile[4][4];
    for (int i = 0; i < 4; i ++) {
        for (int j = 0; j < 4; j ++) {
            multiplied_tile[i][j] = transformed_tile[i][j] * transformed_filter[i * 4 + j];
        }
    }

    // output transformation

    DATA_TYPE tmp_tile_1[2][4], final_tile[2][2];

    // const float At[2][4] {
    //     {1.0f, 1.0f, 1.0f, 0.0f},
    //     {0.0f, 1.0f, -1.0f, -1.0f}
    // }

    // At * I
    for (int j = 0; j < 4; j ++) {
        tmp_tile_1[0][j] = multiplied_tile[0][j] + multiplied_tile[1][j] + multiplied_tile[2][j];
        tmp_tile_1[1][j] = multiplied_tile[1][j] - multiplied_tile[2][j] - multiplied_tile[3][j];
    }
    // I * A
    for (int i = 0; i < 2; i ++) {
        final_tile[i][0] = tmp_tile_1[i][0] + tmp_tile_1[i][1] + tmp_tile_1[i][2];
        final_tile[i][1] = tmp_tile_1[i][1] - tmp_tile_1[i][2] - tmp_tile_1[i][3];
    }

    for (int i = 0; i < 2; i ++) {
        for (int j = 0; j < 2; j ++) {
            int x = 2 * tile_i + i;
            int y = 2 * tile_j + j;
            if (x >= out_map_size || y >= out_map_size) {
                continue;
            }
            output[x * out_map_size + y] = final_tile[i][j];
        }
    }
} 