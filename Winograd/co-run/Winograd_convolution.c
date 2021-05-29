#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

double rtclock() {
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0)
        printf("Error return from gettimeofday: %d", stat);
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

// F(2x2,3x3)

void winograd_GgGt_2x2(float* input, float* output, int K, int C) {
    int total_filter = K * C;
    int in_c_stride = 9, in_k_stride = in_c_stride * C;
    int out_c_stride = 16, out_k_stride = out_c_stride * C;

#pragma omp parallel for
    for (int global_id = 0; global_id < total_filter; global_id++) {
        int k = global_id / C;
        int c = global_id % C;

        float tile[3][3], t_tile[4][3], f_tile[4][4];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                tile[i][j] = input[in_k_stride * k + in_c_stride * c + 3 * i + j];
            }
        }
=
        // G * g
        for (int j = 0; j < 3; j++) {
            t_tile[0][j] = tile[0][j];
            t_tile[1][j] = 0.5f * tile[0][j] + 0.5f * tile[1][j] + 0.5f * tile[2][j];
            t_tile[2][j] = 0.5f * tile[0][j] - 0.5f * tile[1][j] + 0.5f * tile[2][j];
            t_tile[3][j] = tile[2][j];
        }
        // g * Gt
        for (int i = 0; i < 4; i++) {
            f_tile[i][0] = t_tile[i][0];
            f_tile[i][1] = 0.5f * t_tile[i][0] + 0.5f * t_tile[i][1] + 0.5f * t_tile[i][2];
            f_tile[i][2] = 0.5f * t_tile[i][0] - 0.5f * t_tile[i][1] + 0.5f * t_tile[i][2];
            f_tile[i][3] = t_tile[i][2];
        }

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                output[out_k_stride * k + out_c_stride * c + 4 * i + j] = f_tile[i][j];
            }
        }
    }
}

void winograd_BtdB_2x2(float* input, float* output, int batch_size, int C, int tile_n, int map_size) {
    int total_tile = batch_size * C * tile_n * tile_n;
    int in_n_stride = map_size * map_size * C, in_c_stride = map_size * map_size, x_stride = map_size, y_stride = 1;
    int out_n_stride = tile_n * tile_n * 16 * C, out_c_stride = tile_n * tile_n * 16;
    int tilei_stride = tile_n * 16, tilej_stride = 16;

#pragma omp parallel for
    for (int global_id = 0; global_id < total_tile; global_id++) {
        int n = global_id / (C * tile_n * tile_n);
        int remain = global_id % (C * tile_n * tile_n);
        int c = remain / (tile_n * tile_n);
        remain = remain % (tile_n * tile_n);
        int tile_i = remain / tile_n;
        int tile_j = remain % tile_n;

        float tile[4][4], t_tile[4][4];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                int x = 2 * tile_i + i;
                int y = 2 * tile_j + j;
                if (x >= map_size || y >= map_size) {
                    tile[i][j] = 0;
                    continue;
                }
                tile[i][j] = input[n * in_n_stride + c * in_c_stride + x * x_stride + y * y_stride];
            }
        }

        // const float Bt[4][4] = {
        //     {1.0f, 0.0f, -1.0f, 0.0f},
        //     {0.0f, 1.0f, 1.0f, 0.0f},
        //     {0.0f, -1.0f, 1.0f, 0.0f},
        //     {0.0f, 1.0f, 0.0f, -1.0f}
        // }

        // Bt * d
        for (int j = 0; j < 4; j++) {
            t_tile[0][j] = tile[0][j] - tile[2][j];
            t_tile[1][j] = tile[1][j] + tile[2][j];
            t_tile[2][j] = -tile[1][j] + tile[2][j];
            t_tile[3][j] = tile[1][j] - tile[3][j];
        }
        // d * B
        for (int i = 0; i < 4; i++) {
            tile[i][0] = t_tile[i][0] - t_tile[i][2];
            tile[i][1] = t_tile[i][1] + t_tile[i][2];
            tile[i][2] = -t_tile[i][1] + t_tile[i][2];
            tile[i][3] = t_tile[i][1] - t_tile[i][3];
        }

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                output[n * out_n_stride + c * out_c_stride + tile_i * tilei_stride + tile_j * tilej_stride + 4 * i + j] = tile[i][j];
            }
        }
    }
}

void winograd_BtdB_padding_2x2(float* input, float* output, int batch_size, int C, int tile_n, int map_size) {
    int total_tile = batch_size * C * tile_n * tile_n;
    int in_n_stride = map_size * map_size * C, in_c_stride = map_size * map_size, x_stride = map_size, y_stride = 1;
    int out_n_stride = tile_n * tile_n * 16 * C, out_c_stride = tile_n * tile_n * 16;
    int tilei_stride = tile_n * 16, tilej_stride = 16;

#pragma omp parallel for
    for (int global_id = 0; global_id < total_tile; global_id++) {
        int n = global_id / (C * tile_n * tile_n);
        int remain = global_id % (C * tile_n * tile_n);
        int c = remain / (tile_n * tile_n);
        remain = remain % (tile_n * tile_n);
        int tile_i = remain / tile_n;
        int tile_j = remain % tile_n;

        float tile[4][4], t_tile[4][4];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                int x = 2 * tile_i + i;
                int y = 2 * tile_j + j;
                if (x == 0 || y == 0 || x >= (map_size + 1) || y >= (map_size + 1)) {
                    tile[i][j] = 0;
                } else {
                    tile[i][j] = input[n * in_n_stride + c * in_c_stride + (x - 1) * x_stride + (y - 1) * y_stride];
                }
            }
        }

        // const float Bt[4][4] = {
        //     {1.0f, 0.0f, -1.0f, 0.0f},
        //     {0.0f, 1.0f, 1.0f, 0.0f},
        //     {0.0f, -1.0f, 1.0f, 0.0f},
        //     {0.0f, 1.0f, 0.0f, -1.0f}
        // }

        // Bt * d
        for (int j = 0; j < 4; j++) {
            t_tile[0][j] = tile[0][j] - tile[2][j];
            t_tile[1][j] = tile[1][j] + tile[2][j];
            t_tile[2][j] = -tile[1][j] + tile[2][j];
            t_tile[3][j] = tile[1][j] - tile[3][j];
        }
        // d * B
        for (int i = 0; i < 4; i++) {
            tile[i][0] = t_tile[i][0] - t_tile[i][2];
            tile[i][1] = t_tile[i][1] + t_tile[i][2];
            tile[i][2] = -t_tile[i][1] + t_tile[i][2];
            tile[i][3] = t_tile[i][1] - t_tile[i][3];
        }

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                output[n * out_n_stride + c * out_c_stride + tile_i * tilei_stride + tile_j * tilej_stride + 4 * i + j] = tile[i][j];
            }
        }
    }
}

void winograd_outerProduct_AtIA_2x2(float* input, float* weight, float* bias, float* output, int batch_size, int K, int tile_n, int out_map_size, int C) {
    int total_tile = batch_size * K * tile_n * tile_n;
    int c_stride = tile_n * tile_n * 16, in_n_stride = C * c_stride;
    int tilei_stride = tile_n * 16, tilej_stride = 16;
    int w_c_stride = 16, w_k_stride = C * 16;
    int out_k_stride = out_map_size * out_map_size, out_n_stride = out_k_stride * K;
    int x_stride = out_map_size, y_stride = 1;

#pragma omp parallel for
    for (int global_id = 0; global_id < total_tile; global_id++) {
        int n = global_id / (K * tile_n * tile_n);
        int remain = global_id % (K * tile_n * tile_n);
        int k = remain / (tile_n * tile_n);
        remain = remain % (tile_n * tile_n);
        int tile_i = remain / tile_n;
        int tile_j = remain % tile_n;

        float tile[4][4] = {0};
        for (int c = 0; c < C; c++) {
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    tile[i][j] += input[n * in_n_stride + c * c_stride + tile_i * tilei_stride + tile_j * tilej_stride + 4 * i + j] * weight[k * w_k_stride + c * w_c_stride + 4 * i + j];
                }
            }
        }

        // const float At[2][4] {
        //     {1.0f, 1.0f, 1.0f, 0.0f},
        //     {0.0f, 1.0f, -1.0f, -1.0f}
        // }

        float t_tile[2][4], f_tile[2][2];
        // At * I
        for (int j = 0; j < 4; j++) {
            t_tile[0][j] = tile[0][j] + tile[1][j] + tile[2][j];
            t_tile[1][j] = tile[1][j] - tile[2][j] - tile[3][j];
        }
        // I * A
        for (int i = 0; i < 2; i++) {
            f_tile[i][0] = t_tile[i][0] + t_tile[i][1] + t_tile[i][2];
            f_tile[i][1] = t_tile[i][1] - t_tile[i][2] - t_tile[i][3];
        }
        // bias
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                f_tile[i][j] += bias[k];
            }
        }
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                int x = 2 * tile_i + i;
                int y = 2 * tile_j + j;
                if (x >= out_map_size || y >= out_map_size) {
                    continue;
                }
                output[n * out_n_stride + k * out_k_stride + x * x_stride + y * y_stride] = f_tile[i][j];
            }
        }
    }
}

void winograd_convolution_2x2(float* input,  /* NxCxHxW */
                              float* weight, /* KxCx3x3 */
                              float* bias,   /* K */
                              float* my_res, /* NxKxH'xW'*/
                              int batch_size,
                              int C,
                              int K,
                              int map_size,
                              int padding) {
    // filter transformation
    float* trans_filter = (float*)malloc(K * C * 16 * sizeof(float));  // transformed filters
    if (trans_filter == NULL) {
        printf("bad malloc trans_filter\n");
    }
    winograd_GgGt_2x2(weight, trans_filter, K, C);

    int out_map_size = (map_size + padding * 2) - 2;  // kernel size = 3, stride = 1 in Winograd algorithm
    int tile_n = (out_map_size + 1) / 2;

    float* trans_input = (float*)malloc(batch_size * tile_n * tile_n * C * 16 * sizeof(float));  // transformed input
    if (trans_input == NULL) {
        printf("bad malloc trans_input\n");
    }

    // input transformation
    if (padding == 0) {
        winograd_BtdB_2x2(input, trans_input, batch_size, C, tile_n, map_size);
    } else if (padding == 1) {
        winograd_BtdB_padding_2x2(input, trans_input, batch_size, C, tile_n, map_size);
    }

    // element-wise multiplication & output transformation
    winograd_outerProduct_AtIA_2x2(trans_input, trans_filter, bias, my_res, batch_size, K, tile_n, out_map_size, C);

    free(trans_input);
    free(trans_filter);
    return;
}

// F(4x4,3x3)

void winograd_GgGt_4x4(float* input, float* output, int K, int C) {
    int total_filter = K * C;
    int in_c_stride = 9, in_k_stride = in_c_stride * C;
    int out_c_stride = 36, out_k_stride = out_c_stride * C;

#pragma omp parallel for
    for (int global_id = 0; global_id < total_filter; global_id++) {
        int k = global_id / C;
        int c = global_id % C;

        float tile[3][3], t_tile[6][3], f_tile[6][6];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                tile[i][j] = input[in_k_stride * k + in_c_stride * c + 3 * i + j];
            }
        }

        // const float G[6][3] = {
        //     {0.25f, 0.0f, 0.0f},

        //     {-1.0f/6, -1.0f/6, -1.0f/6},
        //     {-1.0f/6, 1.0f/6, -1.0f/6},

        //     {1.0f/24, 1.0f/12, 1.0f/6},
        //     {1.0f/24, -1.0f/12, 1.0f/6},

        //     {0.0f, 0.0f, 1.0f}
        // }

        // G * g
        for (int j = 0; j < 3; j++) {
            t_tile[0][j] = 0.25f * tile[0][j];

            t_tile[1][j] = -1.0f / 6 * tile[0][j] - 1.0f / 6 * tile[1][j] - 1.0f / 6 * tile[2][j];
            t_tile[2][j] = -1.0f / 6 * tile[0][j] + 1.0f / 6 * tile[1][j] - 1.0f / 6 * tile[2][j];

            t_tile[3][j] = 1.0f / 24 * tile[0][j] + 1.0f / 12 * tile[1][j] + 1.0f / 6 * tile[2][j];
            t_tile[4][j] = 1.0f / 24 * tile[0][j] - 1.0f / 12 * tile[1][j] + 1.0f / 6 * tile[2][j];

            t_tile[5][j] = tile[2][j];
        }
        // g * Gt
        for (int i = 0; i < 6; i++) {
            f_tile[i][0] = 0.25f * t_tile[i][0];

            f_tile[i][1] = -1.0f / 6 * t_tile[i][0] - 1.0f / 6 * t_tile[i][1] - 1.0f / 6 * t_tile[i][2];
            f_tile[i][2] = -1.0f / 6 * t_tile[i][0] + 1.0f / 6 * t_tile[i][1] - 1.0f / 6 * t_tile[i][2];

            f_tile[i][3] = 1.0f / 24 * t_tile[i][0] + 1.0f / 12 * t_tile[i][1] + 1.0f / 6 * t_tile[i][2];
            f_tile[i][4] = 1.0f / 24 * t_tile[i][0] - 1.0f / 12 * t_tile[i][1] + 1.0f / 6 * t_tile[i][2];

            f_tile[i][5] = t_tile[i][2];
        }

        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                output[out_k_stride * k + out_c_stride * c + 6 * i + j] = f_tile[i][j];
            }
        }
    }
}

void winograd_BtdB_4x4(float* input, float* output, int batch_size, int C, int tile_n, int map_size) {
    int total_tile = batch_size * C * tile_n * tile_n;
    int in_n_stride = map_size * map_size * C, in_c_stride = map_size * map_size, x_stride = map_size, y_stride = 1;
    int out_n_stride = tile_n * tile_n * 36 * C, out_c_stride = tile_n * tile_n * 36;
    int tilei_stride = tile_n * 36, tilej_stride = 36;

#pragma omp parallel for
    for (int global_id = 0; global_id < total_tile; global_id++) {
        int n = global_id / (C * tile_n * tile_n);
        int remain = global_id % (C * tile_n * tile_n);
        int c = remain / (tile_n * tile_n);
        remain = remain % (tile_n * tile_n);
        int tile_i = remain / tile_n;
        int tile_j = remain % tile_n;

        float tile[6][6], t_tile[6][6];
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                int x = 4 * tile_i + i;
                int y = 4 * tile_j + j;
                if (x >= map_size || y >= map_size) {
                    tile[i][j] = 0;
                    continue;
                }
                tile[i][j] = input[n * in_n_stride + c * in_c_stride + x * x_stride + y * y_stride];
            }
        }

        // const float Bt[6][6] = {
        //     {4.0f, 0.0f, -5.0f, 0.0f, 1.0f, 0.0f},

        //     {0.0f, -4.0f, -4.0f, 1.0f, 1.0f, 0.0f},
        //     {0.0f, 4.0f, -4.0f, -1.0f, 1.0f, 0.0f},

        //     {0.0f, -2.0f, -1.0f, 2.0f, 1.0f, 0.0f},
        //     {0.0f, 2.0f, -1.0f, -2.0f, 1.0f, 0.0f},

        //     {0.0f, 4.0f, 0.0f, -5.0f, 0.0f, 1.0f}
        // }

        // Bt * d
        for (int j = 0; j < 6; j++) {
            t_tile[0][j] = 4.0f * tile[0][j] - 5.0f * tile[2][j] + tile[4][j];

            t_tile[1][j] = -4.0f * tile[1][j] - 4.0f * tile[2][j] + tile[3][j] + tile[4][j];
            t_tile[2][j] = 4.0f * tile[1][j] - 4.0f * tile[2][j] - tile[3][j] + tile[4][j];

            t_tile[3][j] = -2.0f * tile[1][j] - tile[2][j] + 2.0f * tile[3][j] + tile[4][j];
            t_tile[4][j] = 2.0f * tile[1][j] - tile[2][j] - 2.0f * tile[3][j] + tile[4][j];

            t_tile[5][j] = 4.0f * tile[1][j] - 5.0f * tile[3][j] + tile[5][j];
        }
        // d * B
        for (int i = 0; i < 6; i++) {
            tile[i][0] = 4.0f * t_tile[i][0] - 5.0f * t_tile[i][2] + t_tile[i][4];

            tile[i][1] = -4.0f * t_tile[i][1] - 4.0f * t_tile[i][2] + t_tile[i][3] + t_tile[i][4];
            tile[i][2] = 4.0f * t_tile[i][1] - 4.0f * t_tile[i][2] - t_tile[i][3] + t_tile[i][4];

            tile[i][3] = -2.0f * t_tile[i][1] - t_tile[i][2] + 2.0f * t_tile[i][3] + t_tile[i][4];
            tile[i][4] = 2.0f * t_tile[i][1] - t_tile[i][2] - 2.0f * t_tile[i][3] + t_tile[i][4];

            tile[i][5] = 4.0f * t_tile[i][1] - 5.0f * t_tile[i][3] + t_tile[i][5];
        }

        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                output[n * out_n_stride + c * out_c_stride + tile_i * tilei_stride + tile_j * tilej_stride + 6 * i + j] = tile[i][j];
            }
        }
    }
}

void winograd_BtdB_padding_4x4(float* input, float* output, int batch_size, int C, int tile_n, int map_size) {
    int total_tile = batch_size * C * tile_n * tile_n;
    int in_n_stride = map_size * map_size * C, in_c_stride = map_size * map_size, x_stride = map_size, y_stride = 1;
    int out_n_stride = tile_n * tile_n * 36 * C, out_c_stride = tile_n * tile_n * 36;
    int tilei_stride = tile_n * 36, tilej_stride = 36;

#pragma omp parallel for
    for (int global_id = 0; global_id < total_tile; global_id++) {
        int n = global_id / (C * tile_n * tile_n);
        int remain = global_id % (C * tile_n * tile_n);
        int c = remain / (tile_n * tile_n);
        remain = remain % (tile_n * tile_n);
        int tile_i = remain / tile_n;
        int tile_j = remain % tile_n;

        float tile[6][6], t_tile[6][6];
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                int x = 4 * tile_i + i;
                int y = 4 * tile_j + j;
                if (x == 0 || y == 0 || x >= (map_size + 1) || y >= (map_size + 1)) {
                    tile[i][j] = 0;
                } else {
                    tile[i][j] = input[n * in_n_stride + c * in_c_stride + (x - 1) * x_stride + (y - 1) * y_stride];
                }
            }
        }

        // const float Bt[6][6] = {
        //     {4.0f, 0.0f, -5.0f, 0.0f, 1.0f, 0.0f},

        //     {0.0f, -4.0f, -4.0f, 1.0f, 1.0f, 0.0f},
        //     {0.0f, 4.0f, -4.0f, -1.0f, 1.0f, 0.0f},

        //     {0.0f, -2.0f, -1.0f, 2.0f, 1.0f, 0.0f},
        //     {0.0f, 2.0f, -1.0f, -2.0f, 1.0f, 0.0f},

        //     {0.0f, 4.0f, 0.0f, -5.0f, 0.0f, 1.0f}
        // }

        // Bt * d
        for (int j = 0; j < 6; j++) {
            t_tile[0][j] = 4.0f * tile[0][j] - 5.0f * tile[2][j] + tile[4][j];
            t_tile[1][j] = -4.0f * tile[1][j] - 4.0f * tile[2][j] + tile[3][j] + tile[4][j];
            t_tile[2][j] = 4.0f * tile[1][j] - 4.0f * tile[2][j] - tile[3][j] + tile[4][j];
            t_tile[3][j] = -2.0f * tile[1][j] - tile[2][j] + 2.0f * tile[3][j] + tile[4][j];
            t_tile[4][j] = 2.0f * tile[1][j] - tile[2][j] - 2.0f * tile[3][j] + tile[4][j];
            t_tile[5][j] = 4.0f * tile[1][j] - 5.0f * tile[3][j] + tile[5][j];
        }
        // d * B
        for (int i = 0; i < 6; i++) {
            tile[i][0] = 4.0f * t_tile[i][0] - 5.0f * t_tile[i][2] + t_tile[i][4];
            tile[i][1] = -4.0f * t_tile[i][1] - 4.0f * t_tile[i][2] + t_tile[i][3] + t_tile[i][4];
            tile[i][2] = 4.0f * t_tile[i][1] - 4.0f * t_tile[i][2] - t_tile[i][3] + t_tile[i][4];
            tile[i][3] = -2.0f * t_tile[i][1] - t_tile[i][2] + 2.0f * t_tile[i][3] + t_tile[i][4];
            tile[i][4] = 2.0f * t_tile[i][1] - t_tile[i][2] - 2.0f * t_tile[i][3] + t_tile[i][4];
            tile[i][5] = 4.0f * t_tile[i][1] - 5.0f * t_tile[i][3] + t_tile[i][5];
        }

        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                output[n * out_n_stride + c * out_c_stride + tile_i * tilei_stride + tile_j * tilej_stride + 6 * i + j] = tile[i][j];
            }
        }
    }
}

void winograd_outerProduct_AtIA_4x4(float* input, float* weight, float* bias, float* output, int batch_size, int K, int tile_n, int out_map_size, int C) {
    int total_tile = batch_size * K * tile_n * tile_n;
    int c_stride = tile_n * tile_n * 36, in_n_stride = C * c_stride;
    int tilei_stride = tile_n * 36, tilej_stride = 36;
    int w_c_stride = 36, w_k_stride = C * 36;
    int out_k_stride = out_map_size * out_map_size, out_n_stride = out_k_stride * K;
    int x_stride = out_map_size, y_stride = 1;

#pragma omp parallel for
    for (int global_id = 0; global_id < total_tile; global_id++) {
        int n = global_id / (K * tile_n * tile_n);
        int remain = global_id % (K * tile_n * tile_n);
        int k = remain / (tile_n * tile_n);
        remain = remain % (tile_n * tile_n);
        int tile_i = remain / tile_n;
        int tile_j = remain % tile_n;

        float tile[6][6] = {0};
        for (int c = 0; c < C; c++) {
            for (int i = 0; i < 6; i++) {
                for (int j = 0; j < 6; j++) {
                    tile[i][j] += input[n * in_n_stride + c * c_stride + tile_i * tilei_stride + tile_j * tilej_stride + 6 * i + j] * weight[k * w_k_stride + c * w_c_stride + 6 * i + j];
                }
            }
        }

        // const float At[4][6] {
        //     {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f},
        //     {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.0f},
        //     {0.0f, 1.0f, 1.0f, 4.0f, 4.0f, 0.0f},
        //     {0.0f, 1.0f, -1.0f, 8.0f, -8.0f, 1.0f}
        // }

        float t_tile[4][6], f_tile[4][4];
        // At * I
        for (int j = 0; j < 6; j++) {
            t_tile[0][j] = tile[0][j] + tile[1][j] + tile[2][j] + tile[3][j] + tile[4][j];
            t_tile[1][j] = tile[1][j] - tile[2][j] + 2.0f * tile[3][j] - 2.0f * tile[4][j];
            t_tile[2][j] = tile[1][j] + tile[2][j] + 4.0f * tile[3][j] + 4.0f * tile[4][j];
            t_tile[3][j] = tile[1][j] - tile[2][j] + 8.0f * tile[3][j] - 8.0f * tile[4][j] + tile[5][j];
        }
        // I * A
        for (int i = 0; i < 4; i++) {
            f_tile[i][0] = t_tile[i][0] + t_tile[i][1] + t_tile[i][2] + t_tile[i][3] + t_tile[i][4];
            f_tile[i][1] = t_tile[i][1] - t_tile[i][2] + 2.0f * t_tile[i][3] - 2.0f * t_tile[i][4];
            f_tile[i][2] = t_tile[i][1] + t_tile[i][2] + 4.0f * t_tile[i][3] + 4.0f * t_tile[i][4];
            f_tile[i][3] = t_tile[i][1] - t_tile[i][2] + 8.0f * t_tile[i][3] - 8.0f * t_tile[i][4] + t_tile[i][5];
        }
        // bias
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                f_tile[i][j] += bias[k];
            }
        }
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                int x = 4 * tile_i + i;
                int y = 4 * tile_j + j;
                if (x >= out_map_size || y >= out_map_size) {
                    continue;
                }
                output[n * out_n_stride + k * out_k_stride + x * x_stride + y * y_stride] = f_tile[i][j];
            }
        }
    }
}

void winograd_convolution_4x4(float* input,  /* NxCxHxW */
                              float* weight, /* KxCx3x3 */
                              float* bias,   /* K */
                              float* my_res, /* NxKxH'xW'*/
                              int batch_size,
                              int C,
                              int K,
                              int map_size,
                              int padding) {
    // filter transformation
    float* trans_filter = (float*)malloc(K * C * 36 * sizeof(float));  // transformed filters
    if (trans_filter == NULL) {
        printf("bad malloc trans_filter\n");
    }
    winograd_GgGt_4x4(weight, trans_filter, K, C);

    int out_map_size = (map_size + padding * 2) - 2;  // kernel size = 3, stride = 1 in Winograd algorithm
    int tile_n = (out_map_size + 3) / 4;

    float* trans_input = (float*)malloc(batch_size * tile_n * tile_n * C * 36 * sizeof(float));  // transformed input
    if (trans_input == NULL) {
        printf("bad malloc trans_input\n");
    }

    // input transformation
    if (padding == 0) {
        winograd_BtdB_4x4(input, trans_input, batch_size, C, tile_n, map_size);
    } else if (padding == 1) {
        winograd_BtdB_padding_4x4(input, trans_input, batch_size, C, tile_n, map_size);
    }

    // element-wise multiplication & output transformation
    winograd_outerProduct_AtIA_4x4(trans_input, trans_filter, bias, my_res, batch_size, K, tile_n, out_map_size, C);

    free(trans_input);
    free(trans_filter);
    return;
}

// F(6x6,3x3)

void winograd_GgGt_6x6(float* input, float* output, int K, int C) {
    int total_filter = K * C;
    int in_c_stride = 9, in_k_stride = in_c_stride * C;
    int out_c_stride = 64, out_k_stride = out_c_stride * C;

#pragma omp parallel for
    for (int global_id = 0; global_id < total_filter; global_id++) {
        int k = global_id / C;
        int c = global_id % C;

        float tile[3][3], t_tile[8][3], f_tile[8][8];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                tile[i][j] = input[in_k_stride * k + in_c_stride * c + 3 * i + j];
            }
        }

        // const float G[8][3] = {
        //     {   1.0f,     0.0f,     0.0f},

        //     {-2.0f/9,  -2.0f/9,  -2.0f/9},
        //     {-2.0f/9,   2.0f/9,  -2.0f/9},

        //     {1.0f/90,  1.0f/45,  2.0f/45},
        //     {1.0f/90, -1.0f/45,  2.0f/45},

        //     {1.0f/45,  1.0f/90, 1.0f/180},
        //     {1.0f/45, -1.0f/90, 1.0f/180},

        //     {   0.0f,     0.0f,     1.0f}
        // };

        // G * g
        for (int j = 0; j < 3; j++) {
            t_tile[0][j] = tile[0][j];

            t_tile[1][j] = -2.0f / 9 * tile[0][j] - 2.0f / 9 * tile[1][j] - 2.0f / 9 * tile[2][j];
            t_tile[2][j] = -2.0f / 9 * tile[0][j] + 2.0f / 9 * tile[1][j] - 2.0f / 9 * tile[2][j];

            t_tile[3][j] = 1.0f / 90 * tile[0][j] + 1.0f / 45 * tile[1][j] + 2.0f / 45 * tile[2][j];
            t_tile[4][j] = 1.0f / 90 * tile[0][j] - 1.0f / 45 * tile[1][j] + 2.0f / 45 * tile[2][j];

            t_tile[5][j] = 1.0f / 45 * tile[0][j] + 1.0f / 90 * tile[1][j] + 1.0f / 180 * tile[2][j];
            t_tile[6][j] = 1.0f / 45 * tile[0][j] - 1.0f / 90 * tile[1][j] + 1.0f / 180 * tile[2][j];

            t_tile[7][j] = tile[2][j];
        }
        // g * Gt
        for (int i = 0; i < 8; i++) {
            f_tile[i][0] = t_tile[i][0];

            f_tile[i][1] = -2.0f / 9 * t_tile[i][0] - 2.0f / 9 * t_tile[i][1] - 2.0f / 9 * t_tile[i][2];
            f_tile[i][2] = -2.0f / 9 * t_tile[i][0] + 2.0f / 9 * t_tile[i][1] - 2.0f / 9 * t_tile[i][2];

            f_tile[i][3] = 1.0f / 90 * t_tile[i][0] + 1.0f / 45 * t_tile[i][1] + 2.0f / 45 * t_tile[i][2];
            f_tile[i][4] = 1.0f / 90 * t_tile[i][0] - 1.0f / 45 * t_tile[i][1] + 2.0f / 45 * t_tile[i][2];

            f_tile[i][5] = 1.0f / 45 * t_tile[i][0] + 1.0f / 90 * t_tile[i][1] + 1.0f / 180 * t_tile[i][2];
            f_tile[i][6] = 1.0f / 45 * t_tile[i][0] - 1.0f / 90 * t_tile[i][1] + 1.0f / 180 * t_tile[i][2];

            f_tile[i][7] = t_tile[i][2];
        }

        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                output[out_k_stride * k + out_c_stride * c + 8 * i + j] = f_tile[i][j];
            }
        }
    }
}

void winograd_BtdB_6x6(float* input, float* output, int batch_size, int C, int tile_n, int map_size) {
    int total_tile = batch_size * C * tile_n * tile_n;
    int in_n_stride = map_size * map_size * C, in_c_stride = map_size * map_size, x_stride = map_size, y_stride = 1;
    int out_n_stride = tile_n * tile_n * 64 * C, out_c_stride = tile_n * tile_n * 64;
    int tilei_stride = tile_n * 64, tilej_stride = 64;

#pragma omp parallel for
    for (int global_id = 0; global_id < total_tile; global_id++) {
        int n = global_id / (C * tile_n * tile_n);
        int remain = global_id % (C * tile_n * tile_n);
        int c = remain / (tile_n * tile_n);
        remain = remain % (tile_n * tile_n);
        int tile_i = remain / tile_n;
        int tile_j = remain % tile_n;

        float tile[8][8], t_tile[8][8];
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                int x = 6 * tile_i + i;
                int y = 6 * tile_j + j;
                if (x >= map_size || y >= map_size) {
                    tile[i][j] = 0;
                    continue;
                }
                tile[i][j] = input[n * in_n_stride + c * in_c_stride + x * x_stride + y * y_stride];
            }
        }

        // const float Bt[8][8] = {
        //     {1.0f,  0.0f, -5.25f,  0.00f,  5.25f,  0.00f, -1.0f, 0.0f},

        //     {0.0f,  1.0f,  1.00f, -4.25f, -4.25f,  1.00f,  1.0f, 0.0f},
        //     {0.0f, -1.0f,  1.00f,  4.25f, -4.25f, -1.00f,  1.0f, 0.0f},

        //     {0.0f,  0.5f,  0.25f, -2.50f, -1.25f,  2.00f,  1.0f, 0.0f},
        //     {0.0f, -0.5f,  0.25f,  2.50f, -1.25f, -2.00f,  1.0f, 0.0f},

        //     {0.0f,  2.0f,  4.00f, -2.50f, -5.00f,  0.50f,  1.0f, 0.0f},
        //     {0.0f, -2.0f,  4.00f,  2.50f, -5.00f, -0.50f,  1.0f, 0.0f},

        //     {0.0f, -1.0f,  0.00f,  5.25f,  0.00f, -5.25f,  0.0f, 1.0f}
        // };

        // Bt * d
        for (int j = 0; j < 8; j++) {
            t_tile[0][j] = tile[0][j] - 5.25f * tile[2][j] + 5.25 * tile[4][j] - tile[6][j];

            t_tile[1][j] = tile[1][j] + tile[2][j] - 4.25f * tile[3][j] - 4.25f * tile[4][j] + tile[5][j] + tile[6][j];
            t_tile[2][j] = -tile[1][j] + tile[2][j] + 4.25f * tile[3][j] - 4.25f * tile[4][j] - tile[5][j] + tile[6][j];

            t_tile[3][j] = 0.5f * tile[1][j] + 0.25f * tile[2][j] - 2.5f * tile[3][j] - 1.25f * tile[4][j] + 2.0f * tile[5][j] + tile[6][j];
            t_tile[4][j] = -0.5f * tile[1][j] + 0.25f * tile[2][j] + 2.5f * tile[3][j] - 1.25f * tile[4][j] - 2.0f * tile[5][j] + tile[6][j];

            t_tile[5][j] = 2.0f * tile[1][j] + 4.0f * tile[2][j] - 2.5f * tile[3][j] - 5.0f * tile[4][j] + 0.5f * tile[5][j] + tile[6][j];
            t_tile[6][j] = -2.0f * tile[1][j] + 4.0f * tile[2][j] + 2.5f * tile[3][j] - 5.0f * tile[4][j] - 0.5f * tile[5][j] + tile[6][j];

            t_tile[7][j] = -tile[0][j] + 5.25f * tile[2][j] - 5.25 * tile[4][j] + tile[6][j];
        }
        // d * B
        for (int i = 0; i < 8; i++) {
            tile[i][0] = t_tile[i][0] - 5.25f * t_tile[i][2] + 5.25 * t_tile[i][4] - t_tile[i][6];

            tile[i][1] = t_tile[i][1] + t_tile[i][2] - 4.25f * t_tile[i][3] - 4.25f * t_tile[i][4] + t_tile[i][5] + t_tile[i][6];
            tile[i][2] = -t_tile[i][1] + t_tile[i][2] + 4.25f * t_tile[i][3] - 4.25f * t_tile[i][4] - t_tile[i][5] + t_tile[i][6];

            tile[i][3] = 0.5f * t_tile[i][1] + 0.25f * t_tile[i][2] - 2.5f * t_tile[i][3] - 1.25f * t_tile[i][4] + 2.0f * t_tile[i][5] + t_tile[i][6];
            tile[i][4] = -0.5f * t_tile[i][1] + 0.25f * t_tile[i][2] + 2.5f * t_tile[i][3] - 1.25f * t_tile[i][4] - 2.0f * t_tile[i][5] + t_tile[i][6];

            tile[i][5] = 2.0f * t_tile[i][1] + 4.0f * t_tile[i][2] - 2.5f * t_tile[i][3] - 5.0f * t_tile[i][4] + 0.5f * t_tile[i][5] + t_tile[i][6];
            tile[i][6] = -2.0f * t_tile[i][1] + 4.0f * t_tile[i][2] + 2.5f * t_tile[i][3] - 5.0f * t_tile[i][4] - 0.5f * t_tile[i][5] + t_tile[i][6];

            tile[i][7] = -t_tile[i][0] + 5.25f * t_tile[i][2] - 5.25 * t_tile[i][4] + t_tile[i][6];
        }

        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                output[n * out_n_stride + c * out_c_stride + tile_i * tilei_stride + tile_j * tilej_stride + 8 * i + j] = tile[i][j];
            }
        }
    }
}

void winograd_BtdB_padding_6x6(float* input, float* output, int batch_size, int C, int tile_n, int map_size) {
    int total_tile = batch_size * C * tile_n * tile_n;
    int in_n_stride = map_size * map_size * C, in_c_stride = map_size * map_size, x_stride = map_size, y_stride = 1;
    int out_n_stride = tile_n * tile_n * 64 * C, out_c_stride = tile_n * tile_n * 64;
    int tilei_stride = tile_n * 64, tilej_stride = 64;

#pragma omp parallel for
    for (int global_id = 0; global_id < total_tile; global_id++) {
        int n = global_id / (C * tile_n * tile_n);
        int remain = global_id % (C * tile_n * tile_n);
        int c = remain / (tile_n * tile_n);
        remain = remain % (tile_n * tile_n);
        int tile_i = remain / tile_n;
        int tile_j = remain % tile_n;

        float tile[8][8], t_tile[8][8];
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                int x = 6 * tile_i + i;
                int y = 6 * tile_j + j;
                if (x == 0 || y == 0 || x >= (map_size + 1) || y >= (map_size + 1)) {
                    tile[i][j] = 0;
                } else {
                    tile[i][j] = input[n * in_n_stride + c * in_c_stride + (x - 1) * x_stride + (y - 1) * y_stride];
                }
            }
        }

        // const float Bt[8][8] = {
        //     {1.0f,  0.0f, -5.25f,  0.00f,  5.25f,  0.00f, -1.0f, 0.0f},

        //     {0.0f,  1.0f,  1.00f, -4.25f, -4.25f,  1.00f,  1.0f, 0.0f},
        //     {0.0f, -1.0f,  1.00f,  4.25f, -4.25f, -1.00f,  1.0f, 0.0f},

        //     {0.0f,  0.5f,  0.25f, -2.50f, -1.25f,  2.00f,  1.0f, 0.0f},
        //     {0.0f, -0.5f,  0.25f,  2.50f, -1.25f, -2.00f,  1.0f, 0.0f},

        //     {0.0f,  2.0f,  4.00f, -2.50f, -5.00f,  0.50f,  1.0f, 0.0f},
        //     {0.0f, -2.0f,  4.00f,  2.50f, -5.00f, -0.50f,  1.0f, 0.0f},

        //     {0.0f, -1.0f,  0.00f,  5.25f,  0.00f, -5.25f,  0.0f, 1.0f}
        // };

        // Bt * d
        for (int j = 0; j < 8; j++) {
            t_tile[0][j] = tile[0][j] - 5.25f * tile[2][j] + 5.25 * tile[4][j] - tile[6][j];

            t_tile[1][j] = tile[1][j] + tile[2][j] - 4.25f * tile[3][j] - 4.25f * tile[4][j] + tile[5][j] + tile[6][j];
            t_tile[2][j] = -tile[1][j] + tile[2][j] + 4.25f * tile[3][j] - 4.25f * tile[4][j] - tile[5][j] + tile[6][j];

            t_tile[3][j] = 0.5f * tile[1][j] + 0.25f * tile[2][j] - 2.5f * tile[3][j] - 1.25f * tile[4][j] + 2.0f * tile[5][j] + tile[6][j];
            t_tile[4][j] = -0.5f * tile[1][j] + 0.25f * tile[2][j] + 2.5f * tile[3][j] - 1.25f * tile[4][j] - 2.0f * tile[5][j] + tile[6][j];

            t_tile[5][j] = 2.0f * tile[1][j] + 4.0f * tile[2][j] - 2.5f * tile[3][j] - 5.0f * tile[4][j] + 0.5f * tile[5][j] + tile[6][j];
            t_tile[6][j] = -2.0f * tile[1][j] + 4.0f * tile[2][j] + 2.5f * tile[3][j] - 5.0f * tile[4][j] - 0.5f * tile[5][j] + tile[6][j];

            t_tile[7][j] = -tile[0][j] + 5.25f * tile[2][j] - 5.25 * tile[4][j] + tile[6][j];
        }
        // d * B
        for (int i = 0; i < 8; i++) {
            tile[i][0] = t_tile[i][0] - 5.25f * t_tile[i][2] + 5.25 * t_tile[i][4] - t_tile[i][6];

            tile[i][1] = t_tile[i][1] + t_tile[i][2] - 4.25f * t_tile[i][3] - 4.25f * t_tile[i][4] + t_tile[i][5] + t_tile[i][6];
            tile[i][2] = -t_tile[i][1] + t_tile[i][2] + 4.25f * t_tile[i][3] - 4.25f * t_tile[i][4] - t_tile[i][5] + t_tile[i][6];

            tile[i][3] = 0.5f * t_tile[i][1] + 0.25f * t_tile[i][2] - 2.5f * t_tile[i][3] - 1.25f * t_tile[i][4] + 2.0f * t_tile[i][5] + t_tile[i][6];
            tile[i][4] = -0.5f * t_tile[i][1] + 0.25f * t_tile[i][2] + 2.5f * t_tile[i][3] - 1.25f * t_tile[i][4] - 2.0f * t_tile[i][5] + t_tile[i][6];

            tile[i][5] = 2.0f * t_tile[i][1] + 4.0f * t_tile[i][2] - 2.5f * t_tile[i][3] - 5.0f * t_tile[i][4] + 0.5f * t_tile[i][5] + t_tile[i][6];
            tile[i][6] = -2.0f * t_tile[i][1] + 4.0f * t_tile[i][2] + 2.5f * t_tile[i][3] - 5.0f * t_tile[i][4] - 0.5f * t_tile[i][5] + t_tile[i][6];

            tile[i][7] = -t_tile[i][0] + 5.25f * t_tile[i][2] - 5.25 * t_tile[i][4] + t_tile[i][6];
        }

        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                output[n * out_n_stride + c * out_c_stride + tile_i * tilei_stride + tile_j * tilej_stride + 8 * i + j] = tile[i][j];
            }
        }
    }
}

void winograd_outerProduct_AtIA_6x6(float* input, float* weight, float* bias, float* output, int batch_size, int K, int tile_n, int out_map_size, int C) {
    int total_tile = batch_size * K * tile_n * tile_n;
    int c_stride = tile_n * tile_n * 64, in_n_stride = C * c_stride;
    int tilei_stride = tile_n * 64, tilej_stride = 64;
    int w_c_stride = 64, w_k_stride = C * 64;
    int out_k_stride = out_map_size * out_map_size, out_n_stride = out_k_stride * K;
    int x_stride = out_map_size, y_stride = 1;

#pragma omp parallel for
    for (int global_id = 0; global_id < total_tile; global_id++) {
        int n = global_id / (K * tile_n * tile_n);
        int remain = global_id % (K * tile_n * tile_n);
        int k = remain / (tile_n * tile_n);
        remain = remain % (tile_n * tile_n);
        int tile_i = remain / tile_n;
        int tile_j = remain % tile_n;

        float tile[8][8] = {0};
        for (int c = 0; c < C; c++) {
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    tile[i][j] += input[n * in_n_stride + c * c_stride + tile_i * tilei_stride + tile_j * tilej_stride + 8 * i + j] * weight[k * w_k_stride + c * w_c_stride + 8 * i + j];
                }
            }
        }

        // const float At[6][8] = {
        //     {1.0f,  1.0f,   1.0f,   1.0f,   1.0f,  32.0f, 32.0f, 0.0f},
        //     {0.0f,  1.0f,  -1.0f,   2.0f,  -2.0f,  16.0f,-16.0f, 0.0f},
        //     {0.0f,  1.0f,   1.0f,   4.0f,   4.0f,   8.0f,  8.0f, 0.0f},
        //     {0.0f,  1.0f,  -1.0f,   8.0f,  -8.0f,   4.0f, -4.0f, 0.0f},
        //     {0.0f,  1.0f,   1.0f,  16.0f,  16.0f,   2.0f,  2.0f, 0.0f},
        //     {0.0f,  1.0f,  -1.0f,  32.0f, -32.0f,   1.0f, -1.0f, 1.0f}
        // };

        float t_tile[6][8], f_tile[6][6];
        // At * I
        for (int j = 0; j < 8; j++) {
            t_tile[0][j] = tile[0][j] + tile[1][j] + tile[2][j] + tile[3][j] + tile[4][j] + 32.0f * tile[5][j] + 32.0f * tile[6][j];
            t_tile[1][j] = tile[1][j] - tile[2][j] + 2.0f * tile[3][j] - 2.0f * tile[4][j] + 16.0f * tile[5][j] - 16.0f * tile[6][j];
            t_tile[2][j] = tile[1][j] + tile[2][j] + 4.0f * tile[3][j] + 4.0f * tile[4][j] + 8.0f * tile[5][j] + 8.0f * tile[6][j];
            t_tile[3][j] = tile[1][j] - tile[2][j] + 8.0f * tile[3][j] - 8.0f * tile[4][j] + 4.0f * tile[5][j] - 4.0f * tile[6][j];
            t_tile[4][j] = tile[1][j] + tile[2][j] + 16.0f * tile[3][j] + 16.0f * tile[4][j] + 2.0f * tile[5][j] + 2.0f * tile[6][j];
            t_tile[5][j] = tile[1][j] - tile[2][j] + 32.0f * tile[3][j] - 32.0f * tile[4][j] + tile[5][j] - tile[6][j] + tile[7][j];
        }
        // I * A
        for (int i = 0; i < 6; i++) {
            f_tile[i][0] = t_tile[i][0] + t_tile[i][1] + t_tile[i][2] + t_tile[i][3] + t_tile[i][4] + 32.0f * t_tile[i][5] + 32.0f * t_tile[i][6];
            f_tile[i][1] = t_tile[i][1] - t_tile[i][2] + 2.0f * t_tile[i][3] - 2.0f * t_tile[i][4] + 16.0f * t_tile[i][5] - 16.0f * t_tile[i][6];
            f_tile[i][2] = t_tile[i][1] + t_tile[i][2] + 4.0f * t_tile[i][3] + 4.0f * t_tile[i][4] + 8.0f * t_tile[i][5] + 8.0f * t_tile[i][6];
            f_tile[i][3] = t_tile[i][1] - t_tile[i][2] + 8.0f * t_tile[i][3] - 8.0f * t_tile[i][4] + 4.0f * t_tile[i][5] - 4.0f * t_tile[i][6];
            f_tile[i][4] = t_tile[i][1] + t_tile[i][2] + 16.0f * t_tile[i][3] + 16.0f * t_tile[i][4] + 2.0f * t_tile[i][5] + 2.0f * t_tile[i][6];
            f_tile[i][5] = t_tile[i][1] - t_tile[i][2] + 32.0f * t_tile[i][3] - 32.0f * t_tile[i][4] + t_tile[i][5] - t_tile[i][6] + t_tile[i][7];
        }
        // bias
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                f_tile[i][j] += bias[k];
            }
        }
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                int x = 6 * tile_i + i;
                int y = 6 * tile_j + j;
                if (x >= out_map_size || y >= out_map_size) {
                    continue;
                }
                output[n * out_n_stride + k * out_k_stride + x * x_stride + y * y_stride] = f_tile[i][j];
            }
        }
    }
}

void winograd_convolution_6x6(float* input,  /* NxCxHxW */
                              float* weight, /* KxCx3x3 */
                              float* bias,   /* K */
                              float* my_res, /* NxKxH'xW'*/
                              int batch_size,
                              int C,
                              int K,
                              int map_size,
                              int padding) {
    // filter transformation
    float* trans_filter = (float*)malloc(K * C * 64 * sizeof(float));  // transformed filters
    if (trans_filter == NULL) {
        printf("bad malloc trans_filter\n");
    }
    winograd_GgGt_6x6(weight, trans_filter, K, C);

    int out_map_size = (map_size + padding * 2) - 2;  // kernel size = 3, stride = 1 in Winograd algorithm
    int tile_n = (out_map_size + 5) / 6;

    float* trans_input = (float*)malloc(batch_size * tile_n * tile_n * C * 64 * sizeof(float));  // transformed input
    if (trans_input == NULL) {
        printf("bad malloc trans_input\n");
    }

    // input transformation
    if (padding == 0) {
        winograd_BtdB_6x6(input, trans_input, batch_size, C, tile_n, map_size);
    } else if (padding == 1) {
        winograd_BtdB_padding_6x6(input, trans_input, batch_size, C, tile_n, map_size);
    }

    // element-wise multiplication & output transformation
    winograd_outerProduct_AtIA_6x6(trans_input, trans_filter, bias, my_res, batch_size, K, tile_n, out_map_size, C);

    free(trans_input);
    free(trans_filter);
    return;
}

void init(float* A, int size) {
    for (int i = 0; i < size; i++) {
        A[i] = (float)rand() / RAND_MAX;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 7) {
        printf("usage: ./test < N > < C > < H(W) > < K > <padding(0/1)> < m(2/4/6) > \n");
        exit(0);
    }

    int batch_size = atoi(argv[1]);
    int C = atoi(argv[2]);
    int map_size = atoi(argv[3]);
    int K = atoi(argv[4]);
    int padding = atoi(argv[5]);
    int m = atoi(argv[6]);
    double t_start, t_end;

    float* input = (float*)malloc(batch_size * C * map_size * map_size * sizeof(float));
    float* weight = (float*)malloc(K * C * 3 * 3 * sizeof(float));
    float* bias = (float*)malloc(K * sizeof(float));
    float* result = (float*)malloc(batch_size * K * map_size * map_size * sizeof(float));

    init(input, batch_size * C * map_size * map_size);
    init(weight, K * C * 3 * 3);
    init(bias, K);

    t_start = rtclock();
    switch (m) {
        case 2:
            winograd_convolution_2x2(input, weight, bias, result, batch_size, C, K, map_size, padding);
            break;

        case 4:
            winograd_convolution_4x4(input, weight, bias, result, batch_size, C, K, map_size, padding);
            break;

        case 6:
            winograd_convolution_6x6(input, weight, bias, result, batch_size, C, K, map_size, padding);
            break;

        default:
            break;
    }
    t_end = rtclock();
    fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

    free(input);
    free(weight);
    free(bias);
    free(result);
    return 0;
}
