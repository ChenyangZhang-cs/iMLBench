/*
 * kernels.cpp
 *
 *  Created on: 2017/2/20
 *      Author: ZhangHua
 */

#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <atomic>
#include <fstream>
#include <iostream>
#include <sstream>
#include <set>
#include <omp.h>

#include <tensor.hpp>
#include <device_instance.hpp>

using namespace std;
#include <sys/time.h>

double gettime()
{
      struct timeval t;
      gettimeofday(&t, NULL);
      return t.tv_sec + t.tv_usec * 1e-6;
}

double t1 = 0, t2 = 0;

int cpu_offset = 0;
bool cpu_run, gpu_run;

namespace clnet
{
    const char KERNEL_FILE[] = "kernels.cl";
    const string KERNEL_DEF = "kernel void ";
    const vector<cl::Event> no_preconditions;

    unordered_map<string, string> kernels_source;
    unordered_map<Tensor *, size_t> kernels_cost;

    extern int debugger_device_id;
    extern condition_variable breakpoint;
    //unordered_map<Tensor *, cl::Buffer> temp_global_buffer;
    unordered_map<Tensor *, float *> temp_global_buffer;

    mutex breakpoint_mutex;
    Tensor *_current, *_breakpoint = nullptr;
    int64 microseconds = 0, breakpoint_hit_times = -1; //paused on all devices initially

    template <typename T>
    bool read_file_content(const string file, basic_string<T> &content)
    {
        basic_ifstream<T> ifs(file, ios::binary);
        if (ifs)
        {
            basic_ostringstream<T> os;
            os << ifs.rdbuf();
            content.append(os.str());
            return true;
        }
        return false;
    }

    template bool read_file_content<wchar_t>(const string file, basic_string<wchar_t> &content);

    void replace_all(string &content, const string key, const string replace)
    {
        string::size_type pos = content.find(key), length = key.size(), span = replace.size();
        while (pos != string::npos)
        {
            content.replace(pos, length, replace);
            pos = content.find(key, pos + span);
        }
    }

    void load_kernel_source(const string file)
    {
        string source;
        if (!read_file_content<char>(OpenCL.location + file, source) && !read_file_content<char>(OpenCL.location + "kernels/" + file, source) && !read_file_content<char>(OpenCL.location + "src/" + file, source)) //custom for visual studio x64
            throw runtime_error("kernel file " + file + " not found.");
        auto pos = source.find(KERNEL_DEF);
        auto end = source.rfind("}\n", pos) + 2;
        string name = "_header";
        string code = source.substr(0, end);
        kernels_source[name] = code;

        while (pos != string::npos)
        {
            string::size_type begin = pos, name_begin = begin + KERNEL_DEF.size();
            pos = source.find(KERNEL_DEF, name_begin); //next kernel begins
            if (pos == string::npos)
                end = source.length();
            else
                end = source.rfind("}\n", pos) + 2;
            auto name_end = source.find("(", name_begin);
            name = source.substr(name_begin, name_end - name_begin);
            code = source.substr(begin, end - begin);
            kernels_source[name] = code;
        }
    }

    string generate_kernel_sources(DeviceInstance &I, const cl::Device &device, unordered_map<Tensor *, string> &tensor_kernels)
    {
        load_kernel_source(KERNEL_FILE);

        string name = "_header";
        string source;
        string sources = kernels_source[name];
        set<string> kernels; //Note: use c++0x in debug mode. MinGW/GCC has a bug when using set<string> and unordered_map.operator[] simultaneously under c++1y.
        for (auto tensor : Tensor::ALL)
        {
            source = tensor->generate_source_code(I);

            if (source.empty())
                continue;
            size_t pos = 0;
            while ((pos = source.find(KERNEL_DEF, pos)) != string::npos)
            {
                int begin = pos + KERNEL_DEF.size();
                int end = source.find("(", begin);
                name = source.substr(begin, end - begin);
                if (kernels.count(name) == 0)
                {
                    kernels.insert(name);
                    size_t pos2 = source.find(KERNEL_DEF, pos + KERNEL_DEF.size());
                    if (pos2 != string::npos)
                        pos2 = source.rfind("}\n", pos2) + 2;
                    else
                        pos2 = source.size();
                    sources.append("\n").append(source.substr(pos, pos2 - pos));
                }
                //			logger << "****** " << name << endl << code << endl;
                if (!tensor_kernels[tensor].empty())
                    tensor_kernels[tensor].append(" ");
                tensor_kernels[tensor].append(name);
                pos++;
            }
        }
        return sources;
    }

    string gradient_set_type(string key, bool attached)
    {
        string code = kernels_source[key];
        if (!attached)
            return code;
        replace_all(code, "gradient_set_type", "+=");
        replace_all(code, key, key + "_attached");
        return code;
    }

    void Tensor::launch(set<Tensor *> *executed, void *data, function<void(Tensor *, void *)> functor)
    {
        if (executed->count(this) == 0)
        {
            executed->insert(this); //stop cycle access by inserting before inputs access
            for (auto tensor : inputs)
                if (tensor != nullptr)
                    tensor->launch(executed, data, functor);

            _current = this;
            if (CLNET_TENSOR_GLOBALS & CLNET_STEP_INTO_MODE)
            {
                _breakpoint = this;
                breakpoint_hit_times = -1;
            }
            if (this == _breakpoint)
            {
                auto I = static_cast<DeviceInstance *>(data);
                int device_id = I->ID;
                if (breakpoint_hit_times < 0 || (debugger_device_id == device_id && --breakpoint_hit_times == 0))
                {
                    logger << "[debugger] device " << device_id << " break on " << alias << ": " << type_name(this) << endl;
                    unique_lock<mutex> breakpoint_lock(breakpoint_mutex);
                    breakpoint.wait(breakpoint_lock); //No considering spurious wake-up
                    logger << "[debugger] device " << device_id << " continue to run." << endl;
                }
            }
            if (microseconds > 0)
                microseconds = MICROS();
            functor(this, data);
            if (microseconds > 0)
            {
                wait_for_all_kernels_finished(*static_cast<DeviceInstance *>(data));
                auto duration = MICROS(microseconds);
                kernels_cost[this] += duration;
                //			logger << type_name(this) << " " << alias << ": " << duration << " microseconds" << endl;
            }
        }
    }

    void Tensor::upload(DeviceInstance &I, const vector<cl::Event> *preconditions)
    {
        if (size <= 0) //size=0: CL_INVALID_VALUE for clEnqueueWriteBuffer
            return;
        I.pointers[this] = I.buffers[this];
        //    I.queue.enqueueReadBuffer(I.buffers[this], preconditions == nullptr, 0, size, I.pointers[this], preconditions, &I.events[this]);
    }

    void Tensor::download(DeviceInstance &I, float *pointer, const std::vector<cl::Event> *preconditions)
    {
        if (size <= 0) //size=0: CL_INVALID_VALUE for clEnqueueWriteBuffer
            return;
        //preconditions = nullptr;
        //I.queue.enqueueWriteBuffer(I.buffers[this], preconditions == nullptr, 0, size, I.pointers[this], preconditions, &I.events[this]);
        // zcy

        const auto &context = I.queue.getInfo<CL_QUEUE_CONTEXT>();

        I.pointers[this] = (float *)clSVMAlloc(context(), CL_MEM_READ_WRITE, size, 0);
        if (I.pointers[this] == NULL)
        {
            printf("clSVMAlloc failed\n");
        }
        memcpy(I.pointers[this], pointer, size); //initialized by tensor's pointer

        //I->buffers[this] = (float *)clSVMAlloc(context(), CL_MEM_READ_WRITE, size, 0);
        I.buffers[this] = I.pointers[this];
    }

    void Tensor::download(DeviceInstance &I, const vector<cl::Event> *preconditions)
    {
        if (size <= 0) //size=0: CL_INVALID_VALUE for clEnqueueWriteBuffer
            return;
        //I->buffers[this] = (float *)clSVMAlloc(context(), CL_MEM_READ_WRITE, size, 0);
        I.buffers[this] = I.pointers[this];
    }

    int find_proper_local_size(int required, int work_group_size)
    {
        if (required < work_group_size)
        {
            int parallel = 1;
            while (parallel < required)
                parallel <<= 1;
            return parallel;
        }
        else
            return work_group_size;
    }

    cl::Kernel &prepare_for_running_kernel(Tensor *tensor, DeviceInstance &I, int number)
    {
        I.precondition_events.clear();
        for (auto input : tensor->inputs)
            if (input != nullptr && input->volume > 0) //exclude "pure kernel" tensor
                I.precondition_events.push_back(I.events[input]);
        return I.kernels[tensor][number];
    }

#define FULLY_CONNECTED_STD_DIM_IN_UPLIMIT 512
    string type::FullyConnectedLayer::generate_source_code(DeviceInstance &I)
    {
        int dim_in = inputs[1]->dimensions.front(); //inputs[1]: weight
        string code;                                //make a copy
        if (dim_in < FULLY_CONNECTED_STD_DIM_IN_UPLIMIT)
        {
            //Parallel: (batch_size * dim_hidden)
            code = kernels_source["feed_forward_fully_connected_sigmoid"];
            if (dim_in > 0)
            {
                auto dim_in_str = to_string(dim_in);
                replace_once(code, "feed_forward_fully_connected_sigmoid", "feed_forward_fully_connected_sigmoid_" + dim_in_str);
                replace_once(code, "#pragma unroll", "#pragma unroll dim_in");
                replace_once(code, "int dim_in", "int _unused");
                replace_all(code, "dim_in", dim_in_str);
            }
            if (activation != "sigmoid")
                replace_all(code, "sigmoid", activation);
        }
        else
        {
            code = kernels_source["feed_forward_fully_connected_sigmoid"];
            if (activation != "sigmoid")
                replace_all(code, "sigmoid", activation);
            //Parallel: (batch_size * dim_hidden * get_local_size(0))
            //Note: choose local NDRange size near (2 * dim_in) when enqueue ASAP
            //		code = kernels_source["feed_forward_fully_connected_softrelu"];//TODO
            //		if (activation != "softrelu")
            //			replace_all(code, "softrelu", activation);
        }

        return code;
    }

    void feed_forward_fully_connected_sigmoid_omp(float *out, float *in, float *weight,
                                                  float *bias, int dim_hidden, int dim_in,
                                                  clnet::int64 start, clnet::int64 global_size)
    {
        //MLP kernel 1 omp
#pragma omp parallel for
        for (int GID = start; GID < global_size; GID++)
        {
            int n = GID / dim_hidden;
            int hidden = GID % dim_hidden;
            int in_offset = n * dim_in;

            float z = bias != NULL ? bias[hidden] : 0;

            for (int i = 0; i < dim_in; i++)
                z += weight[dim_hidden * i + hidden] * in[in_offset + i];
            out[GID] = 1.0 / (1.0 + exp(-z));
        }
    }

    void type::FullyConnectedLayer::run(DeviceInstance &I)
    {
        //MLP kernel 1
        auto &kernel = prepare_for_running_kernel(this, I);
        int N = inputs[0]->volume / inputs[0]->dimensions.back();
        int HIDDEN = inputs[1]->dimensions.back();
        int dim_in = inputs[1]->dimensions.front();
 
        int cpu_offset = 100;
        cl::Event eventList;
        cl::NDRange global(N * HIDDEN);
        clnet::int64 global_size = global.size();
        clnet::int64 cpu_size = global_size * cpu_offset / 100;
        clnet::int64 cpu_start = global_size - cpu_size;
        clnet::int64 gpu_global_size = cpu_start;
        cl::NDRange gpu_global(gpu_global_size);
//printf("N: %d, HIDDEN: %d, dim_in: %d\n", N, HIDDEN, dim_in);
        if (gpu_run)
        {
            kernel.setArg(0, I.buffers[peers[0]]);
            kernel.setArg(1, I.buffers[inputs[0]]);
            kernel.setArg(2, I.buffers[inputs[1]]);

            if (inputs[2] != nullptr)
                //kernel.setArg(3, inputFloat);
                kernel.setArg(3, I.buffers[inputs[2]]);
            else
                kernel.setArg(3, nullptr);

            int parallel = find_proper_local_size(dim_in, I.work_group_size);
            cl::LocalSpaceArg tmpMem = cl::Local(sizeof(float) * parallel);
            kernel.setArg(4, tmpMem);
            kernel.setArg(5, HIDDEN);
            kernel.setArg(6, dim_in);
            //	if (dim_in < FULLY_CONNECTED_STD_DIM_IN_UPLIMIT) {//TODO

            //printf("kernel 1\n");
            I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, gpu_global, cl::NullRange, NULL, &eventList);
        }
        if (cpu_run)
        {
            if (inputs[2] != NULL)
            {
                feed_forward_fully_connected_sigmoid_omp(I.buffers[peers[0]], I.buffers[inputs[0]], I.buffers[inputs[1]],
                                                         I.buffers[inputs[2]], HIDDEN, dim_in, cpu_start, global_size);
            }
            else
            {
                feed_forward_fully_connected_sigmoid_omp(I.buffers[peers[0]], I.buffers[inputs[0]], I.buffers[inputs[1]],
                                                         NULL, HIDDEN, dim_in, cpu_start, global_size);
            }
        }
        if (gpu_run)
            clWaitForEvents(1, &(eventList()));
    }
    string type::BatchNormalizedLayer::generate_source_code(DeviceInstance &I)
    {
        return kernels_source["feed_forward_batch_normalization"] + "\n" + kernels_source["feed_forward_batch_normalization_small"] + "\n" + kernels_source["feed_forward_batch_normalization_for_inference"];
    }

    void type::BatchNormalizedLayer::run(DeviceInstance &I)
    {

        bool predictOnly = CLNET_TENSOR_GLOBALS & CLNET_PREDICT_ONLY;
        int dim_in = inputs[0]->dimensions.back();
        int N = inputs[0]->volume / dim_in; //(NHW)C or (batch_size)*K
        if (predictOnly)
        {
            auto &kernel = prepare_for_running_kernel(this, I, 2);
            kernel.setArg(0, I.buffers[peers[0]]);  //out
            kernel.setArg(1, I.buffers[peers[3]]);  //moving_mean
            kernel.setArg(2, I.buffers[peers[4]]);  //moving_variance
            kernel.setArg(3, I.buffers[inputs[0]]); //in
            kernel.setArg(4, I.buffers[inputs[1]]); //gamma
            kernel.setArg(5, I.buffers[inputs[2]]); //beta
            kernel.setArg(6, epsilon);
            kernel.setArg(7, dim_in);
            kernel.setArg(8, N);
            cl::NDRange global(inputs[0]->volume);
            //printf("kernel 2\n");
            I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
        }
        else
        {
            bool big = N > I.work_group_size;
            auto &kernel = prepare_for_running_kernel(this, I, big ? 0 : 1);
            kernel.setArg(0, I.buffers[peers[0]]);  //out
            kernel.setArg(1, I.buffers[peers[1]]);  //deviation
            kernel.setArg(2, I.buffers[peers[2]]);  //std_dev
            kernel.setArg(3, I.buffers[peers[3]]);  //moving_mean
            kernel.setArg(4, I.buffers[peers[4]]);  //moving_variance
            kernel.setArg(5, I.buffers[inputs[0]]); //in
            kernel.setArg(6, I.buffers[inputs[1]]); //gamma
            kernel.setArg(7, I.buffers[inputs[2]]); //beta
            kernel.setArg(8, epsilon);
            kernel.setArg(9, momentum);
            kernel.setArg(10, dim_in);
            kernel.setArg(11, N);
            if (big)
            {
                int parallel = find_proper_local_size(N, I.work_group_size);
                cl::NDRange global(dim_in * parallel);
                cl::NDRange local(parallel);
                //printf("kernel 3\n");

                I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
            }
            else
            {
                cl::NDRange global(dim_in);
                //printf("kernel 4\n");

                I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
            }
        }
    }

    string type::DropOut::generate_source_code(DeviceInstance &I)
    {
        string code = kernels_source["feed_forward_dropout"];
        return code;
    }

    void feed_forward_dropout_omp(float *out, float *mask, int num_hidden,
                                  float p, int batch_size,
                                  clnet::int64 start, clnet::int64 global_size)
    {
        //MLP kernel 5 omp
        //int GID = get_global_id(0);
#pragma omp parallel for
        for (int n = 0; n < batch_size; n++)
        {
            for (int GID = start; GID < global_size; GID++)
                out[n * global_size + GID] *= mask[GID];
        }
    }
    void type::DropOut::run(DeviceInstance &I)
    {
        //MLP kernel 5
        if (probability_keep == 1)
            return;
        auto data = inputs[0], mask = inputs[1];
        if (I.buffers.count(this) == 0)
        { //trick: runtime initializing
            refresh_random_numbers(I, no_preconditions);
            I.buffers[this] = I.buffers[data];
        }
        auto &kernel = prepare_for_running_kernel(this, I);
        int batch_size = data->dimensions[0];
        int num_hidden = data->volume / batch_size;
        auto &mask_buffer = I.buffers[mask];

        cl::Event eventList;
        cl::NDRange global(num_hidden);
        clnet::int64 global_size = global.size();
        clnet::int64 cpu_size = global_size * cpu_offset / 100;
        clnet::int64 cpu_start = global_size - cpu_size;
        clnet::int64 gpu_global_size = cpu_start;
        cl::NDRange gpu_global(gpu_global_size);
        //printf("kernel 5\n");
        //static int count2 = 0;
        //count2++;
        //printf("count5: %d\n", count2);

        if (gpu_run)
        {
            kernel.setArg(0, I.buffers[data]);
            kernel.setArg(1, mask_buffer);
            cl::LocalSpaceArg tmpMem = cl::Local(sizeof(float) * I.work_group_size);
            kernel.setArg(2, tmpMem);
            kernel.setArg(3, num_hidden);
            kernel.setArg(4, probability_keep);
            kernel.setArg(5, batch_size);

            I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, gpu_global, cl::NullRange, NULL, &eventList);
        }
        if (cpu_run)
        {
            feed_forward_dropout_omp(I.buffers[data], mask_buffer, num_hidden,
                                     probability_keep, batch_size, cpu_start, global_size);
        }
        if (gpu_run)
        {
            clWaitForEvents(1, &(eventList()));

        }
    }

    void type::DropOut::refresh_random_numbers(DeviceInstance &I, const vector<cl::Event> &preconditions)
    {
        bernoulli_distribution distribution(probability_keep);
        auto mask = inputs[1];
        int N = mask->volume;
        for (float *p = I.pointers[mask], *end = p + N; p < end; p++)
            *p = distribution(generator) ? 1.0f : 0;
        mask->download(I, &preconditions);
    }

    string back::DropOut::generate_source_code(DeviceInstance &I)
    {
        string code = kernels_source["back_propagate_dropout"];
        return code;
    }

    void back::DropOut::run(DeviceInstance &I)
    {
        //printf("kernel 4\n");

        type::DropOut *dropout = static_cast<type::DropOut *>(peers[0]);
        if (dropout->probability_keep == 1)
            return;
        auto data = dropout->inputs[0], mask = dropout->inputs[1];
        auto &kernel = prepare_for_running_kernel(this, I);
        int batch_size = data->dimensions[0];
        int num_hidden = data->volume / batch_size;
        auto &mask_buffer = I.buffers[mask];
        kernel.setArg(0, I.buffers[data]);
        kernel.setArg(1, mask_buffer);

        cl::LocalSpaceArg tmpMem = cl::Local(sizeof(float) * I.work_group_size);
        kernel.setArg(2, tmpMem);
        kernel.setArg(3, num_hidden);
        kernel.setArg(4, dropout->probability_keep);
        kernel.setArg(5, batch_size);
        kernel.setArg(6, 0); //max_norm

        cl::NDRange global(num_hidden);
        I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
        I.precondition_events.clear();
        I.precondition_events.push_back(I.events[data]);
        //I.events[mask].wait(); //should wait for last download to be finished
        dropout->refresh_random_numbers(I, I.precondition_events);
    }

#define FULLY_CONNECTED_STD_DIM_IN_UPLIMIT_BP 4096
    string back::FullyConnectedLayer::generate_source_code(DeviceInstance &I)
    {
        auto weight = peers[2]; //peers[2]: weight
        int dim_in = weight->dimensions.front();
        string code;
        bool attached = inputs.size() > 1 && inputs[1] == this; //TODO: currently only for LSTM
        if (dim_in < FULLY_CONNECTED_STD_DIM_IN_UPLIMIT_BP)
            code = gradient_set_type("back_propagate_fully_connected_softrelu_gradient", attached);
        else
            code = kernels_source["back_propagate_fully_connected_softrelu_gradient_for_bias"] + "\n" +
                   kernels_source["back_propagate_fully_connected_softrelu_gradient_for_weight"] + "\n" +
                   gradient_set_type("back_propagate_fully_connected_softrelu_gradient_for_data", attached);
        if (activation != "softrelu")
            replace_all(code, "softrelu", activation);

        return code;
    }

    inline float softrelu_gradient(float y)
    {
        return 1.0 - exp(-y);
    }

    void back_propagate_fully_connected_softrelu_gradient_omp(float *in_grad, float *weight_grad,
                                                              float *bias_grad, float *weight, float *in,
                                                              float *out, float *out_grad, int dim_out,
                                                              int dim_in, int batch_size,
                                                              clnet::int64 start, clnet::int64 global_size)
    {
        // MLP kernel 7 omp
#pragma omp parallel for
        for (int GID = start; GID < global_size; GID++)
        {
            int k = GID % dim_in;
            int n = GID / dim_in;

            if (n < dim_out)
            {
                float sum_weight_grad = 0, sum_bias_grad = 0;
                for (int j = 0; j < batch_size; j++)
                {
                    float in_j = in[j * dim_in + k];
                    float out_grad_j = (1.0f - exp((-1) * out[j * dim_out + n])) * out_grad[j * dim_out + n];
                    sum_bias_grad += out_grad_j;
                    sum_weight_grad += in_j * out_grad_j;
                }
                if (k == 0 && bias_grad != NULL)
                    bias_grad[n] += sum_bias_grad;
                weight_grad[k * dim_out + n] += sum_weight_grad;
            }

            if (in_grad != NULL && n < batch_size)
            {
                float sum_in_grad = 0;
                for (int j = 0; j < dim_out; j++)
                {
                    float weight_j = weight[k * dim_out + j];
                    float out_grad_j = (1.0f - exp((-1) * out[n * dim_out + j])) * out_grad[n * dim_out + j];
                    sum_in_grad += weight_j * out_grad_j;
                }
                in_grad[n * dim_in + k] = sum_in_grad;
            }
        }
    }

    void back_propagate_fully_connected_softrelu_gradient_for_bias_omp(float *activation_grad, float *bias_grad,
                                                                       float *out, float *out_grad, int dim_out,
                                                                       int dim_in, int batch_size,
                                                                       clnet::int64 start, clnet::int64 global_size)
    {
        //MLP kernel 8 omp
#pragma omp parallel for
        for (int n = start; n < global_size; n++)
        {
            float sum_bias_grad = 0;
            for (int j = 0; j < batch_size; j++)
            {
                float out_grad_j = (1.0f - exp((-1) * out[j * dim_out + n])) * out_grad[j * dim_out + n];
                activation_grad[j * dim_out + n] = out_grad_j;
                sum_bias_grad += out_grad_j;
            }
            if (bias_grad != NULL)
                bias_grad[n] += sum_bias_grad;
        }
    }

    void back_propagate_fully_connected_softrelu_gradient_for_weight_omp(float *weight_grad, float *activation_grad,
                                                                         float *in, int dim_out, int dim_in,
                                                                         int batch_size,
                                                                         clnet::int64 *start, clnet::int64 *global_size)
    {
        int K = global_size[1];
#pragma omp parallel for
        for (int n = start[0]; n < global_size[0]; n++)
        {
            for (int pos = start[1]; pos < global_size[1]; pos++)
            {
                for (int k = pos; k < dim_in; k += K)
                {
                    float sum_weight_grad = 0;
                    for (int j = 0; j < batch_size; j++)
                    {
                        float in_j = in[j * dim_in + k];
                        float out_grad_j = activation_grad[j * dim_out + n];
                        sum_weight_grad += in_j * out_grad_j;
                    }
                    weight_grad[k * dim_out + n] += sum_weight_grad;
                }
            }
        }
    }

    void back_propagate_fully_connected_softrelu_gradient_for_data_omp(float *in_grad, float *weight,
                                                                       float *out, float *out_grad, int dim_out,
                                                                       int dim_in, int batch_size,
                                                                       clnet::int64 start, clnet::int64 global_size)
    {
#pragma omp parallel for
        for (int GID = start; GID < global_size; GID++)
        {
            int k = GID % dim_in;
            int n = GID / dim_in;
            float sum_in_grad = 0;
            for (int j = 0; j < dim_out; j++)
            {
                float weight_j = weight[k * dim_out + j];
                float out_grad_j = softrelu_gradient(out[n * dim_out + j]) * out_grad[n * dim_out + j];
                sum_in_grad += weight_j * out_grad_j;
            }
            if (in_grad != NULL)
                in_grad[GID + k] = sum_in_grad;
        }
    }

    void back::FullyConnectedLayer::run(DeviceInstance &I)
    {
        auto &kernel = prepare_for_running_kernel(this, I);
        auto weight = peers[2];                                 //peers[2]: weight
        int N = peers[3]->volume / peers[3]->dimensions.back(); //peers[3]: in
        int dim_in = weight->dimensions.front();
        int dim_out = weight->dimensions.back();
        auto in_gradient = peers[5];
        auto weight_gradient = peers[0];
        auto bias_gradient = peers[1];

        //MLP kernel 7
        if (dim_in < FULLY_CONNECTED_STD_DIM_IN_UPLIMIT_BP)
        {
            cl::Event eventList;
            clnet::int64 global_size = (N > dim_out ? N : dim_out) * dim_in;
            cl::NDRange global(global_size);
            clnet::int64 cpu_size = global_size * cpu_offset / 100;
            clnet::int64 cpu_start = global_size - cpu_size;
            clnet::int64 gpu_global_size = cpu_start;
            cl::NDRange gpu_global(gpu_global_size);
            if (gpu_run)
            {
                if (in_gradient != nullptr)
                    kernel.setArg(0, I.buffers[in_gradient]);
                else
                    kernel.setArg(0, nullptr);

                kernel.setArg(1, I.buffers[peers[0]]);

                if (bias_gradient != nullptr)
                    kernel.setArg(2, I.buffers[bias_gradient]);
                else
                    kernel.setArg(2, nullptr);

                kernel.setArg(3, I.buffers[peers[2]]);
                kernel.setArg(4, I.buffers[peers[3]]);
                kernel.setArg(5, I.buffers[peers[4]]);
                kernel.setArg(6, I.buffers[inputs[0]]);
                kernel.setArg(7, dim_out);
                kernel.setArg(8, dim_in);
                kernel.setArg(9, N);
          //    static int count7 = 0;
            //  count7++;
             // printf("count7: %d\n", count7);
                I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, gpu_global, cl::NullRange, NULL, &eventList);
            }
            if (cpu_run)
            {
                back_propagate_fully_connected_softrelu_gradient_omp(in_gradient != nullptr ? I.buffers[in_gradient] : nullptr, I.buffers[peers[0]],
                                                                     bias_gradient != nullptr ? I.buffers[bias_gradient] : nullptr, I.buffers[peers[2]], I.buffers[peers[3]], I.buffers[peers[4]],
                                                                     I.buffers[inputs[0]], dim_out, dim_in, N, cpu_start, global_size);
            }
            //printf("kernel 7\n");

            if (gpu_run)
            {
                clWaitForEvents(1, &(eventList()));
             //   if (err != CL_SUCCESS)
   //                 printf("Wait error in kernel 7: %d\n", err);
                //printf("kernel 7 finish\n");
            }

            cl::Event &event = I.events[weight_gradient];
            if (in_gradient != nullptr)
                I.events[in_gradient] = event;
            if (bias_gradient != nullptr)
                I.events[bias_gradient] = event;
        }
        else
        {
            //MLP kernel 8
            //printf("kernel 8\n");
            
            if (temp_global_buffer[this] == 0)
            {
                const auto &context = I.queue.getInfo<CL_QUEUE_CONTEXT>();
                temp_global_buffer[this] = (float *)clSVMAlloc(context(), CL_MEM_READ_WRITE, N * dim_out * sizeof(float), 0);
            }
            int cpu_offset = 100;
            cl::Event eventList;
            cl::NDRange global(dim_out);
            clnet::int64 global_size = global.size();
            clnet::int64 cpu_size = global_size * cpu_offset / 100;
            clnet::int64 cpu_start = global_size - cpu_size;
            clnet::int64 gpu_global_size = cpu_start;
            cl::NDRange gpu_global(gpu_global_size);

            if (gpu_run)
            {
                kernel.setArg(0, temp_global_buffer[this]);
                if (bias_gradient != nullptr)
                    kernel.setArg(1, I.buffers[bias_gradient]);
                else
                    kernel.setArg(1, nullptr);
                kernel.setArg(2, I.buffers[peers[4]]);  //out
                kernel.setArg(3, I.buffers[inputs[0]]); //out_gradient
                kernel.setArg(4, dim_out);
                kernel.setArg(5, dim_in);
                kernel.setArg(6, N);
                /*
                static int count8 = 0;
                count8++;
                printf("count8: %d\n", count8);*/

                I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, gpu_global, cl::NullRange, NULL, &eventList);
            }
            if (cpu_run)
            {
                float *arg2;
                if (bias_gradient != nullptr)
                    arg2 = I.buffers[bias_gradient];
                else
                    arg2 = NULL;
                back_propagate_fully_connected_softrelu_gradient_for_bias_omp(temp_global_buffer[this], arg2,
                                                                              I.buffers[peers[4]], I.buffers[inputs[0]], dim_out, dim_in, N, cpu_start, global_size);
            }
            if (gpu_run)
            {
                clWaitForEvents(1, &(eventList()));
            }

            

            auto &kernel1 = I.kernels[this][1];

            // MLP kernel 9
            {
                cl::Event eventList;
                //printf("kernel 9\n");
                
                //int cpu_offset = 10;
                clnet::int64 global_size[2] = {dim_out, dim_in};
                clnet::int64 cpu_size[2] = {0, dim_in};
                cpu_size[0] = global_size[0] * cpu_offset / 100;
                clnet::int64 cpu_start[2] = {0, 0};
                cpu_start[0] = global_size[0] - cpu_size[0];
                clnet::int64 gpu_global_size[2] = {0, dim_in};
                gpu_global_size[0] = global_size[0] - cpu_size[0];
                cl::NDRange gpu_global(gpu_global_size[0], gpu_global_size[1]);

                if (gpu_run)
                {
                    kernel1.setArg(0, I.buffers[weight_gradient]);
                    kernel1.setArg(1, temp_global_buffer[this]);
                    kernel1.setArg(2, I.buffers[peers[3]]); //in
                    kernel1.setArg(3, dim_out);
                    kernel1.setArg(4, dim_in);
                    kernel1.setArg(5, N);
                    /*
                static int count9 = 0;
                count9++;
                printf("count9: %d\n", count9);*/

                    I.queue.enqueueNDRangeKernel(kernel1, cl::NullRange, gpu_global, cl::NullRange,
                                                 NULL, &eventList);
                }
                if (cpu_run)
                {
                    back_propagate_fully_connected_softrelu_gradient_for_weight_omp(I.buffers[weight_gradient],
                                                                                    temp_global_buffer[this], I.buffers[peers[3]], dim_out, dim_in, N, cpu_start, global_size);
                }
                if (gpu_run)
                {
                    clWaitForEvents(1, &(eventList()));
                 //   if (err != CL_SUCCESS)
       //                 printf("Wait error in kernel 9: %d\n", err);
                   // else
                    //printf("kernel 9 finish\n");
                }
                
            }

            if (in_gradient != nullptr)
            {
                
                auto &kernel2 = prepare_for_running_kernel(this, I, 2);
                //printf("kernel 10\n");
                // MLP kernel 10
                cl::Event eventList;
                //int cpu_offset = 10;

                clnet::int64 global_size = N * dim_in;
                clnet::int64 cpu_size = global_size * cpu_offset / 100;
                clnet::int64 cpu_start = global_size - cpu_size;
                clnet::int64 gpu_global_size = cpu_start;
                cl::NDRange gpu_global(gpu_global_size);
                /*
static int count10 = 0;
count10++;
printf("count10: %d\n", count10);*/

                if (gpu_run)
                {
                    if (in_gradient != nullptr)
                        kernel2.setArg(0, I.buffers[in_gradient]);
                    else
                        kernel2.setArg(0, nullptr);
                    kernel2.setArg(1, I.buffers[weight]);
                    kernel2.setArg(2, I.buffers[peers[4]]);  //out
                    kernel2.setArg(3, I.buffers[inputs[0]]); //out_gradient
                    kernel2.setArg(4, dim_out);
                    kernel2.setArg(5, dim_in);
                    kernel2.setArg(6, N);
                    I.queue.enqueueNDRangeKernel(kernel2, cl::NullRange, gpu_global, cl::NullRange, NULL, &eventList);
                }
                if (cpu_run)
                {
                    float *arg0;
                    if (in_gradient != nullptr)
                        arg0 = I.buffers[in_gradient];
                    else
                        arg0 = NULL;

                    back_propagate_fully_connected_softrelu_gradient_for_data_omp(arg0, I.buffers[weight],
                                                                                  I.buffers[peers[4]], I.buffers[inputs[0]],
                                                                                  dim_out, dim_in, N, cpu_start, global_size);
                }
                if (gpu_run)
                {
                    clWaitForEvents(1, &(eventList()));
                 //   if (err != CL_SUCCESS)
       //                 printf("Wait error in kernel 10: %d\n", err);
                //    else
                    
                        //printf("kernel 10 finish\n");
                    
                }
                
            }
        }
    }

    string back::BatchNormalizedLayer::generate_source_code(DeviceInstance &I)
    {
        bool attached = inputs.size() > 1 && inputs[1] != nullptr; //TODO
        return gradient_set_type("back_propagate_batch_normalization", attached) + "\n" + gradient_set_type("back_propagate_batch_normalization_small", attached);
    }

    void back::BatchNormalizedLayer::run(DeviceInstance &I)
    {
        int dim_in = peers[3]->dimensions.back();
        int N = peers[3]->volume / dim_in; //(NHW)C or (batch_size)*K
        bool big = N > I.work_group_size;
        auto &kernel = prepare_for_running_kernel(this, I, big ? 0 : 1);
        kernel.setArg(0, I.buffers[peers[5]]);  //in_grad
        kernel.setArg(1, I.buffers[peers[0]]);  //gamma_grad
        kernel.setArg(2, I.buffers[peers[1]]);  //beta_grad
        kernel.setArg(3, I.buffers[peers[2]]);  //gamma
        kernel.setArg(4, I.buffers[peers[6]]);  //deviation
        kernel.setArg(5, I.buffers[peers[7]]);  //std_dev
        kernel.setArg(6, I.buffers[inputs[0]]); //out_grad
        if (big)
            kernel.setArg(7, I.buffers[peers[8]]); //deviation_grad
        else
        {
            cl::LocalSpaceArg tmpMem = cl::Local(sizeof(float) * N);
            kernel.setArg(7, tmpMem); //deviation_grad
        }
        kernel.setArg(8, dim_in);
        kernel.setArg(9, N);
        int parallel = find_proper_local_size(N, I.work_group_size);
        cl::NDRange global(dim_in * parallel);
        cl::NDRange local(parallel);
        //printf("kernel 8\n");

        I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
        I.events[peers[5]] = I.events[peers[1]] = I.events[peers[0]];
    }

    size_t &type::IterativeOptimizer::current_epoch(DeviceInstance &I)
    {
        return reinterpret_cast<size_t *>(I.pointers[this])[0];
    }

    size_t type::IterativeOptimizer::milliseconds_since_last(DeviceInstance &I)
    {
        size_t &millis = reinterpret_cast<size_t *>(I.pointers[this])[1];
        size_t last = millis;
        millis = MILLIS(0);
        return millis - last;
    }

    void type::IterativeOptimizer::run(DeviceInstance &I)
    {
        set<Tensor *> visited;
        auto graph = body(); //peers[0]
        auto others = auxiliaries();
        size_t &epoch = current_epoch(I);
        milliseconds_since_last(I);
        MiniBatch *batcher = dynamic_cast<MiniBatch *>(peers[1]);
        if (batcher == nullptr)
            for (; epoch < max_epochs; epoch++)
            {
                visited.clear();
                graph->launch(&visited, &I);

                for (auto aux : others)
                    aux->launch(&visited, &I);
                wait_for_all_kernels_finished(I);
            }
        else
            for (; epoch < max_epochs; epoch++)
            {
                while (batcher->has_next(I))
                {
                    visited.clear();
                    graph->launch(&visited, &I);
                    wait_for_all_kernels_finished(I); //avoid piling up too many events. It's a must for AMD devices.
                }

                for (auto aux : others)
                    aux->launch(&visited, &I);
                batcher->reset(I);
                wait_for_all_kernels_finished(I);
            }
    }

    string type::StochasticGradientDescentUpdater::generate_source_code(DeviceInstance &I)
    {
        return momentum != 0 ? kernels_source["update_parameters_by_stochastic_gradient_descent_with_momentum"] : kernels_source["update_parameters_by_stochastic_gradient_descent"];
    }

    // kernel: update_parameters_by_stochastic_gradient_descent_with_momentum
    void stochastic_gradient_descent_omp(float *params, float *params_grad, float learning_rate,
                                         float weight_decay, float momentum, float *velocity,
                                         clnet::int64 start, clnet::int64 global_size)
    {
        if (velocity == NULL)
        {
            velocity = (float *)calloc(sizeof(float), global_size);
        }
#pragma omp parallel for
        for (int GID = start; GID < global_size; GID++)
        {
            float gradient = params_grad[GID] + weight_decay * params[GID];
            float vt = momentum * velocity[GID] + gradient;
            params[GID] -= learning_rate * vt;
            velocity[GID] = vt;
            params_grad[GID] = 0;
        }
    }

    void type::StochasticGradientDescentUpdater::run(DeviceInstance &I)
    {
        if (I.gradients_state == static_cast<int>(peers.size()))
            for (auto param : peers)
            { //report gradients
                auto grad = param->gradient;
                I.precondition_events.clear();
                I.precondition_events.push_back(I.events[grad]);
                //I.queue.enqueueReadBuffer(I.buffers[grad], CL_FALSE, 0, grad->size, I.pointers[grad], &I.precondition_events, &I.events[grad]);
                // zcy
                I.pointers[grad] = I.buffers[grad];
                I.events[grad].setCallback(CL_COMPLETE, gradients_event_callback, &I);
            }

        if (I.parameters_state == static_cast<int>(peers.size()))
        {
            for (auto param : peers)
            { //load parameters
                I.precondition_events.clear();
                I.precondition_events.push_back(I.events[param->gradient]);
                //I.queue.enqueueWriteBuffer(I.buffers[param], CL_FALSE, 0, param->size, I.pointers[param], &I.precondition_events, &I.events[param]);
                //I.queue.enqueueFillBuffer<float>(I.buffers[param->gradient], 0, 0, param->size, &I.precondition_events, &I.events[param->gradient]);
                // zcy
                I.buffers[param] = I.pointers[param];
                memset(I.buffers[param->gradient], 0, param->size);

                I.events[param].setCallback(CL_COMPLETE, parameters_event_callback, &I);
            }
        }
        else
        {
            auto &kernel = I.kernels[this].front();

            set<Tensor *> gradients;
            
            for (int i = 0, N = peers.size(); i < N; i++)
            {

                auto parameter = peers[i];
                I.precondition_events.clear();
                I.precondition_events.push_back(I.events[parameter->gradient]);
                // MLP kernel 12
                //printf("kernel 12\n");
                cl::Event eventList;
                /*
static int count12 = 0;
count12++;
printf("count12: %d\n", count12);*/

                int cpu_offset = 100;
                clnet::int64 global_size = parameter->volume;
                clnet::int64 cpu_size = global_size * cpu_offset / 100;
                clnet::int64 cpu_start = global_size - cpu_size;
                clnet::int64 gpu_global_size = cpu_start;
                cl::NDRange gpu_global(gpu_global_size);

                if (gpu_run)
                {
                    kernel.setArg(0, I.buffers[parameter]);
                    kernel.setArg(1, I.buffers[parameter->gradient]);
                    kernel.setArg(2, learning_rate);
                    kernel.setArg(3, weight_decay);
                    if (momentum != 0)
                    {
                        auto velocity = inputs[i + N];
                        auto &velocity_buffer = I.buffers[velocity];
                        kernel.setArg(4, momentum);
                        kernel.setArg(5, velocity_buffer);
                    }
                    I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, gpu_global, cl::NullRange, NULL, &eventList);
                }
                if (cpu_run)
                {
                    auto velocity = inputs[i + N];
                    float *velocity_buffer = I.buffers[velocity];
                    stochastic_gradient_descent_omp(I.buffers[parameter], I.buffers[parameter->gradient],
                                                    learning_rate, weight_decay, momentum, velocity_buffer, cpu_start, global_size);
                }
                if (gpu_run)
                {
                    clWaitForEvents(1, &(eventList()));
                    /*
                 //   if (err != CL_SUCCESS)
       //                 printf("Wait error in kernel 12: %d\n", err);
                    else
                        //printf("kernel 12 finish\n");
                        */
                }
                
            }
            
        }
    }

    void type::StochasticGradientDescentUpdater::run_globally(DeviceInstance &I, DeviceInstance &source)
    {
        auto &kernel = I.kernels[this].front();
        cl::Event event;
        vector<cl::Event> preconditions, updates;
        for (int i = 0, N = peers.size(); i < N; i++)
        {
            auto parameter = peers[i];
            auto gradient = parameter->gradient;
            auto &gradient_buffer = I.buffers[gradient];
            auto &parameter_buffer = I.buffers[parameter];
            //I.queue.enqueueWriteBuffer(gradient_buffer, CL_FALSE, 0, gradient->size, source.pointers[gradient], NULL, &event);
            // zcy
            memcpy(gradient_buffer, source.pointers[gradient], gradient->size);

            preconditions.push_back(event);

            kernel.setArg(0, parameter_buffer);
            kernel.setArg(1, gradient_buffer);
            kernel.setArg(2, learning_rate);
            kernel.setArg(3, weight_decay);
            if (momentum != 0)
            {
                auto velocity = inputs[i + N];
                auto &velocity_buffer = I.buffers[velocity];
                kernel.setArg(4, momentum);
                kernel.setArg(5, velocity_buffer);
            }
            cl::NDRange global(parameter->volume);
            //printf("kernel 10\n");

            I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
            updates.push_back(event);
            preconditions.clear();
        }
        /*
        for (auto &ev : updates)
            ev.wait();*/
        source.gradients_state = static_cast<int>(peers.size());

        updates.clear();
        for (int i = 0, N = static_cast<int>(peers.size()); i < N; i++)
        {
            auto parameter = peers[i];
            //I.queue.enqueueReadBuffer(I.buffers[parameter], CL_FALSE, 0, parameter->size, parameter->pointer, NULL, &event); //put into tensor own data pointer
            // zcy ?? memcpy?
            parameter->pointer = I.buffers[parameter];

            updates.push_back(event);
        } /*
        for (auto &ev : updates)
            ev.wait();*/
    }

    void type::Data::run(DeviceInstance &I)
    {
        download(I, &no_preconditions);
    }

    void type::GeneralInitializer::run_globally(DeviceInstance &I)
    {
        default_random_engine generator;
        for (auto tensor : peers)
        {
            if (dynamic_cast<Weight *>(tensor) != nullptr)
            {
                if (tensor->dimensions.size() == 1)
                {
                    if (tensor->gradient != nullptr)
                    {
                        uniform_real_distribution<float> distribution(mu, sigma);
                        for (int64 i = 0; i < tensor->volume; i++)
                            tensor->pointer[i] = (float)distribution(generator);
                    }
                    else
                        for (int64 i = 0; i < tensor->volume; i++)
                            tensor->pointer[i] = 1.0f;
                }
                else
                {
                    int64 fan_in = tensor->volume / tensor->dimensions.back(), fan_out = tensor->volume / tensor->dimensions.front();
                    normal_distribution<float> distribution(mu, sigma * sqrt(2.0f / fan_in)); //Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. He, K. et al. (2015)
                    for (int64 hidden = 0; hidden < fan_out; hidden++)
                        for (int64 k = 0, K = tensor->volume / fan_out; k < K; k++)
                            tensor->pointer[k * fan_out + hidden] = (float)distribution(generator);
                }
            }
            //Bias default initialized to zero
        }
    }

    void type::GeneralInitializer::run(DeviceInstance &I)
    {
        initialization.lock();
        if (!initialized)
        {
            run_globally(I);
            initialized = true;
        }
        initialization.unlock();
        for (auto tensor : peers)
        {
            memcpy(I.pointers[tensor], tensor->pointer, tensor->size);
            tensor->download(I, &no_preconditions);
        }
    }

    string type::LSTMCell::generate_source_code(DeviceInstance &I)
    {
        return kernels_source["feed_forward_LSTM_cell"];
    }

    inline float sigmoid(float z)
    {
        return 1.0 / (1.0 + exp(-z));
    }
    //tanh is predefined
    inline float tanh_gradient(float y)
    {
        return 1.0 - y * y;
    }
    inline float sigmoid_gradient(float y)
    {
        return y * (1.0f - y);
    }
    // LSTM kernel 14
    void feed_forward_LSTM_cell_omp(float *C, float *h, float *gates_data /*cell_no_max * batch_size * 5*dim_hidden*/,
                                    float *z /*batch_size * 4*dim_hidden*/, float *tmp, int dim_hidden, int cell_no,
                                    clnet::int64 start, clnet::int64 global_size)
    {
#pragma omp parallel for
        for (int GID = start; GID < global_size; GID++)
        {
            int batch = GID / dim_hidden;
            int i_g = batch * 4 * dim_hidden + (GID % dim_hidden);
            int i_t = i_g + dim_hidden;
            int f_g = i_t + dim_hidden;
            int o_g = f_g + dim_hidden;
            float in_gate = sigmoid(z[i_g]);
            float C_candicate = std::tanh(z[i_t]);
            float forget_gate = sigmoid(z[f_g]);
            float out_gate = sigmoid(z[o_g]);
            float C_prev = C[GID]; //initialized as zero for first timestamp
            float C_t = forget_gate * C_prev + in_gate * C_candicate;
            float tanh_C_t = std::tanh(C_t);

            if (gates_data != NULL)
            {
                float *data = gates_data + cell_no * global_size * 7;
                float C_grad = out_gate * tanh_gradient(tanh_C_t);
                data[i_g] = C_candicate * sigmoid_gradient(in_gate);
                data[i_t] = in_gate * tanh_gradient(C_candicate);
                data[f_g] = C_prev * sigmoid_gradient(forget_gate);
                data[o_g] = tanh_C_t * sigmoid_gradient(out_gate);

                int p = 4 * global_size + GID;
                int c_g = p + global_size;
                int c_m = c_g + global_size;
                data[p] = h[GID]; //h_prev: initialized as zero for first timestamp
                data[c_g] = C_grad;
                data[c_m] = forget_gate;
            }
            C[GID] = C_t;
            h[GID] = out_gate * tanh_C_t;
        }
    }

    void type::LSTMCell::run(DeviceInstance &I)
    {
        auto &kernel = prepare_for_running_kernel(this, I);
        auto cell_no = I.pointers[peers[3]];
        int batch_size = peers[1]->dimensions[0];
        int dim_hidden = peers[1]->dimensions[1];
        kernel.setArg(0, I.buffers[peers[0]]); //cell
        kernel.setArg(1, I.buffers[peers[1]]); //hidden
        auto gates_data = peers[2];
        if (gates_data != nullptr)
            kernel.setArg(2, I.buffers[gates_data]);
        else
            kernel.setArg(2, nullptr); //prediction need not gates_data
        kernel.setArg(3, I.buffers[inputs[0]]);

        cl::LocalSpaceArg tmpMem = cl::Local(sizeof(float) * I.work_group_size);
        kernel.setArg(4, tmpMem);
        kernel.setArg(5, dim_hidden);
        kernel.setArg(6, static_cast<int>(cell_no[0]));

        cl::NDRange global(batch_size * dim_hidden);

        I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
        I.events[peers[1]] = I.events[peers[0]];
        cell_no[0] += cell_no[1];
    }

    string type::LSTM::generate_source_code(DeviceInstance &I)
    {
        return kernels_source["feed_forward_LSTM_recurrent"];
    }

    void type::LSTMInitializer::run(DeviceInstance &I)
    {
        for (auto tensor : peers)
            tensor->download(I, &no_preconditions); //clear to zero
    }

    vector<Tensor *> type::LSTMInitializer::auxiliaries()
    {
        return peers;
    }

    // LSTM kernel 15
    void feed_forward_LSTM_recurrent_omp(float *x_timestep, float *out, float *x, float *hidden,
                                         int timestep, int sequence_length, int dim_input, int dim_hidden,
                                         clnet::int64 start, clnet::int64 global_size)
    {
        //const int GID = get_global_id(0);
        for (int GID = start; GID < global_size; GID++)
        {
            int batch = GID / dim_hidden;
            int j = GID % dim_hidden;
            int offset = batch * sequence_length + abs(timestep);
            // fetch input timestep from batch-major data
            if (timestep >= 0)
            { //exclude the last call: need not generate x_timestep
                int m = batch * dim_input;
                int n = offset * dim_input;
                for (int index = j; index < dim_input; index += dim_hidden)
                    x_timestep[m + index] = x[n + index]; //collect timestep from batch-major data
            }

            //save hidden result to out
            if (out != NULL)
            { //exclude the first call: no hidden output at this time
                int k = (offset - 1) * dim_hidden + j;
                out[k] = hidden[GID];
            }
        }
    }

    void type::LSTM::run(DeviceInstance &I)
    {
        auto input = inputs[0], input_timestep = peers[0], output_timestep = body(); //peers[1]
        int length = input->dimensions[1];                                           //batch as major index

        auto cell_no = I.pointers[peers[2]];
        cell_no[0] = 0;
        cell_no[1] = 1;

        auto &kernel = prepare_for_running_kernel(this, I);
        int dim_input = input->dimensions.back();
        int dim_hidden = output_timestep->dimensions.back();

        cl::NDRange global(input->dimensions[0] * dim_hidden);
        // kernel 15
        cl::Event eventList;
        clnet::int64 global_size = input->dimensions[0] * dim_hidden;
        clnet::int64 cpu_size = global_size * cpu_offset / 100;
        clnet::int64 cpu_start = global_size - cpu_size;
        clnet::int64 gpu_global_size = cpu_start;
        cl::NDRange gpu_global(gpu_global_size);

        if (gpu_run)
        {
            kernel.setArg(0, I.buffers[input_timestep]);
            kernel.setArg(1, nullptr);
            kernel.setArg(2, I.buffers[input]);
            kernel.setArg(3, I.buffers[output_timestep]);
            kernel.setArg(4, 0);
            kernel.setArg(5, length);
            kernel.setArg(6, dim_input);
            kernel.setArg(7, dim_hidden);

            I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, gpu_global, cl::NullRange, NULL, &eventList);
        }
        if (cpu_run)
        {
            feed_forward_LSTM_recurrent_omp(I.buffers[input_timestep], NULL, I.buffers[input], I.buffers[output_timestep],
                                            0, length, dim_input, dim_hidden, cpu_start, global_size);
        }
        if (gpu_run)
        {
            clWaitForEvents(1, &(eventList()));
//         //   if (err != CL_SUCCESS)
 //  //             printf("Wait error in kernel 15: %d\n", err);

        }

        CLNET_TENSOR_GLOBALS |= CLNET_IN_CYCLE;
        set<Tensor *> visited;
        output_timestep->launch(&visited, &I);

        if (peers.size() > 3) //collect all timestep outputs
            kernel.setArg(1, I.buffers[peers[3]]);
        for (int timestep = 1; timestep < length; timestep++)
        {
            I.precondition_events.clear();
            I.precondition_events.push_back(I.events[output_timestep]);
            kernel.setArg(4, timestep);
            I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
            visited.clear();
            output_timestep->launch(&visited, &I);
        }
        CLNET_TENSOR_GLOBALS ^= CLNET_IN_CYCLE;
        if (peers.size() > 3)
        {
            if (gpu_run)
            {
                kernel.setArg(4, -length);
                I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, gpu_global, cl::NullRange, NULL, &eventList);
            }
            if (cpu_run)
            {
                feed_forward_LSTM_recurrent_omp(I.buffers[input_timestep], I.buffers[peers[3]], I.buffers[input], I.buffers[output_timestep],
                                                0, length, dim_input, dim_hidden, cpu_start, global_size);
            }
            if (gpu_run)
            {
                clWaitForEvents(1, &(eventList()));
             //   if (err != CL_SUCCESS)
   //                 printf("Wait error in kernel 17: %d\n", err);
          //      else
                
                    //printf("kernel 17 finish\n");
                
            }
        }
    }

    string back::LSTM::generate_source_code(DeviceInstance &I)
    {
        return kernels_source["back_propagate_LSTM_recurrent"];
    }

    void back_propagate_LSTM_recurrent_omp(float *hidden_grad, float *x_grad, float *x_timestep, float *out_grad,
                                           float *x_timestep_grad, float *x, int timestep, int sequence_length,
                                           int dim_input, int dim_hidden,
                                           clnet::int64 start, clnet::int64 global_size)
    {
//kernel 19 omp
#pragma omp parallel for
        for (int GID = start; GID < global_size; GID++)
        {
            int batch = GID / dim_hidden;
            int j = GID % dim_hidden;
            int offset = batch * sequence_length + abs(timestep);
            //save hidden result from out_grad
            if (out_grad != NULL)
            { //exclude the first call: no hidden output at this time
                int k = (offset - 1) * dim_hidden + j;
                hidden_grad[GID] += out_grad[k]; //add on back-propagation-through-time gradient
            }

            // put input gradient as batch-major data
            if (timestep > 0)
            { //exclude the last call: need not generate x_timestep_grad
                int m = batch * dim_input, n = offset * dim_input;
                for (int index = j; index < dim_input; index += dim_hidden)
                {
                    int i = m + index, k = n + index;
                    x_timestep[i] = x[k - dim_input]; //recover input
                    x_grad[k] = x_timestep_grad[i];   //restore batch-major data from timestep data
                }
            }

            else if (timestep == 0)
            { //x_timestep is ready in the first time, and need not to be prepared in the last time, so both ignored)
                int m = batch * dim_input, n = offset * dim_input;
                for (int index = j; index < dim_input; index += dim_hidden)
                    x_grad[n + index] = x_timestep_grad[m + index]; //restore batch-major data from timestep data
            }
        }
    }

    void back::LSTM::run(DeviceInstance &I)
    {
        auto input = inputs[0];
        auto input_timestep_gradient = body(); //peers[1]
        auto output_timestep_gradient = peers[2];
        int batch_size = input->dimensions[0]; //batch as major index
        int length = input->dimensions[1];
        int dim_input = input_timestep_gradient->dimensions.back();
        int dim_hidden = output_timestep_gradient->dimensions.back();

        auto cell_no = I.pointers[peers[3]];
        cell_no[1] = -1;

        auto &kernel = prepare_for_running_kernel(this, I);

        cl::NDRange global(batch_size * dim_hidden);

        cl::Event eventList;
        clnet::int64 global_size = global.size();
        clnet::int64 cpu_size = global_size * cpu_offset / 100;
        clnet::int64 cpu_start = global_size - cpu_size;
        clnet::int64 gpu_global_size = cpu_start;
        cl::NDRange gpu_global(gpu_global_size);

        if (gpu_run)
        {
            kernel.setArg(0, I.buffers[output_timestep_gradient]); //hidden_grad
            kernel.setArg(1, I.buffers[peers[0]]);                 //x_grad
            kernel.setArg(2, I.buffers[peers[4]]);                 //x_timestep
            kernel.setArg(3, I.buffers[inputs[1]]);                //out_grad
            kernel.setArg(4, I.buffers[input_timestep_gradient]);  //x_timestep_grad
            kernel.setArg(5, I.buffers[input]);                    //x
            kernel.setArg(6, -length);                             //negative value means the timestep at the end
            kernel.setArg(7, length);
            kernel.setArg(8, dim_input);
            kernel.setArg(9, dim_hidden);
            I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, gpu_global, cl::NullRange, NULL, &eventList);
        }
        if (cpu_run)
        {
            back_propagate_LSTM_recurrent_omp(I.buffers[output_timestep_gradient], I.buffers[peers[0]], I.buffers[peers[4]], I.buffers[inputs[1]],
                                              I.buffers[input_timestep_gradient], I.buffers[input], -length, length, dim_input, dim_hidden, cpu_start, global_size);
        }
        if (gpu_run)
        {
            clWaitForEvents(1, &(eventList()));
         //   if (err != CL_SUCCESS)//
         //       printf("Wait error in kernel 18: %d\n", err);
       //     else
            
                //printf("kernel 18 finish\n");
            
        }

        CLNET_TENSOR_GLOBALS |= CLNET_IN_CYCLE;
        set<Tensor *> visited;
        input_timestep_gradient->launch(&visited, &I);

        for (int timestep = length - 1; timestep > 0; timestep--)
        {
            I.precondition_events.clear();
            I.precondition_events.push_back(I.events[input_timestep_gradient]);

            if (gpu_run)
            {
                kernel.setArg(6, timestep);
                //printf("kernel 19\n");
                I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, gpu_global, cl::NullRange, NULL, &eventList);
            }
            if (cpu_run)
            {
                back_propagate_LSTM_recurrent_omp(I.buffers[output_timestep_gradient], I.buffers[peers[0]], I.buffers[peers[4]], I.buffers[inputs[1]],
                                                  I.buffers[input_timestep_gradient], I.buffers[input], timestep, length, dim_input, dim_hidden, cpu_start, global_size);
            }
            if (gpu_run)
            {
                clWaitForEvents(1, &(eventList()));
             //   if (err != CL_SUCCESS)
             //       printf("Wait error in kernel 19: %d\n", err);
            }

            visited.clear();
            input_timestep_gradient->launch(&visited, &I);
        }

        CLNET_TENSOR_GLOBALS ^= CLNET_IN_CYCLE;
        I.precondition_events.clear();
        I.precondition_events.push_back(I.events[input_timestep_gradient]);

        if (gpu_run)
        {
            kernel.setArg(3, nullptr); //out_grad
            kernel.setArg(6, 0);
            //printf("kernel 20\n");
            I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, gpu_global, cl::NullRange, NULL, &eventList);
        }
        if (cpu_run)
        {
            back_propagate_LSTM_recurrent_omp(I.buffers[output_timestep_gradient], I.buffers[peers[0]],
                                              I.buffers[peers[4]], NULL,
                                              I.buffers[input_timestep_gradient], I.buffers[input], 0, length, dim_input, dim_hidden, cpu_start, global_size);
        }
        if (gpu_run)
        {
            clWaitForEvents(1, &(eventList()));
         //   if (err != CL_SUCCESS)//
               // printf("Wait error in kernel 20: %d\n", err);
          //  else
            
                //printf("kernel 20 finish\n");
            
        }
    }

    string type::Embedding::generate_source_code(DeviceInstance &I)
    {
        return kernels_source["feed_forward_embedding"];
    }
    // kernel 21
    void feed_forward_embedding_omp(float *out, float *input,
                                    float *vector_weight, int dim_in, int vector_length,
                                    clnet::int64 start, clnet::int64 global_size, int local_size)
    {
        int parallel = local_size;
        for (int GID = start; GID < global_size; GID++)
        {
            int weight_offset = GID / parallel;
            for (int index = GID % parallel; index < dim_in; index += parallel)
                out[index * vector_length + weight_offset] = vector_weight[((int)input[index]) * vector_length + weight_offset];
        }
    }

    void type::Embedding::run(DeviceInstance &I)
    {
        auto &kernel = prepare_for_running_kernel(this, I);
        int dim_in = inputs[0]->volume;
        int vector_length = inputs[1]->dimensions[1];
        int parallel = find_proper_local_size(dim_in, I.work_group_size);

        cl::NDRange global(parallel * vector_length);

        cl::Event eventList;
        clnet::int64 global_size = global.size();
        clnet::int64 cpu_size = global_size * cpu_offset / 100;
        clnet::int64 cpu_start = global_size - cpu_size;
        clnet::int64 gpu_global_size = cpu_start;
        cl::NDRange gpu_global(gpu_global_size);

        if (gpu_run)
        {
            kernel.setArg(0, I.buffers[peers[0]]);
            kernel.setArg(1, I.buffers[inputs[0]]);
            kernel.setArg(2, I.buffers[inputs[1]]);

            cl::LocalSpaceArg tmpMem = cl::Local(sizeof(float) * parallel);
            kernel.setArg(3, tmpMem);
            kernel.setArg(4, dim_in);
            kernel.setArg(5, vector_length);

            cl::NDRange local(parallel);

            I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, gpu_global, local, NULL, &eventList);
            //printf("kernel 21\n");
        }
        if (cpu_run)
        {
            feed_forward_embedding_omp(I.buffers[peers[0]], I.buffers[inputs[0]], I.buffers[inputs[1]], dim_in,
                                       vector_length, cpu_start, global_size, parallel);
        }
        if (gpu_run)
        {
            clWaitForEvents(1, &(eventList()));

        }
    }

    string back::Embedding::generate_source_code(DeviceInstance &I)
    {
        return kernels_source["back_propagate_embedding"];
    }
    // kernel 22
    void back_propagate_embedding_omp(float *vector_weight_grad, float *input,
                                      float *out_grad, int dim_in, int vector_length, int dim_vector_num,
                                      clnet::int64 start, clnet::int64 global_size)
    {
        for (int GID = start; GID < global_size; GID++)
        {
            for (int i = 0; i < dim_in; i++)
                vector_weight_grad[((int)input[i]) * vector_length + GID] += out_grad[i * vector_length + GID];
        }
    }

    void back::Embedding::run(DeviceInstance &I)
    {
        auto &kernel = prepare_for_running_kernel(this, I);
        int dim_in = inputs[0]->volume;
        int vector_num = peers[0]->dimensions[0];
        int vector_length = peers[0]->dimensions[1];

        int parallel = find_proper_local_size(dim_in, I.work_group_size);

        cl::NDRange global(vector_length);

        cl::Event eventList;
        clnet::int64 global_size = global.size();
        clnet::int64 cpu_size = global_size * cpu_offset / 100;
        clnet::int64 cpu_start = global_size - cpu_size;
        clnet::int64 gpu_global_size = cpu_start;
        cl::NDRange gpu_global(gpu_global_size);

        if (gpu_run)
        {
            kernel.setArg(0, I.buffers[peers[0]]);
            kernel.setArg(1, I.buffers[inputs[0]]);
            kernel.setArg(2, I.buffers[inputs[1]]);
            cl::LocalSpaceArg tmpMem = cl::Local(sizeof(float) * parallel);
            kernel.setArg(3, tmpMem);
            kernel.setArg(4, dim_in);
            kernel.setArg(5, vector_length);
            kernel.setArg(6, vector_num);

            I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, gpu_global, cl::NullRange, NULL, &eventList);
            //printf("kernel 22\n");
        }
        if (cpu_run)
        {
            back_propagate_embedding_omp(I.buffers[peers[0]], I.buffers[inputs[0]], I.buffers[inputs[1]],
                                         dim_in, vector_length, vector_num, cpu_start, global_size);
        }
        if (gpu_run)
        {
            clWaitForEvents(1, &(eventList()));

        }
    }

    string back::LSTMCell::generate_source_code(DeviceInstance &I)
    {
        return kernels_source["back_propagate_LSTM_cell_gates"]; //another version: back_propagate_fully_connected_LSTM_cell
    }

    void back_propagate_LSTM_cell_gates_omp(float *z_grad, float *h_prev, float *cell_state_grad, float *h_grad,
                                            float *gates_data, int dim_hidden, int batch_size, int cell_no,
                                            clnet::int64 start, clnet::int64 global_size)
    {
//kernel 23 omp
#pragma omp parallel for
        for (int GID = start; GID < global_size; GID++)
        {
            int batch = GID / dim_hidden;
            int i_g = batch * 4 * dim_hidden + (GID % dim_hidden);
            int i_t = i_g + dim_hidden;
            int f_g = i_t + dim_hidden;
            int o_g = f_g + dim_hidden;
            int p = 4 * global_size + GID;
            int c_g = p + global_size;
            int c_m = c_g + global_size;

            float *data = gates_data + cell_no * global_size * 7;
            float h_grad_batch_one = h_grad[GID];
            float C_grad = data[c_g];
            float forget_gate = data[c_m];
            float cell_grad = h_grad_batch_one * C_grad + cell_state_grad[GID];

            z_grad[i_g] = cell_grad * data[i_g];
            z_grad[i_t] = cell_grad * data[i_t];
            z_grad[f_g] = cell_grad * data[f_g];
            z_grad[o_g] = h_grad_batch_one * data[o_g];
            h_prev[GID] = data[p];
            cell_state_grad[GID] = cell_grad * forget_gate;
        }
    }

    void back::LSTMCell::run(DeviceInstance &I)
    {
        auto &kernel = prepare_for_running_kernel(this, I);
        auto cell_no = I.pointers[peers[2]];
        int batch_size = peers[0]->dimensions[0];
        int dim_hidden = inputs[0]->dimensions.back();

        // kernel 23
        cl::Event eventList;
        cl::NDRange global(batch_size * dim_hidden);
        clnet::int64 global_size = global.size();
        clnet::int64 cpu_size = global_size * cpu_offset / 100;
        clnet::int64 cpu_start = global_size - cpu_size;
        clnet::int64 parallel = batch_size * dim_hidden;
        if (cpu_start % parallel != 0)
        {
            cpu_start = (cpu_start / parallel + 1) * parallel;
        }
        clnet::int64 gpu_global_size = cpu_start;
        cl::NDRange gpu_global(gpu_global_size);

        if (gpu_run)
        {
            kernel.setArg(0, I.buffers[peers[0]]);  //z_grad
            kernel.setArg(1, I.buffers[peers[3]]);  //h_prev
            kernel.setArg(2, I.buffers[peers[4]]);  //cell_state_grad
            kernel.setArg(3, I.buffers[inputs[0]]); //h_grad
            kernel.setArg(4, I.buffers[peers[1]]);  //gates_data

            cl::LocalSpaceArg tmpMem = cl::Local(sizeof(float) * I.work_group_size);
            kernel.setArg(5, tmpMem);
            kernel.setArg(6, dim_hidden);
            kernel.setArg(7, batch_size);
            cell_no[0] += cell_no[1];
            kernel.setArg(8, static_cast<int>(cell_no[0]));
            //printf("kernel 23\n");
            I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, gpu_global, cl::NullRange, NULL, &eventList);
        }
        if (cpu_run)
        {
            back_propagate_LSTM_cell_gates_omp(I.buffers[peers[0]], I.buffers[peers[0]], I.buffers[peers[4]], I.buffers[inputs[0]],
                                               I.buffers[peers[1]], dim_hidden, batch_size,
                                               static_cast<int>(cell_no[0]), cpu_start, global_size);
        }
        if (gpu_run)
        {
            clWaitForEvents(1, &(eventList()));

        }
    }

    string back::Loss::generate_source_code(DeviceInstance &I)
    {
        return kernels_source[function + "_loss"];
    }

    void linear_regression_loss_omp(float *out_grad, float *out, float *label, clnet::int64 start, clnet::int64 global_size)
    {
#pragma omp parallel for
        for (int GID = start; GID < global_size; GID++)
        {
            out_grad[GID] = out[GID] - label[GID];
        }
    }

    inline float negative_log_likelihood_gradient(float y, bool i_equal_j)
    {
        return i_equal_j ? y - 1.0f : y;
    }
    //Parallel: (batch_size * get_local_size(0))
    // CNN kernel 24 omp
    void negative_log_likelihood_loss_omp(float *out_grad, float *out, float *in,
                                          float *label, int dim_in, int batch_size,
                                          int local_size, clnet::int64 start, clnet::int64 global_size)
    {
        float *tmp = (float *)malloc(sizeof(float) * local_size);
        int parallel = local_size;
#pragma omp parallel
        {
            for (int GID = start; GID < global_size; GID++)
            {
                int n = GID / parallel;
                int pos = GID % parallel;

                float max_value = in[n * dim_in];
                for (int index = 1; index < dim_in; index++)
                    max_value = max(max_value, in[n * dim_in + index]);

                float sum = 0;
                for (int index = pos; index < dim_in; index += parallel)
                {
                    int k = n * dim_in + index;
                    out[k] = exp(in[k] - max_value);
                    sum += out[k];
                }

                tmp[pos] = sum;
//work_group_barrier(CLK_LOCAL_MEM_FENCE);
#pragma omp barrier
                for (int stride = parallel / 2; stride > 0; stride = stride / 2)
                {
                    if (pos < stride)
                        tmp[pos] += tmp[pos + stride];
#pragma omp barrier
                    //work_group_barrier(CLK_LOCAL_MEM_FENCE);
                }
                sum = tmp[0];

                for (int index = pos; index < dim_in; index += parallel)
                {
                    int i = ((int)label[n]), k = n * dim_in + index;
                    out[k] /= sum;
                    out_grad[k] = negative_log_likelihood_gradient(out[k], index == i) / batch_size;
                }
            }
        }
    }

    void back::Loss::run(DeviceInstance &I)
    {
        if (function == "negative_log_likelihood")
        {
            auto &kernel = prepare_for_running_kernel(this, I);
            int batch_size = peers[0]->dimensions[0];
            int dim_in = peers[0]->dimensions[1];
            int parallel = find_proper_local_size(dim_in, I.work_group_size);

            // CNN kernel 24
            cl::Event eventList;
            //printf("kernel 24\n");

            // cl::NDRange global(batch_size * parallel);
            clnet::int64 global_size = batch_size * parallel;
            clnet::int64 cpu_size = global_size * cpu_offset / 100;
            clnet::int64 cpu_start = global_size - cpu_size;
            if (cpu_start % parallel != 0)
            {
                cpu_start = (cpu_start / parallel + 1) * parallel;
            }
            clnet::int64 gpu_global_size = cpu_start;
            cl::NDRange gpu_global(gpu_global_size);

            if (gpu_run)
            {
                kernel.setArg(0, I.buffers[peers[0]]);  //out_grad
                kernel.setArg(1, I.buffers[peers[1]]);  //out
                kernel.setArg(2, I.buffers[inputs[0]]); //in
                kernel.setArg(3, I.buffers[inputs[1]]); //label
                cl::LocalSpaceArg tmpMem = cl::Local(sizeof(float) * parallel);
                kernel.setArg(4, tmpMem);
                kernel.setArg(5, dim_in);
                kernel.setArg(6, batch_size);
                cl::NDRange local(parallel);
                I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, gpu_global, local, NULL, &eventList);
            }
            if (cpu_run)
            {
                negative_log_likelihood_loss_omp(I.buffers[peers[0]], I.buffers[peers[1]], I.buffers[inputs[0]],
                                                 I.buffers[inputs[1]], dim_in, batch_size, parallel, cpu_start, global_size);
            }
            if (gpu_run)
            {
                clWaitForEvents(1, &(eventList()));

            }
        }
        else if (function == "linear_regression")
        {
            auto &kernel = prepare_for_running_kernel(this, I);
            int parallel = peers[0]->volume;

            cl::NDRange global(parallel);

            // MLP kernel 25
            //printf("kernel 25\n");
            cl::Event eventList;

            clnet::int64 global_size = parallel;
            clnet::int64 cpu_size = global_size * cpu_offset / 100;
            clnet::int64 cpu_start = global_size - cpu_size;
            clnet::int64 gpu_global_size = cpu_start;
            cl::NDRange gpu_global(gpu_global_size);
/*static int count25 = 0;
count25++;
printf("count25: %d\n", count25);
*/
            
            if (gpu_run)
            {
                kernel.setArg(0, I.buffers[peers[0]]);  //out_grad
                kernel.setArg(1, I.buffers[inputs[0]]); //y
                kernel.setArg(2, I.buffers[inputs[1]]); //label
                I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, gpu_global, cl::NullRange, NULL, &eventList);
            }
            if (cpu_run)
            {
                linear_regression_loss_omp(I.buffers[peers[0]], I.buffers[inputs[0]], I.buffers[inputs[1]], cpu_start, global_size);
            }
            if (gpu_run)
            {
                clWaitForEvents(1, &(eventList()));
             //   if (err != CL_SUCCESS)
   //                 printf("Wait error in kernel 25: %d\n", err);
              //  else
                //printf("kernel 25 finish\n");
            }
            
        }
    }

    string type::BinaryOperator::generate_source_code(DeviceInstance &I)
    {
        return kernels_source["parallel_" + function];
    }

    void parallel_plus_omp(float *a, float *b, clnet::int64 start, clnet::int64 global_size)
    {
#pragma omp parallel for
        for (int GID = start; GID < global_size; GID++)
            a[GID] += b[GID];
    }

    void type::BinaryOperator::run(DeviceInstance &I)
    {
        auto &kernel = prepare_for_running_kernel(this, I);
        if (function == "plus")
        {
            int parallel = inputs[0]->volume;
            // CNN kernel 26
            cl::Event eventList;

            clnet::int64 global_size = parallel;
            clnet::int64 cpu_size = global_size * cpu_offset / 100;
            clnet::int64 cpu_start = global_size - cpu_size;
            clnet::int64 gpu_global_size = cpu_start;
            cl::NDRange gpu_global(gpu_global_size);

            if (gpu_run)
            {
                kernel.setArg(0, I.buffers[inputs[0]]); //a
                kernel.setArg(1, I.buffers[inputs[1]]); //b
                I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, gpu_global, cl::NullRange, NULL, &eventList);
            }
            if (cpu_run)
            {
                parallel_plus_omp(I.buffers[inputs[0]], I.buffers[inputs[1]], cpu_start, global_size);
            }
            if (gpu_run)
            {
                clWaitForEvents(1, &(eventList()));
             //   if (err != CL_SUCCESS)
   //                 printf("Wait error in kernel 25: %d\n", err);
              //  else
                //printf("kernel 25 finish\n");
            }
        }
        else if (function == "add")
        {
            int parallel = inputs[0]->volume;
            kernel.setArg(0, I.buffers[peers[0]]);  //z
            kernel.setArg(1, I.buffers[inputs[0]]); //a
            kernel.setArg(2, I.buffers[inputs[1]]); //b

            cl::NDRange global(parallel);
            I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
        }
        else if (function == "multiply")
        {
            int N = inputs[1]->dimensions[0];
            int HIDDEN = inputs[0]->dimensions[0];
            int dim_in = inputs[0]->dimensions[1];
            kernel.setArg(0, I.buffers[peers[0]]);
            kernel.setArg(1, I.buffers[inputs[0]]);
            kernel.setArg(2, I.buffers[inputs[1]]);

            int parallel = find_proper_local_size(dim_in, I.work_group_size);
            cl::LocalSpaceArg tmpMem = cl::Local(sizeof(float) * parallel);
            kernel.setArg(3, tmpMem);
            kernel.setArg(4, HIDDEN);
            kernel.setArg(5, dim_in);
            if (dim_in < 16)
            {
                cl::NDRange global(N * HIDDEN);
                I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
            }
            else
            {
                cl::NDRange local(parallel);
                cl::NDRange global(N * HIDDEN * parallel);
                I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
            }
        }
    }

    string back::BinaryOperator::generate_source_code(DeviceInstance &I)
    {
        return kernels_source["parallel_" + function];
    }

    void back::BinaryOperator::run(DeviceInstance &I)
    {
        auto &kernel = prepare_for_running_kernel(this, I);
        if (function == "plus")
        {
            //zcy
            memcpy(I.buffers[peers[1]], I.buffers[inputs[0]], inputs[0]->size);

            //I.queue.enqueueCopyBuffer(I.buffers[inputs[0]], I.buffers[peers[1]], 0, 0, inputs[0]->size, &I.precondition_events, &I.events[peers[1]]);
        }
        else if (function == "add")
        {
            //zcy
            /*
		I.queue.enqueueCopyBuffer(I.buffers[inputs[0]], I.buffers[peers[0]], 0, 0, inputs[0]->size, &I.precondition_events, &I.events[peers[0]]);
		I.queue.enqueueCopyBuffer(I.buffers[inputs[0]], I.buffers[peers[1]], 0, 0, inputs[0]->size, &I.precondition_events, &I.events[peers[1]]);
	*/
            memcpy(I.buffers[peers[1]], I.buffers[inputs[0]], inputs[0]->size);
            memcpy(I.buffers[peers[0]], I.buffers[inputs[0]], inputs[0]->size);
        }
        else if (function == "multiply")
        {
            int dim_in = peers[0]->dimensions[1];
            int dim_hidden = peers[0]->dimensions[0];
            auto out_gradient = inputs[0];
            kernel.setArg(0, I.buffers[peers[0]]);
            kernel.setArg(1, I.buffers[out_gradient]);
            kernel.setArg(2, I.buffers[inputs[1]]);

            int parallel = find_proper_local_size(dim_hidden / 2, I.work_group_size); //dim_out/2: parallel needs < dim_out
            cl::LocalSpaceArg tmpMem = cl::Local(sizeof(float) * parallel);
            kernel.setArg(3, tmpMem);
            kernel.setArg(4, dim_hidden);
            kernel.setArg(5, dim_in);

            cl::NDRange local(parallel);
            cl::NDRange global(dim_hidden * dim_in);
            I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);

            kernel.setArg(0, I.buffers[peers[1]]);
            kernel.setArg(2, I.buffers[inputs[2]]);
            I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
        }
    }

    float back::Loss::L(DeviceInstance &I) //for training progress evaluation scenario, NOT high speed critical scenario
    {
        auto label = inputs[1];
        if (dynamic_cast<type::Data *>(label) == nullptr)
            label->upload(I);
        if (function == "negative_log_likelihood")
        {
            peers[1]->upload(I);
            float *predict = I.pointers[peers[1]];
            float *target = I.pointers[label];
            int N = peers[1]->dimensions[0];
            int num_classes = peers[1]->dimensions.back();
            float sum = 0;
            for (int i = 0; i < N; ++i)
            {
                int index = static_cast<int>(target[i]);
                float v = predict[i * num_classes + index];
                sum += -log(v + 1e-38f);
            }
            return sum / N;
        }
        else if (function == "linear_regression")
        {
            inputs[0]->upload(I);
            float *predict = I.pointers[inputs[0]];
            float *target = I.pointers[label];
            int N = inputs[1]->dimensions[0];
            float sum = 0;
            for (int i = 0; i < N; ++i)
            {
                float delta = predict[i] - target[i];
                sum += delta * delta;
            }
            return sum / 2 / N;
        }
        return 0;
    }

    string type::ConvolutionLayer::generate_source_code(DeviceInstance &I)
    {
        cl::Platform platform(I.device.getInfo<CL_DEVICE_PLATFORM>());
        if (platform.getInfo<CL_PLATFORM_NAME>().find("NVIDIA") == string::npos)
            cl_build_options += " -DCONVOLUTION_VECTOR"; //screen float16 issue for NVIDIA driver

        string code = kernels_source[volume > 0 ? "feed_forward_convolution_activation_relu_tiling" : "feed_forward_convolution_activation_relu"];
        if (activation != "relu")
            replace_all(code, "relu", activation);

        return code;
    }

    inline float relu(float z)
    {
        return z > 0 ? z : 0;
    }
    // CNN kernel 33 omp
    void feed_forward_convolution_activation_relu_omp(float *out, float *weight /*out_depth * kernel_height * kernel_width * in_depth*/, float *bias,
                                                      float *in /*batch_size * in_height * in_width * in_depth*/,
                                                      int in_height, int in_width, int in_depth,
                                                      int kernel_height, int kernel_width, int stride_height, int stride_width,
                                                      int padding_height, int padding_width, int batch_size,
                                                      clnet::int64 *start, clnet::int64 *global_size)
    {
        int out_height = global_size[1];
        int out_width = global_size[2];
        int out_depth = global_size[0] / batch_size;

        // convolution operation for the image locations centered at (rout, cout)
#pragma omp parallel for
        for (int GID0 = start[0]; GID0 < global_size[0]; GID0++)
        {
            int n = GID0 / out_depth;
            int filter = GID0 % out_depth;
            float sum = bias != NULL ? bias[filter] : 0;
            for (int rout = start[1]; rout < global_size[1]; rout++)
            {
                for (int cout = start[2]; cout < global_size[2]; cout++)
                {
                    int offset = ((n * out_height + rout) * out_width + cout) * out_depth + filter;
                    for (int kr = 0; kr < kernel_height; kr++)
                        for (int kc = 0; kc < kernel_width; kc++)
                        {
                            int rin = rout * stride_height + kr - padding_height;
                            int cin = cout * stride_width + kc - padding_width;
                            if (rin < 0 || rin >= in_height || cin < 0 || cin >= in_width)
                                continue;
                            int weight_index = ((filter * kernel_height + kr) * kernel_width + kc) * in_depth;
                            int in_index = ((n * in_height + rin) * in_width + cin) * in_depth;

                            for (int channel = 0; channel < in_depth; channel++) //cross channel
                                sum += weight[weight_index++] * in[in_index++];
                        }
                    out[offset] = relu(sum);
                }
            }
        }
    }

    void type::ConvolutionLayer::run(DeviceInstance &I)
    {
        auto &kernel = prepare_for_running_kernel(this, I);
        auto input = inputs[0], output = peers[0], weight = inputs[1];
        int in_height = input->dimensions[1], in_width = input->dimensions[2], in_depth = input->dimensions[3];
        int kernel_height = weight->dimensions[1], kernel_width = weight->dimensions[2];
        int batch_size = input->dimensions[0];

        int padding_height = (output->dimensions[1] * stride_size[0] - in_height + weight->dimensions[1]) / 2;
        int padding_weight = (output->dimensions[2] * stride_size[1] - in_width + weight->dimensions[2]) / 2;

        // CNN kernel 33
        cl::Event eventList;
        //printf("kernel 33\n");

        //cl::NDRange global(batch_size * output->dimensions[3], output->dimensions[1], output->dimensions[2]);

        clnet::int64 global_size[3] = {batch_size * output->dimensions[3], output->dimensions[1], output->dimensions[2]};
        clnet::int64 cpu_size[3] = {0, output->dimensions[1], output->dimensions[2]};
        cpu_size[0] = global_size[0] * cpu_offset / 100;
        clnet::int64 cpu_start[3] = {0, 0, 0};
        cpu_start[0] = global_size[0] - cpu_size[0];
        clnet::int64 gpu_global_size[3] = {0, output->dimensions[1], output->dimensions[2]};
        gpu_global_size[0] = global_size[0] - cpu_size[0];
        cl::NDRange gpu_global(gpu_global_size[0], gpu_global_size[1], gpu_global_size[2]);

        if (gpu_run)
        {
            kernel.setArg(0, I.buffers[output]);
            kernel.setArg(1, I.buffers[weight]);
            kernel.setArg(2, I.buffers[inputs[2]]);
            kernel.setArg(3, I.buffers[input]);
            kernel.setArg(4, in_height);
            kernel.setArg(5, in_width);
            kernel.setArg(6, in_depth);
            kernel.setArg(7, kernel_height);
            kernel.setArg(8, kernel_width);
            kernel.setArg(9, stride_size[0]);  //stride_height
            kernel.setArg(10, stride_size[1]); //stride_width
            kernel.setArg(11, padding_height);
            kernel.setArg(12, padding_weight);
            kernel.setArg(13, batch_size);
            I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, gpu_global, cl::NullRange, NULL, &eventList);
        }
        if (cpu_run)
        {
            feed_forward_convolution_activation_relu_omp(I.buffers[output], I.buffers[weight], I.buffers[inputs[2]],
                                                         I.buffers[input], in_height, in_width, in_depth, kernel_height, kernel_width, stride_size[0],
                                                         stride_size[1], padding_height, padding_weight, batch_size, cpu_start, global_size);
        }
        if (gpu_run)
        {
            clWaitForEvents(1, &(eventList()));
         //   if (err != CL_SUCCESS)//
           //     printf("Wait error in kernel 33: %d\n", err);
          //  else
            //printf("kernel 33 finish\n");
        }
    }

    string back::ConvolutionLayer::generate_source_code(DeviceInstance &I)
    {
        bool attached = inputs.size() > 1 && inputs[1] != nullptr; //TODO
        string code = kernels_source["back_propagate_convolution_relu_gradient_for_weight"] + "\n" + gradient_set_type("back_propagate_convolution_relu_gradient_for_input", attached);
        auto &activation = static_cast<type::ConvolutionLayer *>(peers[0])->activation;
        if (activation != "relu")
            replace_all(code, "relu", activation);

        return code;
    }

    inline float relu_gradient(float y)
    {
        return y > 0 ? 1 : 0;
    }
    // CNN kernel 34
    void back_propagate_convolution_relu_gradient_for_weight_omp(float *weight_grad /*out_depth * kernel_height * kernel_width * in_depth*/,
                                                                 float *bias_grad /*out_depth*/, float *in, float *out,
                                                                 float *out_grad, int in_height, int in_width, int in_depth,
                                                                 int out_height, int out_width, int out_depth, int stride_height, int stride_width,
                                                                 int padding_height, int padding_width, int batch_size,
                                                                 clnet::int64 *start, clnet::int64 *global_size)
    {
        int kernel_height = global_size[1];
        int kernel_width = global_size[2];
#pragma omp parallel for
        for (int GID0 = start[0]; GID0 < global_size[0]; GID0++)
        {
            int filter = GID0 / in_depth;
            int kd = GID0 % in_depth;
            for (int kr = start[1]; kr < global_size[1]; kr++)
            {
                for (int kc = start[2]; kc < global_size[2]; kc++)
                {
                    int GID = ((filter * kernel_height + kr) * kernel_width + kc) * in_depth + kd;
                    float sum_weight_grad = 0, sum_bias_grad = 0;
                    int in_offset = kd;
                    int out_offset = filter;
                    for (int n = 0; n < batch_size; n++, in_offset += in_height * in_width * in_depth, out_offset += out_height * out_width * out_depth)
                        for (int rout = 0; rout < out_height; rout++)
                        {
                            int rin = rout * stride_height + kr - padding_height;
                            if (rin < 0 || rin >= in_height)
                                continue;
                            for (int cout = 0; cout < out_width; cout++)
                            {
                                int cin = cout * stride_width + kc - padding_width;
                                if (cin < 0 || cin >= in_width)
                                    continue;
                                int in_index = in_offset + (rin * in_width + cin) * in_depth;
                                int out_index = out_offset + (rout * out_width + cout) * out_depth;
                                float out_gradient = out_grad[out_index];
                                float func_grad = relu_gradient(out[out_index]);
                                float data = in[in_index];
                                float gradient = func_grad * out_gradient;
                                sum_bias_grad += gradient;
                                sum_weight_grad += gradient * data;
                            }
                        }

                    weight_grad[GID] += sum_weight_grad;
                    if (bias_grad != NULL && kr == 0 && kc == 0 && kd == 0)
                        bias_grad[filter] += sum_bias_grad;
                }
            }
        }
    }

    // CNN kernel 35
    void back_propagate_convolution_relu_gradient_for_input_omp(float *in_grad, float *weight, float *out,
                                                                float *out_grad, int kernel_height, int kernel_width, int in_depth,
                                                                int out_height, int out_width, int out_depth, int stride_height, int stride_width,
                                                                int padding_height, int padding_width, int batch_size,
                                                                clnet::int64 *start, clnet::int64 *global_size)
    {
        int in_height = global_size[1];
        int in_width = global_size[2];
#pragma omp parallel for
        for (int GID0 = start[0]; GID0 < global_size[0]; GID0++)
        {
            int n = GID0 / in_depth;
            int channel = GID0 % in_depth;
            for (int rin = start[1]; rin < global_size[1]; rin++)
            {
                for (int cin = start[2]; cin < global_size[2]; cin++)
                {
                    int GID = ((n * in_height + rin) * in_width + cin) * in_depth + channel;
                    float sum_in_grad = 0;
                    int kernel_volume = kernel_height * kernel_width * in_depth;
                    float rout_min = max(0, (rin + padding_height - kernel_height + 1) / stride_height);
                    float rout_max = min(out_height - 1, (rin + padding_height) / stride_height);
                    float cout_min = max(0, (cin + padding_width - kernel_width + 1) / stride_width);
                    float cout_max = min(out_width - 1, (cin + padding_width) / stride_width);
                    for (int rout = rout_min; rout <= rout_max; rout++)
                    {
                        int kr = rin + padding_height - rout * stride_height;
                        if (kr < 0 || kr >= kernel_height)
                            continue;
                        for (int cout = cout_min; cout <= cout_max; cout++)
                        {
                            int kc = cin + padding_width - cout * stride_width;
                            if (kc < 0 || kc >= kernel_width)
                                continue;
                            //if (flag)
                            //	printf("rout:%d cout:%d kr:%d kc:%d\n", rout, cout, kr, kc);
                            int out_index = ((n * out_height + rout) * out_width + cout) * out_depth;
                            int weight_index = (kr * kernel_width + kc) * in_depth + channel;
                            for (int filter = 0; filter < out_depth; filter++, out_index++, weight_index += kernel_volume)
                            {
                                float out_gradient = out_grad[out_index];
                                float func_grad = relu_gradient(out[out_index]);
                                float factor = weight[weight_index];
                                sum_in_grad += func_grad * out_gradient * factor;
                                //				if (flag)
                                //					printf("	out_gradient(%d,%d):%g weight(%d,%d):%g sum_in_grad:%g\n", ((n * out_height + rout) * out_width + cout + 1), filter + 1, out_gradient, ((filter * kernel_height + kr) * kernel_width + kc + 1), channel + 1, factor, sum_in_grad);
                            }
                        }
                    }
                    in_grad[GID] = sum_in_grad;
                    //if (flag)
                    //	printf("in_grad:%g\n", sum_in_grad);
                }
            }
        }
    }

    void back::ConvolutionLayer::run(DeviceInstance &I)
    {
        auto in_gradient = peers[1], weight_gradient = peers[2], bias_gradient = peers[3], input = peers[4], output = peers[5], out_gradient = inputs[0];
        auto tensor = static_cast<type::ConvolutionLayer *>(peers[0]);
        int padding_height = (out_gradient->dimensions[1] * tensor->stride_size[0] - input->dimensions[1] + weight_gradient->dimensions[1]) / 2;
        int padding_weight = (out_gradient->dimensions[2] * tensor->stride_size[1] - input->dimensions[2] + weight_gradient->dimensions[2]) / 2;
        int in_height = input->dimensions[1], in_width = input->dimensions[2], in_depth = input->dimensions[3];
        int out_height = output->dimensions[1], out_width = output->dimensions[2], out_depth = output->dimensions[3];
        int kernel_height = weight_gradient->dimensions[1], kernel_width = weight_gradient->dimensions[2];
        int batch_size = input->dimensions[0];
        //	int local_depth = 10; //Tiling version
        {
            // CNN kernel 34
            cl::Event eventList;
            //printf("kernel 34\n");
            // cl::NDRange global(out_depth * in_depth, kernel_height, kernel_width);

            clnet::int64 global_size[3] = {out_depth * in_depth, kernel_height, kernel_width};
            clnet::int64 cpu_size[3] = {0, kernel_height, kernel_width};
            cpu_size[0] = global_size[0] * cpu_offset / 100;
            clnet::int64 cpu_start[3] = {0, 0, 0};
            cpu_start[0] = global_size[0] - cpu_size[0];
            clnet::int64 gpu_global_size[3] = {0, kernel_height, kernel_width};
            gpu_global_size[0] = global_size[0] - cpu_size[0];
            cl::NDRange gpu_global(gpu_global_size[0], gpu_global_size[1], gpu_global_size[2]);

            if (gpu_run)
            {
                auto &kernel = prepare_for_running_kernel(this, I);
                kernel.setArg(0, I.buffers[weight_gradient]);
                kernel.setArg(1, I.buffers[bias_gradient]);
                kernel.setArg(2, I.buffers[input]);
                kernel.setArg(3, I.buffers[output]);
                kernel.setArg(4, I.buffers[out_gradient]);
                kernel.setArg(5, in_height);
                kernel.setArg(6, in_width);
                kernel.setArg(7, in_depth);
                kernel.setArg(8, out_height);
                kernel.setArg(9, out_width);
                kernel.setArg(10, out_depth);
                kernel.setArg(11, tensor->stride_size[0]); //stride_height
                kernel.setArg(12, tensor->stride_size[1]); //stride_width
                kernel.setArg(13, padding_height);
                kernel.setArg(14, padding_weight);
                kernel.setArg(15, batch_size);
                I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, gpu_global, cl::NullRange, NULL, &eventList);
            }
            if (cpu_run)
            {
                back_propagate_convolution_relu_gradient_for_weight_omp(I.buffers[weight_gradient], I.buffers[bias_gradient],
                                                                        I.buffers[input], I.buffers[output], I.buffers[out_gradient], in_height, in_width, in_depth, out_height,
                                                                        out_width, out_depth, tensor->stride_size[0], tensor->stride_size[1], padding_height, padding_weight,
                                                                        batch_size, cpu_start, global_size);
            }
            if (gpu_run)
            {
                clWaitForEvents(1, &(eventList()));
             //   if (err != CL_SUCCESS)
   //                 printf("Wait error in kernel 34: %d\n", err);
              //  else
                //printf("kernel 34 finish\n");
            }
        }
        if (in_gradient == nullptr)
            return;
        {
            cl::Event eventList;
            //printf("kernel 35\n");
            //cl::NDRange global(batch_size * in_depth, in_height, in_width);

            clnet::int64 global_size[3] = {batch_size * in_depth, in_height, in_width};
            clnet::int64 cpu_size[3] = {0, in_height, in_width};
            cpu_size[0] = global_size[0] * cpu_offset / 100;
            clnet::int64 cpu_start[3] = {0, 0, 0};
            cpu_start[0] = global_size[0] - cpu_size[0];
            clnet::int64 gpu_global_size[3] = {0, in_height, in_width};
            gpu_global_size[0] = global_size[0] - cpu_size[0];
            cl::NDRange gpu_global(gpu_global_size[0], gpu_global_size[1], gpu_global_size[2]);

            if (gpu_run)
            {
                auto &kernel = prepare_for_running_kernel(this, I, 1);
                kernel.setArg(0, I.buffers[in_gradient]);
                kernel.setArg(1, I.buffers[peers[6]]); //weight
                kernel.setArg(2, I.buffers[output]);
                kernel.setArg(3, I.buffers[out_gradient]);
                kernel.setArg(4, kernel_height);
                kernel.setArg(5, kernel_width);
                kernel.setArg(6, in_depth);
                kernel.setArg(7, out_height);
                kernel.setArg(8, out_width);
                kernel.setArg(9, out_depth);
                kernel.setArg(10, tensor->stride_size[0]); //stride_height
                kernel.setArg(11, tensor->stride_size[1]); //stride_width
                kernel.setArg(12, padding_height);
                kernel.setArg(13, padding_weight);
                kernel.setArg(14, batch_size);
                I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, gpu_global, cl::NullRange, NULL, &eventList);
            }
            if (cpu_run)
            {
                back_propagate_convolution_relu_gradient_for_input_omp(I.buffers[in_gradient], I.buffers[peers[6]],
                                                                       I.buffers[output], I.buffers[out_gradient], kernel_height, kernel_width, in_depth, out_height,
                                                                       out_width, out_depth, tensor->stride_size[0], tensor->stride_size[1], padding_height,
                                                                       padding_weight, batch_size, cpu_start, global_size);
            }

            if (gpu_run)
            {
                clWaitForEvents(1, &(eventList()));
             //   if (err != CL_SUCCESS)
   //                 printf("Wait error in kernel 35: %d\n", err);
              //  else
                //printf("kernel 35 finish\n");
            }
        }
    }

    string type::Pooling::generate_source_code(DeviceInstance &I)
    {
        string code = kernels_source["feed_forward_" + type + "_pooling"];
        return code;
    }

    void feed_forward_average_pooling_omp(float *out, float *in /*batch_size * in_depth * in_height * in_width*/,
                                          int in_height, int in_width, int in_depth,
                                          int pool_height, int pool_width, int stride_height, int stride_width,
                                          int padding_height, int padding_width, int batch_size,
                                          clnet::int64 *start, clnet::int64 *global_size)
    {

        int out_height = global_size[1];
        int out_width = global_size[2];
        int out_depth = in_depth;

#pragma omp parallel for
        for (int GID0 = start[0]; GID0 < global_size[0]; GID0++)
        {
            int n = GID0 / out_depth;
            int filter = GID0 % out_depth;
            for (int rout = start[1]; rout < global_size[1]; rout++)
            {
                for (int cout = start[2]; cout < global_size[2]; cout++)
                {
                    int offset = ((n * out_height + rout) * out_width + cout) * out_depth + filter;

                    float sum = 0;
                    for (int pr = 0; pr < pool_height; pr++)
#pragma omp simd reduction(+ \
                           : sum)
                        for (int pc = 0; pc < pool_width; pc++)
                        {
                            int rin = rout * stride_height + pr - padding_height;
                            int cin = cout * stride_width + pc - padding_width;
                            if (rin < 0 || rin >= in_height || cin < 0 || cin >= in_width)
                                continue;
                            int in_index = ((n * in_height + rin) * in_width + cin) * in_depth + filter; //channel==filter
                            sum += in[in_index];
                        }
                    out[offset] = sum / pool_height / pool_width;
                }
            }
        }
    }

    void type::Pooling::run(DeviceInstance &I)
    {
        auto &kernel = prepare_for_running_kernel(this, I);
        auto input = inputs[0], output = peers[0];
        int in_height = input->dimensions[1], in_width = input->dimensions[2], in_depth = input->dimensions[3];
        int batch_size = input->dimensions[0];

        // CNN kernel 36
        //printf("kernel 36\n");
        int padding_height = (output->dimensions[1] * stride_size[0] - in_height + pooling_size[0]) / 2;
        int padding_weight = (output->dimensions[2] * stride_size[1] - in_width + pooling_size[1]) / 2;

        cl::Event eventList;
        //cl::NDRange global(batch_size * output->dimensions[3], output->dimensions[1], output->dimensions[2]);

        clnet::int64 global_size[3] = {batch_size * output->dimensions[3], output->dimensions[1], output->dimensions[2]};
        clnet::int64 cpu_size[3] = {0, output->dimensions[1], output->dimensions[2]};
        cpu_size[0] = global_size[0] * cpu_offset / 100;
        clnet::int64 cpu_start[3] = {0, 0, 0};
        cpu_start[0] = global_size[0] - cpu_size[0];
        clnet::int64 gpu_global_size[3] = {0, output->dimensions[1], output->dimensions[2]};
        gpu_global_size[0] = global_size[0] - cpu_size[0];
        cl::NDRange gpu_global(gpu_global_size[0], gpu_global_size[1], gpu_global_size[2]);

        if (gpu_run)
        {
            kernel.setArg(0, I.buffers[output]);
            kernel.setArg(1, I.buffers[input]);
            kernel.setArg(2, in_height);
            kernel.setArg(3, in_width);
            kernel.setArg(4, in_depth);
            kernel.setArg(5, pooling_size[0]); //pool_height
            kernel.setArg(6, pooling_size[1]); //pool_width
            kernel.setArg(7, stride_size[0]);  //stride_height
            kernel.setArg(8, stride_size[1]);  //stride_width
            kernel.setArg(9, padding_height);
            kernel.setArg(10, padding_weight);
            kernel.setArg(11, batch_size);
            if (type == "max")
                kernel.setArg(12, I.buffers[peers[1]]); //max_index
            I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, gpu_global, cl::NullRange, NULL, &eventList);
        }
        if (cpu_run)
        {
            feed_forward_average_pooling_omp(I.buffers[output], I.buffers[input], in_height, in_width, in_depth,
                                             pooling_size[0], pooling_size[1], stride_size[0], stride_size[1], padding_height,
                                             padding_weight, batch_size, cpu_start, global_size);
        }
        if (gpu_run)
        {
            clWaitForEvents(1, &(eventList()));
         //   if (err != CL_SUCCESS)//
//                printf("Wait error in kernel 36: %d\n", err);
        }
    }

    string back::Pooling::generate_source_code(DeviceInstance &I)
    {
        string code = kernels_source["back_propagate_" + static_cast<type::Pooling *>(peers[0])->type + "_pooling"];
        return code;
    }

    //Parallel: (batch_size * out_height * out_width * out_depth)
    void back_propagate_average_pooling_omp(float *in_grad, float *out_grad,
                                            int out_height, int out_width, int out_depth /*equals to in_depth*/,
                                            int pool_height, int pool_width, int stride_height, int stride_width,
                                            int padding_height, int padding_width, int batch_size,
                                            clnet::int64 *start, clnet::int64 *global_size)
    {

        int in_height = global_size[1];
        int in_width = global_size[2];
        int in_depth = out_depth;

#pragma omp parallel for
        for (int GID0 = start[0]; GID0 < global_size[0]; GID0++)
        {
            int n = GID0 / in_depth;
            int channel = GID0 % in_depth;
            for (int rin = start[1]; rin < global_size[1]; rin++)
            {
                for (int cin = start[2]; cin < global_size[2]; cin++)
                {
                    float gradient = 0;
                    int in_index = ((n * in_height + rin) * in_width + cin) * in_depth + channel;
                    for (int pr = 0; pr < pool_height; pr++)
                        for (int pc = 0; pc < pool_width; pc++)
                        {
                            int rout = (rin - pr + padding_height) / stride_height;
                            int cout = (cin - pc + padding_width) / stride_width;
                            if (rout < 0 || rout >= out_height || cout < 0 || cout >= out_width)
                                continue;
                            int out_index = ((n * out_height + rout) * out_width + cout) * out_depth + channel; //filter==channel
                            gradient += out_grad[out_index];
                        }
                    in_grad[in_index] = gradient / pool_height / pool_width;
                }
            }
        }
    }

    void back::Pooling::run(DeviceInstance &I)
    {
        auto &kernel = prepare_for_running_kernel(this, I);
        auto in_gradient = peers[1], out_gradient = inputs[0];
        auto tensor = static_cast<type::Pooling *>(peers[0]);
        int out_height = out_gradient->dimensions[1], out_width = out_gradient->dimensions[2], out_depth = out_gradient->dimensions[3];
        int batch_size = in_gradient->dimensions[0];
        int padding_height = (out_gradient->dimensions[1] * tensor->stride_size[0] - in_gradient->dimensions[1] + tensor->pooling_size[0]) / 2;
        int padding_weight = (out_gradient->dimensions[2] * tensor->stride_size[1] - in_gradient->dimensions[2] + tensor->pooling_size[1]) / 2;

        // CNN kernel 37

        cl::Event eventList;
        //printf("kernel 37\n");

        //cl::NDRange global(batch_size * in_gradient->dimensions[3], in_gradient->dimensions[1], in_gradient->dimensions[2]);

        clnet::int64 global_size[3] = {batch_size * in_gradient->dimensions[3], in_gradient->dimensions[1], in_gradient->dimensions[2]};
        clnet::int64 cpu_size[3] = {0, in_gradient->dimensions[1], in_gradient->dimensions[2]};
        cpu_size[0] = global_size[0] * cpu_offset / 100;
        clnet::int64 cpu_start[3] = {0, 0, 0};
        cpu_start[0] = global_size[0] - cpu_size[0];
        clnet::int64 gpu_global_size[3] = {0, in_gradient->dimensions[1], in_gradient->dimensions[2]};
        gpu_global_size[0] = global_size[0] - cpu_size[0];
        cl::NDRange gpu_global(gpu_global_size[0], gpu_global_size[1], gpu_global_size[2]);

        if (gpu_run)
        {
            kernel.setArg(0, I.buffers[in_gradient]);
            kernel.setArg(1, I.buffers[out_gradient]);
            kernel.setArg(2, out_height);
            kernel.setArg(3, out_width);
            kernel.setArg(4, out_depth);
            kernel.setArg(5, tensor->pooling_size[0]); //pool_height
            kernel.setArg(6, tensor->pooling_size[1]); //pool_width
            kernel.setArg(7, tensor->stride_size[0]);  //stride_height
            kernel.setArg(8, tensor->stride_size[1]);  //stride_width
            kernel.setArg(9, padding_height);          //padding_height
            kernel.setArg(10, padding_weight);         //padding_width
            kernel.setArg(11, batch_size);
            if (tensor->type == "max")
                kernel.setArg(12, I.buffers[peers[2]]); //max_index

            I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, gpu_global, cl::NullRange, NULL, &eventList);
        }
        if (cpu_run)
        {
            back_propagate_average_pooling_omp(I.buffers[in_gradient], I.buffers[out_gradient], out_height,
                                               out_width, out_depth, tensor->pooling_size[0], tensor->pooling_size[1], tensor->stride_size[0],
                                               tensor->stride_size[1], padding_height, padding_weight, batch_size, cpu_start, global_size);
        }
        if (gpu_run)
        {
            clWaitForEvents(1, &(eventList()));
         //   if (err != CL_SUCCESS)//
//                printf("Wait error in kernel 9: %d\n", err);
        }
    }

    void type::Reshape::run(DeviceInstance &I)
    {
        I.events[this] = I.events[inputs[0]];
    }

    void back::Reshape::run(DeviceInstance &I)
    {
        I.events[peers[0]] = I.events[this];
    }

    string type::Activation::generate_source_code(DeviceInstance &I)
    {
        string code = kernels_source["feed_forward_activation_sigmoid"];
        if (function != "sigmoid")
            replace_all(code, "sigmoid", function);

        return code;
    }

    void type::Activation::run(DeviceInstance &I)
    {
        auto &kernel = prepare_for_running_kernel(this, I);
        auto input = inputs[0], output = peers[0];
        kernel.setArg(0, I.buffers[output]);
        kernel.setArg(1, I.buffers[input]);

        cl::NDRange global(input->volume);
        I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
    }

    string back::Activation::generate_source_code(DeviceInstance &I)
    {
        bool attached = false;
        string code = gradient_set_type("back_propagate_activation_sigmoid", attached);
        if (function != "sigmoid")
            replace_all(code, "sigmoid", function);

        return code;
    }

    void back::Activation::run(DeviceInstance &I)
    {
        auto &kernel = prepare_for_running_kernel(this, I);
        auto out_grad = inputs[0], output = peers[0], in_gradient = peers[1];
        kernel.setArg(0, I.buffers[in_gradient]);
        kernel.setArg(1, I.buffers[out_grad]);
        kernel.setArg(2, I.buffers[output]);

        cl::NDRange global(output->volume);
        I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
    }

    string type::Softmax::generate_source_code(DeviceInstance &I)
    {
        string code = kernels_source["feed_forward_softmax"];
        return code;
    }

    void type::Softmax::run(DeviceInstance &I)
    {
        auto &kernel = prepare_for_running_kernel(this, I);
        int batch_size = peers[0]->dimensions[0];
        int dim_in = peers[0]->dimensions[1];
        kernel.setArg(0, I.buffers[peers[0]]);  //out
        kernel.setArg(1, I.buffers[inputs[0]]); //in

        int parallel = find_proper_local_size(dim_in, I.work_group_size);
        cl::LocalSpaceArg tmpMem = cl::Local(sizeof(float) * parallel);
        kernel.setArg(2, tmpMem);
        kernel.setArg(3, dim_in);
        kernel.setArg(4, batch_size);

        cl::NDRange global(batch_size * parallel);
        cl::NDRange local(parallel);
        I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
    }

    string type::Concatenate::generate_source_code(DeviceInstance &I)
    {
        if (axis == 0 || peers[0]->volume == peers[0]->dimensions.back())
            return string();
        string code = kernels_source["feed_forward_concatenate"];
        return code;
    }

    void type::Concatenate::run(DeviceInstance &I)
    {
        auto out = peers[0];
        int out_offset = 0;
        cl::Event event;

        const auto &context = I.queue.getInfo<CL_QUEUE_CONTEXT>();
        I.events[out] = cl::UserEvent(context);
        //new (I.pointers[this]) AssemblingEvent(inputs.size(), reinterpret_cast<cl::UserEvent *>(&I.events[out]));

        if (axis == 0 || out->volume == out->dimensions.back())
        {
            for (auto input : inputs)
            {
                if (input == nullptr || input->volume == 0) //exclude "pure kernel" tensor
                    continue;
                I.precondition_events.clear();
                I.precondition_events.push_back(I.events[input]);
                //zcy ?? out_offset?
                memcpy(I.buffers[this], I.buffers[input], input->size);
                //I.queue.enqueueCopyBuffer(I.buffers[input], I.buffers[this], 0, out_offset, input->size, &I.precondition_events, &event);
                event.setCallback(CL_COMPLETE, assembling_event_callback, I.pointers[this]);
                out_offset += input->size;
            }
        }
        else
        {
            int out_stride = 1;
            for (size_t i = axis; i < out->dimensions.size(); i++)
                out_stride *= (int)out->dimensions[i];
            int out_num = int(out->volume / out_stride);
            auto &kernel = prepare_for_running_kernel(this, I);
            kernel.setArg(0, I.buffers[peers[0]]); //out
            kernel.setArg(3, out_stride);
            kernel.setArg(4, out_num);
            int parallel = find_proper_local_size(out_num, I.work_group_size);
            cl::NDRange local(parallel);

            for (auto input : inputs)
            {
                int in_size = 1;
                for (size_t i = axis; i < input->dimensions.size(); i++)
                    in_size *= (int)input->dimensions[i];
                kernel.setArg(1, I.buffers[input]); //in
                kernel.setArg(2, out_offset);

                cl::NDRange global(in_size * parallel);
                I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
                event.setCallback(CL_COMPLETE, assembling_event_callback, I.pointers[this]);
                out_offset += in_size;
            }
        }
    }

    string type::Split::generate_source_code(DeviceInstance &I)
    {
        string code = kernels_source["feed_forward_split"];
        return code;
    }

    void type::Split::run(DeviceInstance &I)
    {
        auto in = inputs[0];
        int in_offset = 0;
        if (axis == 0 || in->volume == in->dimensions.back())
        {
            I.precondition_events.clear();
            I.precondition_events.push_back(I.events[in]);
            for (auto output : peers)
            {
                //zcy
                memcpy(I.buffers[output], I.buffers[in], output->size);
                //I.queue.enqueueCopyBuffer(I.buffers[in], I.buffers[output], in_offset, 0, output->size, &I.precondition_events, &I.events[output]);
                in_offset += output->size;
            }
        }
        else
        {
            int in_stride = 1;
            for (size_t i = axis; i < in->dimensions.size(); i++)
                in_stride *= (int)in->dimensions[i];
            int in_num = int(in->volume / in_stride);
            auto &kernel = prepare_for_running_kernel(this, I);
            kernel.setArg(1, I.buffers[inputs[0]]); //in
            kernel.setArg(3, in_stride);
            kernel.setArg(4, in_num);
            int parallel = find_proper_local_size(in_num, I.work_group_size);
            cl::NDRange local(parallel);

            for (auto output : peers)
            {
                int out_size = 1;
                for (size_t i = axis; i < output->dimensions.size(); i++)
                    out_size *= (int)output->dimensions[i];
                kernel.setArg(0, I.buffers[output]); //out
                kernel.setArg(2, in_offset);

                cl::NDRange global(out_size * parallel);
                I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
                in_offset += out_size;
            }
        }
    }

    void type::Collector::run(DeviceInstance &I)
    {
        auto input = inputs[0], output = peers[0];
        I.precondition_events.clear();
        I.precondition_events.push_back(I.events[input]);
        auto &offset = *reinterpret_cast<atomic<int64> *>(I.pointers[this]);
        int64 current = offset;
        while (!offset.compare_exchange_weak(current, current + input->volume))
            current = offset;
        memcpy(I.buffers[output] + current, I.buffers[input], input->volume * sizeof(float));

        //	I.queue.enqueueCopyBuffer(I.buffers[input], I.buffers[output], 0, current * sizeof(float), input->volume * sizeof(float), &I.precondition_events, &I.events[output]);
        if (output->volume - offset < input->volume) //this is the last one, reset the offset
            offset = 0;
    }

} // namespace clnet
