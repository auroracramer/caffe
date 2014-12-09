#include <cuda_runtime.h>
#include <fcntl.h>
#include <google/protobuf/text_format.h>

#include <cstring>
#include <ctime>
#include <cstdio>
#include <sstream>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/solver.hpp"
#include "caffe/vision_layers.hpp"
#include <sys/time.h>
#include <time.h>

using namespace caffe;
using namespace std;

double read_timer(){
    struct timeval start;
    gettimeofday( &start, NULL );
    return (double)((start.tv_sec) + 1.0e-6 * (start.tv_usec)) * 1000; //in milliseconds
}

double timestamp()
{
  struct timeval tv;
  gettimeofday (&tv, 0);
  return tv.tv_sec + 1e-6*tv.tv_usec;
}


double gflops_to_perform(int num, int channels_in, int height_in, int width_in)
{
    return ((float) 4*num*channels_in *height_in *width_in)/1000000000.0f;
}

//set up and benchmark layers without actually having a network.
template<typename Dtype>
int relu_speed_test(int num, int channels_in, int height_in, int width_in, string niceName)
{
    Blob<Dtype>* blob_bottom_ = new Blob<Dtype>(num, channels_in, height_in, width_in);
    Blob<Dtype>* blob_top_ = new Blob<Dtype>();
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
    blob_bottom_vec_.push_back(blob_bottom_); 
    blob_top_vec_.push_back(blob_top_);

    LayerParameter layerParams; 
    layerParams.set_type(LayerParameter_LayerType_RELU);


    ReLULayer<Dtype> reluLayer(layerParams);
    reluLayer.SetUp(blob_bottom_vec_, &(blob_top_vec_));

    // THE BENCHMARK:
    int num_runs = 10;
    double start = read_timer();
    double t0 = timestamp();
    for (int j = 0; j < num_runs; ++j)
    {
        reluLayer.Forward(blob_bottom_vec_, &(blob_top_vec_));
    }
    t0 = timestamp()-t0;
    // CUDA_CHECK(cudaDeviceSynchronize()); //for accurate timing
    double layerTime = (read_timer() - start)/num_runs; 
    double gflops_performed = gflops_to_perform(num, channels_in, height_in, width_in);
    printf("timing RELU ----------------------------------------------: %g seconds\n", t0);
    double gflops_per_sec = gflops_performed / layerTime * 1000; //*1000 for ms to sec 
    LOG(ERROR) << "    " << niceName <<  " forward: " << layerTime << " ms, " << gflops_performed << " gflops ... " << gflops_per_sec << " gflops/sec"; 

    delete blob_bottom_;
    delete blob_top_;
 
    return 0; //TODO: return 1 if error?
}


void vary_input_size(){
    std::cout << "running 'vary input size'";

    for(int inputDim = 8; inputDim <= 512; inputDim = inputDim*2){ 
        ostringstream niceName;
        niceName << "inputDim = " << "50 x 384 x " << inputDim  << " x " << inputDim << ".";

        relu_speed_test<float>(50, 384, inputDim, inputDim, niceName.str());
        LOG(ERROR) << "running";
    }
}


void vary_filter_size(){
    LOG(ERROR) << "running 'vary filter size'";
    for(int filterSize=1; filterSize<10; filterSize++) //out of memory if >10
    { 
        ostringstream niceName;
        niceName << "filterSize = " << filterSize << ".";

        relu_speed_test<float>(50, 384, 55, 55,  niceName.str());
    }
}

void vary_channels_in(){
    LOG(ERROR) << "running 'num input channels'";
    for(int channels_in=4; channels_in <= 2048; channels_in=channels_in*2) //
    { 
        ostringstream niceName;
        niceName << "channels_in = " << channels_in << ".";

        relu_speed_test<float>(50, channels_in, 55, 55, niceName.str());
    }

}

void vary_batch_size()
{
    LOG(ERROR) << "running 'num batch size'";
    for(int NUM_=1; NUM_<60; NUM_+=4)
    { 
        ostringstream niceName;
        niceName << "NUM_ = " << NUM_ << ".";

        relu_speed_test<float>(NUM_, 384, 55, 55,  niceName.str());
    }
}

void vary_num_groups()
{
    LOG(ERROR) << "running 'num groups'";
    for(int group=1; group<=8; group=group*2)
    { 
        ostringstream niceName;
        niceName << "num groups = " << group << ".";

        relu_speed_test<float>(50, 384, 55, 55, niceName.str());
    }
}

void vary_num_filters()
{
    LOG(ERROR) << "running 'num filters'";
    for(int num_output = 2; num_output < 10000; num_output=num_output*2)
    { 
        ostringstream niceName;
        niceName << "num filters = " << num_output << ".";

        relu_speed_test<float>(50, 384, 55, 55,  niceName.str());

    }
}

int main(int argc, char** argv) {
    ::google::InitGoogleLogging(argv[0]);
    cudaSetDevice(0);
    Caffe::set_mode(Caffe::CPU);
    Caffe::set_phase(Caffe::TEST);

    vary_input_size();

    return 0;
}