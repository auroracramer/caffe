#include <algorithm>
#include <vector>
#include <omp.h>
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include <pthread.h>
#include <xmmintrin.h>

#define NTHR 16

namespace caffe {


template <typename Dtype> 
struct worker_t {
    Dtype *top_data;
   const Dtype *bottom_data = NULL; 
  int start;
  int end;
   Dtype negative_slope;
   int tid;
};

template <typename Dtype>
void *relu_worker(void *arg)
{
  worker_t<Dtype> *t = static_cast<worker_t<Dtype>*>(arg);
  Dtype *top_data = t->top_data;
  const Dtype *bottom_data = t->bottom_data;

  for(int i = t->start; i < t->end; i++)
  {

    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + t->negative_slope * std::min(bottom_data[i], Dtype(0));
  }
}


template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  const Dtype negative_slope = this->layer_param_.relu_param().negative_slope();

  const int chunk_size =count/omp_get_num_threads();
  omp_set_num_threads(NTHR);
  int blockSize = 32;
  #pragma omp parallel for schedule(static, chunk_size)
  for (int j = 0; j < count; j += blockSize) {
    for (int i = j; i < std::min (count, j + blockSize); i+=16) {
      __m128 zero = _mm_setzero_ps();
      __m128 bottom0 = _mm_load_ps((const float *) bottom_data+i);
      _mm_store_ps((float *)top_data+i, _mm_add_ps(_mm_max_ps(bottom0,zero), 
        _mm_mul_ps(_mm_set1_ps(negative_slope), _mm_min_ps (bottom0, zero))));
      __m128 bottom1 = _mm_load_ps((const float *) bottom_data+i+4);
      _mm_store_ps((float *)top_data+i, _mm_add_ps(_mm_max_ps(bottom0,zero), 
        _mm_mul_ps(_mm_set1_ps(negative_slope), _mm_min_ps (bottom0, zero))));
      __m128 bottom2 = _mm_load_ps((const float *) bottom_data+i+8);
      _mm_store_ps((float *)top_data+i, _mm_add_ps(_mm_max_ps(bottom0,zero), 
        _mm_mul_ps(_mm_set1_ps(negative_slope), _mm_min_ps (bottom0, zero))));
      __m128 bottom3 = _mm_load_ps((const float *) bottom_data+i+12);
      _mm_store_ps((float *)top_data+i, _mm_add_ps(_mm_max_ps(bottom0,zero), 
        _mm_mul_ps(_mm_set1_ps(negative_slope), _mm_min_ps (bottom0, zero))));
    }
  }
 }



template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = (*bottom)[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const int count = (*bottom)[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
   //omp_set_num_threads(NTHR);
    #pragma omp parallel for
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS(ReLULayer);


}  // namespace caffe
