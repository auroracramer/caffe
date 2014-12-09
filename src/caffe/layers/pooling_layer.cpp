#include <algorithm>
#include <cfloat>
#include <vector>
#include <omp.h>
#include <immintrin.h>
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void PoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  PoolingParameter pool_param = this->layer_param_.pooling_param();
  CHECK(!pool_param.has_kernel_size() !=
      !(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK(pool_param.has_kernel_size() ||
      (pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  CHECK((!pool_param.has_pad() && pool_param.has_pad_h()
      && pool_param.has_pad_w())
      || (!pool_param.has_pad_h() && !pool_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!pool_param.has_stride() && pool_param.has_stride_h()
      && pool_param.has_stride_w())
      || (!pool_param.has_stride_h() && !pool_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  if (pool_param.has_kernel_size()) {
    kernel_h_ = kernel_w_ = pool_param.kernel_size();
  } else {
    kernel_h_ = pool_param.kernel_h();
    kernel_w_ = pool_param.kernel_w();
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!pool_param.has_pad_h()) {
    pad_h_ = pad_w_ = pool_param.pad();
  } else {
    pad_h_ = pool_param.pad_h();
    pad_w_ = pool_param.pad_w();
  }
  if (!pool_param.has_stride_h()) {
    stride_h_ = stride_w_ = pool_param.stride();
  } else {
    stride_h_ = pool_param.stride_h();
    stride_w_ = pool_param.stride_w();
  }
  if (pad_h_ != 0 || pad_w_ != 0) {
    CHECK(this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_AVE
        || this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_MAX)
        << "Padding implemented only for average and max pooling.";
    CHECK_LT(pad_h_, kernel_h_);
    CHECK_LT(pad_w_, kernel_w_);
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
  pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
  if (pad_h_ || pad_w_) {
    // If we have padding, ensure that the last pooling starts strictly
    // inside the image (instead of at the padding); otherwise clip the last.
    if ((pooled_height_ - 1) * stride_h_ >= height_ + pad_h_) {
      --pooled_height_;
    }
    if ((pooled_width_ - 1) * stride_w_ >= width_ + pad_w_) {
      --pooled_width_;
    }
    CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h_);
    CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w_);
  }
  (*top)[0]->Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  if (top->size() > 1) {
    (*top)[1]->ReshapeLike(*(*top)[0]);
  }
  // If max pooling, we will initialize the vector index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_MAX && top->size() == 1) {
    max_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
        pooled_width_);
  }
  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  }
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void PoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const int top_count = (*top)[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top->size() > 1;
  int* mask = NULL;  // suppress warnings about uninitalized variables
  Dtype* top_mask = NULL;
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    if(sizeof(Dtype) == sizeof(double)) {
      // Initialize
      if (use_top_mask) {
        top_mask = (*top)[1]->mutable_cpu_data();
        caffe_set(top_count, Dtype(-1), top_mask);
      } else {
        mask = max_idx_.mutable_cpu_data();
        caffe_set(top_count, -1, mask);
      }
      caffe_set(top_count, Dtype(-FLT_MAX), top_data);
      
      omp_set_num_threads(omp_get_max_threads());
      
      // The main loop
      #pragma omp parallel for collapse(2) schedule(dynamic)
      for (int n = 0; n < bottom[0]->num(); ++n) {
        for (int c = 0; c < channels_; ++c) {
          __m256d max_val, max_ind, v1, v1_ind, v2;
          double result[8];
          int top_offset = (n * channels_ + c) * (*top)[0]->offset(0, 1);
          int bottom_offset = (n * channels_ + c) * bottom[0]->offset(0, 1);
          for (int ph = 0; ph < pooled_height_; ++ph) {
            for (int pw = 0; pw < pooled_width_; ++pw) {
              max_val = _mm256_set1_pd(-DBL_MAX);
              max_ind = _mm256_setzero_pd();
              float curr_max = -DBL_MAX;
              int curr_ind = -1;
              
              const int pool_index = ph * pooled_width_ + pw;
              int hstart = ph * stride_h_ - pad_h_;
              int wstart = pw * stride_w_ - pad_w_;
              int hend = min(hstart + kernel_h_, height_);
              int wend = min(wstart + kernel_w_, width_);
              hstart = max(hstart, 0);
              wstart = max(wstart, 0);
              int index = hstart * width_ + wstart;
              
              for (int h = hstart; h < hend; ++h) {
                int w = wstart;
                for (; w < wend - 3; w+=4) {
                  v1 = _mm256_loadu_pd((const double*)&bottom_data[bottom_offset + index]);
                  v1_ind = _mm256_set_pd(index + 3, index + 2, index + 1, index);
                  
                  // Holds 0xFFFFFFFFFFFFFFFF when max[i] > v1[i]
                  v2 = _mm256_cmp_pd(max_val, v1, _CMP_GT_OS);
                  
                  max_val = _mm256_max_pd(max_val, v1);
                  max_ind = _mm256_or_pd(_mm256_and_pd(v2, max_ind), _mm256_andnot_pd(v2, v1_ind));
                  
                  index += 4;
                }
                for(; w < wend; ++w) {
                  if(bottom_data[index + bottom_offset] > curr_max) {
                    curr_max = bottom_data[index + bottom_offset];
                    curr_ind = index;
                  }
                  ++index;
                }
                index = index - wend + wstart + width_;
              }
              v1 = _mm256_permute2f128_pd(max_val, max_val, _MM_SHUFFLE(0, 1, 0, 1));
              v1_ind = _mm256_permute2f128_pd(max_ind, max_ind, _MM_SHUFFLE(0, 1, 0, 1));
              v2 = _mm256_cmp_pd(max_val, v1, _CMP_GT_OS);
              max_val = _mm256_max_pd(max_val, v1);
              max_ind = _mm256_or_pd(_mm256_and_pd(v2, max_ind), _mm256_andnot_pd(v2, v1_ind));
              
              _mm256_storeu_pd(result, max_val);
              _mm256_storeu_pd(result + 4, max_ind);
              
              if(result[0] > curr_max) {
                curr_max = result[0];
                curr_ind = result[4];
              }
              if(result[1] > curr_max) {
                curr_max = result[1];
                curr_ind = result[5];
              }
              top_data[top_offset + pool_index] = curr_max;
              if (use_top_mask) {
                top_mask[top_offset + pool_index] = static_cast<Dtype>(curr_ind);
              } else {
                mask[top_offset + pool_index] = curr_ind;
              }
            }
          }
        }
      }
    } else if(sizeof(Dtype) == sizeof(float)) {
      // Initialize
      if (use_top_mask) {
        top_mask = (*top)[1]->mutable_cpu_data();
        caffe_set(top_count, Dtype(-1), top_mask);
      } else {
        mask = max_idx_.mutable_cpu_data();
        caffe_set(top_count, -1, mask);
      }
      caffe_set(top_count, Dtype(-FLT_MAX), top_data);
      omp_set_num_threads(omp_get_max_threads());
      
      // The main loop
      #pragma omp parallel for collapse(2) schedule(dynamic)
      for (int n = 0; n < bottom[0]->num(); ++n) {
        for (int c = 0; c < channels_; ++c) {
          __m256 max_val, max_ind, v1, v1_ind, v2;
          float result[16];
          int top_offset = (n * channels_ + c) * (*top)[0]->offset(0, 1);
          int bottom_offset = (n * channels_ + c) * bottom[0]->offset(0, 1);
          for (int ph = 0; ph < pooled_height_; ++ph) {
            for (int pw = 0; pw < pooled_width_; ++pw) {
              // Reset SIMD vectors
              max_val = _mm256_set1_ps(-FLT_MAX);
              max_ind = _mm256_setzero_ps();
              float curr_max = -FLT_MAX;
              int curr_ind = -1;
              
              const int pool_index = ph * pooled_width_ + pw;
              int hstart = ph * stride_h_ - pad_h_;
              int wstart = pw * stride_w_ - pad_w_;
              int hend = min(hstart + kernel_h_, height_);
              int wend = min(wstart + kernel_w_, width_);
              hstart = max(hstart, 0);
              wstart = max(wstart, 0);
              int index = hstart * width_ + wstart;
              
              for (int h = hstart; h < hend; ++h) {
                int w = wstart;
                for (; w < wend - 7; w+=8) {
                  v1 = _mm256_loadu_ps((const float*)&bottom_data[bottom_offset + index]);
                  v1_ind = _mm256_set_ps(index + 7, index + 6, index + 5, index + 4,index + 3, index + 2, index + 1, index);
                  
                  // Holds 0xFFFFFFFF when max[i] > v1[i]
                  v2 = _mm256_cmp_ps(max_val, v1, _CMP_GT_OS);
                  
                  max_val = _mm256_max_ps(max_val, v1);
                  max_ind = _mm256_or_ps(_mm256_and_ps(v2, max_ind), _mm256_andnot_ps(v2, v1_ind));
                  
                  index += 8;
                }
                for(; w < wend; ++w) {
                  if(bottom_data[index + bottom_offset] > curr_max) {
                    curr_max = bottom_data[index + bottom_offset];
                    curr_ind = index;
                  }
                  ++index;
                }
                index = index - wend + wstart + width_;
              }
              v1 = _mm256_permute2f128_ps(max_val, max_val, _MM_SHUFFLE(0, 1, 0, 1));
              v1_ind = _mm256_permute2f128_ps(max_ind, max_ind, _MM_SHUFFLE(0, 1, 0, 1));
              v2 = _mm256_cmp_ps(max_val, v1, _CMP_GT_OS);
              max_val = _mm256_max_ps(max_val, v1);
              max_ind = _mm256_or_ps(_mm256_and_ps(v2, max_ind), _mm256_andnot_ps(v2, v1_ind));
              
              _mm256_storeu_ps(result, max_val);
              _mm256_storeu_ps(result + 8, max_ind);
              
              if(result[0] > curr_max) {
                curr_max = result[0];
                curr_ind = result[8];
              }
              if(result[1] > curr_max) {
                curr_max = result[1];
                curr_ind = result[9];
              }
              if(result[2] > curr_max) {
                curr_max = result[2];
                curr_ind = result[10];
              }
              if(result[3] > curr_max) {
                curr_max = result[3];
                curr_ind = result[11];
              }
              top_data[top_offset + pool_index] = curr_max;
              if (use_top_mask) {
                top_mask[top_offset + pool_index] = static_cast<Dtype>(curr_ind);
              } else {
                mask[top_offset + pool_index] = curr_ind;
              }
            }
          }
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
#pragma omp parallel for
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = 0;
    }
    // The main loop
#pragma omp parallel for
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
	int bottom_offset = (n * channels_ + c) * bottom[0]->offset(0, 1);
	int top_offset = (n * channels_ + c) * (*top)[0]->offset(0, 1);
	for (int ph = 0; ph < pooled_height_; ++ph) {
	  int hstart = ph * stride_h_ - pad_h_;
	  int hend = min(hstart + kernel_h_, height_ + pad_h_);
	  const int hlen = hend - hstart;
	  hstart = max(hstart, 0);
	  hend = min(hend, height_);
          for (int pw = 0; pw < pooled_width_; ++pw) {
	    int wstart = pw * stride_w_ - pad_w_;
	    int wend = min(wstart + kernel_w_, width_ + pad_w_);
            int pool_size = hlen * (wend - wstart);
            wstart = max(wstart, 0);
            wend = min(wend, width_);
	    Dtype sum = top_data[ph * pooled_width_ + pw + top_offset];
	    if (sizeof(Dtype) == 4) {
	      int wlimit = wstart + ((wend - wstart) / 8 * 8);
	      __m256 psum = _mm256_setzero_ps();
	      for (int h = hstart; h < hend; ++h) {
		int index = h * width_ + bottom_offset;
		for (int w = wstart; w < wlimit; w += 8) {
		  psum = _mm256_add_ps(psum, _mm256_load_ps((float *)&bottom_data[index + w]));
		}
		for (int w = wlimit; w < wend; ++w) {
		  sum += bottom_data[index + w];
		}
	      }
	      float temp[8];
	      _mm256_store_ps(temp, psum);
	      sum += temp[0] + temp[1] + temp[2] + temp[3] + \
		temp[4] + temp[5] + temp[6] + temp[7];
	    } else {
	      int wlimit = wstart + ((wend - wstart) / 4 * 4);
	      __m256d psum = _mm256_setzero_pd();
	      for (int h = hstart; h < hend; ++h) {
		int index = h * width_ + bottom_offset;
		for (int w = wstart; w < wlimit; w += 4) {
		  psum = _mm256_add_pd(psum, _mm256_load_pd((double *)&bottom_data[index + w]));
		}
		for (int w = wlimit; w < wend; ++w) {
		  sum += bottom_data[index + w];
		}
	      }
	      double temp[4];
	      _mm256_store_pd(temp, psum);
	      sum += temp[0] + temp[1] + temp[2] + temp[3];
	    }
            top_data[ph * pooled_width_ + pw + top_offset] = sum / pool_size;
          }
        }
        // compute offset
        //bottom_data += bottom[0]->offset(0, 1);
        //top_data += (*top)[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set((*bottom)[0]->count(), Dtype(0), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;  // suppress warnings about uninitialized variables
  const Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // The main loop
    if (use_top_mask) {
      top_mask = top[1]->cpu_data();
    } else {
      mask = max_idx_.cpu_data();
    }
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            const int index = ph * pooled_width_ + pw;
            const int bottom_index =
                use_top_mask ? top_mask[index] : mask[index];
            bottom_diff[bottom_index] += top_diff[index];
          }
        }
        bottom_diff += (*bottom)[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    // The main loop
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                bottom_diff[h * width_ + w] +=
                  top_diff[ph * pooled_width_ + pw] / pool_size;
              }
            }
          }
        }
        // offset
        bottom_diff += (*bottom)[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}


#ifdef CPU_ONLY
STUB_GPU(PoolingLayer);
#endif

INSTANTIATE_CLASS(PoolingLayer);


}  // namespace caffe
