// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
// 

std::vector<float*>
dynamicconv_cpu_forward(float* input, float* filters, int padding_l);

std::vector<float*> dynamicconv_cpu_backward(
    float* gradOutput,
    int padding_l,
    float* input,
    float* filters);

std::vector<float*>
dynamicconv_forward(float* input, float* filters, int padding_l) {
  return dynamicconv_cpu_forward(input, filters, padding_l);
}

std::vector<float*> dynamicconv_backward(
    float* gradOutput,
    int padding_l,
    float* input,
    float* filters) {
  return dynamicconv_cpu_backward(gradOutput, padding_l, input, filters);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &dynamicconv_forward, "dynamicconv forward (CPU)");
  m.def("backward", &dynamicconv_backward, "dynamicconv backward (CPU)");
}
