#pragma once
#ifdef __xpu__
#include <memory>
#include <string>
#include "paddle/fluid/operators/elementwise/elementwise_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_broadcast.cu.h"
#include "paddle/fluid/operators/elementwise/elementwise_xpu.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
#else
#include <algorithm>
#include <utility>
#include "paddle/fluid/operators/elementwise/elementwise_op.h"

// only can include the headers in paddle/phi/include dirs
#include "paddle/phi/kernels/elementwise_grad_kernel.h"
#include "paddle/phi/kernels/math_kernel.h"
#endif
