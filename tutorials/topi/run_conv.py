from __future__ import absolute_import, print_function

import tvm
import topi
import numpy as np

target = "llvm -mcpu=core-avx2"
data_shape = (1, 3, 224, 224)
kernel_shape = (10, 3, 1, 1)
dtype = 'float32'
ctx = tvm.context(target, 0)
data = tvm.placeholder(data_shape)
kernel = tvm.placeholder(kernel_shape)
with tvm.target.create(target):
    conv = topi.nn.conv2d(data, kernel, strides=1, padding=0, dilation=1)
    sconv = tvm.create_schedule(conv.op)
    func = tvm.build(sconv, [data, kernel])
    evaluator = func.time_evaluator(func.entry_name, ctx, number=1000)
    #create dummy data
    a = tvm.nd.array(np.random.rand(*data_shape).astype(dtype), ctx)
    b = tvm.nd.array(np.random.rand(*kernel_shape).astype(dtype), ctx)
    print('Baseline: %f ms' % (evaluator(a, b).mean * 1000))
