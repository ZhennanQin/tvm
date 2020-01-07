# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Compile Tensorflow Models
=========================
This article is an introductory tutorial to deploy tensorflow models with TVM.

For us to begin with, tensorflow python module is required to be installed.

Please refer to https://www.tensorflow.org/install
"""

# tvm, relay
import tvm
from tvm import relay

# os and numpy
import numpy as np
import os.path

# Tensorflow imports
import tensorflow as tf
try:
    tf_compat_v1 = tf.compat.v1
except ImportError:
    tf_compat_v1 = tf

# Tensorflow utility functions
import tvm.relay.testing.tf as tf_testing

######################################################################
# Tutorials
# ---------
# Please refer docs/frontend/tensorflow.md for more details for various models
# from tensorflow.

# Target settings
# Use these commented settings to build for cuda.
#target = 'cuda'
#target_host = 'llvm'
#layout = "NCHW"
#ctx = tvm.gpu(0)
target = 'llvm -mcpu=cascadelake'
target_host = 'llvm'
layout = None
ctx = tvm.cpu(0)

######################################################################
# Import model
# ------------
# Creates tensorflow graph definition from protobuf file.

model_path = "./fp32_NoBiasAdd_optimized_graph.pb"

# To enable tensorboard log

# graph = tf.get_default_graph()
# graph_def = graph.as_graph_def()
# graph_def.ParseFromString(tf_compat_v1.gfile.FastGFile(model_path, 'rb').read())
# tf.import_graph_def(graph_def, name='graph')
# summaryWriter = tf.summary.FileWriter('log/', graph)

with tf_compat_v1.gfile.GFile(model_path, 'rb') as f:
    graph_def = tf_compat_v1.GraphDef()
    graph_def.ParseFromString(f.read())
    graph = tf.import_graph_def(graph_def, name='')
    # Call the utility to import the graph definition into default graph.
    graph_def = tf_testing.ProcessGraphDefParam(graph_def)
    # Add shapes to the graph.
    with tf_compat_v1.Session() as sess:
        graph_def = tf_testing.AddShapesToGraphDef(sess, 'resnet_v2_50/predictions/Reshape_1')

######################################################################
# Decode image
# ------------
# .. note::
#
#   tensorflow frontend import doesn't support preprocessing ops like JpegDecode.
#   JpegDecode is bypassed (just return source node).
#   Hence we supply decoded frame to TVM instead.
#

x = np.random.uniform(size=(1, 224, 224, 3)).astype('float32')

######################################################################
# Import the graph to Relay
# -------------------------
# Import tensorflow graph definition to relay frontend.
#
# Results:
#   sym: relay expr for given tensorflow protobuf.
#   params: params converted from tensorflow params (tensor protobuf).
shape_dict = {'input': x.shape}
dtype_dict = {'input': 'float32'}
mod, params = relay.frontend.from_tensorflow(graph_def,
                                             layout=layout,
                                             shape=shape_dict)

print("Tensorflow protobuf imported to relay frontend.")
######################################################################
# Relay Build
# -----------
# Compile the graph to llvm target with given input specification.
#
# Results:
#   graph: Final graph after compilation.
#   params: final params after compilation.
#   lib: target library which can be deployed on target with TVM runtime.

# enable quantization

with relay.quantize.qconfig(skip_conv_layers=[]):
    mod = relay.quantize.quantize(mod, params=params)

#print(mod)

with relay.build_config(opt_level=3):
    graph, lib, params = relay.build(mod,
                                     target=target,
                                     # target_host=target_host,
                                     params=params)

######################################################################
# Execute the portable graph on TVM
# ---------------------------------
# Now we can try deploying the compiled model on target.

from tvm.contrib import graph_runtime
dtype = 'float32'
m = graph_runtime.create(graph, lib, ctx)
# set inputs
m.set_input('input', tvm.nd.array(x.astype(dtype)))
m.set_input(**params)
# execute
m.run()

# evaluate
print("Evaluate inference time cost...")
ftimer = m.module.time_evaluator("run", ctx, number=100, repeat=3)
prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
        (np.mean(prof_res), np.std(prof_res)))
