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
Compile PyTorch Models
======================
**Author**: `Alex Wong <https://github.com/alexwong/>`_

This article is an introductory tutorial to deploy PyTorch models with Relay.

For us to begin with, PyTorch should be installed.
TorchVision is also required since we will be using it as our model zoo.

A quick solution is to install via pip

.. code-block:: bash

    pip install torch==1.4.0
    pip install torchvision==0.5.0

or please refer to official site
https://pytorch.org/get-started/locally/

PyTorch versions should be backwards compatible but should be used
with the proper TorchVision version.

Currently, TVM supports PyTorch 1.4 and 1.3. Other versions may
be unstable.
"""

import tvm
from tvm import relay

import numpy as np

from tvm.contrib.download import download_testdata
from tvm.relay.frontend.pytorch import get_graph_input_names

# PyTorch imports
import torch
import mobilenetv3

######################################################################
# Load a pretrained PyTorch model
# -------------------------------
model = mobilenetv3.mobilenetv3_small()
model = model.eval()

bs = 128

# We grab the TorchScripted model via tracing
input_shape = [bs, 3, 224, 224]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()

img = np.random.normal(size=(bs, 3, 224, 224))

######################################################################
# Import the graph to Relay
# -------------------------
# Convert PyTorch graph to Relay graph.
input_name = get_graph_input_names(scripted_model)[0]  # only one input
shape_dict = {input_name: img.shape}
mod, params = relay.frontend.from_pytorch(scripted_model,
                                          shape_dict)

######################################################################
# Relay Build
# -----------
# Compile the graph to llvm target with given input specification.
target = 'llvm -mcpu=cascadelake'
with relay.build_config(opt_level=0):
    graph, lib, params = relay.build(mod, target, params=params)

######################################################################
# Execute the portable graph on TVM
# ---------------------------------
# Now we can try deploying the compiled model on target.
# from tvm.contrib import graph_runtime
from tvm.contrib.debugger import debug_runtime as graph_runtime
dtype = 'float32'
ctx = tvm.cpu(0)
m = graph_runtime.create(graph, lib, ctx)
# Set inputs
m.set_input(input_name, tvm.nd.array(img.astype(dtype)))
m.set_input(**params)
# Execute
m.run()
# Get outputs
tvm_output = m.get_output(0)
num_iters = 100
ftimer = m.module.time_evaluator("run", ctx, number=3, repeat=num_iters)
# prof_res = np.array(ftimer().results) * 1000  # multiply 1000 for converting to millisecond
speed = bs / (ftimer().mean)
print("Inference on %d batches, batch size %d, throughput is %f img/sec" % (num_iters, bs, speed))

# get outputs
tvm_output = m.get_output(0)
