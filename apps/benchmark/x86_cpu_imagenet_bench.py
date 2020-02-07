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
"""Benchmark script for ImageNet models on ARM CPU.
see README.md for the usage and results of this script.
"""
import argparse

import numpy as np

import tvm
from tvm.contrib.util import tempdir
import tvm.contrib.graph_runtime as runtime
from tvm import relay

from util import get_network, print_progress

target = 'llvm -mcpu=core-avx2'
def evaluate_network(network, target, target_host, repeat, profile):
    print_progress(network)
    net, params, input_shape, output_shape = get_network(network, batch_size=1)

    print_progress("%-20s building..." % network)
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(
            net, target=target, target_host=target_host, params=params)
    if profile:
        from tvm.contrib.debugger import debug_runtime as graph_runtime
    else:
        from tvm.contrib import graph_runtime

    ctx = tvm.cpu(0)
    dtype = 'float32'
    m = graph_runtime.create(graph, lib, ctx)
    # set inputs
    data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
    m.set_input('data', tvm.nd.array(data_tvm))
    m.set_input(**params)
    # execute
    m.run()

    # evaluate
    print_progress("%-20s evaluating..." % network)
    ftimer = m.module.time_evaluator("run", ctx, number=1, repeat=repeat)
    prof_res = np.array(ftimer().results) * 1000  # multiply 1000 for converting to millisecond
    print("%-20s %-19s (%s)" % (network, "%.2f ms" % np.mean(prof_res), "%.2f ms" % np.std(prof_res)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str, choices=
                        ['resnet-18', 'resnet-34', 'resnet-50',
                         'vgg-16', 'vgg-19', 'densenet-121', 'inception_v3',
                         'mobilenet', 'squeezenet_v1.0', 'squeezenet_v1.1'],
                        help='The name of neural network')
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--profile", type=int, default=0)
    args = parser.parse_args()
    if args.network is None:
        networks = ['squeezenet_v1.1', 'mobilenet', 'resnet-18', 'vgg-16']
    else:
        networks = [args.network]
    target_host = None

    print("--------------------------------------------------")
    print("%-20s %-20s" % ("Network Name", "Mean Inference Time (std dev)"))
    print("--------------------------------------------------")
    for network in networks:
        evaluate_network(network, target, target_host, args.repeat, args.profile)
