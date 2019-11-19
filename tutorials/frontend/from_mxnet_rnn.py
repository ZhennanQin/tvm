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
.. _tutorial-from-mxnet:

Compile MXNet Models
====================
**Author**: `Joshua Z. Zhang <https://zhreshold.github.io/>`_, \
            `Kazutaka Morita <https://github.com/kazum>`_

This article is an introductory tutorial to deploy mxnet models with Relay.

For us to begin with, mxnet module is required to be installed.

A quick solution is

.. code-block:: bash

    pip install mxnet --user

or please refer to offical installation guide.
https://mxnet.incubator.apache.org/versions/master/install/index.html
"""
# some standard imports
import mxnet as mx
import tvm
import tvm.relay as relay
import numpy as np
import time, sys
from os import path as osp
from tvm.contrib import cc

target_dir = "tvm_model"
data_shape = (30, 1, 800)
merge_outputs = True

def get_mxnet_model():
    model = mx.gluon.rnn.LSTM(2, num_layers=6, bidirectional=False)
    model.hybridize(static_alloc=True, static_shape=True)
    return model

def convert_mxnet_lstm(name, fuse, opt_level):
    model = get_mxnet_model()
    net_name = get_tvm_model_name(name, fuse, opt_level)
    if fuse:
        model.initialize(mx.init.One())
        x = mx.nd.ones(shape=data_shape)
        model(x)
        shape_dict = {'data': x.shape}
        mod, params = relay.frontend.from_mxnet(model, shape_dict)
    else:
        model_cell = model._unfuse()
        data = mx.sym.var('data', shape=data_shape)
        inputs, axis, F, _ = mx.gluon.rnn.rnn_cell._format_sequence(
            data_shape[0], data, 'TNC', merge_outputs)
        begin_state = mx.gluon.rnn.rnn_cell._get_begin_state(model_cell, F, None, inputs, data_shape[1])
        y_cell = model_cell.unroll(data_shape[0], data, begin_state, layout='TNC', merge_outputs=merge_outputs)
        mod = mx.mod.Module(symbol=y_cell[0], label_names=None, context=mx.cpu())
        mod.bind(for_training=False,
                inputs_need_grad=False,
                data_shapes=[('data', data_shape)])
        mod.init_params(initializer=mx.init.One())
        mod.save_checkpoint(osp.join(target_dir, net_name), 0)
        x = mx.nd.ones(shape=data_shape)
        sym, arg_params, aux_params = mx.model.load_checkpoint(osp.join(target_dir, net_name), 0)
        mod, params = relay.frontend.from_mxnet(sym, {"data": data_shape}, arg_params=arg_params,
                                                aux_params=aux_params)
    return mod, params

def save_tvm_model(name, graph, lib, params):
    deploy_lib = osp.join(target_dir, name + '.o')
    deploy_so = osp.join(target_dir, name + '.so')
    lib.save(deploy_lib)
    cc.create_shared(deploy_so, [deploy_lib])

    with open(osp.join(target_dir, name + ".json"), "w") as fo:
        fo.write(graph)

    with open(osp.join(target_dir, name + ".params"), "wb") as fo:
        fo.write(relay.save_param_dict(params))

def tvm_load(name):
    graph = open(osp.join(target_dir, name + ".json")).read()
    lib = tvm.module.load(osp.join(target_dir, name + ".so"))
    params = bytearray(open(osp.join(target_dir, name + ".params"), "rb").read())
    return graph, lib, params

def get_tvm_model_name(name, fuse, opt_level):
    return name + "_fuse_" + str(fuse) + "_opt_" + str(opt_level)

def tvm_compile(name, fuse, opt_level):
    mod, params = convert_mxnet_lstm(name, fuse, opt_level)
    net_name = get_tvm_model_name(name, fuse, opt_level)
    ######################################################################
    # now compile the graph
    func = mod["main"]
    # target = 'llvm -mcpu=broadwell'
    target = 'llvm -mcpu=cascadelake'
    print("tvm compiling ...")
    tic = time.time()
    with relay.build_config(opt_level=opt_level):
        graph, lib, params = relay.build(func, target, params=params)
        save_tvm_model(net_name, graph, lib, params)
    tic = time.time() - tic
    print("tvm compiling completed. spent %f seconds" % tic)
    return graph, lib, params

def mxnet_lstm_cell():
    x = mx.nd.ones(shape=data_shape)
    model = get_mxnet_model()
    model_cell = model._unfuse()
    data = mx.sym.var('data', shape=data_shape)
    inputs, axis, F, _ = mx.gluon.rnn.rnn_cell._format_sequence(
        data_shape[0], data, 'TNC', merge_outputs)
    begin_state = mx.gluon.rnn.rnn_cell._get_begin_state(model_cell, F, None, inputs, data_shape[1])
    y_cell = model_cell.unroll(data_shape[0], data, begin_state, layout='TNC', merge_outputs=merge_outputs)
    mod = mx.mod.Module(symbol=y_cell[0], label_names=None, context=mx.cpu())
    mod.bind(for_training=False,
             inputs_need_grad=False,
             data_shapes=[('data', data_shape)])
    mod.init_params(initializer=mx.init.One())
    # get data
    data = [x]
    batch = mx.io.DataBatch(data, [])  # empty label
    dry_run = 5  # use 5 iterations to warm up
    run = 100
    for i in range(dry_run+run):
        if i == dry_run:
            tic = time.time()
        mod.forward(batch, is_train=False)
        for output in mod.get_outputs():
            output.wait_to_read()
    time_iter = (time.time() - tic) * 1000 / run
    print("mxnet lstm_cell: %.2f ms" % time_iter)

def mxnet_lstm_fuse():
    x = mx.nd.ones(shape=data_shape)
    model = get_mxnet_model()
    data = mx.sym.var('data')
    sym = model(data)
    mod = mx.mod.Module(symbol=sym, label_names=None, context=mx.cpu())
    mod.bind(for_training     = False,
             inputs_need_grad = False,
             data_shapes      = [('data', data_shape)])
    mod.init_params(initializer=mx.init.One())
    # get data
    data = [x]
    batch = mx.io.DataBatch(data, []) # empty label
    dry_run = 5  # use 5 iterations to warm up
    run = 100
    for i in range(dry_run+run):
        if i == dry_run:
            tic = time.time()
        mod.forward(batch, is_train=False)
        for output in mod.get_outputs():
            output.wait_to_read()
    time_iter = (time.time() - tic) * 1000 / run
    print("mxnet lstm fuse: %.2f ms" % time_iter)

def tvm_lstm(fuse, opt_level=0, rebuild=True, profile=False):
    name = 'lstm'
    if rebuild:
        graph, lib, params = tvm_compile(name, fuse, opt_level)
    else:
        graph, lib, params = tvm_load(get_tvm_model_name(name, fuse, opt_level))
    ######################################################################
    # Execute the portable graph on TVM
    # ---------------------------------
    # Now, we would like to reproduce the same forward computation using TVM.
    from tvm.contrib import graph_runtime
    from tvm.contrib.debugger import debug_runtime
    ctx = tvm.cpu(0)
    dtype = 'float32'
    if profile:
        m = debug_runtime.create(graph, lib, ctx)
    else:
        m = graph_runtime.create(graph, lib, ctx)
    # set inputs
    x = mx.nd.ones(shape=data_shape)
    m.set_input('data', tvm.nd.array(x.asnumpy().astype(dtype)))
    if rebuild:
        m.set_input(**params)
    else:
        m.load_params(params)
    # execute
    ftimer = m.module.time_evaluator("run", ctx, number=1, repeat=100)
    prof_res = np.array(ftimer().results) * 1000  # multiply 1000 for converting to millisecond
    # m.run()
    # # get outputs
    # tvm_output = m.get_output(0)
    # print(tvm_output)
    if profile:
        m.run()
    if fuse:
        name = "tvm lstm"
    else:
        name = "tvm lstm_cell"
    print("%-20s %-19s (%s)" % ("%s opt=%d" % (name, opt_level), "%.2f ms" %
                                np.mean(prof_res), "%.2f ms" % np.std(prof_res)))

mxnet_lstm_cell()
mxnet_lstm_fuse()
tvm_lstm(fuse=False, opt_level=0)
tvm_lstm(fuse=False, opt_level=3)
tvm_lstm(fuse=True, opt_level=0)
tvm_lstm(fuse=True, opt_level=3)
