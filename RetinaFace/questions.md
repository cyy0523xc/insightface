# 踩坑记录

### 运行久了可能会报显存OOM

```
  File "/usr/local/lib/python3.5/dist-packages/mxnet/base.py", line 252, in check_call
    raise MXNetError(py_str(_LIB.MXGetLastError()))
mxnet.base.MXNetError: [03:58:39] /home/travis/build/dmlc/mxnet-distro/mxnet-build/3rdparty/mshadow/mshadow/././././cuda/tensor_gpu-inl.cuh:110: Check failed: err == cudaSuccess (2 vs. 0) Name: MapPlanKernel ErrStr:out of memory
```

