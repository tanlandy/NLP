# CUDA

## 常用命令

查看CUDA版本

`nvcc --version`

`/usr/local/cuda/bin/nvcc --version`

## 问题解决整理

### `nvcc --version` 显示未安装

1. 定位CUDA版本
`/usr/local/cuda/bin/nvcc --version`

得到

```shell
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Wed_Nov_22_10:17:15_PST_2023
Cuda compilation tools, release 12.3, V12.3.107
Build cuda_12.3.r12.3/compiler.33567101_0
```

2. 修改/root/.bashrc文件

`vim /root/.bashrc`

在文件最下方，新增如下内容

```shell
export PATH="/usr/local/cuda-12.3/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH"
```

3. 保存并关闭后，运行使更改生效

`source /root/.bashrc`

4. 解决返回正常内容

`nvcc --version`
