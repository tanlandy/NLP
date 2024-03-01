# Cuda

## 卸载

sudo apt-get purge nvidia*
sudo apt-get autoremove
sudo apt-get autoclean
sudo rm -rf /usr/local/cuda*

sudo apt-get install linux-headers-$(uname -r)

## 安装

[CUDA Toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local)

pip install -i <http://pypi.douban.com/simple/> --trusted-host pypi.douban.com
