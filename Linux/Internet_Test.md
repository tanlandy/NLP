# Internet_Test

测试交换机内网网速，以设备A与设备B之间网速为例

[参考资料](https://www.slyar.com/blog/iperf-measure-network-performance.html)

## 主要步骤

### 下载iperf

`apt install iperf`

### 其中一台作为服务器，进行监听，该ip地址为192.168.10.201

`iperf -s -u`

### 另一台作为客户端，发送进行测试

格式：

`iperf -u -c 服务端IP -b 1000M -t 60 -i 10`

-b bandwidth，这个是用来表示使用多大带宽进行发包，根据真实环境进行调整

举例：

`iperf -u -c 192.168.10.201 -b 1000M -t 60 -i 10`

### 查看结果

```shell
[  1] local 192.168.10.201 port 5001 connected with 192.168.10.251 port 35957
[ ID] Interval       Transfer     Bandwidth        Jitter   Lost/Total Datagrams
[  1] 0.0000-41.7081 sec  4.62 GBytes   950 Mbits/sec   0.055 ms 22656/3393681 (0.67%)
```
