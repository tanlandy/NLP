# Clash

## 粘贴Clash及其配置文件

cd /opt/clash

## 更新bash文件

vim /root/.bashrc

在文件尾部，增加

alias clash="nohup /opt/clash/clash -d . >output.log 2>&1 &"
alias proxy="export http_proxy=<http://127.0.0.1:7890;export> https_proxy=<http://127.0.0.1:7890>"
alias unproxy="unset http_proxy;unset https_proxy"

## 激活变动

source /root/.bashrc

## 测试网络

```shell
curl -i www.google.com
```
