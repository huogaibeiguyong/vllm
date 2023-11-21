安装环境：
pip install vllm

进入目录：
vllm/vllm/vllm/entrypoints

执行服务器启动命令：
python my_server.py --model 模型位置 --trust-remote-code --max-model-len 8192
