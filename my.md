{
  "lang": "string",
  "prompt": "int i = 1，if",
  "n": 1,
  "temperature": 0.8,
  "top_p": 0.9,
  "top_k": 1 
}

{
  "lang": "string",
  "prompt": "#bubble sort",
  "n": 1,
  "temperature": 0.6,
  "top_p": 0.9,
  "top_k": 3 
}

top_k must be -1 (disable), or at least 1, got -2.0.

服务器要打开所有端口监听：host='0.0.0.0',

实现其他的功能，，基本实现全部框架和功能

服务器启动代码：


CUDA_VISIBLE_DEVICES="1" python my_server.py --model /home/ubuntu/llms/llama/CodeLlama-7b-hf --trust-remote-code --tokenizer /home/ubuntu/llms/Vllm/tokenizer --gpu-memory-utilization 0.5

CUDA_VISIBLE_DEVICES="1" python my_server.py --model /home/ubuntu/llms/llama/llama2-7b-chat-convert --trust-remote-code --gpu-memory-utilization 0.5

CUDA_VISIBLE_DEVICES="1" python api_server.py --model /home/ubuntu/llms/llama/CodeLlama-7b-hf --trust-remote-code --tokenizer /home/ubuntu/llms/Vllm/tokenizer --gpu-memory-utilization 0.5

CUDA_VISIBLE_DEVICES="1" python api_server.py --model /home/ubuntu/llms/llama/CodeLlama-13b-hf --trust-remote-code --tokenizer /home/ubuntu/llms/Vllm/tokenizer --gpu-memory-utilization 0.5

python my_server.py --model /home/ubuntu/llms/llama/CodeLlama-7b-hf --trust-remote-code --tokenizer /home/ubuntu/llms/llama/CodeLlama-7b-hf

python my_server.py --model /home/ubuntu/llms/llama/CodeLlama-7b-hf --trust-remote-code --tokenizer /home/ubuntu/llms/Vllm/tokenizer

CUDA_VISIBLE_DEVICES="1" python my_server.py --model deepseek-ai/deepseek-coder-6.7b-base --trust-remote-code  --gpu-memory-utilization 0.5


服务器端测试的命令
python3 benchmark_serving.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --tokenizer /home/ubuntu/llms/Vllm/tokenizer
python3 benchmark_serving.py --dataset /home/ubuntu/llms/Vllm/vllm-main/benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json --tokenizer /home/ubuntu/llms/Vllm/tokenizer --trust-remote-code 

latency命令
CUDA_VISIBLE_DEVICES="1" python benchmark_latency.py --model /home/ubuntu/llms/llama/CodeLlama-13b-hf --trust-remote-code --tokenizer /home/ubuntu/llms/Vllm/tokenizer --input-len 8142
CUDA_VISIBLE_DEVICES="1" python benchmark_latency.py --model /home/ubuntu/llms/llama/CodeLlama-7b-hf --trust-remote-code --tokenizer /home/ubuntu/llms/Vllm/tokenizer 

throughput命令

CUDA_VISIBLE_DEVICES="1" python benchmark_throughput.py --dataset /home/ubuntu/llms/Vllm/vllm-main/benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json --model /home/ubuntu/llms/llama/CodeLlama-13b-hf --trust-remote-code --tokenizer hf-internal-testing/llama-code-tokenizer


这个才是vllm的真实运行环境
/home/ubuntu/miniconda3/envs/vllm/lib/python3.8/site-packages/vllm/engine/llm_engine.py

--gpu-memory-utilization 这个参数是选择整个gpu使用多少，0.5代表使用gpu全部显存的百分之五十

{
  "lang": "string",
  "prompt": "my name is zcx, what is your name",
  "n": 1,
  "temperature": 0.6,
  "top_p": 0.9,
  "top_k": 1.0 
}

查看进程的命令   ps -L 1511740

@classmethod
里面的cls参数代表了自身的类，该类的方法可以在未创建实例的情况下调用其他的方法
@staticmethod
在未创建实例之前不能调用类中的方法，self表示自身的类

13b,直接生成

latency : 19.86，最大长度：16384
        ：2.67，最大长度：256

throughput

最大长度：2048 Throughput: 4.93 requests/s, 2392.95 tokens/s
最大长度：16384 Throughput: 4.91 requests/s, 2380.74 tokens/s

7b，api

1024:
ternal-testing/llama-tokenizer' instead of the original tokenizer.
Total time: 115.59 s
Throughput: 8.65 requests/s
Average latency: 61.82 s
Average latency per token: 0.20 s
Average latency per output token: 1.04 s

2048:
Total time: 135.42 s
Throughput: 7.38 requests/s
Average latency: 67.32 s
Average latency per token: 0.22 s
Average latency per output token: 1.12 s

8192
Total time: 134.72 s
Throughput: 7.42 requests/s
Average latency: 66.64 s
Average latency per token: 0.21 s
Average latency per output token: 1.10 s

13b,api
8192
Total time: 218.22 s
Throughput: 4.58 requests/s
Average latency: 113.95 s
Average latency per token: 0.38 s
Average latency per output token: 2.19 s

2048
Total time: 219.55 s
Throughput: 4.55 requests/s
Average latency: 114.86 s
Average latency per token: 0.39 s
Average latency per output token: 2.19 s

1024
Total time: 186.17 s
Throughput: 5.37 requests/s
Average latency: 103.38 s
Average latency per token: 0.35 s
Average latency per output token: 1.94 s