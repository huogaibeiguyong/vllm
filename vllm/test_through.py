import subprocess
import time

def get_gpu_model_and_count():
    """
    获取显卡模型名称和数量。

    :return: (显卡模型名称, 显卡数量)
    """
    # 查询显卡模型名称
    result_model = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], stdout=subprocess.PIPE)
    gpu_model = result_model.stdout.decode('utf-8').split('\n')[0].strip().replace(" ", "")

    # 查询显卡数量
    result_count = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv'], stdout=subprocess.PIPE)
    gpu_count = len(result_count.stdout.decode('utf-8').strip().split('\n')) - 1

    return gpu_model, gpu_count


def run_llm_and_test(model_path, tensor_parallel_size, request_rate=10):
    """
    运行LLM并对其进行性能测试。

    :param model_path: LLM模型的路径。
    :param tensor_parallel_size: 并行处理的大小。
    :param request_rate: 请求速率。
    """
    gpu_model, _ = get_gpu_model_and_count()

    # 准备启动LLM的命令
    cmd1 = f"python -m vllm.entrypoints.api_server --model {model_path} --tokenizer /home/ubuntu/llms/Vllm/tokenizer --tensor-parallel-size {tensor_parallel_size} --gpu-memory-utilization 0.5"

    model_name = model_path.split("/")[-1]
    # 定义服务器日志文件的路径
    server_log_filename = f"./logs/server_{model_name}-{gpu_model}x{tensor_parallel_size}.log"

    try:
        # 启动LLM服务器并保存输出到日志文件
        with open(server_log_filename, 'w') as server_log_file:
            server_process = subprocess.Popen(cmd1, shell=True, stdout=server_log_file, stderr=server_log_file, text=True, bufsize=1, universal_newlines=True)

            # 检查LLM服务器的输出，等待"http://localhost:8000"或其他错误信息
            while True:
                with open(server_log_filename, 'r') as f:
                    content = f.read()
                    if "http://localhost:8000" in content:
                        break
                    # 如果检测到错误信息，抛出异常
                    if "Error" in content or "Exception" in content:
                        raise Exception(f"Error detected in cmd1 output. Check {server_log_filename} for details.")
                time.sleep(1)  # 等待1秒后再次检查

            # 定义进行性能测试的命令
            filename = f"./data/{model_name}-{gpu_model}x{tensor_parallel_size}"
            cmd2 = f"python3 benchmark_serving.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --tokenizer /home/ubuntu/llms/Vllm/tokenizer --request-rate {request_rate} --filename {filename}"

            # 执行性能测试命令并保存输出到日志文件
            log_filename = f"./logs/{model_name}-{gpu_model}x{tensor_parallel_size}.log"
            with open(log_filename, 'w') as log_file:
                subprocess.run(cmd2, shell=True, stdout=log_file, stderr=log_file)

    except Exception as e:
        # 输出错误信息并终止LLM服务器
        print(f"Error encountered with model: {model_path} and tensor_parallel_size: {tensor_parallel_size}. Error details: {e}")
        server_process.terminate()
        server_process.wait()
        return  

    server_process.terminate()
    server_process.wait()

if __name__ == "__main__":
    # 定义需要测试的LLM模型路径列表
    models = [
        "/home/ubuntu/llms/llama/CodeLlama-7b-hf",
        "/home/ubuntu/llms/llama/CodeLlama-13b-hf"
    ]

    _, gpu_count = get_gpu_model_and_count()
    # 根据显卡数量定义tensor_parallel_sizes
    tensor_parallel_sizes = list(range(1, gpu_count + 1))

    # 对每个LLM模型进行测试
    for model in models:
        for tensor_parallel_size in tensor_parallel_sizes:
            run_llm_and_test(model, tensor_parallel_size, request_rate=5)