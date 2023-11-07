from vllm import LLM

llm = LLM("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True,tensor_parallel_size=1)
output = llm.generate("#write a bubble sort")
print(output)