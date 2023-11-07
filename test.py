from vllm import LLM, SamplingParams

prompts = [
    "我是",
    "中国人是",
    "百度是",
    "马云是",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=200)
llm = LLM(
    model="/home/ubuntu/llms/llama/Llama2-Chinese-7b-Chat", trust_remote_code=True
)
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
