import argparse
import json
from typing import AsyncGenerator, List

# SYSTEM_PROMPT = """\
# You are a helpful, respectful and honest assistant with a deep knowledge of code and software design. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
# """

import time
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
from pydantic import BaseModel
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from fastapi.middleware.cors import CORSMiddleware

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()

class my_request(BaseModel):
    lang: str = None
    prompt: str = None
    n: int = 0
    temperature: float = 0.6
    top_p: float = 0
    top_k: float = 0


class key_request(BaseModel):
    lang: str = None
    prompt: str = None
    n: int = 0
    apikey: str = None
    apiSecret: str = None
    temperature: float = 0
    top_p: float = 0
    top_k: float = 0

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/multilingual_code_generate_block")
async def generate(request: my_request):
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    # request_dict = await request.json()
    # prompt = request_dict.pop("prompt")
    # stream = request_dict.pop("stream", False)
    # sampling_params = SamplingParams(**request_dict)
    # request_id = random_uuid()
    before = time.time()
    prompt = request.prompt

    # prompt=[f'<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n']
    # prompt.append(f'{message} [/INST]')
    # prompt=''.join(prompt)

    sampling_params = SamplingParams(n=request.n ,
                                     use_beam_search=False,
                                     temperature=request.temperature,
                                     top_p=request.top_p,
                                     top_k=request.top_k)
    stream = False
    request_id = random_uuid()
    # if not engine.is_running:
    #     engine.start_background_loop()

    

    results_generator = engine.generate(prompt, sampling_params, request_id)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            prompt = request_output.prompt
            text_outputs = [
                prompt + output.text for output in request_output.outputs
            ]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\0").encode("utf-8")

    async def abort_request() -> None:
        await engine.abort(request_id)

    if stream:
        background_tasks = BackgroundTasks()
        # Abort the request if the client disconnects.
        background_tasks.add_task(abort_request)
        return StreamingResponse(stream_results(), background=background_tasks)

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        # if await request.is_disconnected():
        #     # Abort the request if the client disconnects.
        #     await engine.abort(request_id)
        #     return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    text_outputs = ""
    for output in final_output.outputs:
        text_outputs = output.text + text_outputs
    # text_outputs = [prompt + output.text for output in final_output.outputs]
    now = time.time()
    all_time = str(now - before)
    answer = {"data": {
        "result": {
            "process_time": all_time,
            "output": {
                "code": [
                    text_outputs
                ]
            }
        },

        "status": 0,
        "message": "success",}
    }
    return answer

# @app.post("/multilingual_code_generate")
# async def generate(request: my_request):
#     before = time.time()
#     prompt = request.prompt
#     sampling_params = SamplingParams(n=request.n ,use_beam_search=False,temperature=request.temperature)
#     stream = False
#     request_id = random_uuid()   

#     results_generator = engine.generate(prompt, sampling_params, request_id)

#     # Streaming case
#     async def stream_results() -> AsyncGenerator[bytes, None]:
#         async for request_output in results_generator:
#             prompt = request_output.prompt
#             text_outputs = [
#                 prompt + output.text for output in request_output.outputs
#             ]
#             ret = {"text": text_outputs}
#             yield (json.dumps(ret) + "\0").encode("utf-8")

#     async def abort_request() -> None:
#         await engine.abort(request_id)

#     if stream:
#         background_tasks = BackgroundTasks()
#         # Abort the request if the client disconnects.
#         background_tasks.add_task(abort_request)
#         return StreamingResponse(stream_results(), background=background_tasks)

#     # Non-streaming case
#     final_output = None
#     async for request_output in results_generator:
#         # if await request.is_disconnected():
#         #     # Abort the request if the client disconnects.
#         #     await engine.abort(request_id)
#         #     return Response(status_code=499)
#         final_output = request_output

#     assert final_output is not None
#     prompt = final_output.prompt
#     for output in final_output.outputs:
#         print (output.text)
#         myoutput = output.text
#     now = time.time()
#     all_time = str(now - before)
#     answer = {"data": {
#         "result": {
#             "process_time": all_time,
#             "output": {
#                 "code": [
#                     myoutput
#                 ]
#             }
#         },
#         "status": 0,
#         "message": "success",}
#     }
#     return answer





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    # engine = AsyncLLMEngine.from_engine_args(engine_args,
    #                                          start_engine_loop=False)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    uvicorn.run(app,
                host='0.0.0.0',
                port=args.port,               
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
