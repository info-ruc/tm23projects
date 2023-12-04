"""
The gradio demo server for chatting with a single model.
"""
from fastapi import FastAPI
app = FastAPI()

import argparse
from collections import defaultdict
import datetime
import json
import os
import time
import uuid

from fastapi.responses import StreamingResponse
import requests
from fastchat.conversation import SeparatorStyle
from fastchat.constants import LOGDIR, WORKER_API_TIMEOUT
from fastchat.model.model_adapter import get_conversation_template
from fastchat.model.model_registry import model_info
from fastchat.serve.gradio_patch import Chatbot as grChatbot
from fastchat.serve.gradio_css import code_highlight_css
from fastchat.utils import (
    build_logger,
    server_error_msg,
    violates_moderation,
    moderation_msg,
    get_window_url_params_js,
)
from fastapi import Body
import urllib.parse


logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "fastchat Client"}


controller_url = "http://10.230.107.102:21001"
enable_moderation = False


def set_global_vars(controller_url_, enable_moderation_):
    global controller_url, enable_moderation
    controller_url = controller_url_
    enable_moderation = enable_moderation_


def post_process_code(code):
    sep = "\n```"
    if sep in code:
        blocks = code.split(sep)
        if len(blocks) % 2 == 1:
            for i in range(1, len(blocks), 2):
                blocks[i] = blocks[i].replace("\\_", "_")
        code = sep.join(blocks)
    return code


def openai_api_stream_iter(model_name, messages, temperature, top_p, max_new_tokens):
    import openai

    print("openai_api_stream_iter:model:%s " %model_name)
    print("openai_api_stream_iter:top_p:%s " %top_p)
    # Make requests
    gen_params = {
        "model": model_name,
        "prompt": messages,
        "temperature": temperature,
        "top_p": top_p,
    }
    logger.info(f"1==== request ====\n{gen_params}")

    res = openai.ChatCompletion.create(
        model=model_name, messages=messages, temperature=temperature, stream=True
    )
    text = ""
    for chunk in res:
        text += chunk["choices"][0]["delta"].get("content", "")
        data = {
            "text": text,
            "error_code": 0,
        }
        yield data


def model_worker_stream_iter_api(model_name, worker_addr, prompt, temperature, top_p, max_new_tokens):
    
    #stop.str=None
    #stop_token_ids=None

    # Make requests
    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "stop": '[END]',#"###", #"[stop]", #conv.stop_str,
        "stop_token_ids": [1],# [2], #None, #conv.stop_token_ids,
        "echo": False,
    }
    logger.info(f"3==== request ====\n{gen_params}")

    # Stream output
    response = requests.post(
        worker_addr + "/worker_generate_stream",
        headers=headers,
        json=gen_params,
        stream=True,
        timeout=WORKER_API_TIMEOUT,
    )
    for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode())
            yield data

@app.get("/chat/streamapitest")
def http_bot_api_test(messages):
    start_tstamp = time.time()
    model_name = "vicuna-13b-v1.5-16k"
    temperature = float(0)
    top_p = float(1.0)
    max_new_tokens = int(512)

    controller_url="http://10.230.107.102:21001"
    conv_id = uuid.uuid4().hex
    prompt=""
    prompt=prompt+"USER: "+ messages
    prompt=prompt+" ASSISTANT: "

        # Query worker address
    ret = requests.post(
        controller_url + "/get_worker_address", json={"model": model_name}
    )
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    prompt = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.  """+prompt

    print("prompt: %s" %prompt)

    stream_iter = model_worker_stream_iter_api(model_name, worker_addr, prompt, temperature, top_p, max_new_tokens)
    #for data in stream_iter:
        #print("data stream: %s" % data)
    #messages = messages + "▌"
    
    yield messages

    try:
        for data in stream_iter:
       #     if data["error_code"] == 0:
            output = data["text"].strip()
            output = post_process_code(output)
            print("output: %s " % output)
            #messages =messages + output + "▌"
            messages = messages + output
            yield messages
            #return StreamingResponse(http_bot_api(messages), media_type='text/event-stream', headers={ 'Cache-Control': 'no-cache', 'Connection': 'keep-alive', 'Access-Control-Allow-Origin': '*' })

            #yield output
            #else:
             #   output = data["text"] + f" (error_code: {data['error_code']})"
              #  messages = messages + output
               # yield messages
                #return
            time.sleep(0.02)
    #except requests.exceptions.RequestException as e:
     #   messages =messages + server_error_msg + f" (error_code: 4)"
      #  yield messages
       # return
    except Exception as e:
        messages = messages+ server_error_msg + f" (error_code: 5, {e})"
        yield messages
        return

    #messages[-1] = messages[-1][-1][:-1]
    #yield messages

    finish_tstamp = time.time()
    #logger.info("output1:"+f"{output}")
    #return StreamingResponse(http_bot_api(messages), media_type='text/event-stream', headers={ 'Cache-Control': 'no-cache', 'Connection': 'keep-alive', 'Access-Control-Allow-Origin': '*' })

def http_bot_api(messages):
    start_tstamp = time.time()
    model_name = "vicuna-13b-v1.5-16k"
    temperature = float(0)
    top_p = float(1.0)
    max_new_tokens = int(5120)

    controller_url="http://10.230.107.102:21001"
    worker_addr = "http://10.230.107.102:21002"
    conv_id = uuid.uuid4().hex
    prompt=""
    prompt=prompt+"USER: "+ messages
    prompt=prompt+" ASSISTANT: "

        # Query worker address
    #ret = requests.post(
    #    controller_url + "/get_worker_address", json={"model": model_name}
    #)
    #worker_addr = ret.json()["address"]
    
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    prompt = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.  """+prompt

    print("prompt: %s" %prompt)

    stream_iter = model_worker_stream_iter_api(model_name, worker_addr, prompt, temperature, top_p, max_new_tokens)
    #for data in stream_iter:
        #print("data stream: %s" % data)
    #messages = messages + "▌"

    return stream_iter



@app.post("/chat/streamingapi")
def qastreaming(messages=Body(1,title="问题及上下文",embed=False)):
    messages = urllib.parse.unquote(messages)
    print("messages: %s" % messages)
    return StreamingResponse(call_http_bot_api(messages), media_type='text/event-stream', headers={'char-set':'utf-8', 'Cache-Control': 'no-cache', 'Connection': 'keep-alive', 'Access-Control-Allow-Origin': '*' })

def call_http_bot_api(messages):
   import time
   #list=["多","个","试","验"]
   #for i in list:
       #yield str(data)
      
    #   yield f"data: event {i}\n\n"
     #  time.sleep(5)
   stream_iter=http_bot_api(messages)
   print("type of stream_iter: %s" % type(stream_iter))
   #print("stream_iter: %s" % stream_iter.text)
   for data in stream_iter:
       output = data["text"].strip()
       output = post_process_code(output)
       print("output: %s " % output)
       #output=output.replace("\n","\\n")
       output=output.replace("\n","<br>")
       output=output+"\n"
       #messages =messages + output + "▌"
       #messages = messages + output
       #time.sleep(0.05)
       yield f"data:{output}\n"
   yield f"&&&"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://10.230.107.102:21001")
    parser.add_argument("--concurrency-count", type=int, default=10)
    parser.add_argument(
        "--model-list-mode", type=str, default="once", choices=["once", "reload"]
    )
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--moderate", action="store_true", help="Enable content moderation"
    )
    parser.add_argument(
        "--add-chatgpt",
        action="store_true",
        help="Add OpenAI's ChatGPT models (gpt-3.5-turbo, gpt-4)",
    )
    parser.add_argument(
        "--add-claude",
        action="store_true",
        help="Add Anthropic's Claude models (claude-v1)",
    )

    args = parser.parse_args()
    logger.info(f"args: {args}")

    set_global_vars(args.controller_url, args.moderate)
    models = get_model_list(args.controller_url)

    if args.add_chatgpt:
        models = ["gpt-3.5-turbo", "gpt-4"] + models
    if args.add_claude:
        models = ["claude-v1"] + models

    demo = build_demo(models)
    demo.queue(
        concurrency_count=args.concurrency_count, status_update_rate=10, api_open=False
    ).launch(
        server_name=args.host, server_port=args.port, share=args.share, max_threads=200
        )
