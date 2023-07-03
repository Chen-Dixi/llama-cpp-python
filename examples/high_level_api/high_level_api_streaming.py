import json
import argparse

from llama_cpp import Llama

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="../models/7B/ggml-models.bin")
parser.add_argument("-p", "--prompt", type=str, default="Question: What are the names of the planets in the solar system? Answer: ")
args = parser.parse_args()

llm = Llama(model_path=args.model)

stream = llm(
    args.prompt,
    max_tokens=128,
    stop=["Q:"],
    stream=True,
)

for output in stream:
    # print(json.dumps(output, indent=2))
    print(output["choices"][0]["text"],end='')
