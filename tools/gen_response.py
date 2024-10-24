import os
import argparse
from datasets import load_from_disk, Dataset
from vllm import LLM, SamplingParams

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='/data/models/vicuna-7b-v1.3')
parser.add_argument('--sam_data_path', type=str, default='sam_data/sam_prompts')
args = parser.parse_args()

sam_dataset = load_from_disk(args.sam_data_path)

prompts = sam_dataset["prompt"]
print("number of prompts: {}".format(len(prompts)))

llm = LLM(model=args.model_name, enable_prefix_caching=True)
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1024)

outputs = llm.generate(prompts, sampling_params)

sam_dialogues = []
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    sam_dialogues.append({
        "prompt": prompt,
        "response": generated_text,
    })

print(sam_dialogues)
sam_dialogues = Dataset.from_list(sam_dialogues)

sam_dialogues.save_to_disk("sam_data/sam_dialogues")
