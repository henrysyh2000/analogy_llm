import os
import re
from typing import *
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_ANABENCH = """
Which of the following is the most analogous story to the target story?
Note: Only generate the index without any additional text. 

Target Story:
%s

Options: 
%s

Answer:
"""

# def checkpoint_exists(ckpt_path):
#     return os.path.exists(ckpt_path)

# def checkpoint_loader(ckpt_path, data_path) -> object:
#     with open(ckpt_path, 'r') as f:
#         lines = f.readlines()
#         processed_indices = set(int(line.strip()) for line in lines)

#     with open(data_path, 'r') as f:
#         data_lines = f.readlines()

#     unprocessed_data = [line for idx, line in enumerate(data_lines) if idx not in processed_indices]
#     return unprocessed_data, processed_indices


def batch_process(rows, prompt, model, tokens=128, temp=0.7):
    for row in rows:
        sentence = row['Sentence']
        story = row.get('Story', '')  # For tasks other than S1, this gets the story.
        options = row['Options']
        out_dict = model.generate(
            prompt=prompt % (story if story else sentence, options),
            max_new_tokens=tokens,
            temperature=temp,
        )
        # Update rows with results
        msg_pairs = out_dict.get("msg_pairs", [])
        if len(msg_pairs) > 0:
            row['Pred_y'] = msg_pairs[-1]["message"]
            # check if there's a reason message before the output
            if len(msg_pairs) > 1:
                row['Reason'] = msg_pairs[-2]["message"]
        else:
            row['Pred_y'] = None
            row['Reason'] = None
    
    return rows


# Matches any special token like <|channel|>, <|message|>, <|end|>, <|foo|>, etc.
ANY_TOKEN_RE = re.compile(r"<\|[^|>]+?\|>")

# Extracts: <|channel|>...<|message|>... [stops at next <|...|> or end-of-string]
PAIR_RE = re.compile(
    r"<\|channel\|>\s*(.*?)\s*<\|message\|>(.*?)(?=(?:<\|[^|>]+?\|>|$))",
    re.DOTALL,
)

def process_out_text(text: str, *, strip_inner_tokens: bool = True) -> Dict[str, Any]:
    """
    Parse all <|channel|> ... <|message|> ... pairs.
    - Never includes trailing special tokens in message content.
    - If strip_inner_tokens=True, also removes any <|...|> tokens that appear inside the message body.
    """
    pairs = []
    for m in PAIR_RE.finditer(text):
        channel = m.group(1).strip()
        message = m.group(2).strip()

        if strip_inner_tokens:
            message = ANY_TOKEN_RE.sub("", message).strip()

        pairs.append({"channel": channel, "message": message})
    return {
        "msg_pairs": pairs,
    }



class Model:
    def __init__(self, model_name, cache_dir='models', dtype="auto", device='cuda', **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device,
        )
        self.kwargs = kwargs

    def generate(self, prompt, max_new_tokens=512, temperature=1.0):

        messages = [
            {"role": "system", "content": "Reasoning: low. Keep your reasoning analysis short and concise in the task. The length of the intermediate reasoning should not exceed 200 words."},
            {"role": "user", "content": prompt},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=self.kwargs.get("tokenize", True),
            return_dict=self.kwargs.get("return_dict", True),
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(**inputs, 
                                max_new_tokens=max_new_tokens,
                                temperature=temperature,
        )

        out_text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
        response_dict = process_out_text(out_text)

        return response_dict
    