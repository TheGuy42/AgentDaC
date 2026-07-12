import pathlib

from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase
from trl.chat_template_utils import get_training_chat_template
from transformers import pipeline

MODEL_NAME = "Qwen/Qwen3.5-4B"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(MODEL_NAME)

FIRST_USER_PROMPT = {"role": "user", "content": "What is 2+2?"}
SECOND_USER_PROMPT = {"role": "user", "content": "And 3+3?"}

qwen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    max_length=1000,
    do_sample=False,
)

# now get only the response tokens, with thinking enabled of course

response_text = qwen_pipeline([FIRST_USER_PROMPT], return_full_text=False)[0]["generated_text"]

print(f"Response text: {response_text!r}")

CONVERSATION = [
    FIRST_USER_PROMPT,
    {"role": "assistant", "content": response_text},
    SECOND_USER_PROMPT
]

tokenizer.chat_template = get_training_chat_template(tokenizer)

ids = tokenizer.apply_chat_template(
    [CONVERSATION],
    add_generation_prompt=True,
    tokenize=True,
    enable_thinking=True,
    return_tensors="pt",
    return_dict=True,
)

tids = ids.input_ids[0]

print("\n=== TRL training chat template ===")
for i, (tid, piece) in enumerate(zip(tids, tokenizer.convert_ids_to_tokens(tids))):
    print(f"{i:4} {tid:>8}  {piece!r}")


# now switch to my patched (prefix-preserving) chat template and print the token ids again
PATCHED_TEMPLATE = pathlib.Path(__file__).resolve().parents[2] / "config_files" / "templates" / "Qwen3.5-4B.jinja"
tokenizer.chat_template = PATCHED_TEMPLATE.read_text()

ids = tokenizer.apply_chat_template(
    [CONVERSATION],
    add_generation_prompt=True,
    tokenize=True,
    enable_thinking=True,
    return_tensors="pt",
    return_dict=True,
)

tids = ids.input_ids[0]

print("\n=== my patched (verbatim) chat template ===")
for i, (tid, piece) in enumerate(zip(tids, tokenizer.convert_ids_to_tokens(tids))):
    print(f"{i:4} {tid:>8}  {piece!r}")
