import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList

model = AutoModelForCausalLM.from_pretrained("../Qwen2-1.5B-Instruct", trust_remote_code = True)
tokenizer = AutoTokenizer.from_pretrained("../Qwen2-1.5B-Instruct", legacy=True, trust_remote_code = True)

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self):
        super().__init__()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, stops=[]):
        #print('-' * 40)
        print(tokenizer.decode(input_ids[0][-1]), end='', flush=True)
        if input_ids[0][-1] == 151643 or input_ids[0][-1] == 151645:
            return True

        return False


model.to("cuda")


ctx = """<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"""

while True:
    user_input = input(f'User: ')
    ctx = ctx + "<|im_start|>user\n" + user_input + "<|im_end|>\n<|im_start|>assistant\n"

    if len(ctx.strip()) > 0:
        batch = tokenizer(ctx, return_tensors="pt").to("cuda")
        print("AI: ", end='')
        result = model.generate(batch["input_ids"],
                                do_sample=True,
                                top_k=50,
                                max_length=2048,
                                top_p=0.95,
                                temperature=0.95,
                                stopping_criteria=StoppingCriteriaList([StoppingCriteriaSub()]),
                                # repetition_penalty=1.17
                                )
        decoded = tokenizer.decode(result[0])
        ctx = decoded
        print('\n', end='', flush=True)