import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList

# Modify modeling_qwen2.py Qwen2ForCausalLM::forward()
# logits = self.lm_head(hidden_states[:, :, :]).float()

model = AutoModelForCausalLM.from_pretrained("/mnt/d/Models/LLM/Qwen2-0.5B-Instruct", trust_remote_code = True)
tokenizer = AutoTokenizer.from_pretrained("/mnt/d/Models/LLM/Qwen2-0.5B-Instruct", legacy=True, trust_remote_code = True)

model.half()

n_layers = 24
n_kv_heads = 2
n_dim = 64

input_ids = torch.zeros(1, 1).long()
attention_mask = torch.zeros(1, 1).long()
position_ids = torch.zeros(1, 1).long()

past_key_values = []
for i in range(n_layers):
    past_key_values.append((torch.zeros(1, n_kv_heads, 1, n_dim).half(), torch.zeros(1, n_kv_heads, 1, n_dim).half()))

input_all = (input_ids, attention_mask, position_ids, past_key_values)

input_names = ['input_ids', 'attention_mask', 'position_ids']
output_names = ['logits']

dynamic_axes = {
                'input_ids' : {0: 'batch_size', 1: 'in_len'},
                'attention_mask' : {0: 'batch_size', 1: 'in_len'},
                'position_ids' : {0: 'batch_size', 1: 'in_len'},
                'logits' : {0: 'batch_size', 1: 'in_len'},
               }

for i in range(n_layers):
    input_names.append("past_key_values_{}_key".format(i))
    input_names.append("past_key_values_{}_value".format(i))
    output_names.append("present_{}_key".format(i))
    output_names.append("present_{}_value".format(i))
    dynamic_axes["past_key_values_{}_key".format(i)] = {0: 'batch_size', 2: 'past_len'}
    dynamic_axes["past_key_values_{}_value".format(i)] = {0: 'batch_size', 2: 'past_len'}
    dynamic_axes["present_{}_key".format(i)] = {0: 'batch_size', 2: 'out_len'}
    dynamic_axes["present_{}_value".format(i)] = {0: 'batch_size', 2: 'out_len'}

torch.onnx.export(model,
                  input_all,
                  "qwen2_0_5b_instruct.onnx",
                  opset_version=16,
                  input_names=input_names,
                  output_names=output_names,
                  dynamic_axes=dynamic_axes
                 )