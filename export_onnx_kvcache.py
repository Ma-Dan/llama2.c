import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList

model = AutoModelForCausalLM.from_pretrained("/mnt/d/llama2/stories42M", trust_remote_code = True)
tokenizer = AutoTokenizer.from_pretrained("/mnt/d/llama2/stories42M", legacy=True, trust_remote_code = True)

input_ids = torch.zeros(1, 1).long()
attention_mask = torch.zeros(1, 1).long()
position_ids = torch.zeros(1, 1).long()
past_key_values_0_key = torch.zeros(1, 8, 1, 64)
past_key_values_0_value = torch.zeros(1, 8, 1, 64)
past_key_values_1_key = torch.zeros(1, 8, 1, 64)
past_key_values_1_value = torch.zeros(1, 8, 1, 64)
past_key_values_2_key = torch.zeros(1, 8, 1, 64)
past_key_values_2_value = torch.zeros(1, 8, 1, 64)
past_key_values_3_key = torch.zeros(1, 8, 1, 64)
past_key_values_3_value = torch.zeros(1, 8, 1, 64)
past_key_values_4_key = torch.zeros(1, 8, 1, 64)
past_key_values_4_value = torch.zeros(1, 8, 1, 64)
past_key_values_5_key = torch.zeros(1, 8, 1, 64)
past_key_values_5_value = torch.zeros(1, 8, 1, 64)
past_key_values_6_key = torch.zeros(1, 8, 1, 64)
past_key_values_6_value = torch.zeros(1, 8, 1, 64)
past_key_values_7_key = torch.zeros(1, 8, 1, 64)
past_key_values_7_value = torch.zeros(1, 8, 1, 64)

input_all = (input_ids, attention_mask, position_ids,
             [(past_key_values_0_key, past_key_values_0_value), (past_key_values_1_key, past_key_values_1_value),
              (past_key_values_2_key, past_key_values_2_value), (past_key_values_3_key, past_key_values_3_value),
              (past_key_values_4_key, past_key_values_4_value), (past_key_values_5_key, past_key_values_5_value),
              (past_key_values_6_key, past_key_values_6_value), (past_key_values_7_key, past_key_values_7_value),
             ]
            )
torch.onnx.export(model,
                  input_all,
                  "stories42M.onnx",
                  opset_version=16,
                  input_names=['input_ids', 'attention_mask', 'position_ids',
                               "past_key_values_0_key", "past_key_values_0_value",
                               "past_key_values_1_key", "past_key_values_1_value",
                               "past_key_values_2_key", "past_key_values_2_value",
                               "past_key_values_3_key", "past_key_values_3_value",
                               "past_key_values_4_key", "past_key_values_4_value",
                               "past_key_values_5_key", "past_key_values_5_value",
                               "past_key_values_6_key", "past_key_values_6_value",
                               "past_key_values_7_key", "past_key_values_7_value"],
                  output_names=['logits',
                                "present_0_key", "present_0_value",
                                "present_1_key", "present_1_value",
                                "present_2_key", "present_2_value",
                                "present_3_key", "present_3_value",
                                "present_4_key", "present_4_value",
                                "present_5_key", "present_5_value",
                                "present_6_key", "present_6_value",
                                "present_7_key", "present_7_value"],
                  dynamic_axes={
                      'input_ids' : {0: 'batch_size'},
                      'attention_mask' : {0: 'batch_size'},
                      'position_ids' : {0: 'batch_size'},
                      'past_key_values_0_key': {0: 'batch_size', 2: 'pos'},
                      'past_key_values_0_value': {0: 'batch_size', 2: 'pos'},
                      'past_key_values_1_key': {0: 'batch_size', 2: 'pos'},
                      'past_key_values_1_value': {0: 'batch_size', 2: 'pos'},
                      'past_key_values_2_key': {0: 'batch_size', 2: 'pos'},
                      'past_key_values_2_value': {0: 'batch_size', 2: 'pos'},
                      'past_key_values_3_key': {0: 'batch_size', 2: 'pos'},
                      'past_key_values_3_value': {0: 'batch_size', 2: 'pos'},
                      'past_key_values_4_key': {0: 'batch_size', 2: 'pos'},
                      'past_key_values_4_value': {0: 'batch_size', 2: 'pos'},
                      'past_key_values_5_key': {0: 'batch_size', 2: 'pos'},
                      'past_key_values_5_value': {0: 'batch_size', 2: 'pos'},
                      'past_key_values_6_key': {0: 'batch_size', 2: 'pos'},
                      'past_key_values_6_value': {0: 'batch_size', 2: 'pos'},
                      'past_key_values_7_key': {0: 'batch_size', 2: 'pos'},
                      'past_key_values_7_value': {0: 'batch_size', 2: 'pos'},
                      'logits' : {0: 'batch_size'},
                      'present_0_key': {0: 'batch_size', 2: 'seq_len'},
                      'present_0_value': {0: 'batch_size', 2: 'seq_len'},
                      'present_1_key': {0: 'batch_size', 2: 'seq_len'},
                      'present_1_value': {0: 'batch_size', 2: 'seq_len'},
                      'present_2_key': {0: 'batch_size', 2: 'seq_len'},
                      'present_2_value': {0: 'batch_size', 2: 'seq_len'},
                      'present_3_key': {0: 'batch_size', 2: 'seq_len'},
                      'present_3_value': {0: 'batch_size', 2: 'seq_len'},
                      'present_4_key': {0: 'batch_size', 2: 'seq_len'},
                      'present_4_value': {0: 'batch_size', 2: 'seq_len'},
                      'present_5_key': {0: 'batch_size', 2: 'seq_len'},
                      'present_5_value': {0: 'batch_size', 2: 'seq_len'},
                      'present_6_key': {0: 'batch_size', 2: 'seq_len'},
                      'present_6_value': {0: 'batch_size', 2: 'seq_len'},
                      'present_7_key': {0: 'batch_size', 2: 'seq_len'},
                      'present_7_value': {0: 'batch_size', 2: 'seq_len'}
                  }
                 )