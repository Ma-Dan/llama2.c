import argparse
import struct
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList

def serialize_fp32(file, tensor):
    d = tensor.flatten()
    """ writes one fp32 tensor to file that is open in wb mode """
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)

def serialize_int8(file, tensor):
    """ writes one int8 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).numpy().astype(np.int8)
    b = struct.pack(f'{len(d)}b', *d)
    file.write(b)

def quantize_q80(w, group_size):
    """
    takes a tensor and returns the Q8_0 quantized version
    i.e. symmetric quantization into int8, range [-127,127]
    """
    assert w.numel() % group_size == 0
    ori_shape = w.shape
    w = w.float() # convert to float32
    w = w.reshape(-1, group_size)
    # find the max in each group
    wmax = torch.abs(w).max(dim=1).values
    # calculate the scaling factor such that float = quant * scale
    scale = wmax / 127.0
    # scale into range [-127, 127]
    quant = w / scale[:,None]
    # round to nearest integer
    int8val = torch.round(quant).to(torch.int8)
    # dequantize by rescaling
    fp32val = (int8val.float() * scale[:,None]).view(-1)
    fp32valr = fp32val.reshape(-1, group_size)
    # calculate the max error in each group
    err = torch.abs(fp32valr - w).max(dim=1).values
    # find the max error across all groups
    maxerr = err.max().item()
    return int8val, scale, maxerr

def main():
    parser = argparse.ArgumentParser(description="Quant test")
    parser.add_argument('--id', default=123)
    parser.add_argument('--quant_bit', default=4)

    args = parser.parse_args()
    id = int(args.id)
    quant_bit = int(args.quant_bit)

    model_path = "./stories42M"
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code = True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=True, trust_remote_code = True)

    weight = model.model.layers[0].self_attn.q_proj.weight.detach()

    weight_file = open("weight.bin", 'wb')
    serialize_fp32(weight_file, weight.numpy())
    weight_file.close()

    id = torch.from_numpy(np.array([id]))

    x = model.model.embed_tokens(id).detach()

    x_file = open("x.bin", 'wb')
    serialize_fp32(x_file, x.numpy())
    x_file.close()

    golden = model.model.layers[0].self_attn.q_proj(x).detach().numpy()

    golden_file = open("golden.bin", 'wb')
    serialize_fp32(golden_file, golden)
    golden_file.close()

    group_size = 64
    q, s, err = quantize_q80(weight, group_size)
    # save the int8 weights to file
    weight_q_file = open("weight_q.bin", 'wb')
    serialize_int8(weight_q_file, q) # save the tensor in int8
    serialize_fp32(weight_q_file, s) # save scale factors
    print(f"quantized to Q8_0 with max error {err}")
    weight_q_file.close()


if __name__ == '__main__':
    main()