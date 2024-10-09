import argparse
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList


def quant(weight, quant_bit, quant_block):
    weight = weight.numpy()
    oc, ic = weight.shape
    if quant_block == 0:
        block_size = ic
    else:
        block_size = quant_block
    block_num = ic // block_size
    weight = weight.reshape(oc, block_num, block_size)
    max_val = np.max(weight, axis=-1, keepdims=True)
    min_val = np.min(weight, axis=-1, keepdims=True)
    offset = 1 << (quant_bit - 1)
    clip_max = offset - 1
    clip_min = -offset
    scale = (max_val - min_val) / (clip_max - clip_min)
    q_weight = np.round((weight - min_val) / scale) + clip_min
    q_weight = (np.clip(q_weight.flatten(), clip_min, clip_max) + offset).astype(np.uint8)
    q_weight = q_weight.reshape(-1, 2)
    if quant_bit == 4:
        q_weight = q_weight[:, 0] * 16 + q_weight[:, 1]
    alpha = np.stack([min_val.flatten(), scale.flatten()], axis=-1).flatten()
    return q_weight, alpha, clip_min

def quant_absmax(x):
    x = x.numpy()
    s_x = np.float32(127 / np.max(np.abs(x)))
    q_x = np.round(s_x * x).astype(np.int8)
    return q_x, s_x

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

    id = torch.from_numpy(np.array([id]))

    x = model.model.embed_tokens(id).detach()

    golden = model.model.layers[0].self_attn.q_proj(x).detach().numpy()

    # 量化weight
    q_weight, alpha, clip_min = quant(weight, quant_bit, 0)

    # 量化activation
    q_x, s_x = quant_absmax(x)

    # W4A8 matmul
    if quant_bit == 4:
        weight_len = q_weight.shape[0]
        q_weight_restore = np.concatenate(((q_weight//16).reshape(weight_len, 1), (q_weight%16).reshape(weight_len, 1)), axis=1).reshape(weight.shape)

        mm_int = np.matmul(q_weight_restore, q_x[0])
    else:
        q_weight_restore = q_weight.reshape(weight.shape)

        mm_int = np.matmul(q_weight_restore.astype(np.uint32), q_x[0])

    # 反量化
    alpha = alpha.reshape(-1, 2)

    result = np.matmul(q_weight_restore * alpha[:, 1].reshape(weight.shape[1], 1) + alpha[:, 0].reshape(weight.shape[1], 1), q_x[0] / s_x)

    result1 = np.matmul(q_weight_restore * alpha[:, 1].reshape(weight.shape[1], 1), q_x[0] / s_x) + alpha[:, 0] * np.sum(q_x[0]) / s_x

    result2 = (mm_int / s_x) * alpha[:, 1] + alpha[:, 0] * np.sum(q_x[0]) / s_x

    diff2 = result2 - golden[0]
    print(diff2.max(), diff2.min())


if __name__ == '__main__':
    main()