# export_to_onnx.py — пример
import torch
from transformers import AutoTokenizer, T5EncoderModel

HF_ID = "ai-forever/FRIDA"
MAX_LEN = 512
ONNX_PATH = "model.onnx"
OPSET = 17

tok = AutoTokenizer.from_pretrained(HF_ID)
mdl = T5EncoderModel.from_pretrained(HF_ID).eval()

# Заглушки входов
dummy = tok(
    ["search_query: пример"],
    max_length=MAX_LEN,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
)
inputs = (dummy["input_ids"], dummy["attention_mask"])

dynamic_axes = {
    "input_ids": {0: "batch", 1: "seq"},
    "attention_mask": {0: "batch", 1: "seq"},
    "last_hidden_state": {0: "batch", 1: "seq"},
}

with torch.no_grad():
    torch.onnx.export(
        mdl,
        inputs,
        ONNX_PATH,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=OPSET,
    )
print("Saved", ONNX_PATH)
