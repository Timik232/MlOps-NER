import os
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import fire


class ONNXExportModel(nn.Module):
    """Wrapper around a Hugging Face causal LM to strip out caching and dict outputs."""
    def __init__(self, model: AutoModelForCausalLM):
        """
        Args:
            model (AutoModelForCausalLM): The pretrained model to wrap.
        """
        super().__init__()
        # disable caching in config
        model.config.use_cache = False
        self.model = model

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor) -> torch.FloatTensor:
        """
        Forward method returning only logits.

        Args:
            input_ids (torch.LongTensor): input token IDs, shape (batch_size, seq_len).
            attention_mask (torch.LongTensor): attention mask, same shape.

        Returns:
            torch.FloatTensor: logits tensor, shape (batch_size, seq_len, vocab_size).
        """
        # Force use_cache=False to avoid past_key_values
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        # outputs.logits is a Tensor
        return outputs.logits


def export_qwen3_to_onnx(
    model_path: str,
    output_path: str = "model.onnx",
    seq_len: int = 2048,
    device: str = "cuda",
) -> None:
    """
    Экспортирует локальную или удалённую модель Qwen-3 в ONNX.

    Args:
        model_path (str): Путь к локальной папке с весами или идентификатор на HuggingFace Hub.
        output_path (str): Путь для сохранения ONNX-файла. По умолчанию "model.onnx".
        seq_len (int): Максимальная длина последовательности. По умолчанию 2048.
        device (str): Устройство для экспорта ('cpu' или 'cuda'). По умолчанию 'cuda'.
    """
    is_local = os.path.isdir(model_path)
    load_kwargs = {"torch_dtype": torch.float16}
    if is_local:
        load_kwargs["local_files_only"] = True

    # Load model & tokenizer
    hf_model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=is_local)
    hf_model.to(device).eval()

    # Wrap to strip caching and dict outputs
    export_model = ONNXExportModel(hf_model).to(device).eval()

    # Prepare dummy inputs
    dummy = tokenizer(
        "Hello, Triton!",
        return_tensors="pt",
        padding="max_length",
        max_length=seq_len,
        truncation=True,
    ).to(device)

    # Export
    torch.onnx.export(
        export_model,
        (dummy["input_ids"], dummy["attention_mask"]),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size", 1: "seq_len"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"ONNX model saved to {output_path}")


def main(
    model_path: str,
    output: str = "model",
    seq_len: int = 2048,
    device: str = "cuda",
) -> None:
    """
    CLI точка входа для fire: экспорт Qwen-3 в ONNX.

    Args:
        model_path (str): Путь к весам Qwen-3.
        output (str): Имя выходного ONNX-файла.
        seq_len (int): Максимальная длина последовательности.
        device (str): Устройство для экспорта.
    """
    export_qwen3_to_onnx(
        model_path=model_path,
        output_path=output,
        seq_len=seq_len,
        device=device,
    )


if __name__ == "__main__":
    fire.Fire({"export": main})
