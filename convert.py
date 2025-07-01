import os

import fire
import onnx
import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
)


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

    def forward(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor
    ) -> torch.FloatTensor:
        """
        Forward method returning only logits.

        Args:
            input_ids (torch.LongTensor): input token IDs, shape (batch_size, seq_len).
            attention_mask (torch.LongTensor): attention mask, same shape.

        Returns:
            torch.FloatTensor: logits tensor, shape (batch_size, seq_len, vocab_size).
        """
        # Force use_cache=False to avoid past_key_values
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        )
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

    # Загрузка модели и токенизатора
    hf_model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=is_local)
    hf_model.to(device).eval()

    # Подготовка модели для экспорта
    export_model = ONNXExportModel(hf_model).to(device).eval()

    # Подготовка входных данных
    dummy = tokenizer(
        "Hello, Triton!",
        return_tensors="pt",
        padding="max_length",
        max_length=seq_len,
        truncation=True,
    ).to(device)

    # Создаем словарь входных данных и извлекаем значения
    inputs = {
        "input_ids": dummy["input_ids"],
        "attention_mask": dummy["attention_mask"],
    }
    args = tuple(inputs.values())
    input_names = list(inputs.keys())
    output_names = ["logits"]

    # Определение динамических осей
    dynamic_axes = {name: {0: "batch_size", 1: "seq_len"} for name in input_names}
    dynamic_axes["logits"] = {0: "batch_size", 1: "seq_len"}

    # Экспорт в ONNX
    torch.onnx.export(
        export_model,
        args,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        export_params=True,
        use_external_data_format=True,
    )
    print(f"ONNX модель сохранена в {output_path}")


def export_causallm(model_checkpoint: str, save_directory: str) -> None:
    r"""
    export model in lower level , using torch.onnx.export
    """
    # generate tokenizer
    # AutoModelForQuestionAnswering
    # model = OPTForCausalLM.from_pretrained(model_checkpoint).eval()

    # model = OPTForQuestionAnswering.from_pretrained(model_checkpoint).eval()
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenizer.save_pretrained(save_directory)

    # generate config.json
    pretrain_cfg = PretrainedConfig.from_pretrained(model_checkpoint, export=True)
    output_config_file = os.path.join(save_directory, "config.json")
    pretrain_cfg.to_json_file(output_config_file, use_diff=True)

    # generate model.onnx
    prompt = "Hey, are you consciours? Can you talk to me?"
    inputs = tokenizer(prompt, return_tensors="pt")

    # print(type(model))
    # exit()
    # x = model(inputs["input_ids"],inputs["attention_mask"])
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint, export=True)
    model.save_pretrained(save_directory)
    exit()
    # print(x)
    # from inspect import getmembers
    # for k,v in getmembers(model):
    # print(k,type(v))

    with torch.no_grad():
        symbolic_names = {0: "batch_size", 1: "sequence_length"}
        torch.onnx.export(
            model,
            args=(inputs["input_ids"], inputs["attention_mask"]),
            f=f"{save_directory}/model.onnx",
            export_params=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["output"],
            dynamic_axes={
                "input_ids": symbolic_names,
                "attention_mask": symbolic_names,
                "output": symbolic_names,
            },
            do_constant_folding=True,
            opset_version=15,  # Use a suitable opset version, such as 12 or 13
        )
    onnx_model = onnx.load(f"{save_directory}/model.onnx")
    onnx.checker.check_model(onnx_model)
    print("Done!")


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
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    export_qwen3_to_onnx(
        model_path=model_path,
        output_path=output,
        seq_len=seq_len,
        device=device,
    )
    # export_CausalLM(model_path, output)


if __name__ == "__main__":
    fire.Fire({"export": main})
