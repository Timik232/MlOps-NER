FROM nvcr.io/nvidia/tritonserver:25.02-vllm-python-py3

WORKDIR /app

COPY ./model_repository /models

RUN pip install --upgrade bitsandbytes transformers vllm