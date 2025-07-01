FROM nvcr.io/nvidia/tritonserver:25.02-vllm-python-py3

WORKDIR /app

# RUN pip install --upgrade bitsandbytes transformers vllm

COPY ./model_repository /models
