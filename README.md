# Triton Inference Server with Monitoring

NVIDIA Triton Inference Server setup with Prometheus monitoring and Grafana dashboards.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Grafana       │◄───┤   Prometheus    │◄───┤ Triton Server   │
│   Port: 3000    │    │   Port: 9090    │    │ Ports: 8900-8902│
│   Dashboards    │    │   Metrics       │    │ Model Serving   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │ Model Repository│
                                              │  - frida_encoder│
                                              │  - ONNX Models  │
                                              └─────────────────┘
```

## Prerequisites
- Docker Compose + NVIDIA Docker runtime
- NVIDIA GPU with CUDA support

## Quick Start

1. **Add Onnx into frida-encode/1 directory**
2. **Clone and Navigate**
   ```bash
   git clone https://github.com/Timik232/MlOps-NER.git
   cd tritonserver
   ```

3. **Start All Services**
   ```bash
   docker-compose up -d
   ```

4. **Access Services**
   - **Triton Server**: `http://localhost:8900` (HTTP), `localhost:8901` (gRPC)
   - **Grafana**: `http://localhost:3000` (admin/admin)
   - **Prometheus**: `http://localhost:9090`

## Services
- **Triton**: Model inference server (ports 8900-8902)
- **Prometheus**: Metrics collection (port 9090)
- **Grafana**: Dashboards (port 3000)
- **TensorRT**: Model optimization


## Adding Models

1. Create directory: `mkdir -p model_repository/your_model/1`
2. Add model: `cp model.onnx model_repository/your_model/1/`
3. Create `config.pbtxt` (see existing models for examples)

## Usage

### Test Model Inference
```bash
curl -X POST http://localhost:8900/v2/models/frida_encoder/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {"name": "input_ids", "data": [1, 2, 3, 4, 5], "datatype": "INT64", "shape": [1, 5]},
      {"name": "attention_mask", "data": [1, 1, 1, 1, 1], "datatype": "INT64", "shape": [1, 5]}
    ]
  }'
```

### Basic Commands
```bash
docker-compose up -d        # Start all services
docker-compose down         # Stop all services
docker-compose logs triton  # View logs
```
