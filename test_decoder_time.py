import torch
import tensorrt as trt
import numpy as np

print('GPU Info:')
print(f'  Name: {torch.cuda.get_device_name(0)}')
print(f'  Compute Capability: {torch.cuda.get_device_capability(0)}')
print(f'  PyTorch CUDA: {torch.version.cuda}')
print(f'  TensorRT: {trt.__version__}')

# Test simple TensorRT inference speed
from core.models.decoder import Decoder

cfg = {'model_path': './checkpoints/ditto_trt_custom2/decoder_fp16.engine', 'device': 'cuda'}
decoder = Decoder(**cfg)

# Dummy input
f_3d = np.random.randn(1, 256, 64, 64).astype(np.float32)

# Warmup
for _ in range(10):
    decoder(f_3d)

# Measure
import time
start = time.time()
for _ in range(100):
    decoder(f_3d)
elapsed = time.time() - start

print(f'\nDecoder TensorRT:')
print(f'  {elapsed/100*1000:.2f} ms per frame')
print(f'  {100/elapsed:.2f} fps')
