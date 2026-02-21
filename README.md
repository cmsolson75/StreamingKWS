


## TRAINING 
1.	Increase batch until GPU utilization stabilizes
2.	Track samples/tokens per second
3.	Stop when throughput stops improving
4.	Tune LR / schedule / regularization around that batch
5.	Leave VRAM headroom



# SPEED CHARTS
- MPS: M4 Max
    - batch_size: 64
    - num_workers: 8
    - persistant_workers: true
    - prefetch_factor: 2
    - amp: true
    - Time: 272.5634 seconds
- GPU: L40
    - batch_size: 64
    - num_workers: 8
    - persistant_workers: true
    - prefetch_factor: 2
    - amp: true
    - pin_memory: True
    - Time: 204.4073 seconds
- GPU: L40
    - batch_size: 64
    - num_workers: 16
    - persistant_workers: true
    - prefetch_factor: 4
    - amp: true
    - pin_memory: True
    - Time: 226.3902 seconds
- GPU: H100 PCI
    - batch_size: 64
    - num_workers: 8
    - persistant_workers: true
    - prefetch_factor: 2
    - amp: true
    - pin_memory: True
    - Time: 294.9295 seconds

