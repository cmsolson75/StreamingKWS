
Check Speed on GPU vs MPS
- MPS: Total wall Time: 915.47 seconds | Throughput: 2097.27 samples/s | Latency: 0.0305 s/step
- ADA 6000: Total wall Time: 297.12 seconds | Throughput: 6462.04 samples/s | Latency: 0.0099 s/step


TUNING
Keep your instinct, but change the order:
	1.	Increase batch until GPU utilization stabilizes
	2.	Track samples/tokens per second
	3.	Stop when throughput stops improving
	4.	Tune LR / schedule / regularization around that batch
	5.	Leave VRAM headroom

At scale, teams:
	1.	Find a batch size that saturates compute
	2.	Measure tokens/sec
	3.	Stop increasing batch once tokens/sec plateaus
	4.	Only push batch higher if required for scaling laws or parallelism efficiency




Inference Testing
- Simple inference engine: manual clean up before running through the model.
- Test model on edge cases
    - Noise
    - Partial words
    - OOD words
    - Silence
- INSPECT: Probabilities and see what is going on.
- SETUP:
    - Streaming Inference Engine - see the edges you hit


For streaming inference engine
- Need to essentially send chunks of data to model 
    - Some form of VAD is helpful to make sure model works less, but model should be able to handle audio chunks

---

---
Setup Docker + ECR
- Get working in RunPod

Setup Augmentations
Setup Sampler: Real, Unknown, Silence
- TEST DIFFERENCE


Model
- Setup TC ResNet


Sweep HPO
- SET MAX BS / What you set to for the sweep
- LR Finder
- HPO - simple random sweep / log spaced grid
- Test
    - 5k steps
    - 10k steps
    - 40k steps
    - Compare the configs found after fully trained


Setup
- Torchscript
- Time differences
- Hook into inference engine


Look into
- Receptive Field