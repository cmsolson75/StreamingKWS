Separate out Eval System
- Scheduler
- Check Speed on GPU vs MPS


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