Separate out Eval System
- best checkpoint
- EMA, Scheduler
- Have the eval take in the best checkpoint 


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


Model
- Setup TC ResNet

Setup Augmentations
Setup Sampler: Real, Unknown, Silence

Sweep HPO
- SET MAX BS
- LR Finder
- HPO - simple random sweep


Setup
- Torchscript

Finalize 
- Inference Engine

