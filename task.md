# Full Scope

Goal
- Get it running on GPU
- Use mixed precision 
- use UV for the project.
- Deploy it with a simple CLI app & a Gradio version as well.
- Compile with torchscript
- Setup S3 bucket sync - when you can.
- Use tee for logging
- Setup checkpointing with most recent checkpoint folder and backlog as well as best
- Deploy to Gradio

Separaet
- Train
- Eval
- Maybe preprocess: I don't think I will do that in this version, can just load data inline.
    - Probably should preprocess: but in tutorial I will not.





Spend time on all components of the data loading process - COLLATE!
- Could move the padding out of the feature extractor and into the collate 
- Do speed checks 



App Runner
- Silence Splitter
    - Segment Audio:
    - Use Adaptive Threshold
    - Have median filter to smooth out detected regions.
    - Merge short gaps
    - Filter short segments
    - Add padding to segments
    - Use RMS for this
    - Wrap in a helpful class for this.


Inference Runner
- torch.inference_mode
- Set to eval
- Load: torch script model


Setup propper checkpoint ystem through - this is good!


Order
- Setup full training setup & inference engine and gradio app.
- Setup Mixed Precision: Look up exactly what this is doing.
- Move over to Run pod
- Train model and make sure it works.
- Setup propper cloud sync.
- Train model!!!!!
    - Add in tests & stuff to verify everything is working when you move to GPU like a simple smoke test.
        - Use pytest for this.
    - Setup overfit to a single batch test
    - Setup initial loss test 



What configs do I want?
- Dataset


Auto Device Detection


Sweep
- Setup a HPO sweep 
- Use a version of successive halving
- Just make it a python script called sweep.py --args
- Wrap in BASH script.

LR Test
- This is for finding a good range of LR



GOALS of this stage
- Move to GPU on run pod with propper cloud sync during training.
- Implement streaming inference - more intresting!!! CLI app that you start and it just listens sound sounddevice.
    - Use global dataset stats to make this work
- Update model: TC ResNet: https://arxiv.org/pdf/1904.03814
- Run HPO manualy
 - LR Range test for stability
- Augmentation - Add in
- Sampler: Silence training, Unknown training
    - Mixed batch sampler
    - p_keyword: float = 0.7, p_unknown: float = 0.2, p_silence: float = 0.1,
- Unknown
    - All other words: Only train on 10 that you select.
    - Unknown pile is just all the other ones

For the first code out
- Go through as normal and code out the VAD system and make that all work.
- Setup high level checkpointing system
    - Go through what I have bellow, will take a long time.
- Seed everything
- Setup . based overwrites, this will help a lot with everything.
- Get torchscript working
- Move to GPU and get all of that working
- Setup mixed precision
- Add in cloud sync for checkpoints
--- LLM okay from this point on
- Build out streaming pipeline: 1 hour on own trying this, see how you get googling - LLM might be super helpful though to get off the ground.
- Setup HPO manually
- Setup Launcher.py - this will be what lanches an experiment - use bash for most of this.
- Add in Augmentation
    - SNR Background mixin: Use a random chunk of the background as well.
    - Time shifting the input
    - Gain: Random
- Code out new model -> TC-ResNet: Follow the paper and try to implement - 1 hour 
- Setup custom sampler - Mix in unknown words 
    - 0.1 for silence, this is 10% I want to mention this
    - 0.2 for unknown
    - 0.7 for in distribution
- Build out final app
- Clean up repo
- Write out all steps for a separate E2E run
- LLM: This is for cleaning up all my implementation stuff - make it all cleaner.


What do I actually want to do
- EMA, Schedulers
- Improved checkpointing: Coding project
- Overwrites
- GPU 100%
- Cloud Sync 
---
After this: Everything is just a "Should" -> So don't do it, get an LLM to help you implement it and have fun and play. You will still learn with an LLM. Make a nice repo and post it to GitHub and be done with it.



--->

run dir setup - Could make this into a libray
- run_naming: {run_id}_{slug}
    - run_id: <UTCtimestamp>_<confighash10>
    - slug: ds=<ds_hash>__base=<base_id>__tag=<tag1+tag2>
        - most systems, the slug is just the tags, nothing else, don't overengineer.
        - Store experiment stuff in the tag.
- console.log
- overrides.txt: what was applied
- config.resolved.json
- provenance
    - code.json
    - code.patch
- metrics.jsonl
- checkpoints
    - model.safetensors
    - train_state.pt
    - manifest.json
- checkpoint dir setup
    - step_000120/
        model.safetensors: use safetensors library for this.
        train_state.pt
        manifest.json
            {
                "run_id": "20260206T184210Z_a13f9c4e7b_91c2a10f3d",
                "step": 240,
                "created_at": "2026-02-06T20:12:00Z",
                "files": [
                    {"path": "model.safetensors", "bytes": 412345678},
                    {"path": "train_state.pt", "bytes": 12345678}
                ]
            }
    - latest.json: pointer file with info like step, and path and when you updated it
    - best.json: same idea 


# example provinance
import json
import os
import platform
import getpass
from datetime import datetime, timezone

def collect_provenance(repo_root="."):
    prov = {
        "created_at": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
        "repo_root": os.path.abspath(repo_root),
        "hostname": platform.node(),
        "user": getpass.getuser(),
        "python": platform.python_version(),
    }

    # git info (optional but recommended)
    prov["git_commit"] = cmd(["git", "rev-parse", "HEAD"], cwd=repo_root)
    status = cmd(["git", "status", "--porcelain"], cwd=repo_root)
    prov["git_dirty"] = bool(status) if status is not None else None

    # torch / device info
    try:
        import torch
        prov["torch"] = torch.__version__
        if torch.cuda.is_available():
            prov["device"] = "cuda"
            prov["cuda"] = torch.version.cuda
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            prov["device"] = "mps"
            prov["cuda"] = None
        else:
            prov["device"] = "cpu"
            prov["cuda"] = None
    except Exception:
        prov["torch"] = None
        prov["device"] = None
        prov["cuda"] = None

    return prov




What I learned from GPU
- Use ToMel & DB in the batched area of the code, so that is training or in the collate
- Transform in the dataloader
- Cache audio data after loaded for speedup, can do this cache inline
- Need to propperly manage torch thread stuff.
- Need to tune in areas of the code & profile slowdown, because it is slow. 
- Update this code to a point where you will get efficient training that beats the training on MPS.
- ONLY Goal is to up throughput




Helpful function
- training_step 
- Make sure you batch everything that can be batched.