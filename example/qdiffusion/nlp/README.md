# QDiffusion-NLP: BM training and generation

This directory contains one natural-language QDiffusion example:

1. train a conditioned Boltzmann-machine energy model;
2. use that BM to select candidates during text generation.

Historical baselines, scalar heads, paper-reproduction scripts, metric
evaluators, and ablations live in the separate `qdiffusion-nlp-experiments`
repository.

## Architecture

The pretrained `kuleshov-group/mdlm-owt` checkpoint is an internal frozen
proposal. It creates noisy/candidate token pairs and supplies transformer
features. The trainable path is:

```text
(noisy tokens, candidate tokens)
  -> conditioned feature encoder
  -> attention pooling
  -> linear projector (768 visible values)
  -> Kaiwu BM (256 hidden values, SA)
  -> sequence energy
```

Visible values stay continuous; the example does not apply a sigmoid or hard
binarization. The NLP code imports generic `QDiffusion` and `EnergyModel`
components from `kaiwu.torch_plugin`.

## Install

```bash
pip install -r example/qdiffusion/requirements-nlp.txt
```

Use Linux/CUDA. Flash Attention must be compiled for the target GPU.

## Train

```bash
python -m example.qdiffusion.nlp.train_bm \
  --input /path/to/openwebtext.jsonl \
  --output /path/to/bm.pt \
  --max-records 100000 \
  --sequence-length 1024 \
  --micro-batch-size 8 \
  --global-batch-size 512 \
  --max-steps 5000 \
  --seed 1444063560
```

The proposal is frozen. The conditioned encoder, projector, and BM are
trained with one-negative binary NCE. Raw and EMA weights are saved in one
compact checkpoint without duplicating proposal weights.

## Generate

```bash
python -m example.qdiffusion.nlp.generate_bm \
  --bm-checkpoint /path/to/bm.pt \
  --output /path/to/samples.jsonl \
  --sequence-length 1024 \
  --steps 128 \
  --num-samples 32 \
  --batch-size 4 \
  --num-candidates 2 \
  --remask-ratio 0.1 \
  --seed 1472685111
```

Each reverse step samples two proposal candidates, scores both with the BM,
and samples a candidate from `softmax(-energy)`.

## Code map

- `train_bm.py`: data preparation, NCE training, EMA, checkpoints.
- `generate_bm.py`: checkpoint loading and JSONL generation.
- `models/proposal.py`: internal adapter for the frozen text proposal.
- `models/features.py`: noisy/candidate feature encoder.
- `models/bm.py`: projector and Kaiwu BM energy.
- `models/generator.py`: concrete `QDiffusion` composition.
- `sampling.py`: BM-guided reverse sampler.
- `checkpoint.py`: compact BM checkpoint I/O.
