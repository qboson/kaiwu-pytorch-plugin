# Injecting Custom Components into QVAE

`QVAE` builds missing components through `_create_encoder`, `_create_decoder`,
`_create_bm`, and `_create_sampler`. Components passed to the constructor take
precedence, so a subclass can reuse an existing encoder, decoder, Boltzmann
machine, or sampler instead of rebuilding it.

This example wires four toy components into a small subclass and checks the
contract end to end.

## Contents

- `custom_qvae.py`: `ToyEncoder`, `ToyDecoder`, `ConstantSampler`,
  `InjectedQVAE`, and a `main()` that runs one forward pass, one loss backward,
  and one energy evaluation.

## Run

```bash
python example/qvae_custom_components/custom_qvae.py
```

Expected output:

```
recon: (5, 6) q: (5, 4) zeta: (5, 4)
encoder sees mean-centered input: True
loss: 4.0923 encoder grads: [True, True]
energy shape: (5,)
custom components OK
```

## Dependencies

Only `torch`, `numpy`, and the installed `kaiwu-torch-plugin`. No dataset,
checkpoint, or Kaiwu license is required.

## Contract checked by the example

- the constructor keeps the supplied encoder, decoder, BM, and sampler;
- `forward` returns `(reconstruction logits, posterior, latent logits, zeta)`
  with the expected shapes;
- with `set_dataset_mean`, the encoder receives `x - mean`;
- `loss` is finite and backpropagates into both encoder and decoder;
- `energy` evaluates the Bernoulli-hardened latent states and returns one value
  per sample.

The BM must cover the latent dimension: `num_visible + num_hidden` equals
`config.num_latent_units`, because `loss` and `energy` evaluate the BM on the
full latent vector.

The same invariants are pinned by
`tests/test_qvae_custom_components_example.py`.
