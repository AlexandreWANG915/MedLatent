# Reproduction Notes

The main paper experiments use:

- `num_hospitals = 3`
- `num_latents = 32`
- one retrieved case per hospital
- maximum prompt length 320
- maximum target length 64
- greedy decoding for diagnosis evaluation
- frozen LLM backbones

Recommended order:

1. Train one MedLatent-H distiller per backbone family.
2. Build family latent caches for encoder families used in MedLatent-X.
3. Train one MedLatent-X projector set per host family.
4. Evaluate diagnosis utility.
5. Evaluate reconstruction leakage on transmitted latent objects.

The repository excludes baseline implementations. Baseline results should be stored separately from this main-method artifact.
