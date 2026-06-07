# Artifact Paths

This file documents the paths expected by the example scripts. It intentionally uses placeholders instead of local absolute paths.

| Variable | Meaning |
|---|---|
| `MEDLATENT_DATA_DIR` | Directory containing bundled diagnosis JSON files, hospital partitions, HPO embeddings, and IC weights. Defaults to `./data`. |
| `MEDLATENT_MODEL_DIR` | Directory containing downloaded frozen LLM backbones. |
| `MEDLATENT_ARTIFACT_DIR` | Directory for trained distillers, projectors, latent caches, and evaluation outputs. |

Expected data files:

```text
${MEDLATENT_DATA_DIR}/train.json
${MEDLATENT_DATA_DIR}/val.json
${MEDLATENT_DATA_DIR}/test.json
${MEDLATENT_DATA_DIR}/hospital_0.json
${MEDLATENT_DATA_DIR}/hospital_1.json
${MEDLATENT_DATA_DIR}/hospital_2.json
${MEDLATENT_DATA_DIR}/hospital_3.json
${MEDLATENT_DATA_DIR}/hospital_4.json
${MEDLATENT_DATA_DIR}/hpo_embeddings.json.gz
${MEDLATENT_DATA_DIR}/hpo_ic.json
```
