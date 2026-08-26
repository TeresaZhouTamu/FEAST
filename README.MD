# FEAST: Federated Activation Steering for Protein Sequence Optimization

FEAST is a lightweight framework for collaborative protein sequence optimization with a frozen ESMC-600M protein language model.

It enables multiple clients to contribute property-specific steering information without sharing raw protein sequences or jointly training model parameters. Datasets can be found in the google drive link here: https://drive.google.com/drive/folders/1MZnz0Dm9meX6eK2o8ahrBr27YopEkMTF?usp=share_link

## Workflows

### Local Steering

Use this workflow when only local labeled data are available.

```text
Local labeled data
        ↓
extract_esmc_sv_local.py
        ↓
Local steering vectors
        ↓
steering_esmc_local.py
        ↓
Optimized sequences
```

- `extract_esmc_sv_local.py`: constructs layer-wise steering vectors from local positive and negative sequences.
- `steering_esmc_local.py`: optimizes target sequences using the local steering vectors.

### Federated Steering

Use this workflow when steering vectors from multiple clients are available.

```text
Client A data ─┐
Client B data ─┼─→ extract_esmc_fed_clientsv.py
Client C data ─┘
                       ↓
             Client steering vectors
                       ↓
      extract_esmc_fedmulti_purified.py
                       ↓
             Fused steering vectors
                       ↓
             steering_esmc_fed.py
                       ↓
              Optimized sequences
```

Each client independently runs `extract_esmc_fed_clientsv.py` on its local labeled data and shares only the resulting steering vectors.

The client performing optimization then runs:

- `extract_esmc_fedmulti_purified.py`: performs target-domain purification and multi-client steering-vector fusion.
- `steering_esmc_fed.py`: performs sequence optimization using the fused steering direction.

### Reliability-Aware Federated Steering

An optional reliability-aware workflow is also provided.

```text
Client data + reliability statistics
                ↓
extract_esmc_reliability_clientsv.py
                ↓
Reliability-aware steering vectors
                ↓
extract_esmc_fedmulti_reliability_purified.py
                ↓
Fused steering vectors
                ↓
steering_esmc_fed.py
```

This variant uses sample-level confidence, correctness, and variability to refine local steering estimation before federated fusion.

## Scripts

| Script | Description |
|---|---|
| `clientsv/extract_esmc_sv_local.py` | Extract local steering vectors |
| `steering_esmc_local.py` | Optimize sequences using local steering |
| `clientsv/extract_esmc_fed_clientsv.py` | Extract client steering vectors for federated steering |
| `fedsv/extract_esmc_fedmulti_purified.py` | Perform target-domain purification and multi-client fusion |
| `steering_esmc_fed.py` | Optimize sequences using federated steering vectors|
| `clientsv/extract_esmc_reliability_clientsv.py` | Extract reliability-aware client steering vectors |
| `fedsv/extract_esmc_fedmulti_reliability_purified.py` | Perform reliability-aware federated fusion |
| `steering_esmc_local.py` | Baseline activation-steering optimization implementation |
