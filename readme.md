# E-TCAV: Formalizing Penultimate Proxies for Efficient Concept-Based Interpretability

Official implementation of:

> **E-TCAV: Formalizing Penultimate Proxies for Efficient Concept-Based Interpretability**

---

## Overview

Testing with Concept Activation Vectors (TCAV) is a concept-based interpretability method that quantifies the alignment between neural network representations and human-understandable concepts. Despite its effectiveness, TCAV suffers from three major limitations:

- high computational overhead,
- instability induced by latent classifier choice,
- and disagreement of TCAV scores across layers.

This repository introduces **E-TCAV**, an efficient approximation framework for TCAV that leverages:

- inter-layer agreement of TCAV scores,
- directional sensitivity degeneracy at the penultimate layer,
- and robust latent classifier analysis,

to significantly reduce computational cost while preserving interpretability fidelity.

E-TCAV enables linearly scaling speed-ups with respect to:

- network depth,
- number of evaluated layers,
- and evaluation sample count.

The framework is evaluated across multiple architectures and datasets spanning both computer vision and natural language tasks.

## Repository Structure

```text
src/
├── classifier/
│   └── signal_cav.py
│
├── concepts/
│   ├── celeba/
│   ├── isic/
│   ├── scdb/
│   ├── wiki/
│
├── config/
│   ├── celeba.json
│   ├── imagenet.json
│   ├── isic.json
│   └── wiki.json
│
├── dataset/
│
├── pipelines/
│   ├── args/
│   ├── trainers/
│   ├── main.py
│   └── evaluate-concepts-main.py
│
└── utils/
    ├── captum_cv.py
    ├── captum_text.py
    ├── common.py
    ├── logger.py
    ├── models.py
    ├── train.py
    └── types.py
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/hasib2003/E_TCAV.git
cd E_TCAV
```

Create a virtual environment:

```bash
python -m venv etcav-env
source etcav-env/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Supported Modalities

### Computer Vision

Supported architectures include:

- ResNet18
- ResNet50
- DenseNet121
- Inception-v3

Supported datasets include:

- CelebA
- ISIC
- SCDB
- ImageNet

### Natural Language Processing

Supported architectures include:

- RoBERTa-base

Supported datasets include:

- Wiki-based concept datasets

---

## Running E-TCAV

### Vision Example

```bash
bash scripts/example_cv.sh scdb resnet50 default
```

### NLP Example

```bash
bash scripts/example_nlp.sh wiki roberta-base default
```

---

## Example Command

```bash
python -m pipelines.main \
    --tcav_mode default \
    --classifier default \
    --input_type image \
    --concept_config config/scdb/config.json \
    --model resnet50 \
    --layers layer4.0 layer4.1 layer4.2 avgpool \
    --checkpoint /path/to/checkpoint.pth \
    --num_classes 2 \
    --save_dir outputs/scdb/resnet50
```

## Citation

```bibtex
@article{etcav2026,
  title={E-TCAV: Formalizing Penultimate Proxies for Efficient Concept-Based Interpretability},
  year={2026}
}
```

## License

MIT License