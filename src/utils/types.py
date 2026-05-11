from dataclasses import dataclass

# from typing import Callable

# GetConceptSigFn = Callable[..., Any]


@dataclass(frozen=False)
class ConceptConfig:
    concepts_dir: str
    concept_name: str
    random_prefix: str
    eval_samples_path: str
    class_idx: int
    num_exps: int
    class_name: str

@dataclass(frozen=True)
class ConceptResponse:
    concept_score: float
    random_score: float
    concept_std: float
    random_std: float
    pval: float
