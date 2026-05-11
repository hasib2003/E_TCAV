"""
contains helper functions for finding TCAV score for nlp inputs
"""

import os
import numpy as np
import torch
from scipy.stats import ttest_ind
from typing import List

# =======================
# Captum imports
# =======================
from captum.concept import Concept, TCAV
from captum.attr import LayerIntegratedGradients



# ============================================================
# Utilities
# ============================================================
from torch.utils.data import IterableDataset, DataLoader
import torch.nn as nn

MODEL_2_PENULTIMATE = {
    "resnet50":"avgpool",
    "densenet121":"avgpool",
    "inception3":"avgpool",
    "roberta-base":"classifier.dropout"
}

def get_classifier(classifier:str):

    if classifier == "default":
        from classifier.default import DefaultClassifier
        return DefaultClassifier()
    if classifier == "signal": 
        from classifier.signal_cav import SignalCav  
        return SignalCav()

    raise ValueError(f"Invalid classifier value {classifier} passed")
    

class CLSExtractor(nn.Module):
    def forward(self, hidden_states):
        # hidden_states: (batch, seq_len, hidden_dim)
        return hidden_states[:, 0, :]


class TextConceptDataset(IterableDataset):
    def __init__(self, texts, tokenizer, device, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len

    def __iter__(self):
        for t in self.texts:
            ids, attn = tokenize_texts([t], self.tokenizer, self.max_len)
            yield (
                ids.to(self.device),
                attn.to(self.device)
            )

def format_float(f):
    return float('{:.3f}'.format(f) if abs(f) >= 5e-4 else '{:.3e}'.format(f))




# ============================================================
# Text loading / tokenization
# ============================================================

def load_text_from_file(path: str, max_lines: int = 500) -> List[str]:
    
    """
    
    path: path to concept file (.txt required)
    read up to max_lines from given path

    """


    assert os.path.isfile(path),f"Expected concept to be contained in a file"
    
    texts = []    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)
            if len(texts) >= max_lines:
                return texts
    return texts


def tokenize_texts(
    texts: List[str],
    tokenizer,
    max_len: int
):
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)


# ============================================================
# Concept assembly
# ============================================================

# def idx_to_tensors(idx,texts,tokenizer, max_len):

#     ids, attn = tokenize_texts([texts[t]], tokenizer, max_len)
#     return (ids.to(device), attn.to(device))

def assemble_text_concept(
    name: str,
    cid: int,
    concepts_dir: str,
    tokenizer,
    device: str,
    max_len: int
) -> Concept:

    concept_path = os.path.join(concepts_dir, name)
    concept_text_list = load_text_from_file(concept_path)

    assert len(concept_text_list) > 0, f"Concept {name} has no text samples"

    dataset = TextConceptDataset(
        concept_text_list,
        tokenizer,
        device,
        max_len
    )

    dataloader = DataLoader(dataset, batch_size=1)

    return Concept(
        id=cid,
        name=name,
        data_iter=dataloader
    )

# ============================================================
# TCAV statistics
# ============================================================

def assemble_scores(scores, experimental_sets, idx, layer, score_type):
    out = []
    for concepts in experimental_sets:
        key = "-".join([str(c.id) for c in concepts])
        out.append(scores[key][layer][score_type][idx].cpu())
    return out


def get_pval(scores, experimental_sets, layer, score_type):
    P1 = assemble_scores(scores, experimental_sets, 0, layer, score_type)
    P2 = assemble_scores(scores, experimental_sets, 1, layer, score_type)

    return test_populations(P1,P2)    


def test_populations(P1,P2):
    P1_np = np.array([x.item() for x in P1])
    P2_np = np.array([x.item() for x in P2])

    # Compute t-test
    _, pval = ttest_ind(P1_np, P2_np)

    # Means and standard deviations
    mean1, mean2 = P1_np.mean(), P2_np.mean()
    std1, std2 = P1_np.std(ddof=1), P2_np.std(ddof=1)  # sample std dev

    return {
        'concept_score': float(mean1),
        'concept_std': float(std1),
        'random_score': float(mean2),
        'random_std': float(std2),
        'pval': format_float(pval)
    }


def get_cls_layer(model, layer_path: str):
    module = model
    for attr in layer_path.split("."):
        module = getattr(module, attr)
    return nn.Sequential(module, CLSExtractor())


# ============================================================
# Main entry point
# ============================================================

def prepare_tcav_experiment(
    *,
    model: torch.nn.Module,
    layers: list[str],
    classifier: str | None,
    concepts_dir: str,
    concept_name: str,
    dump_dir: str,
    num_rand_concepts: int,
    random_prefix: str,
    device: str,
    tokenizer,
    max_len: int,
):
    """
    Prepares TCAV experiment:
    - concepts
    - experimental sets
    - CLS-wrapped layers
    - TCAV object

    Returns:
        tcav: TCAV
        experimental_sets: list[list[Concept]]
    """

    concept_name = concept_name.strip() + ".txt"
    if concept_name not in os.listdir(concepts_dir):
        raise ValueError(f"Concept {concept_name} not found in {concepts_dir}")

    model.eval().to(device)

    # -----------------------
    # Forward function
    # -----------------------
    def forward_func(input_ids, attention_mask):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.logits

    # -----------------------
    # Concepts
    # -----------------------
    target_concept = assemble_text_concept(
        concept_name, 0, concepts_dir, tokenizer, device, max_len
    )

    rand_dirs = [
        d for d in os.listdir(concepts_dir)
        if d.startswith(random_prefix)
    ][:num_rand_concepts]

    if len(rand_dirs) < num_rand_concepts:
        raise ValueError(
            f"Requested {num_rand_concepts} random concepts, "
            f"found only {len(rand_dirs)}"
        )

    random_concepts = [
        assemble_text_concept(
            d, i + 1, concepts_dir, tokenizer, device, max_len
        )
        for i, d in enumerate(rand_dirs)
    ]

    experimental_sets = [
        [target_concept, rc] for rc in random_concepts
    ]

    # -----------------------
    # CLS layers
    # -----------------------
    cls_layers = {
        layer: get_cls_layer(model, layer)
        for layer in layers
    }

    # -----------------------
    # TCAV object
    # -----------------------
    tcav = TCAV(
        model=model,
        layers=cls_layers,
        classifier=get_classifier(classifier),
        save_path=dump_dir,
        layer_attr_method=LayerIntegratedGradients(
            forward_func,
            None,
            multiply_by_inputs=False,
        ),
    )

    return tcav, experimental_sets

def get_concept_significance(
    tcav_mode: str,
    model: torch.nn.Module,
    layers: list[str],
    classifier: str | None,
    concepts_dir: str,
    concept_name: str,
    class_idx: int,
    eval_samples_path: str,
    dump_dir: str,
    score_type: str,
    num_rand_concepts: int,
    random_prefix: str,
    device: str,
    **kwargs,
):


    """
    model               : nn.Module object
    layers              : list of strings, representing layers to extract concepts from
    classifier          : name of the classifier to be used to find CAV
    concepts_dir        : directory where all concept samples are present
    concept_name        : name of the concept, expects a dir or file with concept_name(.txt) in concepts_dir
    class_idx           : index of the class in the logits vector for which TCAV has to be computed
    eval_samples_path   : path to samples (file or dir for txt, img respectively) w.r.t whom directional sensitivity is calculated
    dump_dir            : dir where the activations are stored, as part of TCAV
    score_type          : magnitude or sign_count
    random_prefix       : naming convention of random samples, should be present in concepts_dir; e.g random_prefix1,random_prefix2 etc
    num_rand_concepts   : num of different runs to be executed, each with different sample  
    device              : device for computation

    **kwargs            : additional args required when input_type is text
                                    1) max_len
                                    2) tokenizer
                                    ..
                                    3) last_module_path is required when tcav_mode is fast
                 

    Computes TCAV scores and statistical significance for a text concept.

    Returns:
        dict[layer] = {
        'concept_score': float,
        'concept_std': float,
        'random_score': float,
        'random_std': float,
        'pval': float
    }
    """

    if score_type not in {"magnitude", "sign_count"}:
        raise ValueError(f"Invalid score_type: {score_type}")

    if tcav_mode == "fast":
        assert "last_module_path" in kwargs, f"Fast-TCAV needs the path of the last layer"
        assert len(layers) ==1 , f"Fast-TCAV is only defined for the penultimate layer, got multiple {layers=}"

    tokenizer = kwargs["tokenizer"]
    max_len = kwargs["max_len"]

    tcav, experimental_sets = prepare_tcav_experiment(
        model=model,
        layers=layers,
        classifier=classifier,
        concepts_dir=concepts_dir,
        concept_name=concept_name,
        dump_dir=dump_dir,
        num_rand_concepts=num_rand_concepts,
        random_prefix=random_prefix,
        device=device,
        tokenizer=tokenizer,
        max_len=max_len,
    )

    results = {}
    cavs = tcav.compute_cavs(experimental_sets, force_train=True)

    if tcav_mode == "fast":

        last_module = model
        for attr in kwargs["last_module_path"].split("."):
            last_module = getattr(last_module, attr)

        assert type(last_module) is nn.Linear, (
            f"Expected nn.Linear, got {type(last_module)}"
        )
        
        for layer in layers:
            layer_cavs = torch.stack(
                [
                    cav_obj[layer].stats["weights"][0]
                    for cav_obj in cavs.values()
                ]
            ).to(device)
            
            cav_accs = []
            cav_aucs = []
            for cav_obj in cavs.values():
                cav_accs.append(cav_obj[layer].stats.get("accs",-1))
                cav_aucs.append(cav_obj[layer].stats.get("auc",-1))

                    
            weight_vec = last_module.weight[class_idx].reshape(-1,1)
            dot_product = (layer_cavs @ weight_vec) > 0
            concept_scores = dot_product.to(dtype=torch.float32)
            random_scores = (concept_scores == 0).to(dtype=torch.float32)

            layer_stats =  test_populations(concept_scores,random_scores)
            results[layer] = layer_stats
            results[layer]["num_eval_samples"] = 0
            results[layer]["accs"] = float(np.mean(cav_accs))
            results[layer]["aucs"] = float(np.mean(cav_aucs))

        return results


    # -----------------------
    # Evaluation data
    # -----------------------
    eval_texts = load_text_from_file(eval_samples_path)
    input_ids, attn = tokenize_texts(eval_texts, tokenizer, max_len)

    eval_inputs = (
        input_ids.to(device),
        attn.to(device),
    )

    scores = tcav.interpret(
        eval_inputs,
        experimental_sets,
        class_idx,
        n_steps=5,
    )

    results = {}
    for layer in layers:
        results[layer] = get_pval(
            scores,
            experimental_sets,
            layer,
            score_type,
        )

        cav_accs = []
        cav_aucs = []
        for cav_obj in cavs.values():
            cav_accs.append(cav_obj[layer].stats.get("accs",-1))
            cav_aucs.append(cav_obj[layer].stats.get("auc",-1))

        results[layer]["num_eval_samples"] = len(eval_texts)
        results[layer]["accs"] = float(np.mean(cav_accs))
        results[layer]["aucs"] = float(np.mean(cav_aucs))

    return results

def get_cav(
    model: torch.nn.Module,
    layers: list[str],
    classifier: str | None,
    concepts_dir: str,
    concept_name: str,
    dump_dir: str,
    num_rand_concepts: int,
    random_prefix: str,
    device: str,
    **kwargs,
):
    tokenizer = kwargs["tokenizer"]
    max_len = kwargs["max_len"]

    tcav, experimental_sets = prepare_tcav_experiment(
        model=model,
        layers=layers,
        classifier=classifier,
        concepts_dir=concepts_dir,
        concept_name=concept_name,
        dump_dir=dump_dir,
        num_rand_concepts=num_rand_concepts,
        random_prefix=random_prefix,
        device=device,
        tokenizer=tokenizer,
        max_len=max_len,
    )

    cavs = tcav.compute_cavs(experimental_sets, force_train=True)

    all_cavs = {}
    for layer in layers:
        all_cavs[layer] = torch.stack(
            [
                cav_obj[layer].stats["weights"][0]
                for cav_obj in cavs.values()
            ]
        )

    return all_cavs

if __name__ == "__main__":

    import shutil
    
    from transformers import (
            RobertaTokenizerFast,
            RobertaForSequenceClassification,
            )



    MODEL_NAME = "roberta-base"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT = "/netscratch/aslam/TCAV/fast-tcav/nlp/wiki/checkpoints/best_model.pth"

    LAYERS = ["classifier.dropout"]
    
    # LAYERS = [
    # "roberta.encoder.layer.11.output",
    # "classifier.dense",
    # "classifier.out_proj",
    # ]

    INDICES = [0,1]


    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(DEVICE)

    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)

    checkpoint = torch.load(CHECKPOINT,map_location=DEVICE,weights_only=False)
    model.load_state_dict(checkpoint)
    model = model.to(DEVICE)



    for idx in INDICES:


        result  = get_concept_significance(
            tcav_mode="fast",
            model = model,
            layers= LAYERS ,
            classifier =  "signal",
            concepts_dir = "/netscratch/aslam/TCAV/fast-tcav/nlp/wiki/data/concepts/samples",
            concept_name = "obscene",
            class_idx = idx,
            eval_samples_path ="/netscratch/aslam/TCAV/fast-tcav/nlp/wiki/data/concepts/samples/evaluation-toxic.txt",
            dump_dir = "./test-captum-nlp",
            score_type =  "sign_count",
            num_rand_concepts = 20,
            alpha =  0.05,
            random_prefix = "random_tweets_",
            device = DEVICE,
            max_len = 256,
            tokenizer = tokenizer,
            last_module_path="classifier.out_proj"

        )
        # all_cavs  = get_cav(
        #     model = model,
        #     layers= LAYERS ,
        #     classifier =  "signal",
        #     concepts_dir = "/netscratch/aslam/TCAV/fast-tcav/nlp/wiki/data/concepts/samples",
        #     concept_name = "obscene",
        #     dump_dir = "./test-captum-nlp",
        #     num_rand_concepts = 20,
        #     alpha =  0.05,
        #     random_prefix = "random_tweets_",
        #     device = DEVICE,

        #     max_len = 256,
        #     tokenizer = tokenizer,

        # )
        # # print(f"{all_cavs.shape=}")
        print(result)
        shutil.rmtree("./test-captum-nlp")
        
        
        # num_exps = 20

        # reports = []

        # layer_2_stats = {}
                    
        # for layer,layer_cavs in all_cavs.items():

        #     # print(f"{layer_cavs.device=}")
        #     # print(f"{model.fc.weight[concept["idx"]].device=}")

        #     print(f"{layer_cavs.shape=}")

        #     weight_vec = model.classifier.out_proj.weight[idx].reshape(-1,1).cpu()

        #     concept_scores   = (layer_cavs.cpu() @ weight_vec).cpu()
        #     random_scores    = concept_scores * -1


        #     pos_counts = torch.sum(dot_product > 0).cpu().item()
        #     neq_counts = num_exps - pos_counts

        #     layer_2_stats[layer] = {"concept":pos_counts/num_exps,"random":neq_counts/num_exps}
        
        # # reports.append({"results": layer_2_stats})
        # print(f"{layer=}, stats: {layer_2_stats}")



