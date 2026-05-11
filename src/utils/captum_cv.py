
import os
import numpy as np
# ..........torch imports............
import torch
from torchvision import transforms

#.... Captum imports..................
from captum.concept import Concept
from captum.concept._utils.data_iterator import dataset_to_dataloader, CustomIterableDataset
from captum.attr import  LayerIntegratedGradients
from captum.concept import TCAV

#.... Custom imports..................

from .captum_text import test_populations,get_cls_layer,get_classifier

from PIL import Image

# ..........torch imports............
import numpy as np
from scipy.stats import ttest_ind

def get_transform(size:int):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

def get_tensor_from_filename(filename,transform_fn):
    img = Image.open(filename).convert("RGB")
    return transform_fn(img)

def load_image_tensors(path:str, max_files=200):


    # print("path ",path)
    
    filenames = sorted(os.listdir(path))

    if len(filenames) > max_files:
        rng = np.random.default_rng(42)
        filenames = rng.choice(filenames,size=max_files,replace=False)

    filenames = [os.path.join(path,file) for file in filenames]

    tensors = []
    for filename in filenames:
        img = Image.open(filename).convert('RGB')
        tensors.append(img)
    
    return tensors

def assemble_concept(name, id, concepts_path,transform_fn):
    concept_path = os.path.join(concepts_path, name) + "/"
    # dataset = CustomIterableDataset(get_tensor_from_filename, concept_path)
    
    dataset = CustomIterableDataset(lambda file_name:get_tensor_from_filename(file_name,transform_fn), concept_path)
    concept_iter = dataset_to_dataloader(dataset)

    return Concept(id=id, name=name, data_iter=concept_iter)

def format_float(f):
    return float('{:.3f}'.format(f) if abs(f) >= 0.0005 else '{:.3e}'.format(f))

def assemble_scores(scores, experimental_sets, idx, score_layer, score_type):
    score_list = []
    for concepts in experimental_sets:
        score_list.append(scores["-".join([str(c.id) for c in concepts])][score_layer][score_type][idx].cpu())
        
    return score_list


def get_pval(scores, experimental_sets, layer, score_type):
    P1 = assemble_scores(scores, experimental_sets, 0, layer, score_type)
    P2 = assemble_scores(scores, experimental_sets, 1, layer, score_type)

    # Convert to numpy arrays for easier handling
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
        num_rand_concepts: int ,
        random_prefix: str,
        device: str,
        **kwargs
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

    print(f"device in get_concept_significance is {device=}")

    if tcav_mode == "fast":
        assert "last_module_path" in kwargs, f"Fast-TCAV needs the path of the last layer"
        assert len(layers) ==1 , f"Fast-TCAV is only defined for the penultimate layer, got multiple {layers=}"


    assert concept_name in os.listdir(concepts_dir) , f"Expected concepts to be in {os.listdir(concepts_dir)}"
    assert score_type in ["magnitude","sign_count"]

    img_size = 224
    if "inception" in model.__class__.__name__.lower():
        img_size = 299

    transform_fn = get_transform(img_size)
    
    model.eval()
    model = model.to(device)




    target_concept = assemble_concept(concept_name, 0, concepts_dir,transform_fn)


    all_dirs = os.listdir(concepts_dir)
    all_dirs = [d for d in all_dirs if d.startswith(random_prefix)]
    random_concepts = [assemble_concept(all_dirs[i],(i+2),concepts_dir,transform_fn) for i in range(0, num_rand_concepts)] 
    experimental_sets = [[target_concept, random_concept] for random_concept in random_concepts]

    clf = None
    if classifier is not None:
        clf = get_classifier(classifier)

    if "vit" in model.__class__.__name__.lower():
        print(f"Vision Transformer detected ! Cls token will be hooked from specified layers")
        _layers = {
            layer: get_cls_layer(model, layer)
            for layer in layers
            }
        layers = _layers




    mytcav = TCAV(model=model,
                layers=layers,
                classifier=clf,
                save_path=dump_dir,
                layer_attr_method = LayerIntegratedGradients(
                model, None, multiply_by_inputs=False))

            
    cavs = mytcav.compute_cavs(experimental_sets, force_train=True)    
    results = {}

    if tcav_mode == "fast":
        
        last_module = model
        for attr in kwargs["last_module_path"].split("."):
            last_module = getattr(last_module, attr)

        assert type(last_module) is torch.nn.Linear, (
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
            cav_aucs  = []
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


 
    eval_images = load_image_tensors(path=eval_samples_path)
    eval_tensors = torch.stack([transform_fn(img) for img in eval_images])
    eval_tensors = eval_tensors.to(device)

    assert next(model.parameters()).device == eval_tensors.device, "Model and tensors on different devices!"

    scores = mytcav.interpret(eval_tensors, experimental_sets, class_idx, n_steps=5)

    accs = []
    aucs = []
    
    for layer in layers:
        
        accs = []
        aucs = []
        for cav_obj in cavs.values():
            accs.append(cav_obj[layer].stats["accs"])
            aucs.append(cav_obj[layer].stats["aucs"])
        
        layer_stats = get_pval(scores, experimental_sets, layer, score_type=score_type)
    
        cav_accs = []
        cav_aucs = []
        for cav_obj in cavs.values():
            cav_accs.append(cav_obj[layer].stats.get("accs",-1))
            cav_aucs.append(cav_obj[layer].stats.get("auc",-1))

    
    
        results[layer] = layer_stats
        results[layer]["num_eval_samples"] = len(eval_images)
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
        num_rand_concepts: int ,
        random_prefix: str,
        device: str,
        **kwargs
    ):
    """
    model               : nn.Module object
    layers              : list of strings, representing layers to extract concepts from
    classifier          : name of the classifier to be used to find CAV
    concepts_dir        : directory where all concept samples are present
    concept_name        : name of the concept, expects a dir or file with concept_name(.txt) in concepts_dir
    dump_dir            : dir where the activations are stored, as part of TCAV
    num_rand_concepts   : num of different runs to be executed, each with different sample  
    random_prefix       : naming convention of random samples, should be present in concepts_dir; e.g random_prefix1,random_prefix2 etc
    device              : device for computation

    **kwargs            : additional args required when input_type is text
                                    1) max_len
                                    2) tokenizer
                 
    Computes Concept Activation Vectors

    Returns:
        dict[layer] = {
            "layer_name": tensor (num_rand_concepts * cav_dim)
        }
    """


    assert concept_name in os.listdir(concepts_dir) , f"Expected concepts to be in {os.listdir(concepts_dir)}"

    img_size = 224
    if "inception" in model.__class__.__name__.lower():
        img_size = 299

    transform_fn = get_transform(img_size)
    
    target_concept = assemble_concept(concept_name, 0, concepts_dir,transform_fn)
    model = model.to(device)
    random_concepts = [assemble_concept(random_prefix + str(i+0), (i+2),concepts_dir,transform_fn) for i in range(0, num_rand_concepts)] 
    experimental_sets = [[target_concept, random_concept] for random_concept in random_concepts]

    clf = None
    if classifier is not None:
        clf = get_classifier(classifier)


    model.eval()
    with torch.no_grad():
        mytcav = TCAV(model=model,
                    layers=layers,
                    classifier=clf,
                    save_path=dump_dir,
                    layer_attr_method = LayerIntegratedGradients(
                    model, None, multiply_by_inputs=False))

    cavs = mytcav.compute_cavs(experimental_sets, force_train=True)

    all_cavs = {}

    for layer in layers:

        list_weights = []
        for _, cav_obj in cavs.items():


            list_weights.append(cav_obj[layer].stats["weights"][0])
        all_cavs[layer] = torch.stack(list_weights)


    return all_cavs



if __name__ == "__main__":

    # vit.encoder.layer.0
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


    from utils.models import get_model

    model = get_model("vit_b_16",2,True)

    for name,module in model.named_modules():
        print(name,module)
        print("**")

    # result = get_concept_significance(
    #     tcav_mode="default",
    #     model=model,
    #     classifier="default",
    #     concepts_dir="/netscratch/aslam/TCAV/dataset/textures",
    #     concept_name="striped",
    #     class_idx=340,
    #     eval_samples_path="/ds/images/imagenet/val_folders/n02129604",
    #     dump_dir="./",
    #     score_type="sign_count",
    #     num_rand_concepts=20,
    #     random_prefix="random500_",
    #     layers=["vit.encoder.layer.10","vit.encoder.layer.11"],
    #     device=DEVICE
    # )

    # print(f"{result=}")
    
