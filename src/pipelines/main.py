import os
import csv  
import json
import time
import torch
import shutil      
from datetime import datetime
from utils.models import  get_model, get_tokeinzer
from pipelines.args.tcav import args_evaluate_tcav
from utils.types import ConceptConfig
from utils.models import load_weights
from utils.logger import Logger
import warnings


def concept_kwargs(cfg: ConceptConfig) -> dict:
    return {
        "concepts_dir": cfg.concepts_dir,
        "concept_name": cfg.concept_name,
        "random_prefix": cfg.random_prefix,
        "eval_samples_path": cfg.eval_samples_path,
        "class_idx": cfg.class_idx,
        "class_name": cfg.class_name,
        "num_rand_concepts":cfg.num_exps

    }

MODEL_2_LAST = {
    "roberta-base":"classifier.out_proj",
    "densenet121":"base.classifier",
    "resnet50":"fc",
    "resnet18":"fc",
    "inception_v3":"fc",
}

MODEL_2_PENULTIMATE = {
    "resnet50":"avgpool",
    "resnet18":"avgpool",
    "densenet121":"avgpool",
    "inception_v3":"avgpool",
    "roberta-base":"classifier.dropout"
}

def load_json(path: str):
    """Load target concepts from json file path."""
    concepts = None
    with open(path, 'r') as f:
        concepts = json.load(f)
    
    return concepts

def parse_concept_config(raw: list[dict]) -> list[ConceptConfig]:
    configs = []
    for i, item in enumerate(raw):
        try:
            configs.append(ConceptConfig(**item))
        except TypeError as e:
            raise ValueError(
                f"Invalid concept config at index {i}: {e}"
            ) from None
    return configs

def config_2_report(
    tcav_mode:str,
    concept_config: ConceptConfig,
    get_concept_significance,
    **kwargs
    )->list[dict]:
    

    report = []      
    last_module = MODEL_2_LAST[kwargs["model_name"]]
    layers = kwargs["layers"]
   
    if tcav_mode == "fast" and (len(layers) > 1 or layers[-1] != MODEL_2_PENULTIMATE[kwargs["model_name"]]):
        
        kwargs["layers"] = [MODEL_2_PENULTIMATE[kwargs["model_name"]]]
            
        warnings.warn(f"Fast TCAV is only defined for penultimate layers\nThe requested layers are either gt 1 or don't have valid penultimate layer\nLayer are overriden to {kwargs['layers']}",stacklevel=2)


    ckw = concept_kwargs(concept_config)
    concept_report = get_concept_significance(
        tcav_mode=tcav_mode,
        score_type="sign_count",
        last_module_path = last_module,
        **kwargs,
        **ckw,
    )

    concept_info = {
        k: v for k, v in ckw.items()
        if k != "layers"
    }

    for layer in kwargs["layers"]:
        layer_stats = concept_report[layer]

        row:dict = (
            {"layer": layer}
            | layer_stats
            | concept_info
        )

        report.append(row)

    
    return report
    

def handle_concept_generation(tcav_mode:str,input_type:str,concept_configs:list[ConceptConfig],**kwargs) -> list[dict]:
    """"""
    
    if input_type == "text":
        from utils.captum_text import get_concept_significance
    else:
        from utils.captum_cv import get_concept_significance

    master_report = []

    for concept_config in concept_configs:

        print(f"{'*'*4} evaluating {concept_config.concept_name} on class {concept_config.class_name} " )
        
        # torch.cuda.synchronize()
        start = time.time()
        config_report = config_2_report(tcav_mode,concept_config,get_concept_significance,**kwargs)
        # torch.cuda.synchronize()
        end = time.time()
        exec_time = end - start

        print(f"{'*'*4} exec time {exec_time}",flush=True)

        for item in config_report:
            print(f"\n{'*'*4} Summary " )

            print(f"{item['concept_name']=}")
            print(f"{item['class_name']=}")
            print(f"{item['num_eval_samples']=}")
            print(f"{item['accs']=}")
            print(f"{item['aucs']=}")
            print(f"{item['concept_score']=}")
            print(f"{item['random_score']=}")
            print(f"{item['pval']=}")

            print(f"{'*'*20} ",flush=True)


        for item in config_report:
            item["exec_time"] = exec_time
            item["model"] = kwargs["model_name"]
            item["classifier"] = kwargs["classifier"]
        
        master_report.extend(config_report)


    return master_report


def save_concept_report(concepts_report:list[dict],save_path:str):
        
        if not concepts_report:
            raise ValueError("concepts_report is empty - nothing to save")

        fieldnames = list(concepts_report[0].keys())

        file_exists = os.path.isfile(save_path)

        with open(save_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            # write header only once
            if not file_exists:
                writer.writeheader()

            writer.writerows(concepts_report)
               
    
def main():
    
    args = args_evaluate_tcav()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Print config
    print(f"{'='*20} Config {'='*20}")
    args_dict = vars(args)
    max_key_len = max(len(k) for k in args_dict.keys())
    for key, val in args_dict.items():
        print(f"{key:<{max_key_len + 3}}: {val}")
    print(f"{'='*50}")

    assert os.path.isfile(args.concept_config), f"Invalid path to concept config file"
    json_load = load_json(args.concept_config)
    concept_configs = parse_concept_config(json_load)

    if args.input_type == "text" and (not args.tokenizer or not args.max_len):
        raise ValueError("Expected tokenizer and max_len when input_type is text")



    assert not (args.checkpoint and not os.path.isfile(args.checkpoint)), f"No file found at {args.checkpoint}"
    os.makedirs(args.save_dir, exist_ok=True)

    # create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir,"logs.txt")

    # Logger(log_file=log_path).enable()
    print(f"\n {'=*'*5} Logging at {log_path}")

    print(f"before {args.layers=}")
    if args.model == "densenet121":
        layers = [f"base.{layer}" if "avgpool" != layer else layer for layer in args.layers]
        args.layers = layers
    print(f"after {args.layers=}")



    with open(os.path.join(run_dir,"args.json"),"w") as f:
        json.dump(args_dict,f,indent=4)

    with open(os.path.join(run_dir,"concept-config.json"),"w") as f:
        json.dump(json_load,f,indent=4)
        
    model = get_model(args.model, args.num_classes, args.checkpoint is None)
    tokenizer = None
    if args.tokenizer:
        tokenizer = get_tokeinzer(args.tokenizer)
        print("==== loaded tokenizer successfully")

    if args.checkpoint:
        model = load_weights(model,args.model,args.checkpoint,device)
        print("==== custom checkpoint loaded")

    print(f"==== created model successfully ")

    model = model.to(device)
    print(f"Switched model to {device=}")

    if args.tcav_mode != "both":
        tcav_modes = [args.tcav_mode]
    else:
        tcav_modes = ["default","fast"]

    cav_dump_dir = os.path.join(run_dir,"cav-dump")
    os.makedirs(cav_dump_dir,exist_ok=True)


    try:
        for mode in tcav_modes:    
            concepts_report = handle_concept_generation(
                tcav_mode=mode,
                input_type=args.input_type,
                concept_configs=concept_configs,                
                model_name = args.model,
                model=model,
                layers=args.layers,
                tokenizer=tokenizer,
                max_len=args.max_len,
                classifier=args.classifier,
                dump_dir=cav_dump_dir,
                device=device
                )        
            
            shutil.rmtree(cav_dump_dir) # clear the cav-dump directory
          
            save_path = os.path.join(run_dir,f"results.csv")
            save_concept_report(concepts_report,save_path)
            print(f"{'-'*4} saved to {save_path} ")


    except Exception as e:
        # At least log the error; never leave except empty
        print(f"Error occurred during concept generation: {e}")
        shutil.rmtree(cav_dump_dir)
        raise        

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    main()