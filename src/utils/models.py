import os
import torch
import torchvision.models as models
import torch

CUSTOM_MODELS = ["twoLayer"]

class Densenet121(torch.nn.Module):
    def __init__(self,pretained:bool):
        super().__init__()
        
        weights = None    
        if pretained:
            weights=models.DenseNet121_Weights.IMAGENET1K_V1
        
        self.base = models.densenet121(weights=weights)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        features = self.base.features(x)
        features = torch.nn.functional.relu(features, inplace=True)
        # patch
        pooled = self.avgpool(features)
        flattened = torch.flatten(pooled, 1)
        out = self.base.classifier(flattened)
        return out


def get_patched_densenet121(num_classes: int | None, pretrained: bool = False):
    
    """
    returns a patched version of densenet121, with avgpool layer instead of F.avgpool in forward method as in original
    """

    model  = Densenet121(pretrained)
    if num_classes:
        model.base.classifier = torch.nn.Linear(model.base.classifier.in_features,num_classes)
    else:
        print(f"===== Loaded Densenet with default classifier head")

    print(f"===== Denset121 is patched with adaptive-avgpool-2d ====")
    
    return model

def build_inception(pretrained=True, num_classes=None):
    weights = models.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None

    model = models.inception_v3(
        weights=weights,
        aux_logits=True,   # REQUIRED for pretrained
    )

    # Kill aux head explicitly
    model.AuxLogits = None

    if num_classes is not None:
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    return model

def freeze_backbone(model):
    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the final fully connected layer
    if hasattr(model, "fc") and isinstance(model.fc, torch.nn.Linear):
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        raise ValueError("Model does not have a standard fc layer — check architecture.")
    
    return model

def unfreeze(model):
    # Freeze everything
    for param in model.parameters():
        param.requires_grad = True

    return model

def get_model(name: str, num_classes: int | None, pretrained: bool = False):
    if name not in {
        "densenet121",
        "resnet50",
        "resnet18",
        "inception_v3",
        "roberta-base",
    }:
        raise ValueError(f"Unsupported model '{name}'")

    if name == "roberta-base":

        from transformers import RobertaForSequenceClassification

        if num_classes is None:
            raise ValueError("num_classes must be specified for roberta-base")

        return RobertaForSequenceClassification.from_pretrained(
            name,
            num_labels=num_classes,
        )

    if name == "densenet121":
        return get_patched_densenet121(num_classes,pretrained)

    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)

        if num_classes:
            model.fc = torch.nn.Linear(
                model.fc.in_features,
                num_classes,
            )
        return model

    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)

        if num_classes:
            model.fc = torch.nn.Linear(
                model.fc.in_features,
                num_classes,
            )
        return model

    if name == "inception_v3":
        return build_inception(pretrained,num_classes)

def get_tokeinzer(name:str):

    if name == "roberta-base":
        from transformers import (RobertaTokenizerFast)
        return RobertaTokenizerFast.from_pretrained(name)

    else:
        raise ValueError(f"tokenizer is not defined for {name}")
    


def load_weights(model:torch.nn.Module,model_name:str,checkpoint_path:str,device:str|torch.device):

    checkpoint = torch.load(checkpoint_path,map_location=device,weights_only=False)
    weights = None

    if "model_state_dict" in checkpoint:
        weights = checkpoint["model_state_dict"]
    else:
        weights = checkpoint 

    if model_name == "densenet121":
        
        assert isinstance(model,Densenet121), f"Use the patched version of densenet"
        
        try:
            model.base.load_state_dict(weights)
            print(f"loaded checkpoint was from original torchvision")
        except:
            model.load_state_dict(weights)
            print(f"loaded checkpoint was from patched version")




    else:
        strict= True
        if model_name == "inception_v3":
            strict = False
        model.load_state_dict(weights,strict=strict)

    return model