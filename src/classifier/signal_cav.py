# replacement of SKSG liner model, using cuda support



import torch
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Union
import numpy as np
from captum.concept._utils.classifier import Classifier
from sklearn.metrics import roc_auc_score

class SignalCav(Classifier):
    r"""
    Signal-based Concept Activation Vector (Signal-CAV) implementation.
    Computes pattern vectors (h_pat) via covariance between activations and
    concept labels, following Haufe et al. (2014) and Eq.(3) in the
    Pattern-based CAV paper.
    """

    def __init__(self) -> None:
        self.h_pat: torch.Tensor = torch.empty(0)
        self.class_list : List[np.int32] = []
        self.test_split_ratio = 0.33

    def train_and_eval(
        self,
        dataloader: DataLoader,
        **kwargs: Any,
    ) -> Union[Dict, None]:
        
        test_split_ratio = 0.33
        # 1. Collect everything from the dataloader
        activations, labels = [], []
        for inputs, lbls in dataloader:
            activations.append(inputs)
            labels.append(lbls)

        A = torch.cat(activations).float()  # (N, F)
        t = torch.cat(labels)  # (N,)

        # 2. Check labels and establish mapping
        unique_labels = torch.unique(t, sorted=True)
        assert len(unique_labels) == 2, f"Signal Cav requires exactly 2 labels. Found {unique_labels}"
        self.class_list = torch.tensor(unique_labels).cpu().numpy().astype(np.int32)   

        
        # map the labels to 1 and -1, concept to -1 and random set to 1        
        targets = torch.where(t == unique_labels[1], 1.0, -1.0)

        num_samples = A.shape[0]
        indices = torch.randperm(num_samples)
        test_size = int(num_samples * self.test_split_ratio)
        
        test_idx, train_idx = indices[:test_size], indices[test_size:]
        
        A_train, A_test = A[train_idx], A[test_idx]
        y_train, y_test = targets[train_idx], targets[test_idx]



        y = y_train.cpu().numpy()
        X = A_train.cpu().numpy()

        mean_y = y.mean()
        X_residuals = X - X.mean(axis=0)[None]
        covar = (X_residuals * (y - mean_y)[:, np.newaxis]).sum(axis=0) / (y.shape[0] - 1)
        vary = np.sum((y - mean_y) ** 2, axis=0) / (y.shape[0] - 1)
        w = (covar / vary)
        self.h_pat = torch.tensor(w,dtype=torch.float32).cpu()

        

        # computing metrics using the test set

        scores_test = (A_test.cpu() @ self.h_pat).squeeze()
    
        # Pearson Correlation on Test Set
        # We stack the continuous scores and the {-1, 1} mapped labels
        corr = torch.corrcoef(torch.stack([scores_test, y_test]))[0, 1].item()
        auc = roc_auc_score(y_test.cpu().numpy(), scores_test.cpu().numpy())

        return {"accs": corr,"auc":auc}

    def weights(self) -> torch.Tensor:
        if self.h_pat.numel() == 0:
            raise RuntimeError("You must call train_and_eval() before accessing weights().")
        # Return in shape [C, F]; here we define C=2 (concept, non-concept)
        # so Captum can process it consistently.
        weights= torch.stack([-1*self.h_pat,self.h_pat]).to("cpu")

        # print("weights.shape ",weights.shape)
        # print("weights.dtype ",weights.dtype)

        return weights


    def classes(self) -> List[np.int32]:
        # Captum expects classes in same order as weight rows.
        # Concept (id=0) first, then non-concept (>0)
        # print("classes ",self.class_list)
        # print("classes.shape ",self.class_list.shape)        
        # print("classes.dtype ",self.class_list.dtype)        
        if self.class_list is None: 
            raise ValueError("Class list is not defined")
        return self.class_list


if __name__ == "__main__":

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    # ---- import your SignalCAVClassifier implementation ----
    # from your_module import SignalCAVClassifier
    # (For quick testing, just paste the SignalCAVClassifier class definition above this block)

    # --------------------------------------------------------
    # 1. Generate synthetic activations
    # --------------------------------------------------------
    torch.manual_seed(42)
    N_concept = 50
    N_random = 50
    F = 10  # number of features

    # Concept-present samples: mean shifted along +e0 and +e1
    A_concept = torch.randn(N_concept, F) + torch.tensor([1.5, 1.0] + [0]*(F-2))

    # Non-concept/random samples: centered at zero
    A_random = torch.randn(N_random, F)

    # Stack everything
    A_all = torch.cat([A_concept, A_random], dim=0)
    t_all = torch.cat([torch.zeros(N_concept), torch.ones(N_random)])  # 0 = concept, 1 = non-concept

    dataset = TensorDataset(A_all, t_all)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # --------------------------------------------------------
    # 2. Train the Signal-CAV classifier
    # --------------------------------------------------------
    clf = SignalCav()
    stats = clf.train_and_eval(loader)
    print("Training stats:", stats)

    # --------------------------------------------------------
    # 3. Inspect learned weights
    # --------------------------------------------------------
    weights = clf.weights()
    print("\nWeights shape:", weights.shape)
    print("First 5 values of h_pat:", weights[1, :5])

    # --------------------------------------------------------
    # 4. Test correlation directionality
    # --------------------------------------------------------
    # Projection scores (dot product of activations with h_pat)
    scores = A_all @ weights[1]
    corr = torch.corrcoef(torch.stack([scores, t_all]))[0, 1]
    print(f"\nCorrelation between projection and labels: {corr:.4f}")

    # --------------------------------------------------------
    # 5. Optional sanity check: concept mean vs non-concept mean along h_pat
    # --------------------------------------------------------
    proj_concept = (A_concept @ weights[0]).mean()
    proj_random = (A_random @ weights[1]).mean()
    print(f"\nMean projection (concept):  {proj_concept:.4f}")
    print(f"Mean projection (non-concept): {proj_random:.4f}")
    print(f"Difference (should be positive): {(proj_concept - proj_random):.4f}")
