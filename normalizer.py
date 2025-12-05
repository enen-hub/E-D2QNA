import torch

class SimpleNormalizer:
    def __init__(self, ideal=None, nadir=None):
        self.ideal = torch.tensor(ideal or [0.0, 0.0], dtype=torch.float32)
        self.nadir = torch.tensor(nadir or [800.0, 2500.0], dtype=torch.float32)
    def normalize(self, vec: torch.Tensor, to_negative: bool = True) -> torch.Tensor:
        denom = torch.clamp(self.nadir - self.ideal, min=1e-6)
        norm = (vec - self.ideal.to(vec.device)) / denom.to(vec.device)
        return -norm if to_negative else norm