import torch
from torch.utils.data import DataLoader
from src.model import build_model
from src.train import train_one_epoch

def test_one_epoch_runs_cpu():
    x = torch.randn(4, 3, 224, 224)
    y = torch.randint(0, 2, (4, 14)).float()
    class DS(torch.utils.data.Dataset):
        def __len__(self): return 4
        def __getitem__(self, i): return x[i], y[i], {"sex":"M","age_bin":"40-60"}
    loader = DataLoader(DS(), batch_size=2)
    model = build_model(14, pretrained=False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    loss = train_one_epoch(model, loader, opt, scaler, torch.device("cpu"))
    assert loss > 0
