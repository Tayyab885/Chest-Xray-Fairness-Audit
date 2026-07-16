from src.config import load_config

def test_load_config_has_14_labels():
    cfg = load_config("configs/default.yaml")
    assert len(cfg["labels"]) == 14
    assert cfg["batch_size"] == 16
    assert set(cfg["deep_dive_labels"]).issubset(set(cfg["labels"]))
