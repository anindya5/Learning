from src.utils import load_config

def test_config_load():
    cfg = load_config()
    assert "model_name" in cfg