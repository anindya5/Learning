from src.data_processing import get_dataloaders

def test_dataloader_shape():
    loader = get_dataloaders("data/dataset.csv", "bert-base-uncased", 128, 2)
    batch = next(iter(loader))
    assert batch['input_ids'].shape[0] == 2
    assert batch['input_ids'].shape[1] == 128