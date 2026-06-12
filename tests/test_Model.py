import pytest
from src.data_loader import load_data

def test_data_loading():
    # Test that data loader raises error on fake path
    with pytest.raises(Exception):
        load_data("non_existent_file.csv")