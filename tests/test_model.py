import torch 
from workspace.models.model import myawesomemodel
import pytest



def test_model():
    model = myawesomemodel
    assert model(torch.rand(1, 1, 28, 28)).shape == (1, 10) # Wrong shape 

def test_error_on_wrong_shape():
    model = myawesomemodel
    with pytest.raises(ValueError, match=r'Expected each sample to have shape \[1, 28, 28\]'):
        model(torch.randn(1,2,3))