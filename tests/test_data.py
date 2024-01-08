import torch 
import os
import pytest

# Constants 
N_TRAIN_IMAGES = 30000
N_TEST_IMAGES = 5000

# PATH 
DATA_PATH = "data/processed/"
TRAIN_IMAGES = "train_images.pt"
TEST_IMAGES = "test_images.pt"
TRAIN_TARGETS = "train_target.pt"
TEST_TARGETS = "test_target.pt"

@pytest.mark.skipif(not os.path.exists(DATA_PATH + TRAIN_IMAGES), reason="Data files not found")
@pytest.mark.skipif(not os.path.exists(DATA_PATH + TRAIN_TARGETS), reason="Data files not found")
@pytest.mark.skipif(not os.path.exists(DATA_PATH + TEST_IMAGES), reason="Data files not found")
@pytest.mark.skipif(not os.path.exists(DATA_PATH + TEST_TARGETS), reason="Data files not found")
def test_data():
    # Load data 
    train_images = torch.load(DATA_PATH + TRAIN_IMAGES)
    train_targets = torch.load(DATA_PATH + TRAIN_TARGETS)
    test_images = torch.load(DATA_PATH + TEST_IMAGES)
    test_targets = torch.load(DATA_PATH + TEST_TARGETS)
    
    # Check that data has correct ammount of images 
    assert train_images.shape[0] == N_TRAIN_IMAGES and test_images.shape[0] == N_TEST_IMAGES, "Wrong number of images"
    
    # Check that all images are the correct size 
    assert not sum([dims[0] != dims[1] for img in train_images for dims in zip(img.shape, [1, 28, 28])])
    assert not sum([dims[0] != dims[1] for img in test_images for dims in zip(img.shape, [1, 28, 28])])
    
    # Check that all labels are represented 
    assert set(train_targets.unique().tolist()) == set(range(10))
    assert set(test_targets.unique().tolist()) == set(range(10))
    