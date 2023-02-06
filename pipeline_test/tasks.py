from prefect import task, flow
import numpy as np
import os
from lib import argObj, ImageDataset, transform_alexnet
from torch.utils.data import DataLoader, Dataset
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from pathlib import Path
import yaml

@task
def create_paths_obj():
    params = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "params.yaml")))["prepare"]

    data_dir=params["data_dir"]
    parent_submission_dir=params["parent_submission_dir"]
    subj=params["subj"]

    args = argObj(data_dir, parent_submission_dir, subj)

    return args

@task
def test_val_train_split(args):
    params = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "params.yaml")))["prepare"]

    rand_seed = params["seed"]
    np.random.seed(rand_seed)

    train_img_dir  = os.path.join(args.data_dir, 'training_split', 'training_images')
    test_img_dir  = os.path.join(args.data_dir, 'test_split', 'test_images')

    # Create lists will all training and test image file names, sorted
    train_img_list = os.listdir(train_img_dir)
    train_img_list.sort()
    test_img_list = os.listdir(test_img_dir)
    test_img_list.sort()
    print('Training images: ' + str(len(train_img_list)))
    print('Test images: ' + str(len(test_img_list)))

    # Calculate how many stimulus images correspond to 90% of the training data
    num_train = int(np.round(len(train_img_list) / 100 * 90))
    # Shuffle all training stimulus images
    idxs = np.arange(len(train_img_list))
    np.random.shuffle(idxs)
    # Assign 90% of the shuffled stimulus images to the training partition,
    # and 10% to the test partition
    idxs_train, idxs_val = idxs[:num_train], idxs[num_train:]
    # No need to shuffle or split the test stimulus images
    idxs_test = np.arange(len(test_img_list))

    print('Training stimulus images: ' + format(len(idxs_train)))
    print('\nValidation stimulus images: ' + format(len(idxs_val)))
    print('\nTest stimulus images: ' + format(len(idxs_test)))

    return idxs_train, idxs_val, idxs_test

def create_data_loaders(args, idxs_train, idxs_val, idxs_test):
    train_img_dir  = os.path.join(args.data_dir, 'training_split', 'training_images')
    test_img_dir  = os.path.join(args.data_dir, 'test_split', 'test_images')

    transform=transform_alexnet()

    batch_size = 300 #@param
    # Get the paths of all image files
    train_imgs_paths = sorted(list(Path(train_img_dir).iterdir()))
    test_imgs_paths = sorted(list(Path(test_img_dir).iterdir()))

    # The DataLoaders contain the ImageDataset class
    train_imgs_dataloader = DataLoader(
        ImageDataset(train_imgs_paths, idxs_train, transform), 
        batch_size=batch_size
    )
    val_imgs_dataloader = DataLoader(
        ImageDataset(train_imgs_paths, idxs_val, transform), 
        batch_size=batch_size
    )
    test_imgs_dataloader = DataLoader(
        ImageDataset(test_imgs_paths, idxs_test, transform), 
        batch_size=batch_size
    )

    return train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader

@flow
def test_flow():
    args=create_paths_obj()
    idxs_train, idxs_val, idxs_test=test_val_train_split(args)


test_flow()
