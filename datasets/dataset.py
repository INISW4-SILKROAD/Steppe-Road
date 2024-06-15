import json
from tqdm import tqdm
from torch.utils.data import TensorDataset
import torch

from imagebind.data import load_and_transform_vision_data

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_dataset(train_path, test_path, embed_path):
    train_data, test_data = get_data(train_path, test_path, embed_path)
        
    train_dataset = TensorDataset(
        torch.stack(train_data['img']), 
        torch.Tensor(train_data['portion']), 
        torch.stack(train_data['label']))    
    
    test_dataset = TensorDataset(
        torch.stack(test_data['img']), 
        torch.Tensor(test_data['portion']), 
        torch.stack(test_data['label']))    
    
    return train_dataset, test_dataset

def get_data(train_path, test_path, embed_path):
    img_embed = torch.load(embed_path)

    with open(train_path, 'r') as f: 
        train_json = json.load(f)

    with open(test_path, 'r') as f: 
        test_json = json.load(f)
    
    train_data = {
        "img" : [img_embed[j] for i in train_json for j in i['img']],
        "portion": [i['portion'] for i in train_json for j in i['img']],
        "label" :  [get_vector(i['label']) for i in train_json for j in i['img']]
    }
        
    test_data = {
        "img" :[img_embed[i['img']] for i in test_json],   
        "portion": [i['portion'] for i in test_json],
        "label" :  [get_vector(i['label']) for i in test_json]
    }
    
    return train_data, test_data

def get_vector(label):
    a = [0, 0, 0, 0]
    a.insert(label-1, 1)
    return torch.Tensor(a)