import torch
import numpy as np
import os
import math
import pickle
import matplotlib.pyplot as plt
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import NDArrayDecoder
from ffcv.transforms import ToTensor, ToDevice
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import DataLoader, TensorDataset
from functions_for_NN import *
from constants import *

def visualize_prediction(prediction, ground_truth):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    # Reshape the prediction and ground truth if necessary
    if prediction.ndim == 1:
        prediction = prediction.reshape((20, 20))  # Adjust the shape as needed
    if ground_truth.ndim == 1:
        ground_truth = ground_truth.reshape((20, 20))  # Adjust the shape as needed

    ax[0].imshow(prediction, cmap='gray')
    ax[0].set_title('Prediction')
    ax[1].imshow(ground_truth, cmap='gray')
    ax[1].set_title('Ground Truth')
    plt.show()

def load_dataset(name,i,device):
    
    name_train = f"dataset_{name}{i}.beton"  # Define the path where the dataset will be written
    complete_path_train = os.path.join(FFCV_DIR, name_train)

    train_loader = Loader(complete_path_train, batch_size=64,
    num_workers=8, order=OrderOption.QUASI_RANDOM,
    os_cache=True,
    pipelines={
        'covariate': [NDArrayDecoder(),    # Decodes raw NumPy arrays                    
                    ToTensor(),          # Converts to PyTorch Tensor (1,400,400)
                    ToDevice(device, non_blocking=True)],
        'label': [NDArrayDecoder(),    # Decodes raw NumPy arrays
                ToTensor(),          # Converts to PyTorch Tensor (1,400,400)
                ToDevice(device, non_blocking=True)]
    })

    return train_loader

if __name__ == '__main__':

    number_of_chucks_testset = 1

    # Load model
    model_path = MODEL_DIR
    model_name = 'model_20250227_123854_loss_0.0253'
    model_name = model_name + '.pth'
    model_path = os.path.join(model_path, model_name)
    model = MapToBBModel()
    model.load_state_dict(torch.load(model_path))
    criterion = HungarianMSELoss()
    model.eval()

    # Check if CUDA is available
    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

    # Storing ID of current CUDA device
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device:{torch.cuda.current_device()}")
        
    print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    summary(model, (1, 400, 400))

    predictions = []
   
    for i in range(number_of_chucks_testset): #type: ignore
        
        print(f"\nChunck number {i+1} of {number_of_chucks_testset}")
        
        test_loader = load_dataset('test', i, device)

        print("\nLenght of the datasets:", len(test_loader))

        grid_maps = []
        vertices = []
        
        # Assuming data is a tuple of 64 elements
        for data in test_loader:
            print("Data structure:", type(data), len(data))
            for idx, d in enumerate(data):
                print(f"Element {idx} shape:", d.shape)

            # Separate covariate and label data
            covariate_data = data[0].cpu().numpy()
            label_data = data[1].cpu().numpy()

            # Stack covariate and label data separately
            grid_maps.append(covariate_data)
            vertices.append(label_data)

            # Now covariate_data and label_data are numpy arrays containing all the elements
            print("Covariate data shape:", covariate_data.shape)
            print("Label data shape:", label_data.shape)
        
        grid_maps = np.concatenate(grid_maps, axis=0)
        vertices = np.concatenate(vertices, axis=0)

        print("Grid Maps Shape:", grid_maps.shape)
        print("Vertices Shape:", vertices.shape)
        
        # Make predictions
        with torch.no_grad():
            for data in test_loader:
                inputs, _ = data
                outputs = model(inputs)
                predictions.append(outputs)
        predictions = torch.cat(predictions).cpu().numpy()
        print("Predictions Shape:", predictions.shape)

    # Concatenate predictions
    #predictions = np.concatenate(predictions, axis=0)


    for i in range(predictions.shape[0]):
        
        pred = predictions[i].squeeze()
        gt = vertices[i].squeeze()
        map = grid_maps[i].squeeze()

        print("\nShapes:", pred.shape, gt.shape, map.shape)

        grid_map_recreate_BB_pred = np.full((Y_RANGE, X_RANGE), 0, dtype=float) # type: ignore
        grid_map_recreate_BB_gt = np.full((Y_RANGE, X_RANGE), 0, dtype=float)
        for k in range(len(pred)):
            vertices_pred = np.array(pred[k])
            vertices_gt = np.array(gt[k])

            print(f"Vertices prediction before mult:\n {vertices_pred}")
            print(f"Vertices ground truth before mult:\n {vertices_gt}")
            vertices_pred = np.array(pred[k]) * 399
            vertices_gt = np.array(gt[k]) * 399

            vertices_pred = vertices_pred.astype(int)
            vertices_gt = vertices_gt.astype(int)
            print(f"Vertices shapes: {vertices_pred.shape} {vertices_gt.shape}")
            print(f"Vertices prediction:\n {vertices_pred}")
            print(f"Vertices ground truth:\n {vertices_gt}")
            height_BB = 1  # Assuming all vertices have the same height
            fill_polygon(grid_map_recreate_BB_pred, vertices_pred, height_BB)
            fill_polygon(grid_map_recreate_BB_gt, vertices_gt, height_BB)
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].imshow(map, cmap='gray', alpha=0.5)
        ax[0].imshow(grid_map_recreate_BB_pred, cmap='jet', alpha=0.5)
        ax[0].set_title('Overlay of Original and Prediction Grid Maps')

        ax[1].imshow(map, cmap='gray', alpha=0.5)
        ax[1].imshow(grid_map_recreate_BB_gt, cmap='jet', alpha=0.5)
        ax[1].set_title('Overlay of Original and Ground Truth Grid Maps')
        plt.show()
