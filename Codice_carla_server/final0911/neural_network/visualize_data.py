
import os
import matplotlib.pyplot as plt
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice
from ffcv.fields.decoders import NDArrayDecoder
import torch
from constants import *
from functions_for_NN import *


def load_dataset(name, i, device):
    name_train = f"dataset_train0.beton"  # Define the path where the dataset will be written
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

def visualize_data(loader):
    for batch in loader:
        images, labels = batch
        images = images.cpu().numpy()
        labels = labels.cpu().numpy()

        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")

        for i in range(min(20, len(images))):  # Visualize first 5 images
            
            
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(images[i].squeeze(), cmap='gray', alpha=0.5)
            ax.imshow(grid_map_recreate_BB, cmap='jet', alpha=0.5)
            ax.set_title('Overlay of Original and Prediction Grid Maps')
            plt.show()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = load_dataset('example', 1, device)
    visualize_data(loader)
