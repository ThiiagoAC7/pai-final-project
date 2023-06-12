import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import efficientnet_v2_s
from torch.utils.data.sampler import SubsetRandomSampler
from torchinfo import summary
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import engine
from pathlib import Path

# use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using -> {device}")

# image_size = (384, 384)

def plot_results(hist, epochs_range):
    '''
    Plota um gráfico com os resultados do modelo
    '''
    print('-------Salvando gráfico-------')

    acc = hist['train_acc']
    test_acc = hist['test_acc']
    loss = hist['train_loss']
    test_loss = hist['test_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, test_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, test_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    os.makedirs('_graficos', exist_ok=True)
    plt.savefig('_graficos/test04_25e_4class_l2reg_03dropout.jpg')
    # plt.show()


def dataset_train_test(preprocess):
    """
    Splitando o dataset original em treino e teste
    ---------
    preprocess: transformações feitas pelo modelo original
                ImageClassification(
                    crop_size=[384]
                    resize_size=[384]
                    mean=[0.485, 0.456, 0.406]
                    std=[0.229, 0.224, 0.225]
                    interpolation=InterpolationMode.BILINEAR
                )
    """
    manual_transforms = transforms.Compose([
        transforms.Resize((224, 224)), # usando imagens menores, menor tempo de treino 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])
    _transform = preprocess # usando as mesmas transformações feitas pelo modelo ori
    dataset = ImageFolder('./datasets/dataset_segmented/train/', transform=_transform)
    validation_split = .3
    shuffle = True
    random_seed = 42
    batch_size = 32
    size = len(dataset)
    print(f'Original Dataset size -> {size}, classes -> {dataset.classes}')
    indices = list(range(size))
    split = int(np.floor(validation_split * size))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    print(f'{len(train_indices)} training samples, {len(val_indices)} validation samples')

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

    return train_loader, test_loader, dataset.classes


def show_training_images(loader, classes):
    def imshow(img, title=''):
        img = img / 2 + 0.5     # unnormalize
        print(img.shape)
        npimg = img.numpy()
        plt.title(title)
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    dataiter = iter(loader)
    images, labels = next(dataiter)

    imshow(torchvision.utils.make_grid(images, nrow=16, padding=8), title=[str(classes[labels[j]]) for j in range(32)])


def show_info_model(model):
    """
    Printando informações do modelo
    """
    summary(model=model,
            input_size=(32, 3, 384, 384),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])

def create_model(classes, weights):
    """
    Criando modelo EfficientNetV2-S, retreinando parte fully-conencted
    """
    model = efficientnet_v2_s(weights=weights)

    # freeze pretrained layers
    for param in model.features.parameters():
        param.requires_grad = False

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    output_shape = len(classes)

    # replace fully connected layer
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.3, inplace=True), 
        torch.nn.Linear(in_features=1280, 
                        out_features=output_shape, # same number of output units as number of classes
                        bias=True)).to(device)
    return model


def fit_model(model, train_loader, test_loader, epochs):
    loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001) # l2 regularization
        # Set the random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    start = time.time()
    results = engine.train(model=model,
                           train_dataloader=train_loader,
                           test_dataloader=test_loader,
                           optimizer=optimizer,
                           loss_fn=loss_fn,
                           epochs=epochs,
                           device=device)
    end = time.time()

    print(f"[INFO] Total training time: {end-start:.3f} seconds")
    save_model(model, './models/', f'model_4class_segmented_{epochs}epochs.pth')

    return results


def save_model(model, target_dir, model_name): 
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),f=model_save_path)
    

def main():
    # Total training time: 2300.749 seconds <- 5 epochs, 2 classes
    # Total training time: 2306.996 seconds <- 5 epochs, 4 classes
    # Total training time: 4603.976 seconds <- 10epochs, 4 classes
    # Total training time: 4507.601 seconds <- 10epochs, 2 classes
    # Total training time: 11098.756 seconds <- 25epochs, 4 classes
    weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
    preprocess = weights.transforms()
    train_loader, test_loader, classes = dataset_train_test(preprocess)
    # show_training_images(train_loader, classes)
    model = create_model(classes, weights)
    # show_info_model(model)

    epochs=25
    H = fit_model(model, train_loader, test_loader, epochs)
    plot_results(H, range(epochs))


if __name__ == "__main__":
    main()
