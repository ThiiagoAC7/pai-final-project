import torchvision
import torch
import time
import glob
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import itertools
import numpy as np

from train import create_model, show_info_model, get_transform


weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
VAL_DS = glob.glob('./datasets/dataset_validation/**/*.png')
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_confusion_matrix(cm, classes, title):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def predict_img(img_path, model, transform, class_names):
    img = Image.open(img_path).convert("RGB")

    _classes = {  # usado caso modelo seja binario
        "1": "1_2",
        "2": "1_2",
        "3": "3_4",
        "4": "3_4"
    }

    start = time.time()

    t_img = transform(img).unsqueeze(0)

    model.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(model(t_img), dim=1)

    pred_labels_and_probs = {class_names[i]: float(
        pred_probs[0][i]) for i in range(len(class_names))}
    end = time.time()

    pred_time = end - start

    predicted_label = max(pred_labels_and_probs, key=pred_labels_and_probs.get)

    # pula ./datasets/dataset_validation/
    id = img_path[30:][0]  # retorna somente classe correspondente
    true_label = id[0]
    if len(class_names) == 2:
        true_label = _classes[id[0]]

    print(
        f'({pred_time:.4f}) image -> {img_path[30:]} \n\t predicted_label->{predicted_label}, true_label->{true_label}')

    return predicted_label, true_label, time, pred_labels_and_probs


def predict(img_path, ismanual=False):
    """
    Parameters 
    -------
    ismanual: caso modelo a ser chamado seja 4class_25epochs
    """
    class_names = ["1_2", "3_4"]
    # model_path = './models/model_bin_segmented_25epochs.pth'
    model_path = './models/model_bin_segmented_10epochs.pth'
    if ismanual:
        # model_path = './models/model_4class_segmented_25epochs.pth'
        model_path = './models/model_4class_segmented_10epochs.pth'
        class_names = ["1", "2", "3", "4"]

    print(f'using model -> {model_path}, manual transform ? {str(ismanual)}')
    model = create_model(class_names, weights, gpu=False)
    transform = get_transform(weights)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    return predict_img(img_path, model, transform, class_names)


def classify(ismanual=False):
    """
    Classifica o dataset de Validação
    """

    true_labels = []
    predicted_labels = []

    start = time.time()
    for i in VAL_DS:
        predicted_label, true_label, _, _ = predict(i, ismanual)
        predicted_labels.append(predicted_label)
        true_labels.append(true_label)
    end = time.time()

    labels = ["1_2", "3_4"]
    if ismanual:
        labels = ["1", "2", "3", "4"]

    # metrics
    if ismanual:
        cm = confusion_matrix(true_labels, predicted_labels)
        
        total = np.sum(cm)
        accuracy = np.trace(cm) / total

        specificity = 1 - np.sum(np.diagonal(cm)) / (total - np.sum(np.diagonal(cm)))

        print("Acurácia média", accuracy)
        print("Especificidade", specificity)

        title = f"Acurácia {accuracy:.2f} | Especificidade {specificity:.2f}"
        plot_confusion_matrix(cm, labels, title)
    else:
        tn, fp, fn, tp = confusion_matrix(
            true_labels, predicted_labels, labels=labels).ravel()
        accuracy = (tp + tn) / (tn + fp + fn + tp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
        fone_score = 2 * (precision * recall) / (precision + recall)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "fone_score": fone_score,
            "time": (end - start)
        }

print(classify())
