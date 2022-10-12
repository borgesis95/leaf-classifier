import torch
import numpy as np


def test_classifier(model,loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    predictions = []
    labels = []

    for batch in loader:
        x = batch[0].to(device)
        #etichette
        y = batch[1].to(device)
        output = model(x)


        #prediction
        preds = output.to('cpu').max(1)[1].numpy()
        #etichetta
        label = y.to('cpu').numpy()
        predictions.extend(list(preds))
        labels.extend(list(label))

    return np.array(predictions) , np.array(labels)
