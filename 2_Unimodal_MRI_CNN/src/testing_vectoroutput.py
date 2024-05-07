import torch
import time
import numpy as np

import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from torchmetrics.classification import BinaryConfusionMatrix
from src.utils import AverageMeter, calculate_accuracy

def test_epoch_vectoroutput(data_loader,
              model,
              criterion,
              device,
              logger=None,
              confusion_logger=None,
              early_stopper=None,
              tb_writer=None,
              distributed=False):

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    end_time = time.time()
    bcm = BinaryConfusionMatrix().to(device)

    true_labels = []
    predicted_probs = []
    predicted_probs_softmaxed = []
    feature_vectors = []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)[0]
            featurevector = model(inputs)[1]
            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

            _, pred = outputs.topk(1, 1, largest=True, sorted=True)
            pred = pred.t()
            predicted_labels = pred.view(-1)
            predicted_prob_softmaxed = F.softmax(outputs, dim=1)

            bcm(predicted_labels, targets)

            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))

            true_labels.extend(targets.cpu().numpy())
            predicted_probs.extend(outputs[:, 1].cpu().numpy())
            predicted_probs_softmaxed.extend(predicted_prob_softmaxed.cpu().numpy())
            feature_vectors.extend(featurevector.cpu().numpy())

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      acc=accuracies))

    fcm = bcm.compute()

    # We found that a NaN probability(loss) value pops up once in a blue moon; we convert those NaNs into 0
    predicted_probs = np.nan_to_num(predicted_probs)

    # Calculate ROC curve and AUC using scikit-learn functions
    fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(true_labels, predicted_probs)
    roc_prc = auc(recall, precision)

    if logger is not None:
        logger.log({'loss': losses.avg, 'acc': accuracies.avg, 'roc_auc': roc_auc, 'roc_prc' : roc_prc})

    if confusion_logger is not None:
        confusion_logger.log({'TN': fcm[0][0].item(), 'FP': fcm[0][1].item(),
                              'FN': fcm[1][0].item(), 'TP': fcm[1][1].item()})
    
    if early_stopper is not None:
        early_stopper(roc_auc)

    return feature_vectors