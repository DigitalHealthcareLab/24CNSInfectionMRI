import torch
import time
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from torchmetrics.classification import BinaryConfusionMatrix
from src.utils import AverageMeter, calculate_accuracy

def val_epoch(epoch,
              data_loader,
              model,
              criterion,
              device,
              logger,
              confusion_logger=None,
              early_stopper=None,
              earlystop_criterion=None,
              tb_writer=None,
              distributed=False):

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    end_time = time.time()
    bcm = BinaryConfusionMatrix().to(device)

    true_labels = []  # List to store true labels
    predicted_probs = []  # List to store predicted probabilities

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)[0]
            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

            _, pred = outputs.topk(1, 1, largest=True, sorted=True)
            pred = pred.t()
            predicted_labels = pred.view(-1)

            bcm(predicted_labels, targets)

            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))

            # Append true labels and predicted probabilities
            true_labels.extend(targets.cpu().numpy())
            predicted_probs.extend(outputs[:, 1].cpu().numpy())  # Assuming the second column is for positive class

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch,
                      i + 1,
                      len(data_loader),
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
        logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg, 'roc_auc': roc_auc, 'roc_prc' : roc_prc})

    if confusion_logger is not None:
        confusion_logger.log({'epoch': epoch, 'TN': fcm[0][0].item(), 'FP': fcm[0][1].item(),
                              'FN': fcm[1][0].item(), 'TP': fcm[1][1].item()})

    if tb_writer is not None:
        tb_writer.add_scalar('val/loss', losses.avg, epoch)
        tb_writer.add_scalar('val/acc', accuracies.avg, epoch)
        tb_writer.add_scalar('val/roc_auc', roc_auc, epoch)
    
    if early_stopper is not None:
        if earlystop_criterion == 'AUROC':
            early_stopper(epoch, roc_auc)
        elif earlystop_criterion == 'AUPAC':
            early_stopper(epoch, roc_prc)

    return losses.avg