import torch

class EarlyStopping:
    def __init__(self, model, metric_name, epoch_to_start_ES, patience, path, min_delta=0.05):
        self.model = model
        self.metric_name = metric_name
        self.patience = patience
        self.startEvalOn = epoch_to_start_ES
        self.path = path
        self.min_delta = min_delta
        self.counter = 0
        self.best_metric = float('-inf')
        self.early_stop = False

    def __call__(self, cur_epoch, current_metric):
        improvement = current_metric - self.best_metric
        if improvement > self.min_delta:
            if cur_epoch > int(self.startEvalOn / 2) :
                self.best_metric = current_metric
                self.save_checkpoint()
                self.counter = 0
                print(self.metric_name, 'increased to', current_metric)
            else :
                print(' > Too early for evaluation, evaluation starts at epoch', int(self.startEvalOn / 2))
        else:
            if(cur_epoch > self.startEvalOn):
                self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            print(self.metric_name, 'did not improve significantly, patience counter is now', self.counter)

    def save_checkpoint(self):
        torch.save(self.model.state_dict(), self.path)
        print(f' > Model saved at {self.path}')

    def should_I_Stop(self):
        return self.early_stop
