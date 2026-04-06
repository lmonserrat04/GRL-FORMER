import copy

class EarlyStopping():
    def __init__(self, model, patience, min_delta=0.01):
        self.best_model = copy.deepcopy(model)
        self.patience = patience
        self.counter = 0
        self.min_delta = min_delta
        self.min_val_loss = float('inf')

    def __call__(self, model, avg_val_loss):
        delta = self.min_val_loss - avg_val_loss

        if delta >= self.min_delta:
           
            self.counter = 0
            self.min_val_loss = avg_val_loss          
            self.best_model = copy.deepcopy(model)   
        else:
            
            self.counter += 1
            if self.counter >= self.patience:         
                return self.best_model                

        return None  