import logging
import os


class Logger:

    def __init__(self, log_path, mode='w'):
        self.log_path = log_path

        self.logger = logging.getLogger(log_path)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.FileHandler(log_path, mode=mode)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            self.logger.addHandler(handler)

    def logs(self, about, epoch, train_loss, val_loss, cfg=None):
        if epoch == 0 and cfg is not None:
            cfg_msg = " | ".join(f"{k}:{v}" for k, v in cfg.items())
            self.logger.info(f"[config] {cfg_msg}")

        msg = f"[{about}] Epoch {epoch:04d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        self.logger.info(msg)

    def log_test(self, cfg, fold, acc, precision, recall, f1, auc, ap, fpr, fnr, tpr, tnr):
        # Separador visual inicial
        msg = "\n" + "-"*30 + f" TEST RESULTS FOLD {fold} " + "-"*30 + "\n"
        
        # Configuración formateada
        for k, v in cfg.items():
            msg += f"{k}: {v}\n"
        
        msg += "\n--- Metrics ---\n"
        msg += (
            f"Acc: {acc:.4f}  | Prec: {precision:.4f} | Rec: {recall:.4f} | F1: {f1:.4f}\n"
            f"AUC: {auc:.4f}  | AP: {ap:.4f}\n"
            f"FPR: {fpr:.4f}  | FNR: {fnr:.4f}   | TPR: {tpr:.4f} | TNR: {tnr:.4f}\n"
        )
        msg += "-"*80
        
        self.logger.info(msg)

    def log_summary(self,pt_val_loss,t_val_loss ,mean_acc, std_acc):
        self.logger.info(f"[summary] Mean Acc:{mean_acc:.4f} | Std Acc:{std_acc:.4f} | Pretraining val loss: {pt_val_loss:.4f} | Fine tuning val loss:{t_val_loss:.4f}")