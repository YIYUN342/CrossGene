# -*- coding: utf-8 -*-
"""每 epoch 验证指标监控 + 格式化输出"""

from datetime import datetime
import numpy as np
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve
import sklearn.metrics as m
from ..utils.io import write_jsonl


class KGCNMetric(Callback):
    def __init__(self, x_train, y_train, x_val, y_val, dataset):
        self.x_train = x_train
        self.y_val = y_val
        self.y_train = y_train
        self.x_val = x_val
        self.dataset = dataset
        self._epoch = 0
        self._best = {'auc': 0.0, 'epoch': 0}
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        self._epoch = epoch + 1
        logs = logs or {}

        y_pred = self.model.predict(self.x_val, verbose=0).flatten()
        y_true = self.y_val.flatten()

        auc = roc_auc_score(y_true=y_true, y_score=y_pred)
        precision, recall, _ = precision_recall_curve(y_true=y_true, y_score=y_pred)
        aupr = m.auc(recall, precision)
        y_pred_bin = (y_pred >= 0.5).astype(int)
        acc = accuracy_score(y_true=y_true, y_pred=y_pred_bin)
        f1 = f1_score(y_true=y_true, y_pred=y_pred_bin)

        logs['val_auc'] = float(auc)
        logs['val_aupr'] = float(aupr)
        logs['val_acc'] = float(acc)
        logs['val_f1'] = float(f1)

        if auc > self._best['auc']:
            self._best = {'auc': auc, 'aupr': aupr, 'epoch': self._epoch}

        ts = datetime.now().strftime('%H:%M:%S')
        main = f'auc={auc:.4f}  aupr={aupr:.4f}  acc={acc:.4f}  f1={f1:.4f}'
        aux = '  '.join([f'{k}={logs[k]:.4f}' for k in ['contrast_loss', 'flow_loss']
                         if k in logs])
        best_mark = ' ★' if auc >= self._best['auc'] else ''
        print(f'  [{ts}] epoch {self._epoch:>3d}{best_mark} │ {main}  {aux}')

        record = {
            'epoch': self._epoch, 'timestamp': ts, 'dataset': self.dataset,
            'val_auc': float(auc), 'val_aupr': float(aupr),
            'val_acc': float(acc), 'val_f1': float(f1),
        }
        for k in ['loss', 'val_loss', 'lr', 'contrast_loss', 'flow_loss', 'contrast_temp']:
            if k in logs:
                record[k] = float(logs[k]) if isinstance(logs[k], (int, float, np.floating)) else str(logs[k])
        write_jsonl('log/train_history.jsonl', record)
