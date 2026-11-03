# -*- coding: utf-8 -*-
"""
训练回调 — 每 epoch 验证指标监控 + 日志记录。

改进:
  1. 表格化 epoch 输出 (指标对齐, 一目了然)
  2. 显示更多指标: loss / lr / contrast_loss / flow_loss (如果有)
  3. 训练历史写入 JSONL (便于后续分析)
  4. 清理死代码
"""

from datetime import datetime
import numpy as np
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve
import sklearn.metrics as m
from ..utils.io import write_jsonl


class KGCNMetric(Callback):
    """每 epoch 结束后在验证集上计算 AUC/AUPR/Acc/F1 并打印对齐表格。"""

    def __init__(self, x_train, y_train, x_val, y_val, dataset):
        self.x_train = x_train
        self.y_val = y_val
        self.y_train = y_train
        self.x_val = x_val
        self.dataset = dataset
        self.threshold = 0.5
        self._epoch = 0
        self._best = {'auc': 0.0, 'aupr': 0.0, 'epoch': 0}
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        self._epoch = epoch + 1
        logs = logs or {}

        # 验证集指标


        y_pred = self.model.predict(self.x_val, verbose=0).flatten()
        y_true = self.y_val.flatten()

        auc = roc_auc_score(y_true=y_true, y_score=y_pred)
        precision, recall, _ = precision_recall_curve(y_true=y_true, y_score=y_pred)
        aupr = m.auc(recall, precision)
        y_pred_bin = (y_pred >= self.threshold).astype(int)
        acc = accuracy_score(y_true=y_true, y_pred=y_pred_bin)
        f1 = f1_score(y_true=y_true, y_pred=y_pred_bin)

        logs['val_auc'] = float(auc)
        logs['val_aupr'] = float(aupr)
        logs['val_acc'] = float(acc)
        logs['val_f1'] = float(f1)

        # 追踪最佳


        if auc > self._best['auc']:
            self._best = {'auc': auc, 'aupr': aupr, 'epoch': self._epoch}

        # 终端打印 (对齐)


        ts = datetime.now().strftime('%H:%M:%S')
        # 主要指标行


        main_metrics = (
            f'auc={auc:.4f}  aupr={aupr:.4f}  acc={acc:.4f}  f1={f1:.4f}'
        )
        # 辅助指标 (如果有)


        aux_parts = []
        if 'contrast_loss' in logs:
            aux_parts.append(f'contrast={logs["contrast_loss"]:.4f}')
        if 'flow_loss' in logs:
            aux_parts.append(f'flow={logs["flow_loss"]:.4f}')
        if 'lr' in logs:
            aux_parts.append(f'lr={logs["lr"]:.2e}')
        aux_str = '  ' + '  '.join(aux_parts) if aux_parts else ''

        # 最佳标记


        best_mark = ' ★' if auc >= self._best['auc'] else ''

        print(f'  [{ts}] epoch {self._epoch:>3d}{best_mark} │ {main_metrics}{aux_str}')

        # 写入 JSONL 训练历史


        record = {
            'epoch': self._epoch,
            'timestamp': ts,
            'dataset': self.dataset,
            'val_auc': float(auc),
            'val_aupr': float(aupr),
            'val_acc': float(acc),
            'val_f1': float(f1),
        }
        # 附加 Keras 原生指标


        for k in ['loss', 'val_loss', 'lr', 'contrast_loss', 'flow_loss', 'contrast_temp']:
            if k in logs:
                record[k] = float(logs[k]) if isinstance(logs[k], (int, float, np.floating)) else str(logs[k])
        write_jsonl('log/train_history.jsonl', record)
