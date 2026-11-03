# -*- coding: utf-8 -*-
"""统一 I/O 工具 — 日志、格式化、持久化"""

import os, json, pickle
from datetime import datetime
import numpy as np

# 指标名称映射


_METRIC_NAMES = {
    'auc': 'AUC', 'aupr': 'AUPR', 'acc': 'Acc',
    'f1': 'F1', 'prec': 'Prec', 'rec': 'Rec',
    'sen': 'Sen', 'spe': 'Spe',
}

# Pickle / 路径


def pickle_load(filename: str):
    try:
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        print(f'[加载] {filename}')
    except EOFError:
        print(f'[错误] 无法加载: {filename}')
        obj = None
    return obj

def pickle_dump(filename: str, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
    print(f'[保存] {filename}')

def format_filename(_dir: str, filename_template: str, **kwargs):
    return os.path.join(_dir, filename_template.format(**kwargs))

# JSONL 日志


def write_log(filename: str, log, mode='a'):
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    with open(filename, mode, encoding='utf-8') as f:
        f.write(json.dumps(log, ensure_ascii=False) + '\n')

def write_jsonl(filename: str, record: dict):
    write_log(filename, record, mode='a')

# 指标格式化


def _metric_label(key: str) -> str:
    return _METRIC_NAMES.get(key, key)

def format_metric_row(metrics: dict, keys: list = None, fmt: str = '.4f',
                       separator: str = ' │ ') -> str:
    if keys is None:
        keys = [k for k in metrics if not k.startswith('_')]
    parts = []
    for k in keys:
        if k not in metrics:
            continue
        v = metrics[k]
        label = _metric_label(k)
        parts.append(f'{label}={v:{fmt}}' if isinstance(v, float) else f'{label}={v}')
    return separator.join(parts)

def format_metric_table(rows: list, row_labels: list = None,
                         keys: list = None, fmt: str = '.4f') -> str:
    if not rows:
        return '(空)'
    if keys is None:
        keys = [k for k in rows[0] if not k.startswith('_')]
    if row_labels is None:
        row_labels = [f'第{i+1}折' for i in range(len(rows))]

    header_labels = [_METRIC_NAMES.get(k, k) for k in keys]
    col_widths = []
    for ki, k in enumerate(keys):
        w = len(header_labels[ki])
        for row in rows:
            if k in row:
                s = f'{row[k]:{fmt}}' if isinstance(row[k], float) else str(row[k])
                w = max(w, len(s))
        col_widths.append(w)

    label_width = max(max(len(str(l)) for l in row_labels), 6)
    lines = []
    header = '  ' + ' ' * label_width + ' '
    for ki, k in enumerate(keys):
        header += f'{header_labels[ki]:^{col_widths[ki]}}  '
    lines.append(header.rstrip())

    for ri, row in enumerate(rows):
        line = '  ' + f'{row_labels[ri]:<{label_width}} '
        for ki, k in enumerate(keys):
            if k in row:
                v = row[k]
                s = f'{v:{fmt}}' if isinstance(v, float) else str(v)
            else:
                s = '-'
            line += f'{s:^{col_widths[ki]}}  '
        lines.append(line.rstrip())
    return '\n'.join(lines)

# 汇总


def write_final_summary(log_path: str, dataset: str, results: dict):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    summary = {'dataset': dataset, 'timestamp': now}
    for k in ['auc', 'aupr', 'acc', 'f1', 'prec', 'rec']:
        if k in results and results[k]:
            arr = np.array(results[k])
            summary[f'avg_{k}'] = float(np.mean(arr))
            summary[f'{k}_std'] = float(np.std(arr))
            summary[f'{k}_folds'] = [float(v) for v in arr]
    write_jsonl(log_path, summary)

    n_folds = len(results.get('auc', []))
    print(f'\n  ╔{"═"*58}╗')
    print(f'  ║  ★ {dataset} ({n_folds}折)')
    print(f'  ╠{"═"*58}╣')
    for k in ['auc', 'aupr', 'acc', 'f1']:
        if k in summary and f'avg_{k}' in summary:
            avg, std = summary[f'avg_{k}'], summary.get(f'{k}_std', 0)
            bar = '█' * int(round(avg * 20)) + '░' * (20 - int(round(avg * 20)))
            print(f'  ║  {_metric_label(k):<5s}  {avg:.4f} ± {std:.4f}  {bar}')
    print(f'  ╚{"═"*58}╝\n')

def write_fold_progress(log_path: str, fold_idx: int, n_folds: int, train_log: dict):
    record = {
        'fold': fold_idx + 1, 'total_folds': n_folds,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    for k in ['test_auc', 'test_aupr', 'test_acc', 'test_f1',
              'dev_auc', 'dev_aupr', 'train_time', 'dataset']:
        if k in train_log:
            record[k] = train_log[k]
    write_jsonl(log_path, record)
