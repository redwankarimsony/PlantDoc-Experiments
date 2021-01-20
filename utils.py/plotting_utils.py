import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

data = pd.read_csv('../logs/run_log.csv')
f = plt.figure(figsize=(16,8))
f.add_subplot(1,2, 1)
plt.plot(data['train_acc'], marker='*', linewidth=3, label='train_acc')
plt.plot(data['val_acc'], marker='*', linewidth = 3, label='val_acc')
plt.title('Accuracy')
plt.grid()
plt.legend()

f.add_subplot(1,2, 2)
plt.plot(data['train_loss'], marker='o', linewidth=2, label='train_loss')
plt.plot(data['val_loss'], marker='o', linewidth = 2, label='val_loss')
plt.title('Loss')
plt.grid()
plt.legend()
plt.show(block=True)


