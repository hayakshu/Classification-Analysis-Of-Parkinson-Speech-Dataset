import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, title="Confusion Matrix", label1="Parkinson", label2="No Parkinson", cmap=plt.cm.Blues):
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(2)
	plt.xticks(tick_marks, (label1, label2), rotation=45)
	plt.yticks(tick_marks, (label1, label2))
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.axis('tight')
	plt.show()


