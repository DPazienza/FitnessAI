import os
import numpy as np
from keras.utils import plot_model
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from Dataset import Dataset, calculateWindows
import joblib

windowsSize = 15
df = Dataset()
listDataset, listDatasetLabels = df.calculateListDataset()
listWindows, listWindowsLabels = calculateWindows(listDataset=listDataset,
                                                  listDatasetLabels=listDatasetLabels,
                                                  windowsSize=windowsSize)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(listWindows,
                                                listWindowsLabels, test_size=0.3,
                                                shuffle=True)

Xtrain = np.array(Xtrain)
Ytrain = np.array(Ytrain)

unique_labels, counts = np.unique(Ytrain, return_counts=True)
print(f'Number of occurrences before balancing')
# Print the occurrence count for each label
for label, count in zip(unique_labels, counts):
    print(f'{label}: {count} occurrences')

# Balancing classes through undersampling
min_samples = min(np.bincount(Ytrain))
Xtrain_balanced = []
Ytrain_balanced = []
for label in np.unique(Ytrain):
    indices = np.where(Ytrain == label)[0]
    indices_balanced = resample(indices, replace=False, n_samples=min_samples, random_state=42)
    Xtrain_balanced.append(Xtrain[indices_balanced])
    Ytrain_balanced.append(Ytrain[indices_balanced])

Xtrain_balanced = np.concatenate(Xtrain_balanced)[::10, :, :]
Ytrain_balanced = np.concatenate(Ytrain_balanced)[::10]
Ytest = np.array(Ytest)[::10]

unique_labels, counts = np.unique(Ytrain_balanced, return_counts=True)

print(f'Number of occurrences after balancing')
# Print the occurrence count for each label
for label, count in zip(unique_labels, counts):
    print(f'{label}: {count} occurrences')

# scaler = StandardScaler()

# Reshape and flatten the training data

Xtrain_balanced_reshaped = Xtrain_balanced.reshape(Xtrain_balanced.shape[0], -1)
Xtrain_balanced_scaled = Xtrain_balanced_reshaped
Xtest = np.array(Xtest)[::10, :, :]

# Reshape and flatten the test data
Xtest_reshaped = (Xtest.reshape(Xtest.shape[0], -1))
Xtest_scaled = Xtest_reshaped

svm_model_path = os.path.join('Models', 'SVM', 'svm_model.pkl')
metrics_path = os.path.join('Metrics', 'SVM', 'svm_model')

print("starting SVM")

if os.path.exists(svm_model_path):
    print("loading model")
    svm_model = joblib.load(svm_model_path)
else:
    print("creating model")
    f = open(svm_model_path, "x")
    svm_model = SVC(kernel='rbf', C=10.0, gamma=0.1)
    svm_model.fit(Xtrain_balanced_scaled, Ytrain_balanced)
    joblib.dump(svm_model, svm_model_path)
    print('SVM model saved:', svm_model_path)

Ytrain_pred = svm_model.predict(Xtrain_balanced_scaled)
Ytest_pred = svm_model.predict(Xtest_scaled)

train_accuracy = accuracy_score(Ytrain_balanced, Ytrain_pred)
accuracy = accuracy_score(Ytest, Ytest_pred)

precision = precision_score(Ytest, Ytest_pred, average='macro', zero_division=0)
recall = recall_score(Ytest, Ytest_pred, average='macro')
f1 = f1_score(Ytest, Ytest_pred, average='macro')
matrix = confusion_matrix(Ytest, Ytest_pred)

os.makedirs(metrics_path, exist_ok=True)
df.saveMetrics(metrics_path, accuracy, precision, recall, f1, matrix, None)
# plot_model(svm_model, to_file=os.path.join(metrics_path, 'SVM_model.png'), show_shapes=True, show_layer_names=True)

