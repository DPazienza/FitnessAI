import os
from io import StringIO

import numpy as np

from keras import regularizers
from keras.callbacks import TensorBoard, EarlyStopping
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.models import load_model
from keras.utils import plot_model
from keras.utils.layer_utils import print_summary
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from Dataset import Dataset

windowsSize = 15
df = Dataset()

Xtrain, Xtest, Ytrain, Ytest = df.makeTrainTestForDL(windowsSize)

epochs = 300
batchSize = 128
print(f'Epochs: {epochs}, Batch Size: {batchSize}')
model_name = f'LSTM_ExerciseRecog_{batchSize}_{epochs}.h5'
logDir = os.path.join('Logs', 'LSTM', model_name)
model_path = os.path.join('Models', 'LSTM', model_name)
metrics_path = os.path.join('Metrics', 'LSTM', model_name.replace('.h5', ''))

if os.path.exists(model_path):
    model = load_model(model_path)
else:
    tb_callback = TensorBoard(log_dir=logDir)

    # Definizione dell'architettura della rete neurale LSTM
    model = Sequential()
    model.add(LSTM(64, input_shape=(windowsSize, 67), kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(len(df.exerciseList), activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='loss', patience=10)  # Si ferma dopo 5 epoche senza miglioramenti
    model.fit(Xtrain, Ytrain,
              batch_size=batchSize,
              epochs=epochs,
              callbacks=[tb_callback, early_stopping])
    model.save(model_path)
    print('Model saved:', model_path)

YtestPredict = model.predict(Xtest)
exerciseList = np.array(df.exerciseList)
accuracy = accuracy_score(exerciseList[np.argmax(Ytest, axis=1).tolist()],
                          exerciseList[np.argmax(YtestPredict, axis=1).tolist()])

precision = precision_score(exerciseList[np.argmax(Ytest, axis=1).tolist()],
                            exerciseList[np.argmax(YtestPredict, axis=1).tolist()],
                            average='macro', zero_division=0)

recall = recall_score(exerciseList[np.argmax(Ytest, axis=1).tolist()],
                      exerciseList[np.argmax(YtestPredict, axis=1).tolist()],
                      average='macro')

f1 = f1_score(exerciseList[np.argmax(Ytest, axis=1).tolist()],
              exerciseList[np.argmax(YtestPredict, axis=1).tolist()],
              average='macro')

matrix = confusion_matrix(exerciseList[np.argmax(Ytest, axis=1).tolist()],
                          exerciseList[np.argmax(YtestPredict, axis=1).tolist()])


os.makedirs(metrics_path, exist_ok=True)
df.saveMetrics(metrics_path, accuracy, precision, recall, f1, matrix, model)
plot_model(model, to_file=os.path.join(metrics_path, 'lstm_model.png'), show_shapes=True, show_layer_names=True)



