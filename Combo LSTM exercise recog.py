import os

import numpy as np
from keras import Model, regularizers
from keras.callbacks import TensorBoard, EarlyStopping
from keras.layers import LSTM, Dense, Dropout, Concatenate
from keras.models import Sequential
from keras.models import load_model
from keras.utils import plot_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from Dataset import Dataset

windowsSize = 15
df = Dataset()

Xtrain, Xtest, Ytrain, Ytest = df.makeTrainTestForDL(windowsSize)

Xtrain_balanced_keypoints = Xtrain[:, :, 0:51]
Xtrain_balanced_angles = Xtrain[:, :, 51:]

Xtest_keypoints = Xtest[:, :, 0:51]
Xtest_angles = Xtest[:, :, 51:]

precisions = []
precisions_path = os.path.join('Precisions', 'COMBO LSTM')
if not os.path.exists(precisions_path):
    os.makedirs(precisions_path)

epochs = 300
batchSize = 128
print(f'Epochs: {epochs}, Batch Size: {batchSize}')
model_name = f'Combo_LSTM_ExerciseRecog_{batchSize}_{epochs}.h5'
logDir = os.path.join('Logs', 'COMBO LSTM', model_name)
model_path = os.path.join('Models', 'COMBO LSTM', model_name)
metrics_path = os.path.join('Metrics', 'COMBO LSTM', model_name.replace('.h5', ''))

if os.path.exists(model_path):
    model = load_model(model_path)
else:
    tb_callback = TensorBoard(log_dir=logDir)

    # Definizione dell'architettura della rete neurale LSTM
    input_shape_keypoints = (windowsSize, 51)
    input_shape_angles = (windowsSize, 16)

    # Crea i modelli per i keypoints e gli angoli
    model_keypoints = Sequential()
    model_keypoints.add(LSTM(64, input_shape=input_shape_keypoints, kernel_regularizer=regularizers.l2(0.01)))
    model_keypoints.add(Dropout(0.2))

    model_angles = Sequential()
    model_angles.add(LSTM(64, input_shape=input_shape_angles, kernel_regularizer=regularizers.l2(0.01)))
    model_angles.add(Dropout(0.2))

    # Combina le uscite dei modelli keypoints e angles
    combined = Concatenate()([model_keypoints.output, model_angles.output])

    # Aggiungi il layer di output finale
    x = Dense(len(df.exerciseList), activation='softmax')(combined)

    # Crea il modello finale
    model = Model(inputs=[model_keypoints.input, model_angles.input], outputs=x)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='loss', patience=10)  # Si ferma dopo 10 epoche senza miglioramenti
    model.fit([Xtrain_balanced_keypoints, Xtrain_balanced_angles], Ytrain, batch_size=batchSize, epochs=epochs, callbacks=[tb_callback, early_stopping])
    model.save(model_path)
    print('Model saved:', model_path)

YtestPredict = model.predict([Xtest_keypoints, Xtest_angles])
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
plot_model(model, to_file=os.path.join(metrics_path, 'combo_lstm_model.png'), show_shapes=True, show_layer_names=True)


