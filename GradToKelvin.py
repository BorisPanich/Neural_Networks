''' Для игнорирования необходимости установки CUDA (задействуется память процессора) '''
import imp
import os
from tabnanny import verbose

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense
''' Обучающая область '''

c = np.array([-40, -10, 0, 8, 15, 22, 38])
f = np.array([-40, 14, 32, 46, 59, 72, 100])

model = keras.Sequential()

# units - колич. нейронов, input_shape - сколько входов, activation - активац. функция (f(x)=x)
model.add(Dense(units=1, input_shape=(1, ), activation='linear'))
''' '''
# loss - критерий качества (функция потерь), optimizer - оптимизатор с шагом 0,1
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.1))

# с - вх. параметры, f - выходные параметры, epochs - количество обучений
history = model.fit(c, f, epochs=500, verbose=0)
print("Обучение завершено")

print(model.predict([100]))
print(model.get_weights())

plt.plot(history.history['loss'])
plt.grid(True)
plt.show()