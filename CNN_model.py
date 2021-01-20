import os
import cv2
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout,Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from matplotlib import pyplot

print("processing images...")
# przypisanie argumentów do podstawowych zmiennych
folder_path = 'zdjecia'
img_size = 150
labels = [0, 1]
data = []
label = []

# wczytanie obrazów bez maski
no_mask_folder_path=os.path.join(folder_path,'bez_maski')
file_names=os.listdir(no_mask_folder_path)

# przyporządkowanie danych obrazów wraz z labelami do nich do odpowiednich tablic
for file_name in file_names:
    # wczytanie ścieżki do obrazu
    img_path = os.path.join(no_mask_folder_path, file_name)
    # odczyt obrazu
    img = cv2.imread(img_path)
    # konwersja obrazu z BGR na obraz w skali szarości
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # zmiania rozmiaru żeby przyśpieszyć działanie CNN
    resized_img = cv2.resize(gray_img,(img_size,img_size))
    # przypisanie danych do tablicy
    data.append(resized_img)
    # przypisanie labeli do tablicy
    label.append(0)


# wczytanie obrazów z maską
mask_folder_path=os.path.join(folder_path,'maska')
file_names=os.listdir(mask_folder_path)

# przyporządkowanie danych obrazów wraz z labelami do nich do odpowiednich tablic
for file_name in file_names:
    # wczytanie ścieżki do obrazu
    img_path = os.path.join(mask_folder_path, file_name)
    # odczyt obrazu
    img = cv2.imread(img_path)
    # konwersja obrazu z BGR na obraz w skali szarości
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # zmiania rozmiaru żeby przyśpieszyć działanie CNN
    resized_img = cv2.resize(gray_img,(img_size,img_size))
    # przypisanie danych do tablicy
    data.append(resized_img)
    # przypisanie labeli do tablicy
    label.append(1)


# przekształcenia tablic do odpowiedniego formatu, który będzie przetwarzany przez CNN
data = np.array(data)/255.0
data = np.reshape(data,(data.shape[0],img_size,img_size,1))
label = np.array(label)
label = np_utils.to_categorical(label)

print("Neural network computation...")

# inicializacja modelu
model = Sequential()

# 1 warstwa
model.add(Conv2D(100,(3,3), activation = 'relu', input_shape = data.shape[1:]))
model.add(MaxPooling2D(pool_size = (2,2), strides = None))

# 2 warstwa
model.add(Conv2D(100,(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2), strides = None))

# 3 warstwa
model.add(Conv2D(100,(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2), strides = None))

# przemiana macierzy cech na vektor
model.add(Flatten())

# usunięcie nadmiernego dopasowywania
model.add(Dropout(0.5))

model.add(Dense(50,activation = 'relu'))
model.add(Dense(2,activation = 'softmax'))

# kompilacja modelu
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# podzielenie obrazów na dane do trenowania i dane do walidacji
train_data,test_data,train_label,test_label = train_test_split(data,label,test_size=0.1)

# zapisanie modelów jako plik
checkpoint = ModelCheckpoint('model-{epoch:03d}.model', monitor = 'val_loss', verbose = 0, save_best_only = True, mode = 'auto')

# nauka CNN
history = model.fit(train_data, train_label, epochs = 20, callbacks = [checkpoint], validation_split = 0.1)

# stworzenie wykresu z precyzją CNN
pyplot.plot(history.history['accuracy'],'r',label='training accuracy')
pyplot.plot(history.history['val_accuracy'],label='validation accuracy')
pyplot.xlabel('# epochs')
pyplot.ylabel('loss')
pyplot.legend()
pyplot.show()