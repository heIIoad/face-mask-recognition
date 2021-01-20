import cv2
import numpy as np
import face_recognition
from keras.models import load_model

# wczytanie CNN
model = load_model('model-016.model')

# rozmiar obrazu trafiającego do CNN
img_size = 150

# wybranie podstawowej kamerki
video_capture = cv2.VideoCapture(0)

while(True):
    # czytanie obrazu z kamerki
    ret, frame = video_capture.read()
    # zmniejszenie otrzymanego obrazu dla szybszego działania
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # przekształcenie BGR na RGB
    rgb_small_frame = small_frame[:, :, ::-1]
    # wykrywanie twarzy
    face_locations = face_recognition.face_locations(rgb_small_frame)
    
    for(top,right,bottom,left) in face_locations:
        # reskalowanie obrazu do jego początkowego rozmiaru
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        # konwersja obrazu z BGR na obraz w skali szarości
        gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # wycięcie twarzy
        face_img = gray_img[top:bottom,left:right]
        # zmiana rozmiaru
        resized_img = cv2.resize(face_img,(img_size,img_size))
        # przekształcenie danych do odpowiedniego formatu
        normalized = resized_img/255.0
        reshaped_img = np.reshape(normalized,(1,img_size,img_size,1))
        # wysłanie danych do CNN
        result = model.predict(reshaped_img)
        # przypisanie wyniku do etykiety
        label = np.argmax(result,axis = 1)[0]
        
        # przypisanie nazwy i koloru do narysowania prostokąta
        if label == 0:
            name = 'bez maski'
            color = (230,100,230)
        elif label == 1:
            name = 'z maska'
            color = (110,170,110)
        # narysowanie prostokąta wokół twarzy
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        # dodanie etykiety
        cv2.rectangle(frame, (left, bottom - 40), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)

    # wyświetlenie obrazu z kamerki
    cv2.imshow('Video', frame)
    
    # wyłączenie programu poprzez naciśnięcie q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# zwolnienie kamerki
video_capture.release()

# zamknięcie okna
cv2.destroyAllWindows()