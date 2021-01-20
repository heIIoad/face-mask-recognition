# face-mask-recognition

## Cel
Stworzenie programu umożliwiającego rozpoznanie, czy dana osoba ma założoną maseczkę ochronną, czy nie. 
Weryfikacja wyników odbywać się będzie na zasadzie widoku z kamery internetowej.

## Narzędzia
W celu realizacji zadania posłużyliśmy się następującymi bibliotekami: 
- OpenCV – zapewnia ona funkcje umożliwiające obróbkę dostarczonego przez użytkownika obrazu,
- TensorFlow – biblioteka wykorzystywana w uczeniu maszynowym i głębokich sieciach neuronowych
- Keras – używa się jej przy sieciach neuronowych. Współpracuję z biblioteką TensorFlow. Zaprojektowana z myśląc o szybkim eksperymentowaniu z głęboką siecią neuronową
- Face_recognition – zapewnia ona narzędzia umożliwiające wykrycie twarzy.

## Opis Działania
Proces zaczyna się od pobrania obrazu dostarczonego przez
użytkownika, który jest odpowiednio przetwarzany za pomocą
funkcji biblioteki OpenCV. Następnie dzięki funkcją, w które
zaopatruje nas biblioteka face_recognition wykrywana jest twarz.
Obiekt zostaje skalowany do rozmiaru 150px x 150px i trafia
bezpośrednio do wcześniej przeszkolonej sieci neuronowej CNN.
Jako efekt finalny otrzymujemy prawdopodobieństwo tego, że
osoba na zdjęciu ( klatce ) ma założoną maseczkę ochronną.
