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

## Zbiór Danych
Zbiór danych, które użyliśmy do tego aby przeszkolić nasza sieć
neuronową jest w postaci zdjęć udostępnionych na GitHubie przez
Prajna Bhandary pod linkiem:
https://github.com/prajnasb/observations/tree/master/experiements/data .
Zdjęcia te są różnych rozmiarów, orientacji, składają się z różnych
barw. Ich różnorodność zapewnia lepsze rezultaty. Na tym etapie
zdjęcia przetwarzane są na zdjęcia w skali szarości. Następnie
skaluję się je do rozmiaru 150px x 150px. W ten sposób
przygotowane materiały posłużą do trenowania sieci neuronowej.

## Sieć Neuronowa
Nasza sieć neuronowa składa się z 3 konwolucyjnych warstw.
Wcześniej przygotowane zbiory obrazów zostaną teraz wykorzystane.

Najlepsze rezultaty otrzymaliśmy dla liczby epok równej 16 ( validation
accuracy = 0.9677, training accuracy = 0.9964 ). Im większa jest
liczba epok tym bardziej prawdopodobne, że dojdzie do zjawiska
nadmiernego dopasowania ( overfitting )

## Proces Odróżniania
Stworzony przez nas model powinien zostać załadowany.
Następnie ustawiamy domyślną kamerę. Oznaczamy dwa
prawdopodobne wyniki, które możemy otrzymać ( 0 - dla obiektu z
maską i 1 - dla obiektu bez maski ) i przygotowujemy dla nich
interpretację graficzną korzystając z odpowiedniej funkcji biblioteki
OpenCV - dobieramy kolory ramek identyfikujących twarz. W
naszym przypadku kolor zielony oznacza, że maseczka ochronna
została wykryta, natomiast kolorem fioletowym oznaczony został
drugi przypadek.

## Wyniki
Podczas testowania zostały użyte 4 rodzaje maseczek ochronnych:
- Biała
- Niebieska
- Szara w białe kropki
- Czarna

Dla pierwszych 3 otrzymaliśmy wyniki spełniające nasze
oczekiwania. Natomiast jeżeli chodzi o czarną maseczkę ochronną
funkcje biblioteki face_recognition nie dawały rady z rozpoznaniem
twarzy. Badaliśmy także wpływ oświetlenia, w przypadku
pomarańczowego światła padającego na twarz nie było zbytniej
różnicy w działaniu programu. Radził sobie tak samo dobrze jak w
przypadku światła dziennego.
