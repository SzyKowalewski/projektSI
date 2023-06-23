import numpy as np
import tensorflow as tf

# Krok 1: Zbieranie danych

def zaladuj_historie(file_path):
    historia_zakupow = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            kolumna = line.split(' ')
            klient = kolumna[0]
            produkty = kolumna[1:]
            historia_zakupow[klient] = produkty
    return historia_zakupow

def zaladuj_produkty(file_path):
    produkt_info = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            kolumna = line.split(' ')
            produkt = kolumna[0]
            cechy = [int(cecha) for cecha in kolumna[1:]]
            produkt_info[produkt] = cechy
    return produkt_info


historia_zakupow = zaladuj_historie('historia_zakupow.txt')
#print(historia_zakupow)
produkt_info = zaladuj_produkty('produkt_info.txt')
#print(product_info)
# Krok 2: Przygotowanie danych

# Przyk³adowe kodowanie one-hot dla kategorii produktów
all_products = list(produkt_info.keys())
product_encoder = {product: i for i, product in enumerate(all_products)}

# Przyk³ad kodowania historii zakupów klienta jako wektora one-hot
def encode_history(history):
    encoded_history = np.zeros(len(all_products))
    for product in history:
        encoded_history[product_encoder[product]] = 1
    return encoded_history

# Przygotowanie danych treningowych
X_train = []
y_train = []
for customer, history in historia_zakupow.items():
    encoded_history = encode_history(history)
    for product in history:
        X_train.append(encoded_history)
        y_train.append(produkt_info[product])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Krok 3: Budowa architektury sieci neuronowej

input_dim = len(all_products)
output_dim = len(produkt_info[all_products[0]])

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu', input_dim=input_dim))
model.add(tf.keras.layers.Dense(output_dim, activation='sigmoid'))

# Krok 4: Uczenie sieci neuronowej

model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Krok 5: Ocena modelu

# Wyœwietlanie dostêpnych produktów
print("Lista produktów:")
for produkt in produkt_info:
    print(produkt)

# Wprowadzenie danych testowych
customer_test = input("Podaj nazwê klienta: ")
products_to_buy = input("Podaj produkty, które chcesz kupiæ (oddzielone spacjami): ").split()

# Tworzenie historii zakupów klienta
test_history = historia_zakupow.get(customer_test, [])
test_history.extend(products_to_buy)

# Kodowanie historii zakupów jako wektora one-hot
encoded_test_history = encode_history(test_history)
X_test = np.array([encoded_test_history])

# Generowanie rekomendacji dla u¿ytkownika
predictions = model.predict(X_test)[0]
recommended_products = [product for product, pred in zip(all_products, predictions) if pred >= 0.5]

print("Rekomendowane produkty dla {}: {}".format(customer_test, recommended_products))
