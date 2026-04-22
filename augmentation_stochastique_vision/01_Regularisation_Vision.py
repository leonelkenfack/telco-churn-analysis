import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Silence TF logging
import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import matplotlib.pyplot as plt

print("--- INITIALISATION DU MODULE DE VISION ET D'AUGMENTATION ---")

# 1. Chargement de Fashion MNIST (Vetements)
print("1. Téléchargement et structuration du Dataset (Fashion MNIST)...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalisation et redimensionnement pour le réseau de neurones
x_train = x_train[..., tf.newaxis].astype("float32") / 255.0
x_test = x_test[..., tf.newaxis].astype("float32") / 255.0

# Réduction volontaire du dataset pour "forcer" un apprentissage par cœur immédiat
nb_train = 1000
nb_test = 2000
x_train_sub, y_train_sub = x_train[:nb_train], y_train[:nb_train]
x_test_sub, y_test_sub = x_test[:nb_test], y_test[:nb_test]

print(f"[OK] Sous-ensemble généré : {nb_train} entraînements, {nb_test} tests.")

# 2. Modèle A : DÉTERMINISTE (Aucune Défense)
print("\n2. Entraînement Modèle A (Déterministe - Pas d'augmentation)...")
model_a = models.Sequential([
    layers.InputLayer(shape=(28, 28, 1)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model_a.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# On entraîne pendant 20 époques pour montrer la chute
history_a = model_a.fit(x_train_sub, y_train_sub, epochs=25, validation_data=(x_test_sub, y_test_sub), verbose=0)
print(f"   -> Accuracy de TEST (Déterministe) finale : {history_a.history['val_accuracy'][-1]:.4f}")

# 3. Modèle B : STOCHASTIQUE (Bruit Quantique Intégré)
print("\n3. Entraînement Modèle B (Stochastique - Augmentation aléatoire continue)...")

data_augmentation = tf.keras.Sequential([
  layers.InputLayer(shape=(28, 28, 1)),
  layers.RandomFlip("horizontal"),
  layers.RandomRotation(0.1, fill_mode="constant"),
  layers.RandomZoom(0.1, fill_mode="constant"),
])

model_b = models.Sequential([
    data_augmentation,
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model_b.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_b = model_b.fit(x_train_sub, y_train_sub, epochs=25, validation_data=(x_test_sub, y_test_sub), verbose=0)
print(f"   -> Accuracy de TEST (Stochastique) finale : {history_b.history['val_accuracy'][-1]:.4f}")

# 4. Compilation des actifs analytiques
print("\n4. Génération des preuves pour la soutenance...")
df_epochs = pd.DataFrame({
    "Epoque": range(1, 26),
    "Loss_Train_Det": history_a.history['loss'],
    "Loss_Test_Det": history_a.history['val_loss'],
    "Loss_Train_Sto": history_b.history['loss'],
    "Loss_Test_Sto": history_b.history['val_loss']
})

os.makedirs("presentation_assets", exist_ok=True)
os.makedirs("data", exist_ok=True)
df_epochs.to_csv("data/historique_vision.csv", index=False)

# Export d'une preuve visuelle (Comment l'ordi voit le bruit ?)
plt.figure(figsize=(10, 4))
for i in range(5):
    # Image originale
    ax = plt.subplot(2, 5, i + 1)
    plt.imshow(x_train_sub[i].squeeze(), cmap="gray")
    plt.axis("off")
    if i == 0: plt.title("L'Original")
    
    # Image modifiée par la probabilité (Keras)
    ax = plt.subplot(2, 5, i + 1 + 5)
    augmented_image = data_augmentation(tf.expand_dims(x_train_sub[i], 0))
    plt.imshow(augmented_image[0].numpy().squeeze(), cmap="gray")
    plt.axis("off")
    if i == 0: plt.title("La version Stochastique")

plt.tight_layout()
plt.savefig("presentation_assets/vision_metrics.png", dpi=300)
plt.close()

print("\n[SUCCES] Scénario Deep Learning Vision accompli !")
