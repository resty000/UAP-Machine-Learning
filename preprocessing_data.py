import tensorflow as tf # type: ignore
from tensorflow.keras import layers, Model # type: ignore
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample

# # Fungsi untuk membangun Generator
# def build_generator(latent_dim, input_shape):
#     model = tf.keras.Sequential()
#     model.add(layers.Dense(128, input_dim=latent_dim, activation='relu'))
#     model.add(layers.Dense(256, activation='relu'))
#     model.add(layers.Dense(512, activation='relu'))
#     model.add(layers.Dense(np.prod(input_shape), activation='tanh'))
#     model.add(layers.Reshape(input_shape))
#     return model

# # Fungsi untuk membangun Discriminator
# def build_discriminator(input_shape):
#     model = tf.keras.Sequential()
#     model.add(layers.Flatten(input_shape=input_shape))
#     model.add(layers.Dense(512, activation='relu'))
#     model.add(layers.Dense(256, activation='relu'))
#     model.add(layers.Dense(1, activation='sigmoid'))  # Output: 0 or 1 (real or fake)
#     return model

# # Fungsi untuk melatih GAN
# def train_gan(generator, discriminator, gan, X_train, epochs=100, batch_size=32):
#     for epoch in range(epochs):
#         # Pilih batch acak dari data nyata
#         real_data = X_train[np.random.randint(0, X_train.shape[0], batch_size)]

#         # Buat data palsu dengan generator
#         noise = np.random.normal(0, 1, (batch_size, generator.input_shape[1]))
#         fake_data = generator.predict(noise)

#         # Latih Discriminator: data nyata = 1, data palsu = 0
#         d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
#         d_loss_fake = discriminator.train_on_batch(fake_data, np.zeros((batch_size, 1)))
#         d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

#         # Latih Generator (mencoba menipu Discriminator agar menganggap data palsu nyata)
#         noise = np.random.normal(0, 1, (batch_size, generator.input_shape[1]))
#         g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

#         if epoch % 10 == 0:
#             print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

# # Fungsi untuk melakukan augmentasi data tabular menggunakan GAN
# def augment_data_with_gan(X_train, latent_dim=100, epochs=100, batch_size=32):
#     # Tentukan ukuran input dari data
#     input_shape = X_train.shape[1:]

#     # Bangun Generator dan Discriminator
#     generator = build_generator(latent_dim, input_shape)
#     discriminator = build_discriminator(input_shape)

#     # Tentukan Discriminator yang dilatih secara terpisah
#     discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#     # Bangun GAN (model yang menggabungkan Generator dan Discriminator)
#     discriminator.trainable = False
#     gan_input = layers.Input(shape=(latent_dim,))
#     x = generator(gan_input)
#     gan_output = discriminator(x)
#     gan = Model(gan_input, gan_output)
#     gan.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#     # Latih GAN
#     train_gan(generator, discriminator, gan, X_train, epochs, batch_size)

#     # Augmentasi data: menghasilkan data palsu
#     noise = np.random.normal(0, 1, (X_train.shape[0], latent_dim))
#     augmented_data = generator.predict(noise)

#     return augmented_data

# Fungsi untuk preprocessing data
def preprocess_data(data, feature_columns, target_column):
    # Pilih fitur dan target
    X = data[feature_columns].copy()
    y = data[target_column].copy()

    # Isi nilai kosong
    X.fillna("Unknown", inplace=True)

    # Encode kategori
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    # Normalisasi
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

# Fungsi untuk augmentasi data dengan GAN
def augment_data(X, y):
    # Menggunakan GAN untuk augmentasi data tabular
    augmented_X = augment_data_with_gan(X) # type: ignore

    # Gabungkan data asli dan augmented
    X_combined = np.concatenate([X, augmented_X], axis=0)
    y_combined = np.concatenate([y, y], axis=0)  # Label yang sama untuk data augmented

    return X_combined, y_combined
