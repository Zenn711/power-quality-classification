import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, Reshape, Multiply
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import Sequence

# Augmentasi Data
def frequency_masking(stft, F=10, mF=2):
    freq_dim = stft.shape[0]
    for _ in range(mF):
        f = np.random.randint(0, freq_dim - F)
        stft[f:f+F, :, :] = 0  # Bisa diganti dengan nilai rata-rata
    return stft

def time_masking(stft, T=10, mT=2):
    time_dim = stft.shape[1]
    for _ in range(mT):
        t = np.random.randint(0, time_dim - T)
        stft[:, t:t+T, :] = 0
    return stft

def apply_spec_augment(stft):
    stft = frequency_masking(stft)
    stft = time_masking(stft)
    return stft

# Custom Data Generator untuk Augmentasi
class AugmentedDataGenerator(Sequence):
    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.indices = np.arange(len(X))

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_X = self.X[batch_indices]
        batch_y = self.y[batch_indices]
        augmented_X = np.array([apply_spec_augment(x.copy()) for x in batch_X])
        return augmented_X, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# Squeeze-and-Excitation Block
def se_block(input_tensor, ratio=16):
    channels = input_tensor.shape[-1]
    se = GlobalAveragePooling2D()(input_tensor)
    se = Dense(channels // ratio, activation='relu')(se)
    se = Dense(channels, activation='sigmoid')(se)
    se = Reshape((1,1,channels))(se)
    return Multiply()([input_tensor, se])

# Muat dataset
data = sio.loadmat('DataSet_STFT.mat')
print("Kunci dalam file .mat:", data.keys())

# Asumsikan 'X' adalah data STFT [samples, freq, time], 'y' adalah label [samples,]
X = data['X']
y = data['y'].flatten()

# Tambahkan dimensi kanal
X = X[..., np.newaxis]  # X menjadi [samples, freq, time, 1]

# Cetak informasi dataset
print("Bentuk data X:", X.shape)
print("Label unik:", np.unique(y))

# Periksa distribusi kelas
unique, counts = np.unique(y, return_counts=True)
print("Distribusi kelas:", dict(zip(unique, counts)))

# Bagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Konversi label ke format kategorikal
num_classes = 7  # 0: Normal, 1: Sag, 2: Swell, 3: Harmonics, 4: Transient, 5: Notch, 6: Interruption
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# Hitung class weights untuk menangani ketidakseimbangan kelas
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

# Definisikan model CNN dengan SE blocks
input_shape = X_train.shape[1:]  # (freq, time, 1)
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
    se_block,
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    se_block,
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Kompilasi model dengan metrik tambahan
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'Precision', 'Recall'])

# Buat generator untuk data pelatihan
batch_size = 20
train_generator = AugmentedDataGenerator(X_train, y_train_cat, batch_size)

# Definisikan early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Latih model
epochs = 50
history = model.fit(train_generator, epochs=epochs, validation_data=(X_test, y_test_cat), 
                   class_weight=class_weights_dict, callbacks=[early_stopping])

# Plot riwayat pelatihan
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Akurasi Pelatihan')
plt.plot(history.history['val_accuracy'], label='Akurasi Validasi')
plt.title('Akurasi Model')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Loss Pelatihan')
plt.plot(history.history['val_loss'], label='Loss Validasi')
plt.title('Loss Model')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluasi pada set pengujian
test_loss, test_acc, test_precision, test_recall = model.evaluate(X_test, y_test_cat)
print(f'Akurasi Pengujian: {test_acc:.4f}')
print(f'Precision Pengujian: {test_precision:.4f}')
print(f'Recall Pengujian: {test_recall:.4f}')

# Prediksi pada set pengujian
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Matriks konfusi
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.title('Matriks Konfusi')
plt.show()

# Laporan klasifikasi
print("Laporan Klasifikasi:")
print(classification_report(y_test, y_pred_classes, target_names=['Normal', 'Sag', 'Swell', 'Harmonics', 'Transient', 'Notch', 'Interruption']))
