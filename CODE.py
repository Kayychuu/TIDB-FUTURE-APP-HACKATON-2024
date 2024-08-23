import faiss
import numpy as np
import mysql.connector
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import collections

#çConectar a TiDB Serverless para cargar los embeddings
cnx = mysql.connector.connect(
    user='4AfmDMppLajE5rg.root',
    password='NAVUajR9PvVrkdqN',
    host='gateway01.us-east-1.prod.aws.tidbcloud.com',
    database='test'
)
cursor = cnx.cursor()

#Obtener embeddings y clases
cursor.execute("SELECT mushroom_class, embedding FROM mushroom_embeddings")
results = cursor.fetchall()
cursor.close()
cnx.close()

#Reconstruir embeddings y clases
embeddings = np.array([np.frombuffer(result[1], dtype=np.float32) for result in results])
mushroom_classes = [result[0] for result in results]
class_counts = collections.Counter(mushroom_classes)

#Crear índice FAISS
d = embeddings.shape[1]
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, 50, faiss.METRIC_L2)
index.train(embeddings)
index.add(embeddings)
index.nprobe = 5

#Framework MobileNet
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True

model = models.Sequential([ #Modelo ocmpleto
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),  # Reducimos el número de neuronas
    layers.Dropout(0.5),    layers.Dense(len(np.unique(mushroom_classes)), activation='softmax')
])
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy']) #Model Compiler

#Configuración del dataset (Aumento)
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)
train_gen = datagen.flow_from_directory(
    'C:/Users/maria/OneDrive/Documentos/Hackaton Setas/data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
valid_gen = datagen.flow_from_directory(
    'C:/Users/maria/OneDrive/Documentos/Hackaton Setas/data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Entrenamiento
history = model.fit(train_gen, validation_data=valid_gen, epochs=10)
model.save("mushroom_classifier_simplified.h5")

# Evaluación
results = model.evaluate(valid_gen)
print(f"Precisión final del modelo: {results[1] * 100:.2f}%")

# Visualización de precisión y pérdida
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Precisión')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Pérdida')
plt.legend()
plt.show()

# Búsqueda vectorial en FAISS
test_image, _ = next(valid_gen)
test_embedding = model.layers[0].predict(test_image)
print(f"Dimensión del embedding: {test_embedding.shape}")
distances, indices = index.search(test_embedding, k=5)
for i in range(len(indices[0])):
    print(f"Clase de la seta: {mushroom_classes[indices[0][i]]} (Distancia: {distances[0][i]})")

