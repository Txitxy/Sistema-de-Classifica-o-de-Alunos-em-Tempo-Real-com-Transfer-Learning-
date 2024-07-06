import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
import os

def train_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    x = Flatten()(base_model.output)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Certifique-se de ter as pastas 'data/train' e 'data/validation' com suas imagens
    train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.2, horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory('data/train', target_size=(224, 224), batch_size=32, class_mode='categorical')
    
    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_directory('data/validation', target_size=(224, 224), batch_size=32, class_mode='categorical')
    
    history = model.fit(train_generator, epochs=10, validation_data=validation_generator)
    
    # Certifique-se de que a pasta 'model' existe
    if not os.path.exists('model'):
        os.makedirs('model')
    
    model.save('model/model.h5')
    
    return history

if __name__ == '__main__':
    history = train_model()
    print(history.history)
