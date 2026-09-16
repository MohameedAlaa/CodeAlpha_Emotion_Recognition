import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def build_cnn_model(input_shape: tuple, num_classes: int) -> tf.keras.Model:
    """
    Builds a 1D CNN architecture for Speech Emotion Recognition.
    Original Baseline Architecture.
    
    Args:
        input_shape (tuple): Shape of the input features, e.g., (time_steps, n_mfcc).
        num_classes (int): Number of emotion classes to predict.
        
    Returns:
        tf.keras.Model: A compiled or uncompiled Keras Model.
    """
    model = models.Sequential()
    
    # Conv Block 1
    model.add(layers.Conv1D(64, kernel_size=3, padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))
    
    # Conv Block 2
    model.add(layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))
    
    # Conv Block 3
    model.add(layers.Conv1D(256, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))
    
    # Global Pooling
    model.add(layers.GlobalAveragePooling1D())
    
    # Dense Classifier
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

def build_2d_cnn_model(input_shape: tuple, num_classes: int) -> tf.keras.Model:
    """
    Builds a 2D CNN architecture for Speech Emotion Recognition using Mel-Spectrograms.
    Experiment 2: 2D CNN.
    
    Args:
        input_shape (tuple): Shape of the input features, e.g., (time_steps, n_mels, 1).
        num_classes (int): Number of emotion classes to predict.
        
    Returns:
        tf.keras.Model: A compiled or uncompiled Keras Model.
    """
    model = models.Sequential()
    
    l2_reg = regularizers.l2(0.001)
    
    # Conv Block 1
    model.add(layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape, kernel_regularizer=l2_reg))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.3))
    
    # Conv Block 2
    model.add(layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.3))
    
    # Conv Block 3
    model.add(layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.4))
    
    # Global Pooling
    model.add(layers.GlobalAveragePooling2D())
    
    # Dense Classifier
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=l2_reg))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model
