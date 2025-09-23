import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_and_preprocess_data():
    """Load and preprocess the dataset for CNN training"""
    print("Loading dataset...")
    df = pd.read_csv('patients_features_filtered.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['condition'].value_counts()}")
    
    # Select numerical features for CNN
    feature_columns = ['ACC_RMS_filtered', 'ACC_CV_filtered', 'ACC_AvgFreq_filtered', 
                      'GYRO_RMS_filtered', 'GYRO_CV_filtered', 'GYRO_AvgFreq_filtered']
    
    X = df[feature_columns].values
    y = df['condition'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Classes: {label_encoder.classes_}")
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reshape for CNN (add channel dimension)
    # For 1D CNN, we need (samples, timesteps, features)
    # We'll treat each sample as a 1D sequence with 6 features
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
    
    # Convert to categorical for multi-class classification
    y_categorical = to_categorical(y_encoded)
    
    return X_reshaped, y_categorical, y_encoded, label_encoder, scaler

def create_cnn_model(input_shape, num_classes):
    """Create a 1D CNN model for classification"""
    model = Sequential([
        # First Conv1D layer with smaller kernel
        Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.25),
        
        # Second Conv1D layer
        Conv1D(filters=64, kernel_size=2, activation='relu'),
        BatchNormalization(),
        Dropout(0.25),
        
        # Third Conv1D layer
        Conv1D(filters=128, kernel_size=2, activation='relu'),
        BatchNormalization(),
        Dropout(0.25),
        
        # Global Average Pooling instead of MaxPooling to avoid dimension issues
        tf.keras.layers.GlobalAveragePooling1D(),
        
        # Dense layers
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_model():
    """Main training function"""
    # Load and preprocess data
    X, y, y_encoded, label_encoder, scaler = load_and_preprocess_data()
    
    # Create 80-20 train-test split
    print("\nCreating 80-20 train-test split...")
    X_train, X_test, y_train, y_test, y_train_encoded, y_test_encoded = train_test_split(
        X, y, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    # Create model
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(label_encoder.classes_)
    
    print(f"\nCreating CNN model...")
    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")
    
    model = create_cnn_model(input_shape, num_classes)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    print("\nModel Summary:")
    model.summary()
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=50,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test_encoded, y_pred_classes, 
                              target_names=label_encoder.classes_))
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('cnn_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test_encoded, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('cnn_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, history, train_accuracy, test_accuracy

if __name__ == "__main__":
    print("Starting CNN training for Parkinson's disease classification...")
    model, history, train_acc, test_acc = train_model()
    print(f"\nFinal Results:")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
