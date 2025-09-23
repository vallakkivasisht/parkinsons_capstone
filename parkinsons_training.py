import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

print("🚀 PARKINSON'S DETECTION - 4 MODEL COMPARISON")
print("="*60)

# =========================
# STEP 1: LOAD AND PREPARE DATA
# =========================
def load_and_prepare_data(filepath="bilstm_data_clean.npz", csv_filepath="patients_features_merged_preprocessed.csv"):
    """
    Load data for both sequential (BiLSTM) and traditional ML models
    """
    print("\n📊 LOADING DATA...")
    print("-" * 30)
    
    # Try to load preprocessed sequential data first
    try:
        data = np.load(filepath)
        X_sequential = data['X']  # (n_patients, n_activities, n_features)
        y_sequential = data['y']  # (n_patients, n_classes)
        feature_names = data['feature_names']
        patient_ids = data['patient_ids']
        
        print(f"✅ Loaded sequential data: {X_sequential.shape}")
        print(f"   Features: {list(feature_names)}")
        
        # Convert one-hot to class indices for traditional ML
        y_classes = np.argmax(y_sequential, axis=1)
        
        # Flatten sequential data for traditional ML models
        # Take mean across activities for each patient
        X_traditional = np.mean(X_sequential, axis=1)  # (n_patients, n_features)
        
        print(f"✅ Traditional ML data shape: {X_traditional.shape}")
        
        return X_sequential, X_traditional, y_sequential, y_classes, feature_names, patient_ids
        
    except FileNotFoundError:
        print("⚠️ Sequential data file not found. Loading from CSV...")
        
        # Load from CSV and create both formats
        df = pd.read_csv(csv_filepath)
        
        # Define feature columns
        feature_cols = [col for col in df.columns if 'filtered' in col]
        if not feature_cols:
            # Fallback to your core features
            feature_cols = [
                "ACC_RMS", "ACC_CV", "ACC_AvgFreq",
                "GYRO_RMS", "GYRO_CV", "GYRO_AvgFreq"
            ]
        
        print(f"📋 Using features: {feature_cols}")
        
        # Prepare data
        X_list, y_list = [], []
        
        for patient_id, patient_data in df.groupby("PatientID"):
            features = patient_data[feature_cols].values
            
            # Get label (convert condition to numeric)
            condition = patient_data["condition"].iloc[0].lower().strip()
            if condition == "healthy":
                label = 0
            elif condition == "parkinson's":
                label = 1
            else:
                label = 2  # other conditions
            
            X_list.append(features)
            y_list.append(label)
        
        X_sequential = np.array(X_list)
        y_classes = np.array(y_list)
        
        # Create one-hot for BiLSTM
        y_sequential = to_categorical(y_classes, num_classes=3)
        
        # Traditional ML: take mean across activities
        X_traditional = np.mean(X_sequential, axis=1)
        
        feature_names = feature_cols
        patient_ids = list(range(len(X_list)))
        
        print(f"✅ Created sequential data: {X_sequential.shape}")
        print(f"✅ Created traditional data: {X_traditional.shape}")
        
        return X_sequential, X_traditional, y_sequential, y_classes, feature_names, patient_ids

# =========================
# STEP 2: TRADITIONAL ML MODELS
# =========================
def train_traditional_models(X, y, feature_names):
    """
    Train and evaluate traditional ML models
    """
    print("\n🤖 TRAINING TRADITIONAL ML MODELS...")
    print("-" * 40)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        ),
        'Logistic Regression': LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        ),
        'SVM': SVC(
            kernel='rbf',
            random_state=42,
            class_weight='balanced',
            probability=True
        )
    }
    
    results = {}
    
    # Train each model
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_prob': y_prob,
            'y_test': y_test
        }
        
        print(f"   ✅ Accuracy: {accuracy:.3f}")
        print(f"   📊 CV Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    return results, scaler

# =========================
# STEP 3: BiLSTM MODEL  
# =========================
def create_bilstm_model(input_shape, num_classes=3):
    """
    Create BiLSTM model architecture
    """
    model = Sequential([
        # First BiLSTM layer
        Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3), 
                     input_shape=input_shape),
        
        # Second BiLSTM layer
        Bidirectional(LSTM(32, dropout=0.3, recurrent_dropout=0.3)),
        
        # Dense layers
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(16, activation='relu'),
        Dropout(0.3),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_bilstm_model(X, y):
    """
    Train BiLSTM model
    """
    print("\n🧠 TRAINING BiLSTM MODEL...")
    print("-" * 30)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, 
        stratify=np.argmax(y, axis=1)
    )
    
    print(f"📊 Training shape: {X_train.shape}")
    print(f"📊 Testing shape: {X_test.shape}")
    
    # Create model
    model = create_bilstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    print("\n🏗️ Model Architecture:")
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001,
            verbose=1
        )
    ]
    
    # Train model
    print("\n🚀 Starting training...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    print(f"\n✅ BiLSTM Test Accuracy: {test_accuracy:.3f}")
    
    return {
        'model': model,
        'history': history,
        'accuracy': test_accuracy,
        'y_pred': y_pred,
        'y_test': y_test_classes,
        'y_prob': y_pred_prob
    }

# =========================
# STEP 4: EVALUATION AND COMPARISON
# =========================
def compare_models(traditional_results, bilstm_result):
    """
    Compare all models and create visualizations
    """
    print("\n📊 MODEL COMPARISON RESULTS")
    print("="*50)
    
    # Collect results
    all_results = {}
    
    # Traditional models
    for name, result in traditional_results.items():
        all_results[name] = {
            'accuracy': result['accuracy'],
            'cv_mean': result['cv_mean'],
            'cv_std': result['cv_std']
        }
    
    # BiLSTM
    all_results['BiLSTM'] = {
        'accuracy': bilstm_result['accuracy'],
        'cv_mean': bilstm_result['accuracy'],  # No CV for neural networks typically
        'cv_std': 0.0
    }
    
    # Create comparison table
    results_df = pd.DataFrame(all_results).T
    results_df = results_df.sort_values('accuracy', ascending=False)
    
    print("\n🏆 FINAL RANKINGS:")
    print("-" * 40)
    for i, (model, row) in enumerate(results_df.iterrows(), 1):
        print(f"{i}. {model:<18}: {row['accuracy']:.3f} ± {row['cv_std']:.3f}")
    
    # Visualizations
    create_comparison_plots(traditional_results, bilstm_result, results_df)
    
    return results_df

def create_comparison_plots(traditional_results, bilstm_result, results_df):
    """
    Create comprehensive comparison plots
    """
    print("\n📈 CREATING COMPARISON PLOTS...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Parkinson\'s Detection - Model Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Accuracy Comparison
    models = results_df.index.tolist()
    accuracies = results_df['accuracy'].tolist()
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    
    bars = axes[0, 0].bar(models, accuracies, color=colors[:len(models)])
    axes[0, 0].set_title('Model Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Confusion Matrix for Best Model
    best_model = results_df.index[0]
    
    if best_model == 'BiLSTM':
        cm = confusion_matrix(bilstm_result['y_test'], bilstm_result['y_pred'])
        y_test = bilstm_result['y_test']
        y_pred = bilstm_result['y_pred']
    else:
        result = traditional_results[best_model]
        cm = confusion_matrix(result['y_test'], result['y_pred'])
        y_test = result['y_test']
        y_pred = result['y_pred']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
    axes[0, 1].set_title(f'Confusion Matrix - {best_model}')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Actual')
    
    # Plot 3: BiLSTM Training History (if available)
    if 'history' in bilstm_result:
        history = bilstm_result['history']
        epochs = range(1, len(history.history['accuracy']) + 1)
        
        axes[1, 0].plot(epochs, history.history['accuracy'], 'b-', label='Training Accuracy')
        axes[1, 0].plot(epochs, history.history['val_accuracy'], 'r-', label='Validation Accuracy')
        axes[1, 0].set_title('BiLSTM Training History')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Model Performance Summary
    metrics_data = []
    for name, result in traditional_results.items():
        metrics_data.append({
            'Model': name,
            'Accuracy': result['accuracy'],
            'CV_Mean': result['cv_mean'],
            'Type': 'Traditional ML'
        })
    
    metrics_data.append({
        'Model': 'BiLSTM',
        'Accuracy': bilstm_result['accuracy'],
        'CV_Mean': bilstm_result['accuracy'],
        'Type': 'Deep Learning'
    })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Grouped bar plot
    x = np.arange(len(metrics_df))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, metrics_df['Accuracy'], width, label='Test Accuracy', alpha=0.8)
    axes[1, 1].bar(x + width/2, metrics_df['CV_Mean'], width, label='CV Mean', alpha=0.8)
    
    axes[1, 1].set_xlabel('Models')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Accuracy vs Cross-Validation Comparison')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics_df['Model'], rotation=45)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Plots saved as 'model_comparison_results.png'")

# =========================
# MAIN EXECUTION
# =========================
def main():
    """
    Main execution function
    """
    # Load data
    X_seq, X_trad, y_seq, y_classes, feature_names, patient_ids = load_and_prepare_data()
    
    # Print data summary
    print(f"\n📋 DATA SUMMARY:")
    print(f"   Patients: {len(patient_ids)}")
    print(f"   Activities per patient: {X_seq.shape[1]}")
    print(f"   Features: {X_seq.shape[2]}")
    print(f"   Feature names: {list(feature_names)}")
    print(f"   Class distribution: {np.bincount(y_classes)}")
    
    # Train traditional models
    traditional_results, scaler = train_traditional_models(X_trad, y_classes, feature_names)
    
    # Train BiLSTM
    bilstm_result = train_bilstm_model(X_seq, y_seq)
    
    # Compare models
    comparison_df = compare_models(traditional_results, bilstm_result)
    
    # Generate detailed reports
    print("\n📄 DETAILED CLASSIFICATION REPORTS:")
    print("="*60)
    
    class_names = ['Healthy', 'Parkinsons', 'Other']
    
    for name, result in traditional_results.items():
        print(f"\n{name.upper()}:")
        print("-" * 30)
        print(classification_report(result['y_test'], result['y_pred'], 
                                  target_names=class_names, zero_division=0))
    
    print("\nBiLSTM:")
    print("-" * 30)
    print(classification_report(bilstm_result['y_test'], bilstm_result['y_pred'], 
                              target_names=class_names, zero_division=0))
    
    # Save results
    comparison_df.to_csv('model_comparison_results.csv', index=True)
    print(f"\n💾 Results saved to 'model_comparison_results.csv'")
    
    print("\n🎯 ANALYSIS COMPLETE!")
    print("="*60)
    
    return comparison_df, traditional_results, bilstm_result

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Run comparison
    results_df, trad_results, bilstm_result = main()