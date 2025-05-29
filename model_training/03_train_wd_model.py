#!/usr/bin/env python3

import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model Hyperparameters & Configuration
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10 # Increased from 5
VAL_AUC_MONITOR = 'val_auc' #AUC is tf.keras.metrics.AUC instance, so monitor its name 'auc' or 'val_auc' from history keys

# Relative paths (assuming script is in model_training/)
DEFAULT_DATA_DIR = "data/train_pairs_local"
DEFAULT_MODEL_FILENAME = "pet_model.h5"

# Feature Definitions (must match PairGCPLoad.py output)
# Original names: 'age', 'household_size', 'housing', 'activity_level', 'has_owned_pets', 'pet_type', 'pet_breed', 'pet_color'
# New names from PairGCPLoad.py refactoring:
NUMERICAL_FEATURES = [
    'adopter_age', 
    'adopter_household_size',
    'pet_age_years' # Added
]
CATEGORICAL_FEATURES = [
    'adopter_housing_type', # Was 'housing'
    'adopter_activity_level', # Was 'activity_level'
    'adopter_has_prior_pets', # Was 'has_owned_pets'
    'pet_animal_type', # Was 'pet_type'
    'pet_breed',
    'pet_color',
    'pet_size', # Added
    'pet_activity_needs', # Added
    'pet_needs_experienced_owner', # Added
    'pet_good_with_children' # Added
]
LABEL_COLUMN = 'match_label' # Was 'label'


def load_data(parquet_file_path: str) -> pd.DataFrame:
    """Load training data from parquet file."""
    logger.info(f"Loading data from {parquet_file_path}")
    if not os.path.exists(parquet_file_path):
        logger.error(f"Data file not found: {parquet_file_path}")
        raise FileNotFoundError(f"Data file not found: {parquet_file_path}")
    return pd.read_parquet(parquet_file_path)

def prepare_features(df: pd.DataFrame):
    """Prepare features for the Wide & Deep model."""
    logger.info("Preparing features...")
    
    normalizers = {}
    numerical_inputs = {}
    for feature in NUMERICAL_FEATURES:
        if feature not in df.columns:
            logger.error(f"Numerical feature '{feature}' not found in DataFrame columns: {df.columns.tolist()}")
            raise ValueError(f"Numerical feature '{feature}' not found in DataFrame columns.")
        normalizer = tf.keras.layers.Normalization(axis=None) # axis=None for single feature, -1 if features are last dim of multi-dim tensor
        # Reshape data for adapt method: (num_samples,) -> (num_samples, 1)
        feature_data_for_adapt = df[feature].values.astype(np.float32).reshape(-1, 1)
        normalizer.adapt(feature_data_for_adapt)
        normalizers[feature] = normalizer
        numerical_inputs[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.float32)
    
    categorical_inputs = {}
    categorical_encoded_wide = {}
    categorical_embedded_deep = {}
    
    for feature in CATEGORICAL_FEATURES:
        if feature not in df.columns:
            logger.error(f"Categorical feature '{feature}' not found in DataFrame columns: {df.columns.tolist()}")
            raise ValueError(f"Categorical feature '{feature}' not found in DataFrame columns.")
        
        # Ensure all values are strings for StringLookup, handle potential NaN/None
        vocab = df[feature].astype(str).unique()
        logger.info(f"Feature '{feature}' has {len(vocab)} unique values. Sample: {vocab[:5]}")
        lookup = tf.keras.layers.StringLookup(vocabulary=vocab, name=f"{feature}_lookup", oov_token='[UNK]')
        
        categorical_inputs[feature] = tf.keras.Input(shape=(1,), dtype=tf.string, name=feature)
        lookup_result = lookup(categorical_inputs[feature])
        
        # Wide part: one-hot encoding (or binary encoding)
        onehot = tf.keras.layers.CategoryEncoding(
            num_tokens=lookup.vocabulary_size(), # Includes OOV token if any
            output_mode="binary", # "one_hot" might be very sparse, "binary" is denser
            name=f"{feature}_onehot"
        )(lookup_result)
        categorical_encoded_wide[feature] = onehot
        
        # Deep part: embedding
        # Heuristic for embedding size: min( (vocab_size)^0.25, reasonable_max_dim )
        embedding_size = int(min(max(1, lookup.vocabulary_size()) ** 0.25, 20)) 
        embedding = tf.keras.layers.Embedding(
            input_dim=lookup.vocabulary_size(), # Max integer index + 1
            output_dim=embedding_size,
            name=f"{feature}_embedding"
        )(lookup_result)
        categorical_embedded_deep[feature] = tf.keras.layers.Flatten(name=f"{feature}_flatten")(embedding)
    
    # Combine features for wide part
    # Original wide_inputs used categorical_encoded[f"{feature}_wide"]
    # With new dict names: categorical_encoded_wide[feature]
    wide_feature_list = list(categorical_encoded_wide.values())
    if not wide_feature_list: # Handle case with no categorical features, though unlikely for this model
        wide = None # Or a dummy tensor if Keras requires an input here
        logger.warning("No categorical features for the wide part.")
    else:
        wide = tf.keras.layers.concatenate(wide_feature_list, name="wide_concat") if len(wide_feature_list) > 1 else wide_feature_list[0]
    logger.info(f"Wide input branch shape (after concat if multiple): {wide.shape if wide is not None else 'N/A'}")

    # Combine features for deep part
    # Original normalized_numerical used normalizers[feature](numerical_inputs[feature])
    # Original deep_inputs used normalized_numerical + [categorical_encoded[f"{feature}_deep"]]
    # With new dict names: categorical_embedded_deep[feature]
    processed_numerical_for_deep = [normalizers[feature](numerical_inputs[feature]) 
                                    for feature in NUMERICAL_FEATURES]
    processed_categorical_for_deep = list(categorical_embedded_deep.values())
    
    deep_feature_list = processed_numerical_for_deep + processed_categorical_for_deep
    if not deep_feature_list:
        raise ValueError("No features available for the deep part of the model.")

    deep = tf.keras.layers.concatenate(deep_feature_list, name="deep_concat") if len(deep_feature_list) > 1 else deep_feature_list[0]
    logger.info(f"Deep input branch shape (after concat if multiple): {deep.shape}")
    
    all_inputs = list(numerical_inputs.values()) + list(categorical_inputs.values())
    return all_inputs, wide, deep, normalizers # Return normalizers for saving/serving if needed

def create_model(all_inputs, wide_branch, deep_branch, learning_rate=LEARNING_RATE):
    """Create the Wide & Deep model architecture."""
    logger.info("Creating model...")
    
    # Deep tower
    deep_processed = tf.keras.layers.Dense(512, activation='relu', name='deep_dense_1')(deep_branch)
    deep_processed = tf.keras.layers.BatchNormalization(name='deep_bn_1')(deep_processed)
    deep_processed = tf.keras.layers.Dropout(0.3, name='deep_dropout_1')(deep_processed)
    
    for i, units in enumerate([256, 128, 64], 2):
        deep_processed = tf.keras.layers.Dense(units, activation='relu', name=f'deep_dense_{i}')(deep_processed)
        deep_processed = tf.keras.layers.BatchNormalization(name=f'deep_bn_{i}')(deep_processed)
        deep_processed = tf.keras.layers.Dropout(0.3, name=f'deep_dropout_{i}')(deep_processed)
    
    # Wide tower (if it exists)
    if wide_branch is not None:
        wide_processed = tf.keras.layers.Dense(512, activation='relu', name='wide_dense_1')(wide_branch)
        wide_processed = tf.keras.layers.BatchNormalization(name='wide_bn_1')(wide_processed)
        wide_processed = tf.keras.layers.Dropout(0.3, name='wide_dropout_1')(wide_processed)
        
        for i, units in enumerate([256, 128, 64], 2):
            wide_processed = tf.keras.layers.Dense(units, activation='relu', name=f'wide_dense_{i}')(wide_processed)
            wide_processed = tf.keras.layers.BatchNormalization(name=f'wide_bn_{i}')(wide_processed)
            wide_processed = tf.keras.layers.Dropout(0.3, name=f'wide_dropout_{i}')(wide_processed)
        
        combined = tf.keras.layers.concatenate([wide_processed, deep_processed], name='combine_wide_deep')
    else: # Only deep branch
        combined = deep_processed
        
    combined = tf.keras.layers.Dense(32, activation='relu', name='combined_dense')(combined)
    combined = tf.keras.layers.BatchNormalization(name='combined_bn')(combined)
    combined = tf.keras.layers.Dropout(0.2, name='combined_dropout')(combined)
    
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(combined)
    model = tf.keras.Model(inputs=all_inputs, outputs=output)
    
    model.summary(print_fn=logger.info)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name=VAL_AUC_MONITOR)] # Give AUC a name to monitor
    )
    return model

def prepare_dataset(df: pd.DataFrame):
    """Convert pandas DataFrame to tf.data.Dataset."""
    logger.info(f"Preparing dataset from DataFrame with {len(df)} rows.")
    # Reshape data for dataset features
    numerical_data = {feature: df[feature].values.astype(np.float32).reshape(-1, 1) 
                      for feature in NUMERICAL_FEATURES}
    categorical_data = {feature: df[feature].astype(str).values.reshape(-1, 1) 
                        for feature in CATEGORICAL_FEATURES}
    
    features_dict = {**numerical_data, **categorical_data}
    
    if LABEL_COLUMN not in df.columns:
        logger.error(f"Label column '{LABEL_COLUMN}' not found in DataFrame for dataset preparation.")
        raise ValueError(f"Label column '{LABEL_COLUMN}' not found.")
    labels = df[LABEL_COLUMN].values.astype(np.int32).reshape(-1,1) # Ensure label is also shaped (batch, 1)
    
    dataset = tf.data.Dataset.from_tensor_slices((features_dict, labels))
    return dataset

def main():
    """Main training function."""
    # Determine paths relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, DEFAULT_DATA_DIR, "train_pairs.parquet")
    # Save model in the same directory as the script (model_training/)
    model_save_path = os.path.join(script_dir, DEFAULT_MODEL_FILENAME)
    
    df = load_data(data_path)
    
    # Validate all necessary features and label are in the DataFrame
    all_expected_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [LABEL_COLUMN]
    missing_cols = [col for col in all_expected_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing expected columns in loaded data: {missing_cols}. Available: {df.columns.tolist()}")
        return

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[LABEL_COLUMN] if LABEL_COLUMN in df else None)
    logger.info(f"Training set size: {len(train_df)}, Test set size: {len(test_df)}")

    # Check distribution of labels
    if LABEL_COLUMN in train_df:
        logger.info(f"Train label distribution:\n{train_df[LABEL_COLUMN].value_counts(normalize=True)}")
    if LABEL_COLUMN in test_df:
        logger.info(f"Test label distribution:\n{test_df[LABEL_COLUMN].value_counts(normalize=True)}") 

    # Prepare features (using train_df to adapt normalizers and lookups)
    all_inputs, wide_branch, deep_branch, _ = prepare_features(train_df) # Ignore normalizers for now
    model = create_model(all_inputs, wide_branch, deep_branch)
    
    train_dataset = prepare_dataset(train_df)
    test_dataset = prepare_dataset(test_df)
    
    train_dataset = train_dataset.shuffle(len(train_df)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=VAL_AUC_MONITOR, # Monitor 'val_auc' (or actual name if different, e.g. val_auc_1)
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_save_path,
            monitor=VAL_AUC_MONITOR,
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=VAL_AUC_MONITOR,
            factor=0.2,
            patience=3, # Reduce LR if val_auc doesn't improve for 3 epochs
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    logger.info("Starting model training...")
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info(f"Training history keys: {history.history.keys()}")

    # Load best model for evaluation (if ModelCheckpoint saved it)
    if os.path.exists(model_save_path):
        logger.info(f"Loading best model from {model_save_path} for final evaluation.")
        model.load_weights(model_save_path) # Or use tf.keras.models.load_model(model_save_path) if saving entire model
    else:
        logger.warning("Best model checkpoint not found. Evaluating the model at the end of training.")

    logger.info("Evaluating model on the test set...")
    eval_results = model.evaluate(test_dataset, verbose=1)
    
    # Construct a dictionary for clear logging of evaluation results
    results_log = {history.model.metrics_names[i]: eval_results[i] for i in range(len(eval_results))}
    logger.info(f"Test Results: {results_log}")
    # Example: logger.info(f"Test Loss: {results_log.get('loss', 'N/A'):.4f}")
    # logger.info(f"Test Accuracy: {results_log.get('accuracy', 'N/A'):.4f}")
    # logger.info(f"Test AUC ({VAL_AUC_MONITOR}): {results_log.get(VAL_AUC_MONITOR, 'N/A'):.4f}")

    # Save final model (ModelCheckpoint already saves the best one)
    # If you want to save the very final state regardless of best, you can save again.
    # model.save(model_save_path) # This would overwrite the best model if called here.
    logger.info(f"Best model during training was saved to {model_save_path}")
    logger.info("Training complete!")

if __name__ == "__main__":
    main() 