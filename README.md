# Playlist Track Popularity Prediction

## Project Overview
This project uses a neural network built with PyTorch to predict the popularity of Spotify tracks based on various audio features. The model is trained on historical song data and can predict popularity scores for new tracks.

## Dataset
- **Training data**: `data/songs_train.csv` - Contains Spotify track information with popularity labels
- **Test data**: `data/songs_test.csv` - Test set for predictions
- **Sample submission**: `data/sample_submission.csv` - Reference format for submission

## Data Processing

### 1. **Data Loading**
   - Load training and test data using pandas
   - Configure pandas to display all columns and rows for exploration

### 2. **Data Cleaning**
   - Remove duplicate records
   - Handle missing values using dropna()
   - Check for null values and data types

### 3. **Feature Engineering**
   - Convert `track_album_release_date` to datetime format
   - Normalize the release date to a 0-1 range using min-max normalization
   - Drop non-predictive columns: `id`, `track_id`, `track_name`, `track_album_name`, `track_album_id`, `playlist_id`

### 4. **Encoding**
   - Clean string columns (strip whitespace, convert to lowercase)
   - Apply one-hot encoding to categorical features
   - Combine numeric and encoded features

### 5. **Preprocessing**
   - Split data into training (90%) and validation (10%) sets
   - Apply StandardScaler normalization to all features
   - Convert data to PyTorch tensors with batch size of 256

## Model Architecture

**Neural Network with PyTorch:**
```
Input Layer → Linear (features → 128)
         ↓
    BatchNorm1d(128)
         ↓
    ReLU Activation
         ↓
    Dropout(0.2)
         ↓
    Linear (128 → 64)
         ↓
    ReLU Activation
         ↓
    Dropout(0.2)
         ↓
    Linear (64 → 1) → Output (Popularity Score)
```

- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam (learning rate: 0.0006)
- **Batch Size**: 256
- **Device**: GPU (CUDA) if available, otherwise CPU

## Training

### Hyperparameters:
- **Epochs**: 15 (with early stopping)
- **Patience**: 10 epochs (stop if no improvement)
- **Early Stopping**: Implemented to prevent overfitting

### Training Process:
1. Train the model on the training set for each epoch
2. Validate on the validation set
3. Save the model if validation loss improves
4. Stop early if validation loss doesn't improve for 10 consecutive epochs
5. Load the best model before making predictions

### Evaluation Metrics:
- **MSE (Mean Squared Error)**: Main loss metric
- **RMSE (Root Mean Squared Error)**: More interpretable metric on the same scale as popularity scores

## Prediction & Submission

1. Load the best trained model
2. Make predictions on the test set
3. Post-process predictions:
   - Round to nearest integer
   - Clamp values between 0-100 (valid popularity range)
4. Create submission file with format: `id` and `track_popularity`
5. Save as `submission.csv`

## Output Files
- **`best_model.pth`**: Best model weights saved during training
- **`submission.csv`**: Final predictions for submission containing track IDs and predicted popularity scores

## How to Run
1. Ensure all data files are in the `data/` directory
2. Run the notebook cells sequentially to:
   - Load and explore data
   - Preprocess and feature engineer
   - Train the neural network model
   - Generate predictions and submission file

## Dependencies
- pandas
- scikit-learn (preprocessing, train_test_split)
- torch (PyTorch)
- numpy

## Notes
- The model uses early stopping to prevent overfitting
- Popularity predictions are clamped to the range [0, 100]
- The model saves the best weights based on validation loss
