import pandas as pd
<<<<<<< HEAD
## first commmit
=======
import numpy as np
from datetime import datetime
# NEW: Add ML imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# YOUR EXISTING CODE - UNCHANGED
>>>>>>> f02563dbf5df0f71f886847bccba96c24feb963a
df = pd.read_csv('set A corporate_rating.csv')
print("=" * 60)
print("YOUR ORIGINAL DATA DISPLAY")
print("=" * 60)
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

print("\n" + "🚀 STARTING QUICK ENHANCEMENT")
print("=" * 60)

# Step 1: Create a working copy (preserve original)
enhanced_df = df.copy()
print(f"✓ Created working copy of your data")
print(f"  Original shape: {df.shape}")

# Step 2: Quick cleaning
print("\n📝 QUICK CLEANING:")

# Remove duplicates
duplicates_count = enhanced_df.duplicated().sum()
enhanced_df = enhanced_df.drop_duplicates()
print(f"✓ Removed {duplicates_count} duplicate rows")

# Handle missing values - ENHANCED approach
missing_before = enhanced_df.isnull().sum().sum()
print(f"✓ Missing values before: {missing_before}")

# NEW: Better missing value handling
for column in enhanced_df.columns:
    if enhanced_df[column].dtype == 'object':
        # For categorical columns, fill with mode
        if not enhanced_df[column].mode().empty:
            mode_value = enhanced_df[column].mode()[0]
            enhanced_df[column].fillna(mode_value, inplace=True)
        else:
            enhanced_df[column].fillna('Unknown', inplace=True)
    else:
        # For numerical columns, fill with median
        median_value = enhanced_df[column].median()
        enhanced_df[column].fillna(median_value, inplace=True)

missing_after = enhanced_df.isnull().sum().sum()
print(f"✓ Missing values after: {missing_after}")

# Save enhanced dataset
output_file = "set A corporate_rating_enhanced.csv"
enhanced_df.to_csv(output_file, index=False)
print(f"✅ Enhanced dataset saved: {output_file}")

# YOUR EXISTING REPORT - ENHANCED
report_file = "enhancement_summary.txt"
with open(report_file, 'w') as f:
    f.write("CORPORATE RATING ANALYSIS - SUMMARY REPORT\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"  • Original dataset shape: {df.shape}\n")
    f.write(f"  • Duplicates removed: {duplicates_count}\n")
    f.write(f"  • Missing values handled: {missing_before} → {missing_after}\n")
    f.write(f"  • Data completeness: {100 - (enhanced_df.isnull().sum().sum() / enhanced_df.size * 100):.1f}%\n")

print(f"✅ Enhancement summary saved: {report_file}")

# =============================================================================
# NEW SECTION: MACHINE LEARNING MODELS
# =============================================================================

print("\n" + "🤖 MACHINE LEARNING PREDICTION MODELS")
print("=" * 60)

# Step 1: Prepare data for ML
print("\n📊 PREPARING DATA FOR MACHINE LEARNING:")

# Auto-detect target variable (rating column)
target_columns = [col for col in enhanced_df.columns if 'rating' in col.lower() or 'grade' in col.lower() or 'score' in col.lower()]

if target_columns:
    target_column = target_columns[0]
    print(f"✓ Target variable detected: '{target_column}'")
else:
    # If no rating column found, let user know they need to specify
    print("⚠️  No rating column auto-detected. Please specify your target column:")
    print("Available columns:", list(enhanced_df.columns))
    # For now, use the last column as target
    target_column = enhanced_df.columns[-1]
    print(f"✓ Using '{target_column}' as target variable")

# Check if target column exists
if target_column not in enhanced_df.columns:
    print(f"❌ Error: Column '{target_column}' not found in dataset")
    print("Available columns:", list(enhanced_df.columns))
    exit()

# Separate features and target
X = enhanced_df.drop(columns=[target_column])
y = enhanced_df[target_column]

print(f"✓ Features shape: {X.shape}")
print(f"✓ Target shape: {y.shape}")
print(f"✓ Unique target values: {sorted(y.unique())}")

# Step 2: Encode categorical variables
print("\n🔧 ENCODING CATEGORICAL VARIABLES:")

# Store original column names for later reference
original_columns = X.columns.tolist()

# Encode categorical features
label_encoders = {}
categorical_columns = X.select_dtypes(include=['object']).columns

if len(categorical_columns) > 0:
    for column in categorical_columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column].astype(str))
        label_encoders[column] = le
        print(f"✓ Encoded categorical column: {column}")
else:
    print("✓ No categorical columns to encode")

# Encode target variable if it's categorical
target_encoder = None
if y.dtype == 'object':
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    print(f"✓ Encoded target variable: {target_column}")
    print(f"  Original classes: {list(y.unique())}")
    print(f"  Encoded classes: {list(target_encoder.classes_)}")
else:
    y_encoded = y
    print(f"✓ Target variable is already numerical")

# Step 3: Split data (80% training, 20% testing)
print("\n📊 SPLITTING DATA INTO TRAIN/TEST SETS:")

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"✓ Used stratified split to maintain class distribution")
except:
    # If stratified split fails, use regular split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    print(f"✓ Used regular split")

print(f"✓ Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"✓ Testing set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

# Step 4: Train Random Forest Model
print("\n🌲 TRAINING RANDOM FOREST MODEL:")

# Create and train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1  # Use all CPU cores
)

print("✓ Training Random Forest...")
rf_model.fit(X_train, y_train)
print("✓ Random Forest training completed!")

# Make predictions
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

print(f"📈 Random Forest Results:")
print(f"  • Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")

# Step 5: Train Decision Tree Model
print("\n🌳 TRAINING DECISION TREE MODEL:")

# Create and train Decision Tree
dt_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2
)

print("✓ Training Decision Tree...")
dt_model.fit(X_train, y_train)
print("✓ Decision Tree training completed!")

# Make predictions
dt_predictions = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)

print(f"📈 Decision Tree Results:")
print(f"  • Accuracy: {dt_accuracy:.4f} ({dt_accuracy*100:.2f}%)")

# Step 6: Model Comparison
print("\n⚖️  MODEL COMPARISON:")
print("=" * 40)

models_comparison = pd.DataFrame({
    'Model': ['Random Forest', 'Decision Tree'],
    'Accuracy': [rf_accuracy, dt_accuracy],
    'Accuracy (%)': [rf_accuracy*100, dt_accuracy*100]
})

print(models_comparison)

# Determine best model
best_model_idx = models_comparison['Accuracy'].idxmax()
best_model_name = models_comparison.loc[best_model_idx, 'Model']
best_accuracy = models_comparison.loc[best_model_idx, 'Accuracy']

print(f"\n🏆 Best Performing Model: {best_model_name}")
print(f"🎯 Best Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")

# Step 7: Detailed Model Reports
print("\n📊 DETAILED MODEL EVALUATION:")
print("=" * 50)

print(f"\n🌲 Random Forest - Classification Report:")
print(classification_report(y_test, rf_predictions, zero_division=0))

print(f"\n🌳 Decision Tree - Classification Report:")
print(classification_report(y_test, dt_predictions, zero_division=0))







# Step 10: Save All Results
print("\n💾 SAVING MACHINE LEARNING RESULTS:")
print("=" * 40)

# Prepare full results
full_results = pd.DataFrame({
    'Actual': y_test,
    'Random_Forest_Prediction': rf_predictions,
    'Decision_Tree_Prediction': dt_predictions
})

# Convert predictions back to original format if needed
if target_encoder:
    full_results['Actual'] = target_encoder.inverse_transform(full_results['Actual'])
    full_results['Random_Forest_Prediction'] = target_encoder.inverse_transform(full_results['Random_Forest_Prediction'])
    full_results['Decision_Tree_Prediction'] = target_encoder.inverse_transform(full_results['Decision_Tree_Prediction'])

# Save results
ml_results_file = "ml_predictions_results.csv"
full_results.to_csv(ml_results_file, index=False)
print(f"✅ ML predictions saved: {ml_results_file}")





print(f"✅ Updated summary report: {report_file}")

# Final Summary
print("\n🎉 ANALYSIS COMPLETE!")
print("=" * 50)
print("📊 SUMMARY:")
print(f"  • Data processed: {enhanced_df.shape[0]} records, {enhanced_df.shape[1]} features")
print(f"  • Missing values handled: {missing_before} → {missing_after}")
print(f"  • Best ML model: {best_model_name} ({best_accuracy*100:.2f}% accuracy)")
print(f"  • All results saved to CSV files")
print("\n🎯 Your corporate rating prediction system is ready!")
print("Check the generated files for detailed results and predictions.")