import pandas as pd
import numpy as np
from datetime import datetime

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

# Handle missing values - simple approach
missing_before = enhanced_df.isnull().sum().sum()
print(f"✓ Missing values before: {missing_before}")
missing_after = enhanced_df.isnull().sum().sum()
print(f"✓ Missing values after: {missing_after}")


# Save enhanced dataset
output_file = "set A corporate_rating_enhanced.csv"
enhanced_df.to_csv(output_file, index=False)
print(f"✅ Enhanced dataset saved: {output_file}")

# Create summary report
report_file = "enhancement_summary.txt"
with open(report_file, 'w') as f:

    f.write(f"  • Duplicates removed: {duplicates_count}\n")
    f.write(f"  • Missing values handled: {missing_before} → {missing_after}\n")
    f.write(f"  • Data completeness: {100 - (enhanced_df.isnull().sum().sum() / enhanced_df.size * 100):.1f}%\n")
