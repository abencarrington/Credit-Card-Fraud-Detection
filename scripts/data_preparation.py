"""
Data preparation script for the IEEE-CIS fraud detection dataset.

This script:
1. Performs exploratory data analysis (EDA)
2. Analyzes data quality and completeness
3. Identifies important features through correlation analysis
4. Generates informative visualizations
5. Prepares the data for model training

Usage:
    python scripts/data_preparation.py [--data-dir DATA_DIR] [--output OUTPUT_DIR]
"""

import os
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

from src.data.loader import load_data, downcast_dtypes
from src.bootstrap import set_project_root

# Set project root to resolve imports correctly
set_project_root()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_dtypes(df, output_dir):
    """
    Analyze and visualize column data types.
    
    Args:
        df: DataFrame to analyze
        output_dir: Directory to save visualizations
    """
    logger.info("Analyzing column data types...")
    
    # Count column types
    dtype_counts = df.dtypes.value_counts()
    
    # Visualize
    plt.figure(figsize=(10, 6))
    dtype_counts.plot(kind='bar')
    plt.title('Distribution of Column Data Types')
    plt.xlabel('Data Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dtype_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save to CSV
    dtype_df = pd.DataFrame({
        'column': df.columns,
        'dtype': df.dtypes.astype(str),
        'memory_usage_bytes': df.memory_usage(deep=True)[1:],  # Skip index
        'memory_usage_mb': df.memory_usage(deep=True)[1:] / (1024 * 1024)
    })
    
    # Add nullable info and unique counts
    dtype_df['nullable'] = [df[col].isna().any() for col in df.columns]
    dtype_df['null_count'] = [df[col].isna().sum() for col in df.columns]
    dtype_df['null_percentage'] = [(df[col].isna().sum() / len(df)) * 100 for col in df.columns]
    dtype_df['unique_count'] = [df[col].nunique() for col in df.columns]
    dtype_df['unique_percentage'] = [(df[col].nunique() / len(df)) * 100 for col in df.columns]
    
    # Sort by memory usage
    dtype_df = dtype_df.sort_values('memory_usage_bytes', ascending=False)
    
    # Save to CSV
    dtype_df.to_csv(os.path.join(output_dir, "column_dtypes.csv"), index=False)
    
    # Summary stats
    total_memory = dtype_df['memory_usage_mb'].sum()
    logger.info(f"Total memory usage: {total_memory:.2f} MB")
    logger.info(f"Number of columns: {len(df.columns)}")
    logger.info(f"Number of rows: {len(df)}")
    
    return dtype_df

def analyze_missing_values(df, output_dir):
    """
    Analyze and visualize missing values.
    
    Args:
        df: DataFrame to analyze
        output_dir: Directory to save visualizations
    """
    logger.info("Analyzing missing values...")
    
    # Calculate missing values
    missing = df.isna().sum()
    missing_percent = (missing / len(df)) * 100
    
    # Create DataFrame with missing values info
    missing_df = pd.DataFrame({
        'column': missing.index,
        'count': missing.values,
        'percentage': missing_percent.values
    })
    
    # Sort by missing count
    missing_df = missing_df.sort_values('count', ascending=False)
    
    # Filter columns with missing values
    missing_df = missing_df[missing_df['count'] > 0]
    
    # Save to CSV
    missing_df.to_csv(os.path.join(output_dir, "missing_values.csv"), index=False)
    
    # Visualize top missing columns
    if len(missing_df) > 0:
        # Limit to top 30 columns for readability
        plot_df = missing_df.head(30)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='percentage', y='column', data=plot_df)
        plt.title('Top Columns with Missing Values')
        plt.xlabel('Missing Percentage')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "missing_values.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Found {len(missing_df)} columns with missing values")
        logger.info(f"Top column with missing values: {missing_df.iloc[0]['column']} " +
                   f"({missing_df.iloc[0]['percentage']:.2f}%)")
    else:
        logger.info("No missing values found")
    
    return missing_df

def analyze_fraud_distribution(df, output_dir):
    """
    Analyze and visualize fraud distribution.
    
    Args:
        df: DataFrame with fraud labels
        output_dir: Directory to save visualizations
        
    Returns:
        Dictionary with fraud statistics
    """
    logger.info("Analyzing fraud distribution...")
    
    # Check if fraud labels exist
    if 'isFraud' not in df.columns:
        logger.warning("No fraud labels found in dataset")
        return None
    
    # Calculate fraud distribution
    fraud_counts = df['isFraud'].value_counts()
    fraud_percent = df['isFraud'].value_counts(normalize=True) * 100
    
    # Create summary
    fraud_stats = {
        'total_transactions': len(df),
        'legitimate_count': int(fraud_counts.get(0, 0)),
        'fraud_count': int(fraud_counts.get(1, 0)),
        'legitimate_percent': float(fraud_percent.get(0, 0)),
        'fraud_percent': float(fraud_percent.get(1, 0)),
        'fraud_ratio': float(fraud_counts.get(1, 0) / fraud_counts.get(0, 1))
    }
    
    # Visualize fraud distribution
    plt.figure(figsize=(10, 6))
    ax = fraud_counts.plot(kind='bar', color=['green', 'red'])
    
    # Add count and percentage labels
    for i, count in enumerate(fraud_counts):
        percent = fraud_percent.iloc[i]
        ax.text(i, count + (count * 0.01), f"{count} ({percent:.2f}%)", 
                horizontalalignment='center')
    
    plt.title('Transaction Distribution (Legitimate vs Fraud)')
    plt.xlabel('Is Fraud')
    plt.ylabel('Transaction Count')
    plt.xticks([0, 1], ['Legitimate', 'Fraud'], rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fraud_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save stats
    with open(os.path.join(output_dir, "fraud_stats.json"), 'w') as f:
        import json
        json.dump(fraud_stats, f, indent=2)
    
    logger.info(f"Fraud rate: {fraud_stats['fraud_percent']:.2f}%")
    logger.info(f"Fraud ratio: 1:{1/fraud_stats['fraud_ratio']:.1f}")
    
    return fraud_stats

def analyze_numeric_features(df, output_dir, target='isFraud', n_top=20):
    """
    Analyze numeric features: distribution, correlation with target.
    
    Args:
        df: DataFrame to analyze
        output_dir: Directory to save visualizations
        target: Target column
        n_top: Number of top features to highlight
    """
    logger.info("Analyzing numeric features...")
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # If target is in numeric columns, remove it
    if target in numeric_cols:
        numeric_cols.remove(target)
    
    logger.info(f"Found {len(numeric_cols)} numeric columns")
    
    # Descriptive statistics
    numeric_stats = df[numeric_cols].describe()
    numeric_stats.to_csv(os.path.join(output_dir, "numeric_stats.csv"))
    
    # Correlation with target
    if target in df.columns:
        correlations = []
        for col in tqdm(numeric_cols, desc="Computing correlations"):
            # Skip if column has all same values
            if df[col].nunique() <= 1:
                continue
                
            # Skip columns with high null count
            if df[col].isna().mean() > 0.5:
                continue
            
            # Compute correlation
            try:
                corr = df[[col, target]].corr().iloc[0, 1]
                if not np.isnan(corr):
                    correlations.append({
                        'feature': col,
                        'correlation': corr,
                        'abs_correlation': abs(corr)
                    })
            except:
                continue
        
        # Convert to DataFrame
        corr_df = pd.DataFrame(correlations)
        
        # Sort by absolute correlation
        corr_df = corr_df.sort_values('abs_correlation', ascending=False)
        
        # Save to CSV
        corr_df.to_csv(os.path.join(output_dir, "feature_correlations.csv"), index=False)
        
        # Plot top correlations
        plt.figure(figsize=(12, 10))
        top_corr = corr_df.head(n_top)
        sns.barplot(x='correlation', y='feature', data=top_corr)
        plt.title(f'Top {n_top} Features Correlated with {target}')
        plt.xlabel('Correlation Coefficient')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "top_correlations.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log top features
        logger.info(f"Top positively correlated feature: {corr_df.iloc[0]['feature']} " +
                  f"({corr_df.iloc[0]['correlation']:.4f})")
        
        # Find top negative correlation
        top_neg = corr_df[corr_df['correlation'] < 0].sort_values('correlation').head(1)
        if not top_neg.empty:
            logger.info(f"Top negatively correlated feature: {top_neg.iloc[0]['feature']} " +
                      f"({top_neg.iloc[0]['correlation']:.4f})")
        
        return corr_df
    
    return None

def analyze_categorical_features(df, output_dir, target='isFraud'):
    """
    Analyze categorical features: distribution, relationship with target.
    
    Args:
        df: DataFrame to analyze
        output_dir: Directory to save visualizations
        target: Target column
    """
    logger.info("Analyzing categorical features...")
    
    # Select categorical columns (object and category)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    logger.info(f"Found {len(cat_cols)} categorical columns")
    
    # Save feature cardinality
    cardinality = []
    for col in cat_cols:
        cardinality.append({
            'feature': col,
            'unique_count': df[col].nunique(),
            'unique_percent': (df[col].nunique() / len(df)) * 100,
            'null_count': df[col].isna().sum(),
            'null_percent': (df[col].isna().sum() / len(df)) * 100
        })
    
    # Convert to DataFrame
    card_df = pd.DataFrame(cardinality)
    
    # Sort by cardinality
    card_df = card_df.sort_values('unique_count', ascending=False)
    
    # Save to CSV
    card_df.to_csv(os.path.join(output_dir, "categorical_cardinality.csv"), index=False)
    
    # Plot top high cardinality features
    plt.figure(figsize=(12, 8))
    top_card = card_df.head(20)
    sns.barplot(x='unique_count', y='feature', data=top_card)
    plt.title('Top 20 Categorical Features by Cardinality')
    plt.xlabel('Unique Value Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "categorical_cardinality.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # For low-cardinality features, analyze relationship with target
    if target in df.columns:
        low_card_cols = card_df[card_df['unique_count'] <= 10]['feature'].tolist()
        
        # Limit to a manageable number
        low_card_cols = low_card_cols[:10]
        
        for col in low_card_cols:
            # Contingency table
            cont_table = pd.crosstab(
                df[col], df[target], 
                normalize='index',
                dropna=False
            ) * 100
            
            # Save table
            cont_table.to_csv(os.path.join(output_dir, f"{col}_vs_{target}.csv"))
            
            # Plot
            plt.figure(figsize=(12, 6))
            cont_table.plot(kind='bar', stacked=True)
            plt.title(f'{col} vs {target}')
            plt.xlabel(col)
            plt.ylabel('Percentage')
            plt.legend(title=target)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{col}_vs_{target}.png"), dpi=300, bbox_inches='tight')
            plt.close()
    
    return card_df

def analyze_time_patterns(df, output_dir, time_col='TransactionDT', target='isFraud'):
    """
    Analyze time-based patterns in fraud activity.
    
    Args:
        df: DataFrame to analyze
        output_dir: Directory to save visualizations
        time_col: Time column 
        target: Target column
    """
    logger.info("Analyzing time patterns...")
    
    # Check if required columns exist
    if time_col not in df.columns or target not in df.columns:
        logger.warning(f"Required columns {time_col} or {target} not found")
        return None
    
    # Convert to datetime (assuming start date is 2017-12-01 as per IEEE-CIS dataset)
    df_temp = df.copy()
    
    # Convert seconds to datetime
    df_temp['datetime'] = pd.to_datetime('2017-12-01') + pd.to_timedelta(df_temp[time_col], unit='s')
    
    # Extract time components
    df_temp['hour'] = df_temp['datetime'].dt.hour
    df_temp['day'] = df_temp['datetime'].dt.day
    df_temp['weekday'] = df_temp['datetime'].dt.weekday
    df_temp['week'] = df_temp['datetime'].dt.isocalendar().week
    df_temp['month'] = df_temp['datetime'].dt.month
    
    # Hour of day analysis
    hour_fraud = df_temp.groupby(['hour', target])[target].count().unstack().fillna(0)
    hour_fraud_rate = hour_fraud[1] / (hour_fraud[0] + hour_fraud[1]) * 100
    
    plt.figure(figsize=(14, 7))
    
    # Plot fraud count by hour
    ax1 = plt.subplot(1, 2, 1)
    hour_fraud.plot(kind='bar', stacked=True, ax=ax1, color=['green', 'red'])
    ax1.set_title('Transaction Count by Hour of Day')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Transaction Count')
    ax1.legend(['Legitimate', 'Fraud'])
    
    # Plot fraud rate by hour
    ax2 = plt.subplot(1, 2, 2)
    hour_fraud_rate.plot(kind='line', marker='o', ax=ax2, color='red')
    ax2.set_title('Fraud Rate by Hour of Day')
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Fraud Rate (%)')
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fraud_by_hour.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Day of week analysis
    weekday_fraud = df_temp.groupby(['weekday', target])[target].count().unstack().fillna(0)
    weekday_fraud_rate = weekday_fraud[1] / (weekday_fraud[0] + weekday_fraud[1]) * 100
    
    plt.figure(figsize=(14, 7))
    
    # Plot fraud count by weekday
    ax1 = plt.subplot(1, 2, 1)
    weekday_fraud.plot(kind='bar', stacked=True, ax=ax1, color=['green', 'red'])
    ax1.set_title('Transaction Count by Day of Week')
    ax1.set_xlabel('Day of Week')
    ax1.set_ylabel('Transaction Count')
    ax1.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    ax1.legend(['Legitimate', 'Fraud'])
    
    # Plot fraud rate by weekday
    ax2 = plt.subplot(1, 2, 2)
    weekday_fraud_rate.plot(kind='line', marker='o', ax=ax2, color='red')
    ax2.set_title('Fraud Rate by Day of Week')
    ax2.set_xlabel('Day of Week')
    ax2.set_ylabel('Fraud Rate (%)')
    ax2.set_xticks(range(7))
    ax2.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fraud_by_weekday.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a time pattern dataframe with insights
    time_patterns = pd.DataFrame({
        'hour': range(24),
        'transactions': hour_fraud.sum(axis=1).values,
        'fraud_rate': hour_fraud_rate.values
    })
    
    # Add weekday patterns
    weekday_patterns = pd.DataFrame({
        'weekday': range(7),
        'weekday_name': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        'transactions': weekday_fraud.sum(axis=1).values,
        'fraud_rate': weekday_fraud_rate.values
    })
    
    # Save to CSV
    time_patterns.to_csv(os.path.join(output_dir, "hourly_patterns.csv"), index=False)
    weekday_patterns.to_csv(os.path.join(output_dir, "weekday_patterns.csv"), index=False)
    
    # Find peak fraud hours
    peak_hour = hour_fraud_rate.idxmax()
    peak_rate = hour_fraud_rate.max()
    
    logger.info(f"Peak fraud hour: {peak_hour} ({peak_rate:.2f}%)")
    
    # Find weekend vs weekday difference
    weekday_mean = weekday_fraud_rate.iloc[:5].mean()
    weekend_mean = weekday_fraud_rate.iloc[5:].mean()
    
    logger.info(f"Weekday fraud rate: {weekday_mean:.2f}%")
    logger.info(f"Weekend fraud rate: {weekend_mean:.2f}%")
    
    return time_patterns, weekday_patterns

def analyze_amount_patterns(df, output_dir, amount_col='TransactionAmt', target='isFraud'):
    """
    Analyze transaction amount patterns.
    
    Args:
        df: DataFrame to analyze
        output_dir: Directory to save visualizations
        amount_col: Transaction amount column
        target: Target column
    """
    logger.info("Analyzing transaction amount patterns...")
    
    # Check if required columns exist
    if amount_col not in df.columns or target not in df.columns:
        logger.warning(f"Required columns {amount_col} or {target} not found")
        return None
    
    # Calculate amount statistics by fraud status
    amount_stats = df.groupby(target)[amount_col].describe()
    amount_stats.to_csv(os.path.join(output_dir, "amount_stats_by_fraud.csv"))
    
    # Create histograms of transaction amounts
    plt.figure(figsize=(14, 7))
    
    # Regular scale
    ax1 = plt.subplot(1, 2, 1)
    for fraud_val in df[target].unique():
        label = 'Fraud' if fraud_val == 1 else 'Legitimate'
        color = 'red' if fraud_val == 1 else 'green'
        sns.histplot(
            data=df[df[target] == fraud_val], 
            x=amount_col,
            bins=50,
            alpha=0.5,
            label=label,
            color=color,
            ax=ax1
        )
    
    ax1.set_title('Transaction Amount Distribution')
    ax1.set_xlabel('Amount')
    ax1.set_ylabel('Count')
    ax1.legend()
    
    # Log scale
    ax2 = plt.subplot(1, 2, 2)
    for fraud_val in df[target].unique():
        label = 'Fraud' if fraud_val == 1 else 'Legitimate'
        color = 'red' if fraud_val == 1 else 'green'
        sns.histplot(
            data=df[df[target] == fraud_val], 
            x=amount_col,
            bins=50,
            alpha=0.5,
            label=label,
            color=color,
            ax=ax2,
            log_scale=True
        )
    
    ax2.set_title('Transaction Amount Distribution (Log Scale)')
    ax2.set_xlabel('Amount (Log Scale)')
    ax2.set_ylabel('Count')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "amount_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create amount bins and analyze fraud rate by bin
    df_temp = df.copy()
    
    # Create bins based on percentiles
    percentiles = [0, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    bins = np.percentile(df_temp[amount_col], percentiles)
    bin_labels = [f'P{percentiles[i]}-P{percentiles[i+1]}' for i in range(len(percentiles)-1)]
    
    df_temp['amount_bin'] = pd.cut(df_temp[amount_col], bins=bins, labels=bin_labels)
    
    # Calculate fraud rate by bin
    bin_fraud = df_temp.groupby(['amount_bin', target])[target].count().unstack().fillna(0)
    bin_fraud_rate = bin_fraud[1] / (bin_fraud[0] + bin_fraud[1]) * 100
    
    plt.figure(figsize=(12, 6))
    bin_fraud_rate.plot(kind='bar', color='red')
    plt.title('Fraud Rate by Transaction Amount Percentile')
    plt.xlabel('Amount Percentile Bin')
    plt.ylabel('Fraud Rate (%)')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fraud_by_amount_bin.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save fraud rate by bin
    bin_fraud_df = pd.DataFrame({
        'amount_bin': bin_labels,
        'min_amount': bins[:-1],
        'max_amount': bins[1:],
        'transaction_count': bin_fraud.sum(axis=1).values,
        'fraud_count': bin_fraud[1].values,
        'fraud_rate': bin_fraud_rate.values
    })
    
    bin_fraud_df.to_csv(os.path.join(output_dir, "fraud_by_amount_bin.csv"), index=False)
    
    # Find high fraud rate amount bins
    high_fraud_bins = bin_fraud_df.sort_values('fraud_rate', ascending=False).head(3)
    
    logger.info("Top 3 amount bins by fraud rate:")
    for _, row in high_fraud_bins.iterrows():
        logger.info(f"  {row['amount_bin']}: ${row['min_amount']:.2f}-${row['max_amount']:.2f} " +
                   f"({row['fraud_rate']:.2f}%)")
    
    return bin_fraud_df

def analyze_card_patterns(df, output_dir, card_col='card1', target='isFraud'):
    """
    Analyze card-level patterns and identify high-risk cards.
    
    Args:
        df: DataFrame to analyze
        output_dir: Directory to save visualizations
        card_col: Card identifier column
        target: Target column
    """
    logger.info("Analyzing card-level patterns...")
    
    # Check if required columns exist
    if card_col not in df.columns or target not in df.columns:
        logger.warning(f"Required columns {card_col} or {target} not found")
        return None
    
    # Card transaction frequency
    card_counts = df[card_col].value_counts()
    
    # Plot distribution of transactions per card
    plt.figure(figsize=(10, 6))
    sns.histplot(card_counts, bins=50, kde=True)
    plt.title('Distribution of Transactions per Card')
    plt.xlabel('Number of Transactions')
    plt.ylabel('Number of Cards')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "transactions_per_card.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate fraud rate per card
    card_fraud = df.groupby(card_col)[target].mean() * 100
    
    # Plot distribution of fraud rate per card
    plt.figure(figsize=(10, 6))
    sns.histplot(card_fraud, bins=50, kde=True)
    plt.title('Distribution of Fraud Rate per Card')
    plt.xlabel('Fraud Rate (%)')
    plt.ylabel('Number of Cards')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fraud_rate_per_card.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Identify high-volume cards
    high_volume_cards = card_counts[card_counts >= 10].index.tolist()
    
    # Calculate fraud statistics for high-volume cards
    high_volume_stats = []
    for card in high_volume_cards:
        card_df = df[df[card_col] == card]
        fraud_rate = card_df[target].mean() * 100
        fraud_count = card_df[target].sum()
        transaction_count = len(card_df)
        
        high_volume_stats.append({
            'card_id': card,
            'transaction_count': transaction_count,
            'fraud_count': fraud_count,
            'fraud_rate': fraud_rate,
            'avg_transaction_amount': card_df['TransactionAmt'].mean()
        })
    
    # Convert to DataFrame
    high_volume_df = pd.DataFrame(high_volume_stats)
    
    # Sort by fraud rate
    high_volume_df = high_volume_df.sort_values('fraud_rate', ascending=False)
    
    # Save to CSV
    high_volume_df.to_csv(os.path.join(output_dir, "high_volume_cards.csv"), index=False)
    
    # Plot top cards by fraud rate
    plt.figure(figsize=(12, 8))
    top_cards = high_volume_df.head(20)
    sns.barplot(x='fraud_rate', y='card_id', data=top_cards)
    plt.title('Top 20 High-Volume Cards by Fraud Rate')
    plt.xlabel('Fraud Rate (%)')
    plt.ylabel('Card ID')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_fraud_cards.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Log summary statistics
    logger.info(f"Total cards: {df[card_col].nunique()}")
    logger.info(f"Cards with 10+ transactions: {len(high_volume_cards)}")
    logger.info(f"Max transactions per card: {card_counts.max()}")
    logger.info(f"Average transactions per card: {card_counts.mean():.2f}")
    
    return high_volume_df

def generate_report(output_dir, insights):
    """
    Generate a summary report with key insights.
    
    Args:
        output_dir: Directory to save the report
        insights: Dictionary of insights from analyses
    """
    logger.info("Generating summary report...")
    
    # Create report markdown
    report = [
        "# IEEE-CIS Fraud Detection Dataset Analysis",
        "\n## Dataset Overview",
        f"- Total transactions: {insights.get('total_transactions', 'N/A')}",
        f"- Fraud rate: {insights.get('fraud_rate', 'N/A'):.2f}%",
        f"- Fraud ratio: 1:{insights.get('fraud_ratio_inverse', 'N/A'):.1f}"
    ]
    
    # Key insights
    report.append("\n## Key Insights")
    
    # Add time insights
    if 'peak_fraud_hour' in insights:
        report.append(f"- Peak fraud hour: {insights['peak_fraud_hour']} ({insights['peak_fraud_rate']:.2f}%)")
    
    if 'weekday_fraud_rate' in insights and 'weekend_fraud_rate' in insights:
        report.append(f"- Weekday fraud rate: {insights['weekday_fraud_rate']:.2f}%, Weekend fraud rate: {insights['weekend_fraud_rate']:.2f}%")
    
    # Add amount insights
    if 'high_fraud_amount_bins' in insights:
        report.append("- High fraud rate amount bins:")
        for bin_info in insights['high_fraud_amount_bins']:
            report.append(f"  * {bin_info['bin']}: ${bin_info['min']:.2f}-${bin_info['max']:.2f} ({bin_info['rate']:.2f}%)")
    
    # Add feature insights
    if 'top_correlated_features' in insights:
        report.append("- Top features correlated with fraud:")
        for feature in insights['top_correlated_features']:
            report.append(f"  * {feature['name']}: {feature['correlation']:.4f}")
    
    # Add data quality insights
    if 'missing_values' in insights:
        report.append(f"- Columns with missing values: {insights['missing_values']}")
        report.append(f"- Top column with missing values: {insights['top_missing_column']} ({insights['top_missing_percent']:.2f}%)")
    
    # Save report
    report_path = os.path.join(output_dir, "data_analysis_report.md")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    logger.info(f"Report saved to {report_path}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Prepare and analyze IEEE-CIS fraud detection dataset")
    parser.add_argument("--data-dir", default="./data/raw", help="Directory containing dataset")
    parser.add_argument("--output-dir", default="./artifacts/data_analysis", help="Directory to save analysis results")
    parser.add_argument("--sample", type=int, default=None, help="Sample size (None for all data)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {args.data_dir}...")
    df = load_data(args.data_dir, train=True)
    
    # Downsample if requested
    if args.sample is not None and args.sample < len(df):
        logger.info(f"Sampling {args.sample} transactions...")
        df = df.sample(args.sample, random_state=args.seed)
    
    logger.info(f"Dataset shape: {df.shape}")
    
    # Save a sample
    df.head(100).to_csv(os.path.join(args.output_dir, "data_sample.csv"), index=False)
    
    # Run analyses
    dtype_df = analyze_dtypes(df, args.output_dir)
    missing_df = analyze_missing_values(df, args.output_dir)
    fraud_stats = analyze_fraud_distribution(df, args.output_dir)
    corr_df = analyze_numeric_features(df, args.output_dir)
    cat_df = analyze_categorical_features(df, args.output_dir)
    time_patterns, weekday_patterns = analyze_time_patterns(df, args.output_dir)
    amount_patterns = analyze_amount_patterns(df, args.output_dir)
    card_patterns = analyze_card_patterns(df, args.output_dir)
    
    # Generate insights dictionary
    insights = {}
    
    if fraud_stats:
        insights.update({
            'total_transactions': fraud_stats['total_transactions'],
            'fraud_rate': fraud_stats['fraud_percent'],
            'fraud_ratio_inverse': 1 / fraud_stats['fraud_ratio']
        })
    
    if time_patterns is not None:
        peak_hour = time_patterns['fraud_rate'].idxmax()
        insights.update({
            'peak_fraud_hour': peak_hour,
            'peak_fraud_rate': time_patterns.loc[peak_hour, 'fraud_rate']
        })
    
    if weekday_patterns is not None:
        insights.update({
            'weekday_fraud_rate': weekday_patterns.iloc[:5]['fraud_rate'].mean(),
            'weekend_fraud_rate': weekday_patterns.iloc[5:]['fraud_rate'].mean()
        })
    
    if amount_patterns is not None:
        high_fraud_bins = amount_patterns.sort_values('fraud_rate', ascending=False).head(3)
        insights['high_fraud_amount_bins'] = [
            {
                'bin': row['amount_bin'],
                'min': row['min_amount'],
                'max': row['max_amount'],
                'rate': row['fraud_rate']
            }
            for _, row in high_fraud_bins.iterrows()
        ]
    
    if corr_df is not None:
        insights['top_correlated_features'] = [
            {'name': row['feature'], 'correlation': row['correlation']}
            for _, row in corr_df.head(5).iterrows()
        ]
    
    if missing_df is not None and not missing_df.empty:
        insights.update({
            'missing_values': len(missing_df),
            'top_missing_column': missing_df.iloc[0]['column'],
            'top_missing_percent': missing_df.iloc[0]['percentage']
        })
    
    # Generate report
    generate_report(args.output_dir, insights)
    
    logger.info(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()