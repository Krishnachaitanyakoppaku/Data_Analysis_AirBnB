"""
Exploratory Data Analysis functions for NYC Airbnb analysis.

This module contains functions for creating visualizations and summaries
to explore patterns in the Airbnb dataset including price distributions,
geographic patterns, and market analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import plugins
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


def setup_plotting_style():
    """Set up consistent plotting style for all visualizations."""
    plt.style.use('default')
    sns.set_palette('husl')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10


def plot_listings_by_borough(df: pd.DataFrame, save_path: str = 'outputs/figures/listings_by_borough.png') -> None:
    """
    Create a bar plot showing number of listings by borough.
    
    Args:
        df (pd.DataFrame): Cleaned dataset
        save_path (str): Path to save the plot
    """
    setup_plotting_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Count plot
    borough_counts = df['neighbourhood_group'].value_counts()
    colors = sns.color_palette('husl', len(borough_counts))
    
    bars = ax1.bar(borough_counts.index, borough_counts.values, color=colors)
    ax1.set_title('Number of Listings by Borough', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Borough')
    ax1.set_ylabel('Number of Listings')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 100,
                f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    ax2.pie(borough_counts.values, labels=borough_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax2.set_title('Market Share by Borough', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Listings by borough plot saved to {save_path}")


def plot_price_distribution(df: pd.DataFrame, save_path: str = 'outputs/figures/price_distribution.png') -> None:
    """
    Create price distribution plots with both linear and log scales.
    
    Args:
        df (pd.DataFrame): Cleaned dataset
        save_path (str): Path to save the plot
    """
    setup_plotting_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Linear scale histogram
    ax1.hist(df['price'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(df['price'].mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: ${df["price"].mean():.0f}')
    ax1.axvline(df['price'].median(), color='orange', linestyle='--', linewidth=2,
                label=f'Median: ${df["price"].median():.0f}')
    ax1.set_title('Price Distribution (Linear Scale)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Price ($)')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Log scale histogram
    ax2.hist(df['price'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_yscale('log')
    ax2.axvline(df['price'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: ${df["price"].mean():.0f}')
    ax2.axvline(df['price'].median(), color='orange', linestyle='--', linewidth=2,
                label=f'Median: ${df["price"].median():.0f}')
    ax2.set_title('Price Distribution (Log Scale)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Price ($)')
    ax2.set_ylabel('Frequency (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Price distribution plot saved to {save_path}")


def plot_price_by_room_type(df: pd.DataFrame, save_path: str = 'outputs/figures/price_by_room_type.png') -> None:
    """
    Create box plots showing price distribution by room type.
    
    Args:
        df (pd.DataFrame): Cleaned dataset
        save_path (str): Path to save the plot
    """
    setup_plotting_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Box plot
    sns.boxplot(data=df, x='room_type', y='price', ax=ax1)
    ax1.set_title('Price Distribution by Room Type', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Room Type')
    ax1.set_ylabel('Price ($)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Violin plot for better distribution visualization
    sns.violinplot(data=df, x='room_type', y='price', ax=ax2)
    ax2.set_title('Price Density by Room Type', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Room Type')
    ax2.set_ylabel('Price ($)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Price by room type plot saved to {save_path}")


def plot_top_neighborhoods(df: pd.DataFrame, top_n: int = 10, 
                          save_path: str = 'outputs/figures/top_neighborhoods.png') -> None:
    """
    Create plots showing top neighborhoods by number of listings and average price.
    
    Args:
        df (pd.DataFrame): Cleaned dataset
        top_n (int): Number of top neighborhoods to show
        save_path (str): Path to save the plot
    """
    setup_plotting_style()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top neighborhoods by listing count
    top_neighborhoods_count = df['neighbourhood'].value_counts().head(top_n)
    colors1 = sns.color_palette('viridis', len(top_neighborhoods_count))
    
    bars1 = ax1.barh(range(len(top_neighborhoods_count)), top_neighborhoods_count.values, color=colors1)
    ax1.set_yticks(range(len(top_neighborhoods_count)))
    ax1.set_yticklabels(top_neighborhoods_count.index)
    ax1.set_title(f'Top {top_n} Neighborhoods by Number of Listings', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Number of Listings')
    
    # Add value labels
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(width + 20, bar.get_y() + bar.get_height()/2,
                f'{int(width):,}', ha='left', va='center', fontweight='bold')
    
    # Top neighborhoods by average price (min 50 listings)
    neighborhood_prices = df.groupby('neighbourhood').agg({
        'price': ['count', 'mean']
    }).round(2)
    neighborhood_prices.columns = ['count', 'avg_price']
    neighborhood_prices = neighborhood_prices[neighborhood_prices['count'] >= 50]
    top_expensive = neighborhood_prices.nlargest(top_n, 'avg_price')
    
    colors2 = sns.color_palette('plasma', len(top_expensive))
    bars2 = ax2.barh(range(len(top_expensive)), top_expensive['avg_price'], color=colors2)
    ax2.set_yticks(range(len(top_expensive)))
    ax2.set_yticklabels(top_expensive.index)
    ax2.set_title(f'Top {top_n} Most Expensive Neighborhoods (min 50 listings)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Average Price ($)')
    
    # Add value labels
    for i, bar in enumerate(bars2):
        width = bar.get_width()
        ax2.text(width + 5, bar.get_y() + bar.get_height()/2,
                f'${int(width)}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Top neighborhoods plot saved to {save_path}")


def plot_price_vs_reviews(df: pd.DataFrame, save_path: str = 'outputs/figures/price_vs_reviews.png') -> None:
    """
    Create scatter plot showing relationship between price and number of reviews.
    
    Args:
        df (pd.DataFrame): Cleaned dataset
        save_path (str): Path to save the plot
    """
    setup_plotting_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Price vs Number of Reviews
    ax1.scatter(df['number_of_reviews'], df['price'], alpha=0.5, s=20, color='steelblue')
    
    # Add trendline
    z = np.polyfit(df['number_of_reviews'], df['price'], 1)
    p = np.poly1d(z)
    ax1.plot(df['number_of_reviews'], p(df['number_of_reviews']), "r--", alpha=0.8, linewidth=2)
    
    ax1.set_title('Price vs Number of Reviews', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Number of Reviews')
    ax1.set_ylabel('Price ($)')
    ax1.grid(True, alpha=0.3)
    
    # Calculate correlation
    correlation = df['price'].corr(df['number_of_reviews'])
    ax1.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax1.transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Price vs Reviews per Month
    ax2.scatter(df['reviews_per_month'], df['price'], alpha=0.5, s=20, color='darkorange')
    
    # Add trendline
    z2 = np.polyfit(df['reviews_per_month'], df['price'], 1)
    p2 = np.poly1d(z2)
    ax2.plot(df['reviews_per_month'], p2(df['reviews_per_month']), "r--", alpha=0.8, linewidth=2)
    
    ax2.set_title('Price vs Reviews per Month', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Reviews per Month')
    ax2.set_ylabel('Price ($)')
    ax2.grid(True, alpha=0.3)
    
    # Calculate correlation
    correlation2 = df['price'].corr(df['reviews_per_month'])
    ax2.text(0.05, 0.95, f'Correlation: {correlation2:.3f}', transform=ax2.transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Price vs reviews plot saved to {save_path}")


def plot_folium_map(df: pd.DataFrame, limit: int = 2000, 
                   save_path: str = 'outputs/figures/map.html') -> None:
    """
    Create an interactive Folium map showing listing locations colored by price.
    
    Args:
        df (pd.DataFrame): Cleaned dataset
        limit (int): Maximum number of markers to display
        save_path (str): Path to save the HTML map
    """
    # Sample data to avoid performance issues
    df_sample = df.sample(n=min(limit, len(df)), random_state=42)
    
    # Create base map centered on NYC
    nyc_center = [40.7128, -74.0060]
    m = folium.Map(location=nyc_center, zoom_start=11, tiles='OpenStreetMap')
    
    # Define color mapping for price ranges
    price_ranges = [
        (0, 75, 'green', 'Budget ($0-75)'),
        (75, 150, 'blue', 'Mid-range ($75-150)'),
        (150, 250, 'orange', 'Premium ($150-250)'),
        (250, float('inf'), 'red', 'Luxury ($250+)')
    ]
    
    # Add markers for each price range
    for min_price, max_price, color, label in price_ranges:
        if max_price == float('inf'):
            subset = df_sample[df_sample['price'] >= min_price]
        else:
            subset = df_sample[(df_sample['price'] >= min_price) & (df_sample['price'] < max_price)]
        
        if len(subset) > 0:
            marker_cluster = plugins.MarkerCluster(name=label).add_to(m)
            
            for idx, row in subset.iterrows():
                popup_text = f"""
                <b>{row['name'][:50]}...</b><br>
                Price: ${row['price']}/night<br>
                Room Type: {row['room_type']}<br>
                Borough: {row['neighbourhood_group']}<br>
                Neighborhood: {row['neighbourhood']}<br>
                Reviews: {row['number_of_reviews']}
                """
                
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=f"${row['price']}/night - {row['room_type']}",
                    icon=folium.Icon(color=color, icon='home')
                ).add_to(marker_cluster)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add a legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 150px; height: 120px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><b>Price Ranges</b></p>
    <p><i class="fa fa-circle" style="color:green"></i> Budget ($0-75)</p>
    <p><i class="fa fa-circle" style="color:blue"></i> Mid-range ($75-150)</p>
    <p><i class="fa fa-circle" style="color:orange"></i> Premium ($150-250)</p>
    <p><i class="fa fa-circle" style="color:red"></i> Luxury ($250+)</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    m.save(save_path)
    print(f"Interactive map saved to {save_path} (showing {len(df_sample)} listings)")


def correlation_matrix(df: pd.DataFrame, save_path: str = 'outputs/figures/correlation_heatmap.png') -> None:
    """
    Create correlation heatmap for numeric variables.
    
    Args:
        df (pd.DataFrame): Cleaned dataset
        save_path (str): Path to save the plot
    """
    setup_plotting_style()
    
    # Select numeric columns for correlation
    numeric_cols = [
        'price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
        'calculated_host_listings_count', 'availability_365', 'price_per_minimum_night',
        'active_days', 'reviews_per_year', 'host_productivity'
    ]
    
    # Filter columns that exist in the dataframe
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    # Calculate correlation matrix
    corr_matrix = df[available_cols].corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, fmt='.2f', cbar_kws={"shrink": .8})
    
    plt.title('Correlation Matrix of Numeric Variables', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Correlation heatmap saved to {save_path}")


def create_summary_by_borough(df: pd.DataFrame, save_path: str = 'outputs/summary_by_borough.csv') -> pd.DataFrame:
    """
    Create comprehensive summary statistics by borough.
    
    Args:
        df (pd.DataFrame): Cleaned dataset
        save_path (str): Path to save the CSV summary
        
    Returns:
        pd.DataFrame: Summary statistics by borough
    """
    summary = df.groupby('neighbourhood_group').agg({
        'price': ['count', 'mean', 'median', 'std', 'min', 'max'],
        'minimum_nights': ['mean', 'median'],
        'number_of_reviews': ['mean', 'median'],
        'reviews_per_month': ['mean', 'median'],
        'availability_365': ['mean', 'median'],
        'calculated_host_listings_count': ['mean', 'median'],
        'active_days': ['mean', 'median']
    }).round(2)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    
    # Add market share
    total_listings = len(df)
    summary['market_share_percent'] = (summary['price_count'] / total_listings * 100).round(1)
    
    # Save summary
    summary.to_csv(save_path)
    print(f"Borough summary saved to {save_path}")
    
    return summary


def run_eda_pipeline(df: pd.DataFrame) -> None:
    """
    Run the complete EDA pipeline with all visualizations.
    
    Args:
        df (pd.DataFrame): Cleaned dataset
    """
    print("=== STARTING EDA PIPELINE ===")
    
    # Create all plots
    plot_listings_by_borough(df)
    plot_price_distribution(df)
    plot_price_by_room_type(df)
    plot_top_neighborhoods(df, top_n=10)
    plot_price_vs_reviews(df)
    plot_folium_map(df, limit=2000)
    correlation_matrix(df)
    
    # Create summary statistics
    summary = create_summary_by_borough(df)
    
    print("\n=== EDA PIPELINE COMPLETE ===")
    print("All visualizations and summaries have been created!")
    
    return summary