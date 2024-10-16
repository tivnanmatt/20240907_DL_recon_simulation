import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Function to load AUC results from CSV
def load_auc_data(datasets, original_ovr_file, original_ovo_file):
    auc_data = []
    
    # Load the original AUC data for OvR and OvO separately
    original_ovr_df = pd.read_csv(original_ovr_file)
    original_ovo_df = pd.read_csv(original_ovo_file)
    
    # Add a 'Reconstruction Type' column for original data
    original_ovr_df['Reconstruction Type'] = 'Original'
    original_ovo_df['Reconstruction Type'] = 'Original'
    
    # Add a 'Comparison' column for original data
    original_ovr_df['Comparison'] = 'OvR'
    original_ovo_df['Comparison'] = 'OvO'
    
    # Append original data to the list
    auc_data.append(original_ovr_df)
    auc_data.append(original_ovo_df)
    
    # Load reconstructed AUC data
    for dataset_name in datasets:
        # Define file paths for each dataset
        ovr_file = f'results/ovr_auc_{dataset_name}_reconstructions.csv'
        ovo_file = f'results/ovo_auc_{dataset_name}_reconstructions.csv'
        
        # Load One-vs-Rest (OvR) and One-vs-One (OvO) AUC results
        ovr_df = pd.read_csv(ovr_file)
        ovo_df = pd.read_csv(ovo_file)
        
        # Prepare a unified dataframe for plotting
        ovr_df['Comparison'] = 'OvR'
        ovo_df['Comparison'] = 'OvO'
        
        # Add dataset type to dataframe
        ovr_df['Reconstruction Type'] = dataset_name
        ovo_df['Reconstruction Type'] = dataset_name
        
        # Concatenate both AUC types into one dataframe
        auc_data.append(pd.concat([ovr_df[['Hemorrhage Type', 'AUC', 'Comparison', 'Reconstruction Type']], 
                                   ovo_df[['Hemorrhage Type Pair', 'AUC', 'Comparison', 'Reconstruction Type']]], 
                                   axis=0, ignore_index=True))
    
    # Combine all datasets into a single dataframe
    combined_auc_df = pd.concat(auc_data, axis=0, ignore_index=True)
    
    return combined_auc_df

# Function to create grouped bar chart for OvO AUC results
def plot_grouped_bar_auc_ovo(auc_data):
    # Ensure column names are correctly mapped
    auc_data = auc_data.rename(columns={
        'Hemorrhage Type': 'Label',
        'Hemorrhage Type Pair': 'Label'
    })

    # Filter for OvO comparisons
    ovr_data = auc_data[auc_data['Comparison'] == 'OvR']

    # Drop any duplicate columns
    ovr_data = ovr_data.loc[:, ~ovr_data.columns.duplicated()]

    # Print the DataFrame and column names for debugging
    print(ovr_data.head())
    print(ovr_data.columns)

    # Check unique values in relevant columns
    print(ovr_data['Label'].dropna().unique())
    print(ovr_data['Reconstruction Type'].dropna().unique())

    plt.figure(figsize=(9, 6))
    
    # Use Seaborn to plot the grouped bar chart
    sns.barplot(data=ovr_data, x='Label', y='AUC', hue='Reconstruction Type', palette='Set2', ci=None)
    
    # Customize the plot
    plt.title('OvO AUC Scores by Hemorrhage and Reconstruction', fontsize=16)
    plt.xlabel('Hemorrhage Type')
    plt.ylabel('AUC Score')
    plt.legend(title='Reconstruction Type', title_fontsize='13', fontsize='11')
    plt.xticks(rotation=10)  # Rotate x-axis labels for better readability
    plt.ylim(0.8, 1)  # AUC scores are between 0 and 1
    
    plt.savefig('figures/ovo_auc_grouped_bar_charts_by_label.png', dpi=300)
    # plt.show()


    # now same for ovr
    plt.figure(figsize=(9, 6))

    sns.barplot(data=ovr_data, x='Label', y='AUC', hue='Reconstruction Type', palette='Set2', ci=None)

    plt.title('OvR AUC Scores by Hemorrhage and Reconstruction', fontsize=16)
    plt.xlabel('Hemorrhage Type')
    plt.ylabel('AUC Score')
    plt.legend(title='Reconstruction Type', title_fontsize='13', fontsize='11')
    plt.xticks(rotation=10)
    plt.ylim(0.8, 1)

    plt.savefig('figures/ovr_auc_grouped_bar_charts_by_label.png', dpi=300)
    # plt.show()

# Main function to process datasets and generate plots
def main():
    datasets = ['FBP', 'MBIR', 'DLR']  # List of dataset names
    original_ovr_file = 'results/ovr_auc_128.csv'  # Path to the original OvR AUC data file
    original_ovo_file = 'results/ovo_auc_128.csv'  # Path to the original OvO AUC data file
    
    # Load AUC data
    auc_data = load_auc_data(datasets, original_ovr_file, original_ovo_file)
    
    # Plot grouped bar chart for OvO AUC results
    plot_grouped_bar_auc_ovo(auc_data)
    print("Grouped bar chart with OvO AUC data by label saved as 'figures/ovr_auc_grouped_bar_charts_by_label.png'.")

if __name__ == "__main__":
    main()
