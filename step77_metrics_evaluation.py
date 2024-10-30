import pandas as pd

# Load the CSV file
df = pd.read_csv('results/combined_metrics_dlr.csv')

# Clean up 'mean_occlusion' and 'variance_occlusion' columns from tensor formatting
df['mean_occlusion'] = df['mean_occlusion'].apply(lambda x: float(str(x).split('(')[-1].split(',')[0]))
df['variance_occlusion'] = df['variance_occlusion'].apply(lambda x: float(str(x).split('(')[-1].split(',')[0]))

# Define columns to calculate the mean and variance for each label by saliency type
saliency_metrics = {
    'confidence_score': 'confidence_score',
    'mean_saliency_vanilla': 'saliency_vanilla',
    'variance_saliency_vanilla': 'saliency_vanilla',
    'mean_guided_backprop': 'guided_backprop',
    'variance_guided_backprop': 'guided_backprop',
    'mean_gradcam_reg': 'gradcam_reg',
    'variance_gradcam_reg': 'gradcam_reg',
    'mean_gradcam_plus': 'gradcam_plus',
    'variance_gradcam_plus': 'gradcam_plus',
    'mean_gradcam_full': 'gradcam_full',
    'variance_gradcam_full': 'gradcam_full',
    'mean_occlusion': 'occlusion',
    'variance_occlusion': 'occlusion'
}

# Initialize a list to store results
results = []

# Loop through each label to calculate mean and variance for each saliency type
for label in df['label'].unique():
    label_df = df[df['label'] == label]
    label_summary = {'label': label}
    
    for saliency_type in set(saliency_metrics.values()):
        # Extract mean and variance columns for the current saliency type
        mean_cols = [col for col, sal_type in saliency_metrics.items() if sal_type == saliency_type and 'mean' in col]
        var_cols = [col for col, sal_type in saliency_metrics.items() if sal_type == saliency_type and 'variance' in col]
        
        # Calculate mean for the confidence score separately and store in the summary dictionary
        if saliency_type == 'confidence_score':
            label_summary['confidence_score_mean'] = label_df['confidence_score'].mean()
        else:
            # Calculate mean and variance for each saliency type and store in the summary dictionary
            label_summary[f'{saliency_type}_mean'] = label_df[mean_cols].mean().mean()
            label_summary[f'{saliency_type}_variance'] = label_df[var_cols].mean().mean()
    
    # Append the summary for this label to the results list
    results.append(label_summary)

# Convert the list of dictionaries to a DataFrame and save it to a new CSV file
results_df = pd.DataFrame(results)

results_df.to_csv('results/saliency_summary_dlr.csv', index=False)
print("Analysis complete. Results saved to 'saliency_summary_original.csv'")
