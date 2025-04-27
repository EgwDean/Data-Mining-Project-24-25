import pandas as pd
import pathlib
import matplotlib as plt
import io

# Define the dataset path
input_path = pathlib.Path(__file__).parent.parent / 'data' / 'small_data.csv'

# Check if the file exists
if not input_path.exists():
    raise FileNotFoundError(f'The file {input_path} does not exist.')

# Read the CSV file into a DataFrame
df = pd.read_csv(input_path)

# Deifine the path for the info file
info_path = pathlib.Path(__file__).parent.parent / 'data' / 'info.csv'

# Save the info to a csv file via buffer
buffer = io.StringIO()
df.info(buf=buffer)
info_str = buffer.getvalue()
info_df = pd.DataFrame({'Info': info_str.split('\n')})
info_df.to_csv(info_path, index=False)
print(f'DataFrame info saved to {info_path}')

# Define the path for the statistics file
statistics_path = pathlib.Path(__file__).parent.parent / 'data' / 'statistics.csv'

# Save the statistics to an excel file
df.describe(include='all').to_csv(statistics_path)
print(f'DataFrame statistics saved to {statistics_path}')

# Define a list of categorical columns
categorical_columns = [
    'Label', 'Traffic Type', 'Traffic Subtype', 
    'Timestamp', 'Flow ID', 'Src IP', 'Dst IP'
    ]

# Define the paths for the histograms, boxplots and barplots
histograms_path = pathlib.Path(__file__).parent.parent / 'data' / 'histograms'
boxplots_path = pathlib.Path(__file__).parent.parent / 'data' / 'boxplots'
barplots_path = pathlib.Path(__file__).parent.parent / 'data' / 'barplots'

# Ensure the directories for plots exist
histograms_path.mkdir(parents=True, exist_ok=True)
boxplots_path.mkdir(parents=True, exist_ok=True)
barplots_path.mkdir(parents=True, exist_ok=True)

# Create histograms, boxplots, and barplots for each column
for column in df.columns:
    # Sanitize column name for file paths
    sanitized_column = column.replace(' ', '_').replace('/', '_')

    if column in categorical_columns:
        # Bar plot for categorical data
        bar_plot_path = barplots_path / f'{sanitized_column}_barplot.png'
        df[column].value_counts().plot(kind='bar').get_figure().savefig(bar_plot_path)
        plt.pyplot.close()
    else:
        # Histogram for numeric data
        histogram_path = histograms_path / f'{sanitized_column}_histogram.png'
        df[column].hist().get_figure().savefig(histogram_path)
        plt.pyplot.close() 

        # Boxplot for detecting outliers
        boxplot_path = boxplots_path / f'{sanitized_column}_boxplot.png'
        df.boxplot(column=column).get_figure().savefig(boxplot_path)
        plt.pyplot.close()

print(f'Bar plots saved to {barplots_path}')
print(f'Histograms saved to {histograms_path}')
print(f'Boxplots saved to {boxplots_path}')

# Drop categorical columns from the DataFrame
df.drop(columns=categorical_columns, inplace=True, errors='ignore')

# Define the path for the column correlation matrix file
column_correlation_path = pathlib.Path(__file__).parent.parent / 'data' / 'column_correlation_matrix.xlsx'

# Compute the correlation matrix for the remaining numeric columns
column_correlation_matrix = df.corr()

# Rank the correlation pairs
correlation_pairs = (
    column_correlation_matrix
    .unstack()
    .reset_index()
    .rename(columns={0: 'Correlation', 'level_0': 'Variable 1', 'level_1': 'Variable 2'})
)

# Remove duplicate pairs and self-correlations
correlation_pairs = correlation_pairs[correlation_pairs['Variable 1'] != correlation_pairs['Variable 2']]
correlation_pairs = correlation_pairs.drop_duplicates(subset=['Correlation'], keep='first')

# Sort by absolute correlation values in descending order
correlation_pairs['Absolute Correlation'] = correlation_pairs['Correlation'].abs()
correlation_pairs = correlation_pairs.sort_values(by='Absolute Correlation', ascending=False)

# Save the ranked correlation pairs to an Excel file
with pd.ExcelWriter(column_correlation_path) as writer:
    column_correlation_matrix.to_excel(writer, sheet_name='Correlation Matrix')
    correlation_pairs.to_excel(writer, sheet_name='Ranked Correlations', index=False)

print(f'Column correlation matrix and ranked correlations saved to {column_correlation_path}')

# Define the path for the statistics correlation matrix file
statistics_correlation_path = pathlib.Path(__file__).parent.parent / 'data' / 'statistics_correlation_matrix.xlsx'

# Compute the correlation matrix for the statistics
statistics_correlation_matrix = df.describe().drop('count').transpose().corr()

# Rank the correlation pairs for the statistics
statistics_correlation_pairs = (
    statistics_correlation_matrix
    .unstack()
    .reset_index()
    .rename(columns={0: 'Correlation', 'level_0': 'Variable 1', 'level_1': 'Variable 2'})
)

# Remove duplicate pairs and self-correlations
statistics_correlation_pairs = statistics_correlation_pairs[statistics_correlation_pairs['Variable 1'] != statistics_correlation_pairs['Variable 2']]
statistics_correlation_pairs = statistics_correlation_pairs.drop_duplicates(subset=['Correlation'], keep='first')

# Sort by absolute correlation values in descending order
statistics_correlation_pairs['Absolute Correlation'] = statistics_correlation_pairs['Correlation'].abs()
statistics_correlation_pairs = statistics_correlation_pairs.sort_values(by='Absolute Correlation', ascending=False)

# Save the ranked correlation pairs for the statistics to an Excel file
with pd.ExcelWriter(statistics_correlation_path) as writer:
    statistics_correlation_matrix.to_excel(writer, sheet_name='Correlation Matrix')
    statistics_correlation_pairs.to_excel(writer, sheet_name='Ranked Correlations', index=False)

print(f'Statistics correlation matrix and ranked correlations saved to {statistics_correlation_path}')