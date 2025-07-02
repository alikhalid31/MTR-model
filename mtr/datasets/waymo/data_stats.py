import pandas as pd
import matplotlib.pyplot as plt


def get_valid_timestamps_ego(df,ego=True):
    """
    Get the count of each agent type in the DataFrame.
    """
    df_filtered = df[df['to_predict']==ego]

    df_past_stamps= df['past_valid_stamps'].value_counts().sort_index()
    df_future_stamps= df['future_valid_stamps'].value_counts().sort_index()
    df_current_stamps= df['current_valid_stamps'].value_counts().sort_index()
    return df_past_stamps, df_future_stamps, df_current_stamps

def get_agent_and_ego_count(df,ego_col='ego', agents_col='agents'):
    """
    Get the count of each agent type in the DataFrame.
    """
    grouped_counts = df.groupby('scenario')['to_predict'].value_counts().unstack(fill_value=0)
    grouped_counts = grouped_counts.rename(columns={True: ego_col, False: agents_col})
    # print(grouped_counts)
    return grouped_counts

def get_frequency_of_unique_values(df, column_name):
    """
    Get the frequency of unique values in a specified column.
    """
    return df[column_name].value_counts().sort_index()


def plot_hitogram(df, xlabel='', ylabel='', title='', output_file='histogram.png'):
    """
    Plot a histogram (bar chart) from a Series or DataFrame.
    """
    plt.figure()  # Create a new figure to avoid overlaps
    df.plot(kind='bar', figsize=(10, 5))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()  # Optional: auto-adjust layout

    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()  #


def get_ego_histogram(df, column_name, xlabel='', ylabel='', title='', output_file='histogram.png', prefix=''):

    # Get frequency of unique values in 'scenario' column
    hist_ego = get_frequency_of_unique_values(df, column_name)

    plot_hitogram(hist_ego,
    xlabel=xlabel,
    ylabel=ylabel,
    title=title,
    output_file=prefix+output_file)

def get_agent_histogram(df, column_name, xlabel='', ylabel='', title='', output_file='histogram.png', prefix='', filter=0):

    if filter > 0:
        # Filter the DataFrame to only include rows where the 'agents' column is greater than the filter value
        df = df[df['ego'] == filter]
    # Bin directly on the agents column in df_count
    bins = [0, 10, 20, 30, 50, 100, 200, 1000]  # 8 edges
    labels = ['0–10', '11–20', '21–30', '31–50', '51–100', '101–200', '200+']  # 7 labels
    df['agents_binned'] = pd.cut(df['agents'], bins=bins, labels=labels, right=False)

    hist_agents_binned = get_frequency_of_unique_values(df,'agents_binned')
    plot_hitogram(hist_agents_binned, 
    xlabel=xlabel,
    ylabel=ylabel,
    title=title,
    output_file=prefix+output_file)


def get_interactions_histogram(df, column_name, xlabel='', ylabel='', title='', output_file='histogram.png', prefix='', filter=0):

    if filter > 0:
        # Filter the DataFrame to only include rows where the 'agents' column is greater than the filter value
        df = df[df['to_predict_count'] == filter]

    hist_agents_binned = get_frequency_of_unique_values(df,column_name)
    plot_hitogram(hist_agents_binned, 
    xlabel=xlabel,
    ylabel=ylabel,
    title=title,
    output_file=prefix+output_file)

def get_count_histograms(df):
    """
    Get histograms for the count of agents and ego vehicles in the DataFrame.
    """
    # Get agent and ego counts
    df_count = get_agent_and_ego_count(df, ego_col='ego', agents_col='agents')

    # save histograms
    get_ego_histogram(df_count, 'ego',
        xlabel='Number of Ego Tracks',
        ylabel='Number of Scenarios',
        title='Histogram of Ego Vehicle Count per Scenario',
        output_file='ego_histogram.png', prefix='validation_dataset_')

    get_agent_histogram(df_count, 'agents',
        xlabel='Number of Agent Tracks',
        ylabel='Number of Scenarios',
        title='Histogram of Agent Count per Scenario',
        output_file='agent_histogram.png', prefix='validation_dataset_')
    
    for i in range(1,9):
        get_agent_histogram(df_count, 'agents',
            xlabel='Number of Agent Tracks',
            ylabel='Number of Scenarios',
            title='Histogram of Agent Count per Scenario',
            output_file='agent_histogram.png', prefix='validation_dataset_ego'+str(i)+'_', filter=i)
    
    
def get_timestamps_histograms(df):
    df_filtered_ego = df[df['to_predict']== True]
    get_ego_histogram(df_filtered_ego, 'past_valid_stamps',
    xlabel='Number of valid timestamps',
    ylabel='Number of ego vehicles',
    title='Histogram of past valid timestamps for ego vehicles',
    output_file='ego_past_time_histogram.png', prefix='validation_dataset_')

    get_ego_histogram(df_filtered_ego, 'future_valid_stamps',
    xlabel='Number of valid timestamps',
    ylabel='Number of ego vehicles',
    title='Histogram of future valid timestamps for ego vehicles',
    output_file='ego_future_time_histogram.png', prefix='validation_dataset_')

    df_filtered_agents = df[df['to_predict']== False]
    get_ego_histogram(df_filtered_agents, 'past_valid_stamps',
    xlabel='Number of valid timestamps',
    ylabel='Number of agent vehicles',
    title='Histogram of past valid timestamps for agent vehicles',
    output_file='agent_past_time_histogram.png', prefix='validation_dataset_')

    get_ego_histogram(df_filtered_agents, 'future_valid_stamps',
    xlabel='Number of valid timestamps',
    ylabel='Number of agent vehicles',
    title='Histogram of future valid timestamps for agent vehicles',
    output_file='agent_future_time_histogram.png', prefix='validation_dataset_')


def summarize_by_scenario(df):
    # Ensure boolean column is actually boolean (in case it's read as string)
    df['to_predict'] = df['to_predict'].astype(bool)

    # Group by scenario
    grouped = df.groupby('scenario').agg(
        to_predict_count=('to_predict', lambda x: x.sum()),     # Count of True values
        interactions_count=('interactions_count', 'first')      # Assumes same value for the group
    ).reset_index()

    return grouped

def get_interaction_histograms(df):
    df_summarized = summarize_by_scenario(df)

    for i in range(1,9):
        get_interactions_histogram(df_summarized, 'interactions_count',
            xlabel='Number of interaction Tracks',
            ylabel='Number of Scenarios',
            title='Histogram of interaction Count per Scenario',
            output_file='interaction_histogram.png', prefix='validation_dataset_ego'+str(i)+'_', filter=i)
    




    
def main():
    # Example usage
    csv_file_path = '../../../data/waymo/processed_scenarios_validation.csv'
    df = pd.read_csv(csv_file_path)
    get_count_histograms(df)

    get_timestamps_histograms(df)

    get_interaction_histograms(df)

    





if __name__ == "__main__":  
    main()




