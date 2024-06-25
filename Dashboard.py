import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Function to generate a dashboard and report
def generate_dashboard_and_report(data, report_path):
    df = pd.DataFrame(data, columns=['Frame', 'PeopleCount', 'MaxPeopleCount', 'Timestamp'])
    summary = df.describe()

    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Frame', y='PeopleCount', data=df, label='People Count')
    sns.lineplot(x='Frame', y='MaxPeopleCount', data=df, label='Max People Count', linestyle='--')
    plt.xlabel('Frame')
    plt.ylabel('Count')
    plt.title('People Count Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(report_path.replace('.txt', '_plot.png'))
    plt.show()

    with open(report_path, 'w') as f:
        f.write("Summary Statistics:\n")
        f.write(summary.to_string())
        f.write("\n\nData:\n")
        f.write(df.to_string())

# Load frame data
with open('frame_data.pkl', 'rb') as f:
    frame_data = pickle.load(f)

# Generate the dashboard
report_path = 'report.txt'
generate_dashboard_and_report(frame_data, report_path)
