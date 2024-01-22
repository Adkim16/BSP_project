import os
import pandas as pd
import matplotlib.pyplot as plt

experiments_folder = "C:/Users/Nico/Documents/Python Projects/BSP/experiments/"
output_folder = "BSP/results/"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"\nFolder \"{output_folder}\" has been created.")

dfs = []

for folder in os.listdir(experiments_folder):
    folder_path = os.path.join(experiments_folder, folder)

    if os.path.isdir(folder_path):
        result_file_path = os.path.join(folder_path, 'results.csv')

        if os.path.isfile(result_file_path):
            result_df = pd.read_csv(result_file_path, index_col=0, header=None, names=[folder]).T

            result_df['sample_id'] = folder

            dfs.append(result_df)

# concatenate the list of DataFrames into a single DataFrame
df = pd.concat(dfs, ignore_index=True)

# create columns for differences between before and after metrics
for metric in ["accuracy", "precision", "distance"]:
    before_col = f'before_{metric}'
    after_col = f'after_{metric}'
    diff_col = f'diff_{metric}'

    if metric == "distance":
        df[diff_col] = round(abs(df[after_col]) - abs(df[before_col]), 2)
    else:
        df[diff_col] = round(df[after_col] - df[before_col], 2)

# Plot the diff_accuracy
plt.plot(df.index, df['diff_accuracy'], color='blue')
plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
plt.axhline(y=df['diff_accuracy'].mean(), color='green', linestyle='-', linewidth=2)
plt.xlabel('Index')
plt.ylabel('Accuracy gained (in %)')
plt.title('Difference in Accuracy')
plt.savefig(output_folder + "diff_accuracy.png")
plt.close()

# Plot the diff_precision
plt.plot(df.index, df['diff_precision'], color='blue')
plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
plt.axhline(y=df['diff_precision'].mean(), color='green', linestyle='-', linewidth=2)
plt.xlabel('Index')
plt.ylabel('Precision gained (in %)')
plt.title('Difference in Precision')
plt.savefig(output_folder + "diff_precision.png")
plt.close()

# Plot the diff_distance
plt.plot(df.index, df['diff_distance'], color='blue')
plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
plt.axhline(y=df['diff_distance'].mean(), color='green', linestyle='-', linewidth=2)
plt.xlabel('Index')
plt.ylabel('Distance change')
plt.title('Difference in Distance of detection')
plt.savefig(output_folder + "diff_distance.png")
plt.close()

# Plot the BCR
plt.plot(df.index, df['BCR'], color='blue')
plt.axhline(y=df['BCR'].mean(), color='green', linestyle='-', linewidth=2)
plt.xlabel('Index')
plt.ylabel('BCR')
plt.title('Baseline Correction Ratio')
plt.savefig(output_folder + "BCR.png")
plt.close()

# Plot the NSR
plt.plot(df.index, df['NSR'], color='blue')
plt.axhline(y=df['NSR'].mean(), color='green', linestyle='-', linewidth=2)
plt.xlabel('Index')
plt.ylabel('NSR')
plt.title('Noise Suppression Ratio')
plt.savefig(output_folder + "NSR.png")
plt.close()

# Plot the SDR
plt.plot(df.index, df['SDR'], color='blue')
plt.axhline(y=df['SDR'].mean(), color='green', linestyle='-', linewidth=2)
plt.xlabel('Index')
plt.ylabel('SDR')
plt.title('Signal Distortion Ratio')
plt.savefig(output_folder + "SDR.png")
plt.close()
