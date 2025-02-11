import pandas as pd

# Load the CSV file
csv_file_path = 'Cap3D_automated_Objaverse_full.csv'  # Replace with your CSV file path
captions_df = pd.read_csv(csv_file_path)

# Ensure that the column name is correct (in this case, 'caption')
# If your column name is different, replace 'caption' with the actual column name.

# Specify the output file path
txt_file_path = 'objaverse_prompt.txt'  # Replace with your desired output text file path

# Open the text file in write mode
with open(txt_file_path, 'w') as f:
    for index, row in captions_df.iterrows():
        value = row.iloc[1]
        f.write(str(value) + '\n')
        # print(f"Row {index}, Value: {value}")

print(f"Captions have been successfully written to {txt_file_path}")
