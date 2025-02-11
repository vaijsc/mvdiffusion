import json

# Load the JSON data from a file
input_json_path = "/workspace/env/data/eval/coco.json"  # Replace with the path to your JSON file
output_txt_path = "coco_prompts.txt"  # Replace with your desired output file path

with open(input_json_path, "r") as json_file:
    data = json.load(json_file)

# Extract prompts and save to a text file
with open(output_txt_path, "w") as txt_file:
    for item in data["labels"]:
        txt_file.write(item[1].strip() + "\n")

print(f"Prompts have been saved to {output_txt_path}")
