import csv

# Function to combine files into one
def combine_files(input_files, output_file):
    with open(output_file, 'w') as outfile:
        for file in input_files:
            with open(file, 'r') as infile:
                outfile.write(infile.read())

# Function to process each line
def process_line(line):
    parts = line.split(' ')
    cleaned_line = ' '.join(parts[1:])
    cleaned_line = cleaned_line.replace('[', '').replace(']', '')
    return cleaned_line

# Define file paths
anti_files = [
    '../raw/nlp/anti_stereotyped_type1.txt.dev',
    '../raw/nlp/anti_stereotyped_type1.txt.test',
    '../raw/nlp/anti_stereotyped_type2.txt.dev',
    '../raw/nlp/anti_stereotyped_type2.txt.test'
]
combined_anti_file = '../raw/nlp/combined_anti_file.txt'
combine_files(anti_files, combined_anti_file)

pro_files = [
    '../raw/nlp/pro_stereotyped_type1.txt.dev',
    '../raw/nlp/pro_stereotyped_type1.txt.test',
    '../raw/nlp/pro_stereotyped_type2.txt.dev',
    '../raw/nlp/pro_stereotyped_type2.txt.test'
]
combined_pro_file = '../raw/nlp/combined_pro_file.txt'
combine_files(pro_files, combined_pro_file)

# Input and output file paths
output_combined_file = '../biased/nlp/cleaned_combined_data.csv'

# Write to the output file
with open(output_combined_file, 'w', newline='') as outfile:
    csv_writer = csv.writer(outfile)
    csv_writer.writerow(['Text'])  # Write header

    # Write only the first 525 rows from anti_files
    with open(combined_anti_file, 'r') as infile:
        for i, line in enumerate(infile):
            if i < 525:
                cleaned_line = process_line(line)
                csv_writer.writerow([cleaned_line.strip()])

    # Write all rows from pro_files
    with open(combined_pro_file, 'r') as infile:
        for line in infile:
            cleaned_line = process_line(line)
            csv_writer.writerow([cleaned_line.strip()])

print("525 rows from anti_files and all rows from pro_files have been copied and saved to the output file.")
