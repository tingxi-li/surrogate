import csv
import json
import argparse

def csv_to_jsonl(csv_file, jsonl_file, delimiter=','):
    """
    Convert a CSV file to JSONL format, preserving empty cells as empty strings.
    
    Args:
        csv_file (str): Path to the input CSV file
        jsonl_file (str): Path to the output JSONL file
        delimiter (str, optional): CSV delimiter. Defaults to ','.
    """
    try:
        with open(csv_file, 'r', encoding='utf-8') as csvfile:
            # Read CSV with DictReader to automatically map headers to values
            reader = csv.DictReader(csvfile, delimiter=delimiter)
            
            with open(jsonl_file, 'w', encoding='utf-8') as jsonlfile:
                for row in reader:
                    # Replace None values with empty strings
                    for key in row:
                        if row[key] is None:
                            row[key] = ""
                    
                    # Write each row as a JSON line
                    jsonlfile.write(json.dumps(row) + '\n')
        
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Convert CSV to JSONL with empty cells preserved')
    parser.add_argument('csv_file', help='Input CSV file path')
    parser.add_argument('jsonl_file', help='Output JSONL file path')
    parser.add_argument('-d', '--delimiter', default=',', help='CSV delimiter (default: ,)')
    
    args = parser.parse_args()
    
    # Run the conversion
    success = csv_to_jsonl(args.csv_file, args.jsonl_file, args.delimiter)
    
    if success:
        print(f"Successfully converted {args.csv_file} to {args.jsonl_file}")
    else:
        print("Conversion failed")

if __name__ == "__main__":
    main()