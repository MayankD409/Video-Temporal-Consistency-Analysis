import json
import re
import google.generativeai as genai
import os
from dotenv import load_dotenv
import argparse

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def extract_predicted_number_simple(response_text):
    # Use Gemini to extract the numeric answer
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"""From the following response, extract the numeric answer:
    {response_text}
    
    If the response mentions a multiple choice option (A, B, C, D, E, F), return the corresponding number.
    If a direct numeric value is mentioned, return that number.
    If no clear numeric answer is found, return None.
    
    Respond ONLY with the number or None."""

    try:
        response = model.generate_content(prompt)
        extracted_num = response.text.strip()
        
        # Additional parsing to handle various response formats
        if extracted_num.lower() == 'none':
            return None
        
        try:
            return int(extracted_num)
        except ValueError:
            # Handle letter to number conversion
            letter_to_number = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6}
            if extracted_num.lower() in letter_to_number:
                return letter_to_number[extracted_num.lower()]
            
            return None
    
    except Exception as e:
        print(f"Error in Gemini extraction: {e}")
        return None

# Rest of the code remains the same as in the original script
def parse_gt(gt_str):
    parts = gt_str.split('.')
    number = parts[1].strip()
    return int(number)

def compute_numeric_distance_score(pred_num, gt_num):
    diff = abs(pred_num - gt_num)
    score = 1 - (diff / (gt_num + 1))
    if score < 0: score = 0.0
    return score

def evaluate_jsonl_with_numeric(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line.strip())
            response = data['response']
            gt_str = data['gt']

            gt_num = parse_gt(gt_str)
            pred_num = extract_predicted_number_simple(response)
            
            if pred_num is not None:
                score = compute_numeric_distance_score(pred_num, gt_num)
            else:
                score = 0.0

            data['numeric_distance_score'] = score
            outfile.write(json.dumps(data) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Compute Numeric Distance Score for Model Predictions")
    parser.add_argument('input_file', help='Input JSONL file with predictions')
    parser.add_argument('output_file', help='Output JSONL file with numeric distance scores')
    
    args = parser.parse_args()
    
    evaluate_jsonl_with_numeric(args.input_file, args.output_file)
    print(f"Numeric distance scores computed. Output saved to {args.output_file}")

if __name__ == "__main__":
    main()
