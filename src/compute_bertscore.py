import os
import json
import re
from bert_score import score
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

########################
# Gemini Client Setup  #
########################
# Assuming a similar API to genai or openai
# Modify as per your actual Gemini API client usage.
import google.generativeai as genai

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# Model and generation config for Gemini
GENERATION_CONFIG = genai.types.GenerationConfig(
    candidate_count=1,
    max_output_tokens=128,
    temperature=0.0,  # deterministic
    top_p=1.0
)

########################
# Helper Functions     #
########################

def parse_gt(gt_str):
    # Example: "D. 3"
    parts = gt_str.split('.')
    # parts[0] = "D"
    # parts[1] = "3"
    number = parts[1].strip()
    return number

def extract_predicted_answer_option_and_number(response_text):
    """
    Attempt to extract a chosen option and a numeric answer from the response.
    The response typically ends with something like:
    "The correct answer is:\n\n**D: 5**"
    or "The correct answer is D: 5".
    Adjust regex as needed.
    """
    match = re.search(r"answer\s+is:\s*\**([A-F])\**[:\s]+(\d+)", response_text, re.IGNORECASE)
    if match:
        chosen_option = match.group(1).upper()
        predicted_number = match.group(2)
        return chosen_option, predicted_number

    # fallback: just find a number
    match = re.search(r"(\d+)", response_text)
    if match:
        return None, match.group(1)
    return None, None

def call_gemini_for_reference(question, gt_number):
    """
    Call Gemini-1.5-Flash API to produce a reference sentence that states the correct numeric answer.
    
    Prompt the LLM to produce a natural sentence. For example:
    "Given the question and the correct numeric answer, produce a factual, self-contained sentence that describes the answer.
    Do not mention options. Just plainly state the correct count or relevant info."

    Adjust the prompt as needed for your dataset.
    """
    prompt = f"""
You are given a question from a video-based scenario and the correct numeric answer.
Your task is to produce a factual, self-contained sentence that describes the correct answer in a natural manner.
Do not refer to any multiple-choice options. Just state the correct quantity or fact clearly.

Question: {question}
Correct answer (number): {gt_number}

Your response should be a single sentence that incorporates the correct number in context.
"""
    # Call Gemini model
    # Assuming a method similar to genai model
    # If your model is different, adjust accordingly
    model_name = "gemini-1.5-flash"  # or the correct model name you use for generation
    flash = genai.GenerativeModel('gemini-1.5-flash')
    response = flash.generate_content(
        prompt.strip(),
        generation_config=GENERATION_CONFIG
    )
    
    # response.text contains the generated sentence
    ref_sentence = response.text.strip()
    return ref_sentence

def compute_bertscore(pred_sentence, ref_sentence):
    P, R, F = score([pred_sentence], [ref_sentence], lang='en', verbose=False)
    return float(F[0])

########################
# Main Evaluation Code #
########################

def evaluate_with_gemini_and_bertscore(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in tqdm(infile):
            data = json.loads(line.strip())
            question = data['question']
            response = data['response']
            gt_str = data['gt'] # e.g., "D. 3"
            
            # Parse ground-truth number
            gt_number = parse_gt(gt_str)

            # Call Gemini to get a reference sentence
            ref_sentence = call_gemini_for_reference(question, gt_number)

            # pred_sentence is just the model's response
            pred_sentence = response

            # Compute BERTScore
            bscore = compute_bertscore(pred_sentence, ref_sentence)

            data['bertscore_gemini'] = bscore
            data['reference_sentence'] = ref_sentence
            outfile.write(json.dumps(data) + "\n")


########################
# Example Usage
########################
# python this_script.py input.jsonl output.jsonl
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python script.py input.jsonl output.jsonl")
        sys.exit(1)
    input_jsonl = sys.argv[1]
    output_jsonl = sys.argv[2]
    evaluate_with_gemini_and_bertscore(input_jsonl, output_jsonl)
