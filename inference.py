import argparse
from pydoc import text
from tqdm import tqdm
import csv
import os
import gc
from utils import Model, batch_process, PROMPT_ANABENCH
from datasets import load_dataset
import datetime

# Format: YYYY-MM-DD
# DATE = datetime.date.today().strftime("%m-%d")
DATE = "09-22"

def main():
    ''' 
    code below is used to free cache from gpu
    '''
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    # torch.cuda.empty_cache()
    # gc.collect()
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo id or local path")
    ap.add_argument("--sentence_length", type=str, default="S10")
    ap.add_argument("--batch_size", type=int, default=10)
    ap.add_argument("--verbose", type=bool, default=False)
    # ap.add_argument("--prompt", default="Write a short haiku about lakes.")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    # ap.add_argument("--use_4bit", action="store_true", help="load 4-bit quantized (requires bitsandbytes)")
    args = ap.parse_args()
    
    print(f"Loading model...{args.model}")
    model = Model(args.model)
    print("Model loaded.")
    model_name = args.model.split('/')[-1] if '/' in args.model else args.model
    sentence_length = args.sentence_length
    batch_size = args.batch_size
    output_file = f'results/analobench/T1{sentence_length}-{model_name}-{DATE}.csv'
    verbose = args.verbose
    dataset = load_dataset("jhu-clsp/AnaloBench", f"T1{sentence_length}-Subset")

    # Check for existing results to resume
    processed_indices = set()
    file_exists = os.path.exists(output_file)
    if file_exists:
        with open(output_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Assuming 'Index' is a unique identifier in your data
                if 'Index' in row and row['Index']:
                    processed_indices.add(int(row['Index']))
        if verbose:
            print(f"Resuming. Found {len(processed_indices)} processed rows in {output_file}")

    with open(output_file, 'a', newline='') as outfile:
        reader = dataset["train"]
        fields = ["Index", "Sentence", "Story", "Options", "Label", "Pred_y", "Reason"]
        writer = csv.DictWriter(outfile, fieldnames=fields)
        
        if not file_exists or not processed_indices:
            writer.writeheader()
            if verbose:
                print("Creating new file... CSV header written.")


        batch = []
        print(f"Starting processing with batch size {batch_size}...")
        for row in tqdm(reader):
            # Skip rows that are already processed
            if row['Index'] in processed_indices:
                continue

            batch.append(row)
            if len(batch) >= batch_size:
                processed_batch = batch_process(batch, 
                                                PROMPT_ANABENCH, model, 
                                                tokens=args.max_new_tokens, 
                                                temp=args.temperature)
                writer.writerows(processed_batch)
                outfile.flush() # Ensure data is written to disk
                batch = []
                print(f"Processed {batch_size} rows and wrote to {output_file}")

        # Process the remaining batch
        if batch:
            if verbose:
                print(f"Processing the final batch of {len(batch)} rows...")
            processed_batch = batch_process(batch, 
                                            PROMPT_ANABENCH, model, 
                                            tokens=args.max_new_tokens, temp=args.temperature)
            writer.writerows(processed_batch)
            if verbose:
                print(f"Processed final {len(processed_batch)} rows and appended to {output_file}")
        

if __name__ == "__main__":
    main()