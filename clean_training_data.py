"""
Clean training data - remove special tokens
"""

import json
from pathlib import Path
import re

def clean_content(text):
    """Remove special tokens and problematic characters"""
    # Remove special tokens
    special_tokens = [
        '<|im_start|>', '<|im_end|>', '<|endoftext|>',
        '<|system|>', '<|user|>', '<|assistant|>',
        '<|fim_prefix|>', '<|fim_middle|>', '<|fim_suffix|>'
    ]
    
    for token in special_tokens:
        text = text.replace(token, '')
    
    # Remove other problematic patterns
    text = re.sub(r'<\|.*?\|>', '', text)  # Any <|...|> pattern
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def clean_training_file(input_file, output_file):
    """Clean a JSONL training file"""
    print(f"Cleaning: {input_file}")
    
    cleaned_count = 0
    valid_examples = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Total examples: {len(lines)}")
    
    for i, line in enumerate(lines, 1):
        try:
            data = json.loads(line)
            
            # Clean all message content
            cleaned = False
            for message in data['messages']:
                original = message['content']
                message['content'] = clean_content(message['content'])
                
                if original != message['content']:
                    cleaned = True
                    cleaned_count += 1
                
                # Skip if too short after cleaning
                if len(message['content']) < 20:
                    raise ValueError("Content too short after cleaning")
            
            # Check total length
            total_chars = sum(len(m['content']) for m in data['messages'])
            if total_chars < 50 or total_chars > 15000:
                continue
            
            valid_examples.append(data)
            
        except Exception as e:
            print(f"  Skipping line {i}: {str(e)}")
    
    print(f"\n✅ Cleaned {cleaned_count} examples")
    print(f"✅ Valid examples: {len(valid_examples)}")
    
    # Save cleaned data
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in valid_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"💾 Saved to: {output_file}")
    return len(valid_examples)

# Clean training and validation files
train_in = Path("data/training/fine_tuning/train.jsonl")
train_out = Path("data/training/fine_tuning/train_cleaned.jsonl")

val_in = Path("data/training/fine_tuning/validation.jsonl")
val_out = Path("data/training/fine_tuning/validation_cleaned.jsonl")

if train_in.exists():
    train_count = clean_training_file(train_in, train_out)
    
    if train_count >= 10:
        print(f"\n✅ Training file ready: {train_count} examples")
        # Replace original
        train_out.replace(train_in)
    else:
        print(f"\n❌ Not enough examples: {train_count} (need 10+)")
else:
    print("❌ Training file not found!")

if val_in.exists():
    val_count = clean_training_file(val_in, val_out)
    if val_count > 0:
        val_out.replace(val_in)

print("\n🎉 Data cleaning complete!")
print("Now run: python src\\fine_tuning\\openai_finetune.py")