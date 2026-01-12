#!/usr/bin/env python3
"""
Convert chat.json (JSON array) to chat.jsonl (JSONL format) for LLaVA dataset.
"""
import json
import jsonlines
import sys
import os

def convert_json_to_jsonl(json_path, jsonl_path):
    """Convert JSON array file to JSONL format."""
    print(f"Reading {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} samples")
    print(f"Writing to {jsonl_path}...")
    
    with jsonlines.open(jsonl_path, mode='w') as writer:
        for item in data:
            writer.write(item)
    
    print(f"✓ Conversion complete! Created {jsonl_path} with {len(data)} samples")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_chat_json_to_jsonl.py <input.json> <output.jsonl>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found")
        sys.exit(1)
    
    convert_json_to_jsonl(input_file, output_file)
