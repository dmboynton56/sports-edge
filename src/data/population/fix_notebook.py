import json
import os

nb_path = 'notebooks/model_story.ipynb'
try:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Find first code cell
    patched = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            # Check if already added
            if not any('load_dotenv' in line for line in source[:5]):
                print("Injecting load_dotenv()...")
                new_source = ["from dotenv import load_dotenv\n", "load_dotenv()\n"] + source
                cell['source'] = new_source
                patched = True
            else:
                print("Notebook already patched.")
            break
    
    if patched:
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print("Notebook saved.")

except Exception as e:
    print(f"Error: {e}")
