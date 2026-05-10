import json
import os

def generate_static_data():
    """
    Generate STATIC_DATA JavaScript array by scanning assets/ folder
    and reading all manifest.json files
    """
    
    assets_path = 'assets'
    static_data = []
    
    if not os.path.exists(assets_path):
        print(f"Error: {assets_path} folder not found!")
        return
    
    # Get all datasets (folders in assets/)
    datasets = [d for d in os.listdir(assets_path) 
                if os.path.isdir(os.path.join(assets_path, d))]
    
    print(f"Found datasets: {datasets}")
    
    for dataset in datasets:
        dataset_path = os.path.join(assets_path, dataset)
        
        # Get all model folders (excluding GT folder)
        models = [m for m in os.listdir(dataset_path) 
                  if os.path.isdir(os.path.join(dataset_path, m)) and m != 'GT']
        
        print(f"  {dataset} models: {models}")
        
        for model in models:
            manifest_path = os.path.join(dataset_path, model, 'manifest.json')
            
            if os.path.exists(manifest_path):
                print(f"    Reading {manifest_path}")
                
                try:
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                    
                    # Process each item in the manifest
                    for item in manifest.get('items', []):
                        # Extract filename without extension for index
                        index = item['input_image'].replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
                        
                        entry = {
                            'dataset': dataset,
                            'model': model,
                            'index': index,
                            'result_source_dir': f'assets/{dataset}/GT/{item["input_image"]}',
                            'result_target_dir': f'assets/{dataset}/{model}/{item["output_image"]}',
                            'metrics': item.get('results', {}),
                            'category': 'pair',
                            'prompt': f'{dataset} - {model} - {index}'
                        }
                        static_data.append(entry)
                        
                except json.JSONDecodeError as e:
                    print(f"    Error reading {manifest_path}: {e}")
                except Exception as e:
                    print(f"    Error processing {manifest_path}: {e}")
            else:
                print(f"    Warning: {manifest_path} not found")
    
    # Generate JavaScript output
    print(f"\nGenerated {len(static_data)} entries")
    print("\n" + "="*50)
    print("Copy the following into your static HTML file:")
    print("="*50)
    
    print("const STATIC_DATA = [")
    for i, entry in enumerate(static_data):
        # Format the entry as JavaScript object
        js_entry = json.dumps(entry, indent=4)
        # Indent each line for proper formatting
        indented_entry = '\n'.join('    ' + line for line in js_entry.split('\n'))
        print(indented_entry, end="")
        
        if i < len(static_data) - 1:
            print(",")
        else:
            print("")
    print("];")
    
    print("\n" + "="*50)
    print(f"Total entries: {len(static_data)}")
    
    # Print summary by dataset and model
    summary = {}
    for entry in static_data:
        dataset = entry['dataset']
        model = entry['model']
        if dataset not in summary:
            summary[dataset] = {}
        if model not in summary[dataset]:
            summary[dataset][model] = 0
        summary[dataset][model] += 1
    
    print("\nSummary:")
    for dataset, models in summary.items():
        print(f"  {dataset}:")
        for model, count in models.items():
            print(f"    {model}: {count} samples")

def save_to_file():
    """
    Alternative function to save the output directly to a file
    """
    import sys
    from io import StringIO
    
    # Capture the output
    old_stdout = sys.stdout
    sys.stdout = buffer = StringIO()
    
    generate_static_data()
    
    # Get the captured output
    output = buffer.getvalue()
    sys.stdout = old_stdout
    
    # Extract just the JavaScript part
    lines = output.split('\n')
    start_idx = None
    end_idx = None
    
    for i, line in enumerate(lines):
        if line.strip() == 'const STATIC_DATA = [':
            start_idx = i
        elif line.strip() == '];' and start_idx is not None:
            end_idx = i + 1
            break
    
    if start_idx and end_idx:
        js_code = '\n'.join(lines[start_idx:end_idx])
        
        with open('static_data.js', 'w') as f:
            f.write(js_code)
        
        print(f"JavaScript code saved to static_data.js")
        print("You can copy this content into your static HTML file.")
    else:
        print("Error: Could not extract JavaScript code")

if __name__ == "__main__":
    print("Generating STATIC_DATA from assets/ folder...")
    print("Make sure you're running this script from the directory containing the assets/ folder")
    print()
    
    # Choose one of these:
    generate_static_data()  # Print to console
    
    # Uncomment the line below if you want to save to a file instead:
    save_to_file()