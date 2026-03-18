import json
from pathlib import Path

def main():
    sources = [
        "reports/data_dictionary.md",
        "reports/eda_report.md",
        "reports/exp_week04.md",
        "reports/exp_week05.md",
        "reports/exp_week06.md",
        "reports/model_explain.md"
    ]

    knowledge_base = []
    
    for source in sources:
        filepath = Path(source)
            
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        current_title = filepath.stem 
        current_content_lines = []
        

        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line: 
                continue
                

            is_hash_title = line.startswith('#')

            
            if is_hash_title:
                if current_content_lines:
                    record = {
                        "source": filepath.name,
                        "title": current_title,
                        "content": "\n".join(current_content_lines)
                    }
                    knowledge_base.append(record)
                    current_content_lines = [] 
                

                if is_hash_title:
                    current_title = line.strip('#').strip()
            else:
                current_content_lines.append(line)
                

        if current_content_lines:
            record = {
                "source": filepath.name,
                "title": current_title,
                "content": "\n".join(current_content_lines)
            }
            knowledge_base.append(record)

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "knowledge_base.json"
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, ensure_ascii=False, indent=2)
        


if __name__ == "__main__":
    main()