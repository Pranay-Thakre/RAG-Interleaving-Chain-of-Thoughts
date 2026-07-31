import os
import json

def setup_data_files():
    datasets = ["hotpotqa", "2wikimultihopqa", "musique", "iirc"]
    
    for ds in datasets:
        proc_file = os.path.join("processed_data", ds, "dev_subsampled.jsonl")
        raw_dir = os.path.join("raw_data", ds)
        os.makedirs(raw_dir, exist_ok=True)
        
        if os.path.exists(proc_file):
            items = []
            with open(proc_file, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        qid = data.get("_id") or data.get("id") or data.get("question_id")
                        data["_id"] = qid
                        data["id"] = qid
                        data["question_id"] = qid
                        if "answer" not in data:
                            if "answers_objects" in data and len(data["answers_objects"]) > 0:
                                data["answer"] = data["answers_objects"][0].get("spans", [""])[0]
                            elif "answers" in data and len(data["answers"]) > 0:
                                data["answer"] = data["answers"][0]
                            else:
                                data["answer"] = ""
                        if "supporting_facts" not in data:
                            supp_facts = []
                            for ctx in data.get("contexts", []):
                                if ctx.get("is_supporting"):
                                    supp_facts.append([ctx["title"], 0])
                            data["supporting_facts"] = supp_facts
                        items.append(data)
            
            # Write matching raw files for official evaluation
            if ds == "hotpotqa":
                with open(os.path.join(raw_dir, "hotpot_dev_distractor_v1.json"), "w") as f:
                    json.dump(items, f, indent=2)
            elif ds == "2wikimultihopqa":
                with open(os.path.join(raw_dir, "dev.json"), "w") as f:
                    json.dump(items, f, indent=2)
                with open(os.path.join(raw_dir, "id_aliases.json"), "w") as f:
                    json.dump({}, f)
            elif ds == "musique":
                with open(os.path.join(raw_dir, "musique_ans_v1.0_dev.jsonl"), "w") as f:
                    for item in items:
                        f.write(json.dumps(item) + "\n")

    print("Official raw evaluation data files synchronized with processed dev_subsampled datasets!")

if __name__ == "__main__":
    setup_data_files()
