import os
import sys
import json
import time
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

# Add repository root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retriever_server.unified_retriever import UnifiedRetriever
from commaqa.models.mock_generator import MockGenerator

app = FastAPI(title="IRCoT Web Demo Server", description="Interactive Web Interface for Interleaved Retrieval-Guided Chain-of-Thought Reasoning")

# Mount static directory
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

retriever = UnifiedRetriever()
mock_generator = MockGenerator()

class QueryRequest(BaseModel):
    question: str
    dataset: str = "hotpotqa"
    system_type: str = "ircot_qa"  # ircot_qa, oner_qa, nor_qa
    retrieval_type: str = "bm25"    # bm25, vector, hybrid
    max_steps: int = 3
    provided_context: Optional[List[Dict]] = None

@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>IRCoT Web Demo Server Running</h1>")

@app.get("/api/datasets")
async def get_datasets():
    sample_data_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "setup_demo_data.py")
    return {
        "datasets": [
            {
                "id": "hotpotqa",
                "name": "HotpotQA",
                "description": "Multi-hop question answering dataset requiring reasoning over Wikipedia paragraphs.",
                "samples": [
                    "Were Scott Derrickson and Ed Wood of the same nationality?",
                    "Which magazine was founded first, Arthur's Magazine or First for Women?"
                ]
            },
            {
                "id": "2wikimultihopqa",
                "name": "2WikiMultihopQA",
                "description": "Complex multi-hop dataset with explicitly annotated reasoning paths and supporting facts.",
                "samples": [
                    "Who is the mother of the director of movie The Dark Knight?"
                ]
            },
            {
                "id": "musique",
                "name": "MuSiQue",
                "description": "Multi-hop dataset composed of 2-4 hop questions designed to minimize single-hop shortcuts.",
                "samples": [
                    "Where was the performer of song Thriller born?"
                ]
            },
            {
                "id": "iirc",
                "name": "IIRC",
                "description": "Incomplete Information Reading Comprehension dataset requiring retrieving external context.",
                "samples": [
                    "How many years after the founding of Harvard University was Yale University founded?"
                ]
            }
        ]
    }

@app.post("/api/query")
async def process_query(req: QueryRequest):
    start_time = time.time()
    question = req.question.strip()
    dataset = req.dataset
    system_type = req.system_type
    
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    steps = []
    retrieved_paragraphs = []
    selected_titles = []
    
    # 1. Initial State
    steps.append({
        "step_num": 1,
        "type": "thought",
        "title": "Initial Reasoning & Goal",
        "content": f"Analyzing question: '{question}'. Identifying entity relations and required multi-hop evidence.",
        "timestamp": round(time.time() - start_time, 2)
    })

    if system_type == "nor_qa":
        # No retrieval baseline
        llm_seq = mock_generator.generate_text_sequence(question)
        final_ans = llm_seq[0][0]
        steps.append({
            "step_num": 2,
            "type": "answer",
            "title": "Direct Synthesis (No Retrieval)",
            "content": f"Answer: {final_ans}",
            "timestamp": round(time.time() - start_time, 2)
        })
        return {
            "question": question,
            "system_type": system_type,
            "final_answer": final_ans,
            "steps": steps,
            "retrieved_paragraphs": [],
            "execution_time_sec": round(time.time() - start_time, 2)
        }

    elif system_type == "oner_qa":
        # One-step retrieval baseline
        query_text = question
        results = retriever.retrieve_from_elasticsearch(query_text=query_text, corpus_name=dataset, max_hits_count=3)
        for r in results:
            src = r.get("_source", r)
            retrieved_paragraphs.append({
                "title": src.get("title", ""),
                "text": src.get("paragraph_text", ""),
                "score": round(r.get("_score", 0.0), 3)
            })
            selected_titles.append(src.get("title", ""))

        steps.append({
            "step_num": 2,
            "type": "retrieval",
            "title": "One-Shot Retrieval",
            "content": f"Retrieved {len(retrieved_paragraphs)} paragraphs for query '{query_text}'.",
            "paragraphs": retrieved_paragraphs,
            "timestamp": round(time.time() - start_time, 2)
        })

        context_str = "\n".join([f"Wikipedia Title: {p['title']}\n{p['text']}" for p in retrieved_paragraphs])
        llm_seq = mock_generator.generate_text_sequence(f"{context_str}\nQuestion: {question}\nAnswer:")
        final_ans = llm_seq[0][0]

        steps.append({
            "step_num": 3,
            "type": "answer",
            "title": "Final Answer Synthesis",
            "content": final_ans,
            "timestamp": round(time.time() - start_time, 2)
        })

        return {
            "question": question,
            "system_type": system_type,
            "final_answer": final_ans,
            "steps": steps,
            "retrieved_paragraphs": retrieved_paragraphs,
            "execution_time_sec": round(time.time() - start_time, 2)
        }

    else:
        # Full IRCoT Interleaved Loop
        # Step 1: First Thought & Initial Retrieval Query Generation
        q1 = question
        res1 = retriever.retrieve_from_elasticsearch(query_text=q1, corpus_name=dataset, max_hits_count=2)
        for r in res1:
            src = r.get("_source", r)
            if src.get("title") not in selected_titles:
                selected_titles.append(src.get("title", ""))
                retrieved_paragraphs.append({
                    "title": src.get("title", ""),
                    "text": src.get("paragraph_text", ""),
                    "score": round(r.get("_score", 1.0), 3)
                })

        steps.append({
            "step_num": 2,
            "type": "retrieval",
            "title": "IRCoT Hop 1 Retrieval",
            "content": f"Generated sub-query: '{q1}'. Retrieved initial context paragraphs.",
            "paragraphs": list(retrieved_paragraphs),
            "timestamp": round(time.time() - start_time, 2)
        })

        # Step 2: Intermediate Reasoning Step
        context_so_far = "\n".join([f"Title: {p['title']}\n{p['text']}" for p in retrieved_paragraphs])
        cot_prompt = f"Context:\n{context_so_far}\nQuestion: {question}\nThought:"
        cot_gen = mock_generator.generate_text_sequence(cot_prompt)[0][0]

        steps.append({
            "step_num": 3,
            "type": "thought",
            "title": "IRCoT Chain-of-Thought Hop 1",
            "content": cot_gen,
            "timestamp": round(time.time() - start_time, 2)
        })

        # Step 3: Follow-up query & Hop 2 Retrieval
        q2 = f"{question} {cot_gen}"
        res2 = retriever.retrieve_from_elasticsearch(query_text=q2, corpus_name=dataset, max_hits_count=2)
        new_paras = []
        for r in res2:
            src = r.get("_source", r)
            if src.get("title") not in selected_titles:
                selected_titles.append(src.get("title", ""))
                p_item = {
                    "title": src.get("title", ""),
                    "text": src.get("paragraph_text", ""),
                    "score": round(r.get("_score", 1.0), 3)
                }
                retrieved_paragraphs.append(p_item)
                new_paras.append(p_item)

        steps.append({
            "step_num": 4,
            "type": "retrieval",
            "title": "IRCoT Hop 2 Retrieval",
            "content": f"Follow-up retrieval guided by CoT reasoning. Added {len(new_paras)} new paragraphs.",
            "paragraphs": new_paras if new_paras else retrieved_paragraphs,
            "timestamp": round(time.time() - start_time, 2)
        })

        # Final Answer
        final_prompt = f"Context:\n{context_so_far}\nQuestion: {question}\nAnswer:"
        final_ans = mock_generator.generate_text_sequence(final_prompt)[0][0]

        steps.append({
            "step_num": 5,
            "type": "answer",
            "title": "IRCoT Final Answer Synthesis",
            "content": final_ans,
            "timestamp": round(time.time() - start_time, 2)
        })

        return {
            "question": question,
            "system_type": system_type,
            "final_answer": final_ans,
            "steps": steps,
            "retrieved_paragraphs": retrieved_paragraphs,
            "execution_time_sec": round(time.time() - start_time, 2)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)
