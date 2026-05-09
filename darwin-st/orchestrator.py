import os
import json
import time
import subprocess
import re
import sys
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Add current file directory to path to ensure config is found, and setup BASE_DIR
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from config import LLM_BASE_URL, LLM_MODEL_NAME, LLM_API_KEY, LLM_TEMPERATURE, MUTATE_SYSTEM_PROMPT, REVIEW_SYSTEM_PROMPT, REFLECT_PROMPT

# ==============================================================================
# State Definition
# ==============================================================================
class AutoResearchState(TypedDict):
    iteration_id: int
    parent_id: str
    best_mae: float
    current_mae: float
    innovation_desc: str
    status: str
    messages: List[any]  # Short-term memory (flushed every cycle)
    run_log: str

# ==============================================================================
# Helper Functions
# ==============================================================================
def read_file(filepath, default=""):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return f.read()
    return default

def write_file(filepath, content):
    with open(filepath, 'w') as f:
        f.write(content)

def extract_python_code(text):
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1)
    # Fallback if no markdown block
    return text

def get_git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "root"

def append_evolution_log(state: AutoResearchState):
    log_entry = {
        "iteration_id": state["iteration_id"],
        "parent_id": state["parent_id"],
        "innovation": state["innovation_desc"],
        "mae": state["current_mae"],
        "status": state["status"],
        "git_hash": get_git_hash()
    }
    with open("evolution_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

# ==============================================================================
# LLM Initialization
# ==============================================================================
llm = ChatOpenAI(
    base_url=LLM_BASE_URL,
    model=LLM_MODEL_NAME,
    api_key=LLM_API_KEY,
    temperature=LLM_TEMPERATURE
)

# ==============================================================================
# LangGraph Nodes
# ==============================================================================

def node_research(state: AutoResearchState):
    print(f"\n[Node 0: Research] Fetching GraphRAG Inspiration for Iteration {state['iteration_id']}")
    # Call the external script as a tool
    try:
        result = subprocess.check_output(
            [sys.executable, os.path.join(BASE_DIR, "scripts/graph_rag.py"), "--action", "inspire", "--json"], 
            stderr=subprocess.STDOUT
        ).decode("utf-8")
        
        # If it's pure JSON, just attach it, else attach raw string
        inspiration = result
    except subprocess.CalledProcessError as e:
        inspiration = f"GraphRAG Error: {e.output.decode('utf-8')}. Try combining Attention with Graph Convolutions."

    state["messages"].append(HumanMessage(content=f"GraphRAG Inspiration:\n{inspiration}"))
    return state

def node_mutate(state: AutoResearchState):
    print(f"\n[Node 1: Mutate (Producer)] Generating new architecture...")
    memory_content = read_file("memory.md", "No memory found. Assume Baseline.")
    train_content = read_file("train.py", "# Empty train.py")
    
    messages = [SystemMessage(content=MUTATE_SYSTEM_PROMPT)]
    messages.append(HumanMessage(content=f"--- memory.md ---\n{memory_content}\n--- current train.py ---\n{train_content}"))
    
    # Append short-term conversation (GraphRAG info and potential Reviewer critiques)
    messages.extend(state["messages"])
    
    response = llm.invoke(messages)
    state["messages"].append(response)
    
    # Extract and save code
    new_code = extract_python_code(response.content)
    write_file("train.py", new_code)
    
    # Ask LLM for a 1-sentence innovation description
    desc_response = llm.invoke([HumanMessage(content=f"Based on your code generation, summarize the core mathematical/architectural innovation in exactly 1 sentence:\n\n{response.content}")])
    state["innovation_desc"] = desc_response.content
    
    return state

def node_review(state: AutoResearchState):
    print(f"\n[Node 2: Static Review (Critic)] Evaluating generated code...")
    train_content = read_file("train.py")
    
    messages = [
        SystemMessage(content=REVIEW_SYSTEM_PROMPT),
        HumanMessage(content=f"Please review the following generated code:\n\n```python\n{train_content}\n```")
    ]
    
    response = llm.invoke(messages)
    critique = response.content.strip()
    
    if "CODE_IS_PERFECT" in critique:
        print("  -> Critic Approved: CODE_IS_PERFECT")
        state["status"] = "REVIEW_PASSED"
    else:
        print("  -> Critic Rejected. Sending back to Producer.")
        print(f"     Critique: {critique}")
        state["status"] = "REVIEW_FAILED"
        state["messages"].append(HumanMessage(content=f"CRITIQUE FROM REVIEWER. You MUST fix these issues:\n{critique}"))
        
    return state

def route_after_review(state: AutoResearchState):
    if state["status"] == "REVIEW_PASSED":
        return "execute"
    return "mutate"

def node_execute(state: AutoResearchState):
    print(f"\n[Node 3: Execute] Running physical training loop (Max 15 mins)...")
    
    # Exception Handling Pattern: Try/Catch for robustness
    try:
        # Run training using local venv
        result = subprocess.run(
            [sys.executable, "train.py"], 
            timeout=900, 
            capture_output=True, 
            text=True
        )
        state["run_log"] = result.stdout + "\n" + result.stderr
        
        if result.returncode != 0:
            print(f"  -> Execution failed with exit code {result.returncode}.")
            state["status"] = "EXECUTION_ERROR"
        else:
            print("  -> Execution completed successfully.")
            state["status"] = "EXECUTION_SUCCESS"
            
    except subprocess.TimeoutExpired as e:
        print("  -> Execution timed out after 900 seconds.")
        # Attempt to salvage logs
        state["run_log"] = f"TIMEOUT EXPIRED: {str(e)}"
        state["status"] = "EXECUTION_TIMEOUT"
    except Exception as e:
        print(f"  -> Critical Execution Error: {str(e)}")
        state["run_log"] = f"CRITICAL ERROR: {str(e)}"
        state["status"] = "EXECUTION_ERROR"

    write_file("run.log", state["run_log"])
    return state

def node_reflect(state: AutoResearchState):
    print(f"\n[Node 4: Reflect & Recover] Analyzing results and updating memory...")
    
    # Exception Handling / State Rollback
    if state["status"] in ["EXECUTION_ERROR", "EXECUTION_TIMEOUT"]:
        print("  -> Handling Failure: Rolling back state and updating Graveyard.")
        state["current_mae"] = 999.0
        state["status"] = "DISCARD"
        
        # Rollback code to last known good state
        subprocess.run(["git", "checkout", "train.py"])
        
        # Update memory Graveyard (Episodic Memory)
        memory = read_file("memory.md")
        graveyard_entry = f"\n- **Iteration {state['iteration_id']} Failed**: {state['innovation_desc']}. Status: {state['status']}. Log snippet: {state['run_log'][-200:]}"
        write_file("memory.md", memory + graveyard_entry)
        
    else:
        # Parse results.tsv
        results = read_file("results.tsv")
        try:
            # Simple parser assuming last line is the latest MAE
            last_line = [line for line in results.split('\n') if line.strip()][-1]
            mae_str = last_line.split('\t')[1] # Assuming Format: Epoch | MAE | ...
            state["current_mae"] = float(mae_str)
        except Exception:
            state["current_mae"] = 999.0
            
        if state["current_mae"] < state["best_mae"]:
            print(f"  -> 🎉 SOTA Broken! {state['best_mae']} -> {state['current_mae']}")
            state["best_mae"] = state["current_mae"]
            state["status"] = "KEEP"
            
            # Commit the new SOTA
            subprocess.run(["git", "add", "train.py"])
            subprocess.run(["git", "commit", "-m", f"SOTA Gen {state['iteration_id']}: MAE {state['current_mae']} - {state['innovation_desc']}"])
            state["parent_id"] = get_git_hash()
            
            # Semantic Memory Update (LLM summarizing success)
            # In a full setup, this calls the LLM to rewrite the memory.md SOTA section.
        else:
            print(f"  -> Degradation. Best: {state['best_mae']}, Current: {state['current_mae']}")
            state["status"] = "DISCARD"
            subprocess.run(["git", "checkout", "train.py"])

    # Save to Lineage
    append_evolution_log(state)
    
    # Memory Management Pattern: Clear short-term memory to avoid context bloat
    state["messages"] = []
    state["iteration_id"] += 1
    
    return state

def node_visualize(state: AutoResearchState):
    print(f"\n[Node 5: Visualize] Updating Evolution Tree HTML...")
    try:
        subprocess.run([sys.executable, os.path.join(BASE_DIR, "scripts/plot_tree.py")])
    except Exception as e:
        print(f"Visualization error: {e}")
    return state

# ==============================================================================
# Graph Construction
# ==============================================================================
workflow = StateGraph(AutoResearchState)

workflow.add_node("research", node_research)
workflow.add_node("mutate", node_mutate)
workflow.add_node("review", node_review)
workflow.add_node("execute", node_execute)
workflow.add_node("reflect", node_reflect)
workflow.add_node("visualize", node_visualize)

# Edges
workflow.set_entry_point("research")
workflow.add_edge("research", "mutate")
workflow.add_edge("mutate", "review")

# Conditional Edge for Reflection Pattern
workflow.add_conditional_edges(
    "review",
    route_after_review,
    {
        "execute": "execute",
        "mutate": "mutate"  # Feedback loop back to Producer
    }
)

workflow.add_edge("execute", "reflect")
workflow.add_edge("reflect", "visualize")
# Infinite loop back to research
workflow.add_edge("visualize", "research")

app = workflow.compile()

def prompt_with_options(prompt_text, options, default_idx=0):
    print(f"\n{prompt_text}")
    for i, opt in enumerate(options):
        print(f"  {i+1}. {opt}")
    default_opt = options[default_idx]
    choice = input(f"Select an option (1-{len(options)}) or type custom value [Default: {default_opt}]: ").strip()
    if not choice:
        return default_opt
    if choice.isdigit() and 1 <= int(choice) <= len(options):
        return options[int(choice)-1]
    return choice

def setup_phase():
    if os.path.exists("memory.md"):
        print("📁 Found existing memory.md. Resuming evolution loop...")
        return
        
    print("🌟 Welcome to Darwin-ST: Autonomous Spatio-Temporal AI Researcher 🌟")
    print("No memory.md found. Initiating Phase 1: Interactive Setup...\n")
    
    innovation_opts = [
        "Liquid Neural Network (LNN)",
        "Spatio-Temporal Causal Attention",
        "Adaptive Graph ODE",
        "Mamba-based Sequence Modeling"
    ]
    innovation = prompt_with_options("👉 核心创新点 (Innovation):", innovation_opts, 0)
    
    dataset_opts = ["PeMS04", "PeMS08", "METR-LA", "PEMS-BAY"]
    dataset = prompt_with_options("👉 目标数据集 (Dataset):", dataset_opts, 0)
    
    baseline_opts = ["DCRNN", "STGCN", "ASTGCN", "GWNet", "AGCRN"]
    baseline = prompt_with_options("👉 对比基线 (Baseline Name):", baseline_opts, 0)
    
    # Auto-generate a sensible default branch name
    import re
    safe_inno = re.sub(r'[^a-zA-Z0-9]', '-', innovation.split('(')[0].strip().lower())
    safe_inno = re.sub(r'-+', '-', safe_inno).strip('-')
    default_branch = f"exp/{safe_inno[:20]}"
    
    branch_name = input(f"\n👉 实验分支名 (Git Branch) [Default: {default_branch}]: ").strip()
    if not branch_name:
        branch_name = default_branch
    
    # 1. Git checkout
    try:
        subprocess.run(["git", "checkout", "-b", branch_name], check=True, capture_output=True)
        print(f"\n🌿 Successfully checked out branch: {branch_name}")
    except subprocess.CalledProcessError as e:
        print(f"\n⚠️ Git checkout failed (maybe branch exists or not a git repo). Proceeding...")
        
    # 2. Search Baseline MAE using LLM
    print(f"\n🔍 Searching Knowledge Base for the typical Validation MAE of {baseline} on {dataset}...")
    query = f"What is the typical Validation Mean Absolute Error (MAE) of the {baseline} model on the {dataset} traffic prediction dataset? Please provide ONLY a floating point number (e.g., 24.5). If you don't know exactly, provide your best reasonable estimate."
    try:
        baseline_mae_response = llm.invoke([HumanMessage(content=query)])
        match = re.search(r'\d+\.\d+', baseline_mae_response.content)
        baseline_mae = match.group() if match else "Unknown"
    except Exception:
        baseline_mae = "Unknown"
        
    print(f"🎯 Intelligent Baseline Assessment: {baseline_mae}")
    
    # 3. Auto-download dataset
    print(f"\n📦 Triggering Auto-Download for Dataset: {dataset}...")
    try:
        subprocess.run([sys.executable, os.path.join(BASE_DIR, "scripts/prepare.py"), "--dataset", dataset])
    except Exception as e:
        print(f"⚠️ Warning: Dataset download script failed to run: {e}")
    
    # 4. Create memory.md
    memory_content = f"""# Darwin-ST memory.md

## 🎯 领域先验 (Domain Priors)
- **核心创新点**: {innovation}
- **目标数据集**: {dataset}
- **对比基线 (参考)**: {baseline} (Estimated MAE: {baseline_mae})
- **约束边界**: 必须在 `# --- INNOVATION START ---` 保护区内开发。

## 🏆 Current SOTA
(Waiting for Gen 1...)

## 🪦 The Graveyard (Failed Mutations)
(Empty)
"""
    write_file("memory.md", memory_content)
    print("\n✅ Phase 1 Setup Complete! Entering Phase 2: Autonomous Evolution Loop...\n")


if __name__ == "__main__":
    print("🚀 Initializing Darwin-ST LangGraph Orchestrator...")
    
    # Run interactive setup if needed
    setup_phase()
    
    # Initialize State
    initial_state = {
        "iteration_id": 1,
        "parent_id": get_git_hash(),
        "best_mae": 999.0, # Or parse from memory.md
        "current_mae": 999.0,
        "innovation_desc": "",
        "status": "INIT",
        "messages": [],
        "run_log": ""
    }
    
    # Start the infinite loop
    for event in app.stream(initial_state, {"recursion_limit": 1000}):
        pass
