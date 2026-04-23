import json
import os
from pyecharts import options as opts
from pyecharts.charts import Tree

LOG_FILE = "evolution_log.jsonl"
OUTPUT_HTML = "evolution_tree.html"

def load_lineage():
    if not os.path.exists(LOG_FILE):
        return []
    
    logs = []
    with open(LOG_FILE, 'r') as f:
        for line in f:
            if line.strip():
                logs.append(json.loads(line.strip()))
    return logs

def build_tree_dict(logs):
    # Create a mapping of parent_id to its children
    children_map = {}
    node_map = {}
    
    # Add an artificial root if empty
    if not logs:
        return [{"name": "Seed", "value": "N/A"}]

    for log in logs:
        git_hash = log.get("git_hash", f"gen_{log['iteration_id']}")
        parent = log.get("parent_id", "root")
        status = log.get("status", "DISCARD")
        mae = log.get("mae", "N/A")
        desc = log.get("innovation", "Unknown")
        
        # Color coding: Green for KEEP (SOTA), Gray for DISCARD (Graveyard)
        color = "#4CAF50" if status == "KEEP" else "#9E9E9E"
        
        node = {
            "name": f"Gen {log['iteration_id']}\n{git_hash[:7]}",
            "value": mae,
            "desc": desc,
            "status": status,
            "itemStyle": {"color": color},
            "label": {
                "formatter": "{b}\nMAE: {c}"
            }
        }
        
        node_map[git_hash] = node
        
        if parent not in children_map:
            children_map[parent] = []
        children_map[parent].append(node)

    # Recursive function to build the nested dictionary required by pyecharts
    def get_children(parent_id):
        if parent_id not in children_map:
            return []
        
        res = []
        for child in children_map[parent_id]:
            # Extract git hash from node name text (hacky but works since we know the format)
            c_hash = child["name"].split("\n")[1]
            # Fetch grandchildren
            grandchildren = get_children(c_hash)
            if grandchildren:
                child["children"] = grandchildren
            res.append(child)
        return res

    # Find the root node(s). The first log is usually the root.
    root_parent = logs[0].get("parent_id", "root")
    tree_data = get_children(root_parent)
    
    # If tree_data is empty, fallback to the first node as standalone
    if not tree_data and logs:
         first = logs[0]
         tree_data = [node_map[first.get("git_hash", f"gen_{first['iteration_id']}")] ]

    return tree_data

def plot():
    logs = load_lineage()
    if not logs:
        print("No evolution logs found. Skipping visualization.")
        return

    tree_data = build_tree_dict(logs)
    
    c = (
        Tree(init_opts=opts.InitOpts(width="1200px", height="800px", page_title="Darwin-ST Evolution Tree"))
        .add(
            "",
            tree_data,
            orient="TB", # Top to Bottom
            initial_tree_depth=-1, # Expand all
            label_opts=opts.LabelOpts(
                position="top",
                horizontal_align="right",
                vertical_align="middle",
                rotate=-90,
            ),
            leaves_label_opts=opts.LabelOpts(
                position="bottom",
                horizontal_align="center",
                vertical_align="middle",
                rotate=0,
            ),
            tooltip_opts=opts.TooltipOpts(
                is_show=True,
                formatter=lambda params: f"<b>{params.data.get('name')}</b><br/>"
                                         f"Status: {params.data.get('status')}<br/>"
                                         f"MAE: {params.data.get('value')}<br/>"
                                         f"Innovation: {params.data.get('desc')}"
            )
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Darwin-ST Autonomous Evolution Lineage")
        )
    )
    c.render(OUTPUT_HTML)
    print(f"🧬 Evolution tree rendered to {OUTPUT_HTML}")

if __name__ == "__main__":
    plot()
