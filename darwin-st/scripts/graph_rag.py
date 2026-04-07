import os
from neo4j import GraphDatabase

# ================= 配置区 =================
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "darwin_st_2024")

class DarwinSTGraphRAG:
    """
    专门为 Darwin-ST Agent 打造的 GraphRAG 检索接口。
    用于在模型架构演化 (Architecture Evolution) 和超参数优化时提供“诊断与处方”。
    """
    def __init__(self, uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def query(self, cypher, parameters=None):
        """执行基础 Cypher 查询"""
        with self.driver.session() as session:
            result = session.run(cypher, parameters or {})
            return [record.data() for record in result]

    def diagnose_and_prescribe(self, dataset_type, benchmark_name=None, symptom_keyword=None):
        """
        场景 1: “对症下药” (核心能力)
        Agent 输入当前的数据集类型 (如 Graph)、基准测试名称 (如 PeMS04)，以及遇到的瓶颈关键词 (如 "长时预测")。
        RAG 返回适用的结构修改建议和超参数代码。
        """
        cypher = """
        MATCH (d:Dataset)<-[:OBSERVED_IN]-(s:Symptom)<-[:RESOLVES]-(a:Action)-[:BASED_ON]->(c:Concept)
        WHERE d.type = $dataset_type
        """
        params = {"dataset_type": dataset_type}

        if benchmark_name:
            cypher += " AND $benchmark_name IN d.benchmark_names"
            params["benchmark_name"] = benchmark_name
        
        if symptom_keyword:
            cypher += " AND (s.name CONTAINS $keyword OR s.description CONTAINS $keyword)"
            params["keyword"] = symptom_keyword
            
        cypher += """
        RETURN a.name AS Action, a.description AS Mechanism, a.effectiveness AS Effectiveness, 
               a.math_or_code_structure AS Code_Structure, a.recipe AS Hyperparameters, 
               a.anti_pattern AS Anti_Pattern, c.id AS Base_Concept, s.name AS Target_Symptom
        """
        return self.query(cypher, params)

    def find_solutions_by_concept(self, concept_id):
        """
        场景 2: “理论验证”
        Agent 想要尝试一个具体的概念 (如 dynamic_graph_convolution)，想知道在这个概念下，前人是怎么实现并取得什么效果的。
        """
        cypher = """
        MATCH (c:Concept {id: $concept_id})<-[:BASED_ON]-(a:Action)-[:APPLIED_TO]->(d:Dataset)
        MATCH (a)-[:RESOLVES]->(s:Symptom)
        RETURN a.name AS Action, a.math_or_code_structure AS Implementation, a.effectiveness AS Effectiveness,
               d.domain AS Domain, s.name AS Solved_Issue
        """
        return self.query(cypher, {"concept_id": concept_id})

    def get_anti_patterns(self):
        """
        场景 3: “雷区排查”
        在 Agent 决定突变模型架构之前，拉取所有的“反面经验”(消融实验教训)，确保变异操作不会触发已知的失败路径。
        """
        cypher = """
        MATCH (a:Action)
        WHERE a.anti_pattern IS NOT NULL AND a.anti_pattern <> ""
        RETURN a.name AS Action, a.anti_pattern AS Avoid_These_Mistakes
        """
        return self.query(cypher)

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Darwin-ST Agent GraphRAG Interface")
    parser.add_argument("--action", type=str, required=True, choices=["diagnose", "verify", "antipatterns"], help="The GraphRAG action to perform.")
    parser.add_argument("--dataset_type", type=str, default="Graph", help="Dataset type (e.g., Graph)")
    parser.add_argument("--benchmark", type=str, help="Optional benchmark name (e.g., PeMS04)")
    parser.add_argument("--symptom", type=str, help="Keyword for the symptom/bottleneck to diagnose")
    parser.add_argument("--concept", type=str, help="The theoretical concept ID to verify")
    parser.add_argument("--json", action="store_true", help="Output raw JSON for agent digestion")
    
    args = parser.parse_args()
    rag = DarwinSTGraphRAG()
    
    results = []
    try:
        if args.action == "diagnose":
            if not args.symptom:
                raise ValueError("Diagnose action requires --symptom argument.")
            results = rag.diagnose_and_prescribe(dataset_type=args.dataset_type, benchmark_name=args.benchmark, symptom_keyword=args.symptom)
        elif args.action == "verify":
            if not args.concept:
                raise ValueError("Verify action requires --concept argument.")
            results = rag.find_solutions_by_concept(concept_id=args.concept)
        elif args.action == "antipatterns":
            results = rag.get_anti_patterns()
            
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            if not results:
                print("未从文献图谱中检索到相关的处方或经验。建议 Agent 依靠自身创造力探索，或拓宽搜索关键词。")
            for idx, res in enumerate(results):
                print(f"=== [文献图谱处方 {idx+1}] ===")
                for k, v in res.items():
                    if v:
                        print(f"{k}: {v}")
                print("-" * 40)
    except Exception as e:
        print(f"GraphRAG Query Error: {str(e)}")
    finally:
        rag.close()
