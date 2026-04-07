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
        RAG 返回适用的结构修改建议和超参数代码。具备软性分词模糊匹配能力。
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
            # 分词器：去掉停用词，只保留长度大于2的单词
            stop_words = ['and', 'with', 'for', 'the', 'use', 'to', 'of', 'in', 'on', 'a']
            keywords = [kw.lower() for kw in symptom_keyword.replace('-', ' ').split() if len(kw) > 2 and kw.lower() not in stop_words]
            if keywords:
                conditions = []
                for i, kw in enumerate(keywords):
                    params[f"kw_{i}"] = kw
                    conditions.append(f"(toLower(s.name) CONTAINS $kw_{i} OR toLower(s.description) CONTAINS $kw_{i})")
                cypher += " AND (" + " OR ".join(conditions) + ")"
            
        cypher += """
        RETURN a.name AS Action, a.description AS Mechanism, a.effectiveness AS Effectiveness, 
               a.math_or_code_structure AS Code_Structure, a.recipe AS Hyperparameters, 
               a.anti_pattern AS Anti_Pattern, c.id AS Base_Concept, s.name AS Target_Symptom
        """
        return self.query(cypher, params)

    def find_solutions_by_concept(self, concept_id):
        """
        场景 2: “理论验证”
        具备柔性模糊匹配能力：将长句 concept_id 拆解为短 Token 在图谱中执行正则/包含查询。
        """
        cypher = """
        MATCH (c:Concept)<-[:BASED_ON]-(a:Action)-[:APPLIED_TO]->(d:Dataset)
        MATCH (a)-[:RESOLVES]->(s:Symptom)
        """
        params = {}
        stop_words = ['and', 'with', 'for', 'the', 'use', 'to', 'of', 'in', 'on', 'a']
        keywords = [kw.lower() for kw in concept_id.replace('-', ' ').split() if len(kw) > 2 and kw.lower() not in stop_words]
        
        if keywords:
            conditions = []
            for i, kw in enumerate(keywords):
                params[f"kw_{i}"] = kw
                conditions.append(f"(toLower(c.id) CONTAINS $kw_{i} OR toLower(c.name) CONTAINS $kw_{i})")
            cypher += " WHERE " + " OR ".join(conditions)
        else:
            cypher += " WHERE toLower(c.id) CONTAINS toLower($concept_id)"
            params["concept_id"] = concept_id

        cypher += """
        RETURN a.name AS Action, a.math_or_code_structure AS Implementation, a.effectiveness AS Effectiveness,
               d.domain AS Domain, s.name AS Solved_Issue
        """
        return self.query(cypher, params)

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

    def get_inspirations(self):
        """
        场景 4: “灵感漫游超市” (Primary Drive)
        不查错误，不验真伪，抛开目的性。无差别地丛图谱里随机抽样返回一批不同的模型构建方法，
        包含基础机制和顶级架构，供 Agent 自由提取和发散组合。
        """
        cypher = """
        MATCH (c:Concept)<-[:BASED_ON]-(a:Action)
        WITH c, a, rand() as r
        ORDER BY r
        LIMIT 10
        RETURN c.id AS Methodology_Family, a.name AS Specific_Mechanism, a.description AS How_It_Works, a.effectiveness AS Claimed_Effect
        """
        return self.query(cypher)

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Darwin-ST Agent GraphRAG Interface")
    parser.add_argument("--action", type=str, required=True, choices=["diagnose", "verify", "antipatterns", "inspire"], help="The GraphRAG action to perform.")
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
        elif args.action == "inspire":
            results = rag.get_inspirations()
            
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            if not results:
                print("未从文献图谱中检索到相关的精准处方或经验。这可能是因为您传入的句子过长，导致无法在图谱中定位具体的概念节点。")
                print("💡 [GraphRAG 提示]: 请不要灰心！请尝试从您的长句中提取 1~2 个最核心机制单词进行重新查询！")
                print("-" * 40)
                try:
                    hints = rag.query("MATCH (c:Concept) RETURN c.id AS concept_id LIMIT 10")
                    if hints:
                        print("👉 知识图谱中存在的【部分学术概念】示例：")
                        for h in hints:
                            print(f"   - {h['concept_id']}")
                except Exception:
                    pass
                print("-" * 40)
            else:
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
