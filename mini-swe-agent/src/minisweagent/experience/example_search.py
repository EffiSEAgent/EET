#!/usr/bin/env python3
"""示例：从经验文件中加载并检索经验"""
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from minisweagent.experience.store import ExperienceStore
from minisweagent.experience.models import Experience


def main():
    # 初始化存储，指定经验文件路径
    storage_path = Path(__file__).parent / "extracted_experiences_lite_llm.jsonl"
    store = ExperienceStore(storage_path=str(storage_path))
    
    # 加载所有经验
    all_experiences = store.get_all()
    print(f"加载了 {len(all_experiences)} 条经验记录\n")
    
    # 示例1: 按 issue_id 检索
    print("=" * 60)
    print("示例1: 按 issue_id 检索")
    print("=" * 60)
    issue_id = "psf__requests-1963"
    experiences = store.get_by_issue_id(issue_id)
    print(f"Issue ID: {issue_id}")
    print(f"找到 {len(experiences)} 条相关经验:")
    for i, exp in enumerate(experiences[:3], 1):
        print(f"\n{i}. {exp.task_summary}")
        print(f"   置信度: {exp.confidence}")
        print(f"   命令: {exp.metadata.get('command', 'N/A')}")
    
    # 示例2: 按任务相似度搜索
    print("\n" + "=" * 60)
    print("示例2: 按任务相似度搜索")
    print("=" * 60)
    query = "read code file examine"
    results = store.search_by_task_similarity(
        query=query,
        top_k=5,
        min_similarity=0.1,
        use_tfidf=False
    )
    print(f"查询: '{query}'")
    print(f"找到 {len(results)} 条相似经验:")
    for i, (exp, similarity) in enumerate(results, 1):
        print(f"\n{i}. 相似度: {similarity:.3f}")
        print(f"   {exp.task_summary}")
        print(f"   Issue: {exp.issue_id}")
        print(f"   置信度: {exp.confidence}")
    
    # 示例3: 按 Issue 描述相似度搜索
    print("\n" + "=" * 60)
    print("示例3: 按 Issue 描述相似度搜索")
    print("=" * 60)
    issue_query = "redirect request method selection"
    results = store.search_by_issue_similarity(
        query=issue_query,
        top_k=5,
        min_similarity=0.1,
        use_tfidf=True
    )
    print(f"查询: '{issue_query}'")
    print(f"找到 {len(results)} 条相似经验:")
    for i, (exp, similarity) in enumerate(results, 1):
        print(f"\n{i}. 相似度: {similarity:.3f}")
        print(f"   Issue: {exp.issue_id}")
        print(f"   任务: {exp.task_summary}")
        print(f"   描述: {exp.issue_description[:100]}...")
    
    # 示例4: 自定义过滤条件
    print("\n" + "=" * 60)
    print("示例4: 自定义过滤条件")
    print("=" * 60)
    high_confidence = store.filter(lambda exp: exp.confidence >= 85)
    print(f"高置信度经验 (>=85): {len(high_confidence)} 条")
    
    read_code_tasks = store.filter(
        lambda exp: "read code" in exp.task_summary.lower() or 
                   "examine" in exp.task_summary.lower()
    )
    print(f"读取代码相关任务: {len(read_code_tasks)} 条")
    
    # 示例5: 统计信息
    print("\n" + "=" * 60)
    print("示例5: 统计信息")
    print("=" * 60)
    stats = store.get_statistics()
    print(f"总记录数: {stats['total_experiences']}")
    print(f"唯一 Issue 数: {stats['unique_issues']}")
    print(f"平均置信度: {stats['avg_confidence']:.2f}")
    print(f"每个 Issue 平均步骤数: {stats['avg_steps_per_issue']:.2f}")
    
    # 示例6: 查找特定命令的经验
    print("\n" + "=" * 60)
    print("示例6: 查找包含特定命令的经验")
    print("=" * 60)
    grep_experiences = store.filter(
        lambda exp: exp.metadata.get('command', '').startswith('grep')
    )
    print(f"包含 'grep' 命令的经验: {len(grep_experiences)} 条")
    for i, exp in enumerate(grep_experiences[:3], 1):
        print(f"\n{i}. {exp.task_summary}")
        print(f"   命令: {exp.metadata.get('command')}")
        print(f"   Issue: {exp.issue_id}")


if __name__ == "__main__":
    main()

