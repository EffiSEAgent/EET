#!/usr/bin/env python3
"""
测试 experience 模块的功能
"""
import os
import tempfile
import json
from pathlib import Path

from experience import Experience, ExperienceStore, TextSimilarity


def test_experience_model():
    """测试 Experience 数据模型"""
    print("=" * 60)
    print("测试 Experience 数据模型")
    print("=" * 60)
    
    # 测试创建 Experience
    exp = Experience(
        issue_id="test-001",
        issue_description="Fix login bug",
        task_summary="Check user authentication logic",
        confidence=85,
        output="Fixed authentication issue",
        metadata={"step": 1, "tool": "grep"}
    )
    print(f"✓ 创建 Experience: {exp}")
    
    # 测试 to_dict
    exp_dict = exp.to_dict()
    assert isinstance(exp_dict, dict)
    assert exp_dict["issue_id"] == "test-001"
    assert exp_dict["confidence"] == 85
    print("✓ to_dict() 正常工作")
    
    # 测试 from_dict
    exp2 = Experience.from_dict(exp_dict)
    assert exp2.issue_id == exp.issue_id
    assert exp2.confidence == exp.confidence
    print("✓ from_dict() 正常工作")
    
    # 测试 to_json 和 from_json
    json_str = exp.to_json()
    assert isinstance(json_str, str)
    exp3 = Experience.from_json(json_str)
    assert exp3.issue_id == exp.issue_id
    assert exp3.confidence == exp.confidence
    print("✓ to_json() 和 from_json() 正常工作")
    
    # 测试置信度验证
    try:
        invalid_exp = Experience(
            issue_id="test-002",
            issue_description="test",
            task_summary="test",
            confidence=150  # 无效值
        )
        assert False, "应该抛出 ValueError"
    except ValueError as e:
        print(f"✓ 置信度验证正常工作: {e}")
    
    print()


def test_text_similarity():
    """测试文本相似度计算"""
    print("=" * 60)
    print("测试文本相似度计算")
    print("=" * 60)
    
    # 测试分词
    text = "Hello world, this is a test!"
    tokens = TextSimilarity.tokenize(text)
    assert isinstance(tokens, list)
    assert "hello" in tokens
    assert "world" in tokens
    print(f"✓ tokenize() 正常工作: {tokens}")
    
    # 测试简单相似度
    text1 = "Fix login bug"
    text2 = "Fix login authentication bug"
    similarity = TextSimilarity.simple_similarity(text1, text2)
    assert 0 <= similarity <= 1
    print(f"✓ simple_similarity() 正常工作: {similarity:.3f}")
    
    # 测试 TF-IDF
    docs = [
        TextSimilarity.tokenize("Fix login bug"),
        TextSimilarity.tokenize("Check user authentication logic"),
        TextSimilarity.tokenize("Update database connection")
    ]
    idf = TextSimilarity.compute_idf(docs)
    assert isinstance(idf, dict)
    print(f"✓ compute_idf() 正常工作，IDF词汇数: {len(idf)}")
    
    tfidf = TextSimilarity.compute_tfidf(docs[0], idf)
    assert isinstance(tfidf, dict)
    print(f"✓ compute_tfidf() 正常工作")
    
    # 测试余弦相似度
    vec1 = {"a": 0.5, "b": 0.3, "c": 0.2}
    vec2 = {"a": 0.4, "b": 0.4, "c": 0.2}
    cos_sim = TextSimilarity.cosine_similarity(vec1, vec2)
    assert 0 <= cos_sim <= 1
    print(f"✓ cosine_similarity() 正常工作: {cos_sim:.3f}")
    
    print()


def test_experience_store_basic():
    """测试 ExperienceStore 基本功能"""
    print("=" * 60)
    print("测试 ExperienceStore 基本功能")
    print("=" * 60)
    
    # 使用临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_path = f.name
    
    try:
        store = ExperienceStore(storage_path=temp_path)
        print(f"✓ ExperienceStore 初始化成功: {temp_path}")
        
        # 测试添加单个经验
        exp1 = Experience(
            issue_id="issue-001",
            issue_description="Fix login bug",
            task_summary="Check user authentication logic",
            confidence=85
        )
        store.add(exp1)
        print("✓ add() 正常工作")
        
        # 测试获取所有经验
        all_exps = store.get_all()
        assert len(all_exps) == 1
        assert all_exps[0].issue_id == "issue-001"
        print(f"✓ get_all() 正常工作，获取到 {len(all_exps)} 条记录")
        
        # 测试批量添加
        exp2 = Experience(
            issue_id="issue-001",
            issue_description="Fix login bug",
            task_summary="Update password validation rules",
            confidence=90
        )
        exp3 = Experience(
            issue_id="issue-002",
            issue_description="Optimize database query performance",
            task_summary="Add index",
            confidence=75
        )
        store.add_batch([exp2, exp3])
        all_exps = store.get_all()
        assert len(all_exps) == 3
        print(f"✓ add_batch() 正常工作，现在有 {len(all_exps)} 条记录")
        
        # 测试根据 issue_id 获取
        issue_exps = store.get_by_issue_id("issue-001")
        assert len(issue_exps) == 2
        print(f"✓ get_by_issue_id() 正常工作，获取到 {len(issue_exps)} 条记录")
        
        # 测试过滤
        filtered = store.filter(lambda exp: exp.confidence >= 85)
        assert len(filtered) == 2
        print(f"✓ filter() 正常工作，获取到 {len(filtered)} 条记录")
        
        # 测试获取所有 issue IDs
        issues = store.get_issues()
        assert len(issues) == 2
        assert "issue-001" in issues
        assert "issue-002" in issues
        print(f"✓ get_issues() 正常工作: {issues}")
        
        # 测试统计信息
        stats = store.get_statistics()
        assert stats["total_experiences"] == 3
        assert stats["unique_issues"] == 2
        assert stats["avg_confidence"] > 0
        print(f"✓ get_statistics() 正常工作:")
        print(f"  - 总经验数: {stats['total_experiences']}")
        print(f"  - 唯一Issue数: {stats['unique_issues']}")
        print(f"  - 平均置信度: {stats['avg_confidence']:.2f}")
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    print()


def test_experience_store_update_delete():
    """测试 ExperienceStore 更新和删除功能"""
    print("=" * 60)
    print("测试 ExperienceStore 更新和删除功能")
    print("=" * 60)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_path = f.name
    
    try:
        store = ExperienceStore(storage_path=temp_path)
        
        # 添加测试数据
        exp1 = Experience(
            issue_id="issue-001",
            issue_description="Fix login bug",
            task_summary="Check user authentication logic",
            confidence=85
        )
        exp2 = Experience(
            issue_id="issue-001",
            issue_description="Fix login bug",
            task_summary="Update password validation rules",
            confidence=90
        )
        store.add_batch([exp1, exp2])
        
        # 测试更新
        updated_count = store.update(
            condition=lambda exp: exp.confidence < 90,
            update_func=lambda exp: Experience(
                issue_id=exp.issue_id,
                issue_description=exp.issue_description,
                task_summary=exp.task_summary,
                confidence=95,  # 更新置信度
                output=exp.output,
                metadata=exp.metadata
            )
        )
        assert updated_count == 1
        all_exps = store.get_all()
        assert all(exp.confidence >= 90 for exp in all_exps)
        print(f"✓ update() 正常工作，更新了 {updated_count} 条记录")
        
        # 测试删除
        deleted_count = store.delete_by_issue_id("issue-001")
        assert deleted_count == 2
        all_exps = store.get_all()
        assert len(all_exps) == 0
        print(f"✓ delete_by_issue_id() 正常工作，删除了 {deleted_count} 条记录")
        
        # 测试清空
        store.add(exp1)
        store.clear()
        assert len(store.get_all()) == 0
        print("✓ clear() 正常工作")
        
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    print()


def test_experience_store_search():
    """测试 ExperienceStore 搜索功能"""
    print("=" * 60)
    print("测试 ExperienceStore 搜索功能")
    print("=" * 60)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_path = f.name
    
    try:
        store = ExperienceStore(storage_path=temp_path)
        
        # 添加测试数据
        experiences = [
            Experience(
                issue_id="issue-001",
                issue_description="Fix login bug, user cannot login",
                task_summary="Check user authentication logic and password verification",
                confidence=85
            ),
            Experience(
                issue_id="issue-002",
                issue_description="Optimize database query performance",
                task_summary="Add index to improve query speed",
                confidence=75
            ),
            Experience(
                issue_id="issue-003",
                issue_description="Fix login bug, password verification failed",
                task_summary="Fix password validation rules",
                confidence=90
            ),
        ]
        store.add_batch(experiences)
        
        # 测试基于任务相似度搜索（简单方法）
        results = store.search_by_task_similarity(
            query="Check user authentication",
            top_k=2,
            use_tfidf=False
        )
        assert len(results) > 0
        assert isinstance(results[0], tuple)
        assert isinstance(results[0][0], Experience)
        assert isinstance(results[0][1], float)
        print(f"✓ search_by_task_similarity() (简单方法) 正常工作，找到 {len(results)} 条结果")
        for exp, score in results:
            print(f"  - 相似度: {score:.3f}, 任务: {exp.task_summary}")
        
        # 测试基于任务相似度搜索（TF-IDF方法）
        results_tfidf = store.search_by_task_similarity(
            query="Check user authentication",
            top_k=2,
            use_tfidf=True
        )
        assert len(results_tfidf) > 0
        print(f"✓ search_by_task_similarity() (TF-IDF方法) 正常工作，找到 {len(results_tfidf)} 条结果")
        for exp, score in results_tfidf:
            print(f"  - 相似度: {score:.3f}, 任务: {exp.task_summary}")
        
        # 测试基于Issue相似度搜索
        results_issue = store.search_by_issue_similarity(
            query="Fix login bug",
            top_k=2,
            use_tfidf=False
        )
        assert len(results_issue) > 0
        print(f"✓ search_by_issue_similarity() 正常工作，找到 {len(results_issue)} 条结果")
        for exp, score in results_issue:
            print(f"  - 相似度: {score:.3f}, Issue: {exp.issue_description[:30]}...")
        
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    print()


def test_experience_store_group():
    """测试 ExperienceStore 分组功能"""
    print("=" * 60)
    print("测试 ExperienceStore 分组功能")
    print("=" * 60)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_path = f.name
    
    try:
        store = ExperienceStore(storage_path=temp_path)
        
        # 添加相似的任务
        experiences = [
            Experience(
                issue_id="issue-001",
                issue_description="Fix login bug",
                task_summary="Check user authentication logic",
                confidence=85
            ),
            Experience(
                issue_id="issue-002",
                issue_description="Fix login bug",
                task_summary="Check user authentication and password verification",
                confidence=90
            ),
            Experience(
                issue_id="issue-003",
                issue_description="Optimize database query",
                task_summary="Add database index",
                confidence=75
            ),
        ]
        store.add_batch(experiences)
        
        # 测试分组
        groups = store.group_by_task_similarity(
            similarity_threshold=0.3,
            use_tfidf=False
        )
        assert len(groups) > 0
        print(f"✓ group_by_task_similarity() 正常工作，找到 {len(groups)} 个组")
        for i, group in enumerate(groups):
            print(f"  组 {i+1}: {len(group)} 条经验")
            for exp in group:
                print(f"    - {exp.task_summary}")
        
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    print()


def test_experience_store_import_export():
    """测试 ExperienceStore 导入导出功能"""
    print("=" * 60)
    print("测试 ExperienceStore 导入导出功能")
    print("=" * 60)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_path = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        export_path = f.name
    
    try:
        store = ExperienceStore(storage_path=temp_path)
        
        # 添加测试数据
        exp1 = Experience(
            issue_id="issue-001",
            issue_description="Fix login bug",
            task_summary="Check user authentication logic",
            confidence=85
        )
        exp2 = Experience(
            issue_id="issue-002",
            issue_description="Optimize database query",
            task_summary="Add index",
            confidence=75
        )
        store.add_batch([exp1, exp2])
        
        # 测试导出
        store.export_to_json(export_path)
        assert os.path.exists(export_path)
        with open(export_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            assert len(data) == 2
        print(f"✓ export_to_json() 正常工作，导出到 {export_path}")
        
        # 测试导入（替换模式）
        store2 = ExperienceStore(storage_path=temp_path + ".new")
        store2.import_from_json(export_path, merge=False)
        assert len(store2.get_all()) == 2
        print("✓ import_from_json() (替换模式) 正常工作")
        
        # 测试导入（合并模式）
        store2.add(exp1)
        assert len(store2.get_all()) == 3
        store2.import_from_json(export_path, merge=True)
        # 合并后应该有 3 + 2 = 5 条（因为exp1和导入的记录可能不完全相同）
        assert len(store2.get_all()) == 5
        print("✓ import_from_json() (合并模式) 正常工作")
        
    finally:
        for path in [temp_path, export_path, temp_path + ".new"]:
            if os.path.exists(path):
                os.unlink(path)
    
    print()


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始测试 experience 模块")
    print("=" * 60 + "\n")
    
    try:
        test_experience_model()
        test_text_similarity()
        test_experience_store_basic()
        test_experience_store_update_delete()
        test_experience_store_search()
        test_experience_store_group()
        test_experience_store_import_export()
        
        print("=" * 60)
        print("✓ 所有测试通过！")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ 测试失败: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

