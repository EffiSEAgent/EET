#!/usr/bin/env python3
"""
从 extracted_experiences_llm.jsonl 中提取经验，使用LLM总结issue description，
并生成类似 extracted_experiences_summarized_gpt_5_mini.jsonl 格式的输出文件
"""
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
from datetime import datetime
import re
import argparse

# 添加项目路径以支持导入
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import litellm  # type: ignore
    USE_LITELLM = True
except ImportError:
    print("Error: Please install litellm: pip install litellm")
    sys.exit(1)


# LLM总结issue description的提示词
SUMMARY_PROMPT_TEMPLATE = """You are an expert at summarizing technical issues. Your task is to create a concise, clear summary of a programming issue that captures:
1. The core problem
2. The root cause (if mentioned)
3. The impact or expected behavior

## Instructions
- Write a clear, technical summary (2-4 sentences)
- Focus on the essential problem, not implementation details
- Include root cause if it's clearly stated
- Use technical terminology appropriately
- Keep it concise but informative

## Original Issue Description:
{issue_description}

## Task Summary Steps:
{task_summaries}

Now provide a concise summary of this issue:"""


def load_experiences(input_file: str) -> List[Dict[str, Any]]:
    """加载extracted_experiences_llm.jsonl文件"""
    experiences = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                exp = json.loads(line)
                experiences.append(exp)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line: {e}")
                continue
    return experiences


def group_by_issue_id(experiences: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """按issue_id分组"""
    grouped = defaultdict(list)
    for exp in experiences:
        issue_id = exp.get('issue_id', 'unknown')
        grouped[issue_id].append(exp)
    return dict(grouped)


def extract_original_description(experiences: List[Dict[str, Any]]) -> str:
    """从经验记录中提取原始issue description"""
    # 取第一个记录的issue_description（它们应该相同）
    if not experiences:
        return "No description available"
    
    desc = experiences[0].get('issue_description', '')
    
    if not desc:
        return "No description available"
    
    # 清理描述：移除"We're currently solving..."前缀
    if "We're currently solving" in desc:
        # 查找实际的issue描述
        patterns = [
            r"Here's the issue text:\s*(.*?)(?=\n\n|\Z)",
            r"Description\s*\n\s*(.*?)(?=\n\n|\Z)",
            r"Description\s*\n\t\s*(.*?)(?=\n\n|\Z)",
        ]
        for pattern in patterns:
            match = re.search(pattern, desc, re.DOTALL)
            if match:
                desc = match.group(1).strip()
                # 移除可能的标签和多余空白
                desc = re.sub(r'^\s*\(last modified by.*?\)\s*', '', desc, flags=re.MULTILINE)
                desc = re.sub(r'\n\s*\n', '\n', desc)  # 合并多个空行
                break
    
    # 如果描述太长，截断但保留完整句子
    if len(desc) > 2000:
        truncated = desc[:2000]
        # 尝试在句号处截断
        last_period = truncated.rfind('.')
        if last_period > 1500:  # 如果句号位置合理
            desc = truncated[:last_period + 1] + "..."
        else:
            desc = truncated + "..."
    
    return desc.strip()


def summarize_issue_description(
    original_description: str,
    task_summaries: List[str],
    model_name: str = "openai/gpt-5-mini"
) -> str:
    """使用LLM总结issue description"""
    # 构建任务摘要文本
    task_summaries_text = "\n".join([f"- {ts}" for ts in task_summaries[:10]])  # 限制前10个
    
    prompt = SUMMARY_PROMPT_TEMPLATE.format(
        issue_description=original_description,
        task_summaries=task_summaries_text
    )
    
    try:
        response = litellm.completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3  # 较低温度以获得更一致的总结
        )
        summary = response.choices[0].message.content.strip()  # type: ignore
        
        # 清理总结：移除可能的引号或代码块标记
        summary = re.sub(r'^["\']|["\']$', '', summary)
        summary = re.sub(r'```.*?```', '', summary, flags=re.DOTALL)
        summary = summary.strip()
        
        return summary
    except Exception as e:
        print(f"Error calling LLM for summarization: {e}")
        import traceback
        traceback.print_exc()
        # 如果LLM调用失败，返回原始描述的简化版本
        return original_description[:500] + "..." if len(original_description) > 500 else original_description


def calculate_confidence(experiences: List[Dict[str, Any]]) -> int:
    """计算平均confidence"""
    if not experiences:
        return 50
    
    confidences = [exp.get('confidence', 50) for exp in experiences if 'confidence' in exp]
    if not confidences:
        return 50
    
    # 计算加权平均（可以根据需要调整权重）
    # 简单平均
    avg_confidence = sum(confidences) / len(confidences)
    
    # 四舍五入到整数
    return int(round(avg_confidence))


def process_issue(
    issue_id: str,
    experiences: List[Dict[str, Any]],
    model_name: str = "openai/gpt-5-mini"
) -> Optional[Dict[str, Any]]:
    """处理单个issue，生成总结记录"""
    if not experiences:
        return None
    
    # 提取原始描述
    original_description = extract_original_description(experiences)
    
    # 提取所有task_summary
    task_summaries = []
    seen_summaries = set()  # 去重
    for exp in experiences:
        task_summary = exp.get('task_summary', '').strip()
        if task_summary and task_summary not in seen_summaries:
            task_summaries.append(task_summary)
            seen_summaries.add(task_summary)
    
    if not task_summaries:
        print(f"Warning: No task summaries found for {issue_id}")
        return None
    
    # 使用LLM总结issue description
    print(f"Summarizing issue description for {issue_id}...")
    summarized_description = summarize_issue_description(
        original_description,
        task_summaries,
        model_name
    )
    
    # 计算平均confidence
    avg_confidence = calculate_confidence(experiences)
    
    # 获取最早的created_at作为基准时间
    created_ats = [exp.get('created_at', '') for exp in experiences if exp.get('created_at')]
    created_at = min(created_ats) if created_ats else datetime.now().isoformat()
    
    # 构建输出记录
    record = {
        "issue_id": issue_id,
        "issue_description": summarized_description,
        "task_summary": task_summaries,
        "confidence": avg_confidence,
        "created_at": created_at,
        "metadata": {
            "original_count": len(experiences),
            "summarization_method": "llm",
            "model": model_name
        }
    }
    
    return record


def main():
    parser = argparse.ArgumentParser(
        description="Summarize experiences from extracted_experiences_llm.jsonl"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Input JSONL file (extracted_experiences_llm.jsonl)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="extracted_experiences_summarized.jsonl",
        help="Output JSONL file (default: extracted_experiences_summarized.jsonl)"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="openai/gpt-5-mini",
        help="LLM model name (default: openai/gpt-5-mini)"
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Limit number of issues to process (for testing)"
    )
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)
    
    # 加载经验记录
    print(f"Loading experiences from {args.input_file}...")
    experiences = load_experiences(args.input_file)
    print(f"Loaded {len(experiences)} experience records")
    
    # 按issue_id分组
    grouped = group_by_issue_id(experiences)
    print(f"Found {len(grouped)} unique issues")
    
    # 限制处理数量（用于测试）
    if args.limit:
        issue_ids = list(grouped.keys())[:args.limit]
        grouped = {issue_id: grouped[issue_id] for issue_id in issue_ids}
        print(f"Processing limited to {len(grouped)} issues")
    
    # 处理每个issue
    output_records = []
    for idx, (issue_id, issue_experiences) in enumerate(grouped.items(), 1):
        print(f"\n[{idx}/{len(grouped)}] Processing {issue_id}...")
        record = process_issue(issue_id, issue_experiences, args.model)
        if record:
            output_records.append(record)
    
    # 写入输出文件
    print(f"\nWriting {len(output_records)} summarized records to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        for record in output_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"Done! Output written to {args.output}")


if __name__ == "__main__":
    main()

