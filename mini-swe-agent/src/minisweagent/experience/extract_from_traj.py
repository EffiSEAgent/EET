"""
从轨迹文件中提取经验记录
"""
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import re

# 添加项目路径以支持导入
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from minisweagent.experience.models import Experience
except ImportError:
    # 如果作为独立脚本运行
    from models import Experience


def extract_command_from_content(content: str) -> Optional[str]:
    """从 assistant 消息中提取命令"""
    # 查找 bash 代码块
    bash_pattern = r'```bash\s*\n(.*?)\n```'
    matches = re.findall(bash_pattern, content, re.DOTALL)
    if matches:
        return matches[0].strip()
    return None


def extract_thought_from_content(content: str) -> str:
    """从 assistant 消息中提取思考部分"""
    # 提取 THOUGHT 部分
    thought_pattern = r'THOUGHT:\s*(.*?)(?:\n\n```|$)'
    match = re.search(thought_pattern, content, re.DOTALL)
    if match:
        thought = match.group(1).strip()
        # 简化思考内容，只保留关键信息
        # 移除过长的内容
        if len(thought) > 500:
            thought = thought[:500] + "..."
        return thought
    return ""


def extract_issue_description(messages: List[Dict]) -> str:
    """从消息列表中提取 Issue 描述（problem_statement）"""
    # 第一个 user 消息通常包含 pr_description，其中包含 problem_statement
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            # 查找 <pr_description> 标签内的内容
            # 内容格式通常是：标题 + \n + 详细描述
            pr_pattern = r'<pr_description>\s*(.*?)\s*</pr_description>'
            match = re.search(pr_pattern, content, re.DOTALL)
            if match:
                desc = match.group(1).strip()
                # 移除 "Consider the following PR description:" 这样的前缀
                desc = re.sub(r'^Consider the following PR description:\s*', '', desc, flags=re.IGNORECASE)
                desc = desc.strip()
                # 简化描述，保留前500字符
                if len(desc) > 500:
                    desc = desc[:500] + "..."
                return desc
    return ""


def calculate_confidence(
    exit_status: str,
    is_last_step: bool,
    has_output: bool,
    output_success: bool = False
) -> int:
    """
    根据上下文计算置信度
    
    Args:
        exit_status: 任务的退出状态
        is_last_step: 是否是最后一步
        has_output: 是否有输出
        output_success: 输出是否成功（returncode == 0）
    """
    base_confidence = 50
    
    # 如果任务成功提交，增加置信度
    if exit_status == "Submitted":
        base_confidence = 85
    
    # 如果命令执行成功，增加置信度
    if output_success:
        base_confidence += 10
    
    # 如果有输出结果，增加置信度
    if has_output:
        base_confidence += 5
    
    # 限制在 1-100 范围内
    return min(100, max(1, base_confidence))


def classify_task_type(command: str, thought: str) -> str:
    """
    根据命令和思考内容分类任务类型
    
    Returns:
        任务类型分类（如 "查找文件", "读取代码", "修改代码", "运行测试" 等）
    """
    command_lower = command.lower() if command else ""
    thought_lower = thought.lower() if thought else ""
    
    # 查找文件/目录
    if any(cmd in command_lower for cmd in ["find", "ls", "locate", "which", "whereis"]):
        return "查找文件或目录"
    
    # 读取文件内容
    if any(cmd in command_lower for cmd in ["cat", "head", "tail", "grep", "sed -n", "nl", "less", "more", "view"]):
        if "grep" in command_lower:
            return "搜索代码模式"
        return "读取文件内容"
    
    # 修改代码
    if any(cmd in command_lower for cmd in ["sed -i", "vim", "nano", "ed", "python -c", "cat <<"]):
        if "cat <<" in command_lower or "python -c" in command_lower:
            return "创建或修改文件"
        return "编辑代码文件"
    
    # 运行测试/脚本
    if any(cmd in command_lower for cmd in ["python", "pytest", "test", "run", "bash", "sh", "./"]):
        if "test" in command_lower or "pytest" in command_lower:
            return "运行测试"
        return "执行脚本或命令"
    
    # Git 操作
    if command_lower.startswith("git "):
        if "diff" in command_lower:
            return "查看代码变更"
        elif "add" in command_lower:
            return "提交代码变更"
        return "Git 操作"
    
    # 分析/理解代码
    if any(keyword in thought_lower for keyword in ["understand", "analyze", "examine", "review", "检查", "分析", "理解"]):
        return "分析代码逻辑"
    
    # 默认根据思考内容推断
    if "fix" in thought_lower or "修改" in thought_lower or "修复" in thought_lower:
        return "修复代码问题"
    if "test" in thought_lower or "测试" in thought_lower:
        return "编写或运行测试"
    if "create" in thought_lower or "创建" in thought_lower:
        return "创建新文件"
    
    return "执行命令"


def extract_experiences_from_traj_file(
    traj_path: str,
    problem_statement: str = None
) -> List[Experience]:
    """
    从单个轨迹文件中提取经验记录
    
    Args:
        traj_path: 轨迹文件路径
        
    Returns:
        经验记录列表
    """
    with open(traj_path, 'r', encoding='utf-8') as f:
        traj_data = json.load(f)
    
    experiences = []
    
    # 提取基本信息
    info = traj_data.get("info", {})
    # instance_id 在顶层，不在 info 中（因为 save_traj 使用 **kwargs）
    issue_id = traj_data.get("instance_id", "")
    
    # 如果顶层没有，尝试从文件名提取
    if not issue_id:
        filename = os.path.basename(traj_path)
        # 例如: astropy__astropy-6938.traj.json -> astropy__astropy-6938
        if filename.endswith(".traj.json"):
            issue_id = filename[:-10]  # 移除 .traj.json
    
    exit_status = info.get("exit_status", "")
    
    messages = traj_data.get("messages", [])
    if not messages:
        return experiences
    
    # 提取 Issue 描述
    # 优先使用从数据集加载的 problem_statement，否则从消息中提取
    if problem_statement:
        issue_description = problem_statement
        if len(issue_description) > 500:
            issue_description = issue_description[:500] + "..."
    else:
        issue_description = extract_issue_description(messages)
    
    # 遍历消息，提取每一步的经验
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        
        content = msg.get("content", "")
        if not content:
            continue
        
        # 提取思考过程和命令
        thought = extract_thought_from_content(content)
        command = extract_command_from_content(content)
        
        if not thought and not command:
            continue
        
        # 获取下一步的用户消息（包含命令输出）
        output = ""
        output_success = False
        if i + 1 < len(messages):
            next_msg = messages[i + 1]
            if next_msg.get("role") == "user":
                next_content = next_msg.get("content", "")
                # 提取输出
                output_pattern = r'<output>\s*(.*?)\s*</output>'
                output_match = re.search(output_pattern, next_content, re.DOTALL)
                if output_match:
                    output = output_match.group(1).strip()
                    # 限制输出长度
                    if len(output) > 1000:
                        output = output[:1000] + "..."
                
                # 检查返回码
                returncode_pattern = r'<returncode>(\d+)</returncode>'
                returncode_match = re.search(returncode_pattern, next_content)
                if returncode_match:
                    returncode = int(returncode_match.group(1))
                    output_success = (returncode == 0)
        
        # 分类任务类型
        task_type = classify_task_type(command, thought)
        
        # 构建任务摘要：任务类型 + 简要描述
        if thought:
            # 提取思考中的关键信息（前100个字符）
            thought_brief = thought[:100].strip()
            if len(thought) > 100:
                thought_brief += "..."
            task_summary = f"{task_type}: {thought_brief}"
        elif command:
            # 如果没有思考，使用命令的简要描述
            command_brief = command[:80].strip()
            if len(command) > 80:
                command_brief += "..."
            task_summary = f"{task_type}: {command_brief}"
        else:
            continue
        
        # 计算置信度
        is_last_step = (i == len([m for m in messages if m.get("role") == "assistant"]) - 1)
        confidence = calculate_confidence(
            exit_status=exit_status,
            is_last_step=is_last_step,
            has_output=bool(output),
            output_success=output_success
        )
        
        # 创建经验记录
        exp = Experience(
            issue_id=issue_id,
            issue_description=issue_description,
            task_summary=task_summary,
            confidence=confidence,
            output=output,
            metadata={
                "command": command,
                "step_index": i,
                "exit_status": exit_status,
                "traj_file": os.path.basename(traj_path)
            }
        )
        
        experiences.append(exp)
    
    return experiences


def load_dataset_for_instances(
    dataset_name: str = "princeton-nlp/SWE-Bench_Lite",
    split: str = "test"
) -> Dict[str, Dict]:
    """
    从数据集加载实例信息，返回 instance_id -> instance 的映射
    
    Args:
        dataset_name: 数据集名称
        split: 数据集分割
        
    Returns:
        实例字典，key 为 instance_id
    """
    try:
        from datasets import load_dataset
        dataset = load_dataset(dataset_name, split=split)
        instances = {}
        for instance in dataset:
            instance_id = instance.get("instance_id", "")
            if instance_id:
                instances[instance_id] = instance
        print(f"从数据集 {dataset_name} 加载了 {len(instances)} 个实例")
        return instances
    except Exception as e:
        print(f"警告: 无法加载数据集 {dataset_name}: {e}")
        return {}


def load_filter_ids(filter_file: str | None) -> set[str]:
    """
    从文件中加载要过滤的 issue ID 列表
    
    Args:
        filter_file: 包含 issue ID 的文件路径，每行一个 ID
        
    Returns:
        issue ID 集合
    """
    if not filter_file or not Path(filter_file).exists():
        return set()
    
    filter_ids = set()
    with open(filter_file, 'r', encoding='utf-8') as f:
        for line in f:
            issue_id = line.strip()
            if issue_id:
                filter_ids.add(issue_id)
    
    print(f"从 {filter_file} 加载了 {len(filter_ids)} 个过滤 ID")
    return filter_ids


def extract_experiences_from_directory(
    directory: str,
    pattern: str = "*.traj.json",
    dataset_name: str = None,
    dataset_split: str = "test",
    filter_ids_file: str = None
) -> List[Experience]:
    """
    从目录中的所有轨迹文件提取经验
    
    Args:
        directory: 目录路径
        pattern: 文件匹配模式
        dataset_name: 数据集名称（可选），如果提供，将从数据集加载 problem_statement
        dataset_split: 数据集分割
        filter_ids_file: 过滤文件路径，只提取文件中列出的 issue ID
        
    Returns:
        经验记录列表
    """
    directory_path = Path(directory)
    all_experiences = []
    
    # 加载过滤 ID（如果提供）
    filter_ids = load_filter_ids(filter_ids_file) if filter_ids_file else set()
    
    # 加载数据集（如果提供）
    dataset_instances = {}
    if dataset_name:
        dataset_instances = load_dataset_for_instances(dataset_name, dataset_split)
    
    # 查找所有轨迹文件
    traj_files = list(directory_path.rglob(pattern))
    
    print(f"找到 {len(traj_files)} 个轨迹文件")
    if filter_ids:
        print(f"将只处理 {len(filter_ids)} 个指定的 issue ID")
    
    processed_count = 0
    skipped_count = 0
    
    for traj_file in traj_files:
        try:
            # 从文件名提取 instance_id
            # 文件路径格式：{instance_id}/{instance_id}.traj.json
            # 父目录名就是 instance_id
            instance_id = traj_file.parent.name
            
            # 如果提供了过滤文件，只处理列出的 ID
            if filter_ids and instance_id not in filter_ids:
                skipped_count += 1
                continue
            
            # 如果数据集已加载，尝试获取 problem_statement
            instance_data = dataset_instances.get(instance_id, {})
            problem_statement = instance_data.get("problem_statement", "")
            
            experiences = extract_experiences_from_traj_file(
                str(traj_file),
                problem_statement=problem_statement
            )
            all_experiences.extend(experiences)
            processed_count += 1
            print(f"  ✓ {traj_file.name}: 提取了 {len(experiences)} 条经验")
        except Exception as e:
            print(f"  ✗ {traj_file.name}: 提取失败 - {e}")
    
    if filter_ids:
        print(f"\n处理了 {processed_count} 个文件，跳过了 {skipped_count} 个文件")
    
    return all_experiences


def main():
    """主函数：从 swe_lite_test 目录提取经验"""
    import sys
    
    # 默认使用 swe_lite_test 目录
    if len(sys.argv) > 1:
        traj_directory = sys.argv[1]
    else:
        # 假设在 mini-swe-agent 项目根目录下运行
        traj_directory = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "swe_lite_test"
        )
    
    # 数据集名称（可选）
    dataset_name = os.environ.get("SWEBENCH_DATASET", "princeton-nlp/SWE-Bench_Lite")
    
    # 过滤文件路径（可选）
    filter_ids_file = None
    if len(sys.argv) > 2:
        filter_ids_file = sys.argv[2]
    else:
        # 默认查找同目录下的 exp_ids.txt
        default_filter_file = Path(__file__).parent / "exp_ids.txt"
        if default_filter_file.exists():
            filter_ids_file = str(default_filter_file)
    
    print(f"从目录提取经验: {traj_directory}")
    if dataset_name:
        print(f"使用数据集: {dataset_name}")
    if filter_ids_file:
        print(f"使用过滤文件: {filter_ids_file}")
    print("=" * 60)
    
    # 提取经验
    experiences = extract_experiences_from_directory(
        traj_directory,
        dataset_name=dataset_name if dataset_name else None,
        filter_ids_file=filter_ids_file
    )
    
    print("=" * 60)
    print(f"总共提取了 {len(experiences)} 条经验记录")
    
    # 保存到 ExperienceStore
    # 确保路径正确以便导入
    project_root = Path(__file__).parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # 直接导入模块，避免相对导入问题
    import importlib.util
    models_path = Path(__file__).parent / "models.py"
    store_path = Path(__file__).parent / "store.py"
    
    # 加载 models 模块
    models_spec = importlib.util.spec_from_file_location("experience_models", models_path)
    models_module = importlib.util.module_from_spec(models_spec)
    models_module.__package__ = "minisweagent.experience"
    models_spec.loader.exec_module(models_module)
    Experience = models_module.Experience
    
    # 加载 store 模块（需要先设置 models）
    import sys
    sys.modules['minisweagent.experience.models'] = models_module
    
    store_spec = importlib.util.spec_from_file_location("experience_store", store_path)
    store_module = importlib.util.module_from_spec(store_spec)
    store_module.__package__ = "minisweagent.experience"
    store_module.Experience = Experience  # 手动设置依赖
    store_spec.loader.exec_module(store_module)
    ExperienceStore = store_module.ExperienceStore
    
    # 确定存储路径
    store_path = os.path.join(
        os.path.dirname(__file__),
        "extracted_experiences.jsonl"
    )
    
    store = ExperienceStore(storage_path=store_path)
    
    # 如果提供了过滤文件，清空旧数据只保留过滤后的经验
    if filter_ids_file:
        print("清空旧数据，只保留过滤后的经验...")
        store.clear()
    
    # 批量添加
    store.add_batch(experiences)
    
    print(f"经验已保存到: {store_path}")
    
    # 显示统计信息
    stats = store.get_statistics()
    print("\n统计信息:")
    print(f"  总记录数: {stats['total_experiences']}")
    print(f"  唯一 Issue 数: {stats['unique_issues']}")
    print(f"  平均置信度: {stats['avg_confidence']:.2f}")
    print(f"  每个 Issue 平均步骤数: {stats['avg_steps_per_issue']:.2f}")


if __name__ == "__main__":
    main()

