from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseTask(ABC):
    """
    Dual-Track RLVR (Reinforcement Learning with Verifiable Rewards) 任务抽象基类。
    统一规范强化学习环境的数据供给 (Data Provider) 与真理判定 (Verifiable Reward Evaluator)。
    """

    @abstractmethod
    def get_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        获取一个批次的强化学习训练样本。
        
        Args:
            batch_size: 批次样本数量。
            
        Returns:
            List[Dict[str, Any]]: 每个字典元素必须包含:
              - 'prompt': 格式化后的输入提示词 (统一以 'Assistant: <think>\\n' 结尾诱导思维链)
              - 'ground_truth': 用于客观真理核对的基准元数据 (如 GSM8K 的最终标准数值，或代码断言用例)
              - 其他可选任务特定元数据 (如 'question', 'raw_answer' 等)
        """
        pass

    @abstractmethod
    def compute_rewards(
        self,
        prompts: List[str],
        completions: List[str],
        metadata: List[Dict[str, Any]]
    ) -> List[float]:
        """
        批量计算模型生成完成文本 (Completion) 的客观奖励值。
        
        Args:
            prompts: 输入提示词列表
            completions: 模型自回归生成的文本列表 (不含 prompt)
            metadata: 对应样本的元数据 (包含 ground_truth 等)
            
        Returns:
            List[float]: 每个 completion 对应的浮点奖励分值列表。
        """
        pass

    def __len__(self) -> int:
        """任务数据集大小，未知或无限流生成时返回 0"""
        return 0
