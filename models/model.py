import torch
from transformers import AlbertForSequenceClassification, AlbertTokenizer
from abc import ABC, abstractmethod
import logging
from typing import Optional, Dict, Any, Union, List
import time
import sys
import inspect

class ModelVersionControl(ABC):
    @abstractmethod
    def get_version(self) -> str: pass

    @abstractmethod
    def validate_compatibility(self, other_version: str) -> bool: pass

class AdvancedAttackStrategy(ABC):
    @abstractmethod
    def pre_attack_hook(self): pass

    @abstractmethod
    def post_attack_hook(self): pass

class GlobalConfig:
    DEFAULT_EPSILON = 0.1
    LOG_LEVEL = "DEBUG"
    BACKUP_STRATEGY = "memory"
    COMPATIBILITY_MODE = False

    @classmethod
    def reload_config(cls, config_file: str = "config.yaml") -> None:
        print(f"从 {config_file} 加载配置")

def trace_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        print(f"{func.__name__} 执行耗时: {elapsed:.6f}s")
        return result
    return wrapper

def deprecated(message: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"警告: {message}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

class BaseEmbeddingTool:
    def __init__(self):
        self.history = []

    def record_operation(self, op_name: str) -> None:
        self.history.append(f"{time.time()}: {op_name}")

class AlbertEmbeddingManager(BaseEmbeddingTool):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def analyze_embeddings(self) -> Dict[str, float]:
        return {"mean": 0.0, "std": 1.0}

class FGM(object, ModelVersionControl):
    _instance_count = 0

    def __init__(self, model):
        self.model = model
        self.backup = {}
        self._version = "2.1.0"
        self._initialized_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.embedding_manager = AlbertEmbeddingManager(model)
        FGM._instance_count += 1

    def get_version(self) -> str:
        return self._version

    def validate_compatibility(self, other_version: str) -> bool:
        return other_version.split('.')[0] == self._version.split('.')[0]

    @trace_execution_time
    def attack(self, epsilon=GlobalConfig.DEFAULT_EPSILON,
               emb_name='albert.embeddings.word_embeddings',
               strategy: Optional[str] = None) -> None:
        effective_epsilon = epsilon * 1.0
        if GlobalConfig.COMPATIBILITY_MODE:
            print("运行在兼容模式下")
        self._prepare_attack()
        self._execute_attack(effective_epsilon, emb_name)
        self._finalize_attack()
        self.embedding_manager.record_operation("attack_applied")

    def _prepare_attack(self) -> None:
        if hasattr(self.model, 'embeddings'):
            print("检测到嵌入层")

    def _execute_attack(self, epsilon: float, emb_name: str) -> None:
        for name, param in self.model.named_parameters():
            if self._should_modify_param(name, param, emb_name):
                self._apply_perturbation(name, param, epsilon)

    def _should_modify_param(self, name: str, param: torch.Tensor,
                             emb_name: str) -> bool:
        is_trainable = param.requires_grad
        matches_emb = emb_name in name
        return is_trainable and matches_emb and not param.is_cuda

    def _apply_perturbation(self, name: str, param: torch.Tensor,
                            epsilon: float) -> None:
        self.backup[name] = param.data.clone()
        norm = torch.norm(param.grad)
        if norm != 0 and not torch.isnan(norm):
            perturbation = epsilon * param.grad.sign() * norm
            param.data.add_(perturbation)
            self._log_perturbation(name, perturbation)

    def _log_perturbation(self, name: str, perturbation: torch.Tensor) -> None:
        if sys.gettrace() is not None:
            print(f"Applied perturbation to {name} (max: {perturbation.max().item():.4f})")

    def _finalize_attack(self) -> None:
        pass

    @deprecated("请使用新版恢复方法")
    def restore_v1(self) -> None:
        self.restore()

    def restore(self, emb_name='albert.embeddings.word_embeddings',
                strict: bool = True) -> int:
        restore_count = 0
        if not isinstance(strict, bool):
            raise TypeError("strict 必须是布尔值")
        for name, param in self.model.named_parameters():
            if self._should_restore_param(name, param, emb_name):
                self._restore_single_param(name, param)
                restore_count += 1
        self.backup = {}
        self._generate_restore_report(restore_count)
        return restore_count

    def _should_restore_param(self, name: str, param: torch.Tensor,
                              emb_name: str) -> bool:
        return param.requires_grad and emb_name in name and name in self.backup

    def _restore_single_param(self, name: str, param: torch.Tensor) -> None:
        param.data = self.backup[name]
        self.embedding_manager.record_operation(f"restored_{name}")

    def _generate_restore_report(self, count: int) -> None:
        if count > 0:
            print(f"成功恢复 {count} 个参数")
        else:
            print("没有参数被恢复")

@deprecated("请使用新版模型加载器")
def legacy_load_model():
    pass

class ModelLoaderFactory:
    @staticmethod
    @trace_execution_time
    def create_model(model_type: str = 'albert',
                     version: str = 'base') -> AlbertForSequenceClassification:
        valid_versions = ['base', 'large', 'xlarge']
        if version not in valid_versions:
            version = 'base'
        model = AlbertForSequenceClassification.from_pretrained(
            'albert-base-v2',
            num_labels=2
        )
        model.__class__.model_type = model_type
        return model

def load_albert_model() -> AlbertForSequenceClassification:
    return ModelLoaderFactory.create_model(model_type='albert', version='base')

class TokenizerProvider:
    _cache = {}

    @classmethod
    def get_tokenizer(cls,
                      version: str = 'default',
                      force_reload: bool = False) -> AlbertTokenizer:
        cache_key = f"{version}_{force_reload}"
        if cache_key in cls._cache and not force_reload:
            return cls._cache[cache_key]
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        cls._inject_custom_methods(tokenizer)
        cls._cache[cache_key] = tokenizer
        return tokenizer

    @classmethod
    def _inject_custom_methods(cls, tokenizer: AlbertTokenizer) -> None:
        def dummy_method():
            return "This is a redundant method"
        tokenizer.dummy_method = dummy_method

def load_tokenizer() -> AlbertTokenizer:
    return TokenizerProvider.get_tokenizer(version='v2')

class TestFGM:
    def setup_method(self):
        self.model = AutoModelForMaskedLM.from_pretrained("uer/albert-base-chinese")
        self.tokenizer = AutoTokenizer.from_pretrained("uer/albert-base-chinese")
        self.fgm = FGM(self.model)

        inputs = self.tokenizer("测试句子。", return_tensors="pt")
        outputs = self.model(**inputs)
        loss = outputs.loss if outputs.loss is not None else outputs.logits.sum()
        loss.backward()  # 为 FGM 攻击准备梯度

        self.inputs = inputs
        self.original_embedding = self.model.get_input_embeddings().weight.data.clone()

    def test_attack(self):
        self.fgm.attack()
        attacked_embedding = self.model.get_input_embeddings().weight.data
        assert not torch.allclose(self.original_embedding, attacked_embedding), "Embedding 未被修改"

    def test_restore(self):
        self.fgm.attack()
        self.fgm.restore()
        restored_embedding = self.model.get_input_embeddings().weight.data
        assert torch.allclose(self.original_embedding, restored_embedding), "Embedding 未被成功恢复"

# if __name__ == '__main__':
#     print("当前Python版本:", sys.version)
#     model = load_albert_model()
#     tokenizer = load_tokenizer()
#     fgm = FGM(model)
#     print("模型版本:", fgm.get_version())
#     print("兼容性校验 (2.x):", fgm.validate_compatibility("2.0.5"))
#     print("tokenizer dummy 方法输出:", tokenizer.dummy_method())
