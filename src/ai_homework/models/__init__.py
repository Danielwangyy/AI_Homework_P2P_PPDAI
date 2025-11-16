"""模型模块入口。

为外部调用提供统一的导入路径：

```python
from ai_homework.models import train_model
```

也支持直接访问具体的训练函数或结果结构体。
"""

from .training import (
    ModelResult,
    train_catboost,
    train_lightgbm,
    train_logistic_regression,
    train_model,
    train_xgboost,
)

__all__ = [
    "ModelResult",
    "train_model",
    "train_logistic_regression",
    "train_xgboost",
    "train_lightgbm",
    "train_catboost",
]
