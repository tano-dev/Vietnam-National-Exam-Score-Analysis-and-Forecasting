# 📦 Python OOP – Project-wide Coding Standards & Class Layout Guide

> Áp dụng cho toàn bộ dự án: bố cục OOP chuẩn, quy ước đặt tên, docstring, và hướng dẫn triển khai

Lưu ý thuật ngữ trong Python:
- `_name`: quy ước **protected/internal** (dùng nội bộ, vẫn truy cập được).
- `__name`: **private (name mangling)** – tránh truy cập/override ngoài ý muốn.
- Python **không có** ‘protected’ thực sự bằng `__`; double underscore là **private**, single underscore là **protected theo quy ước**.

# 🧭 Quy ước đặt tên (Naming Conventions)

- **Tên lớp (Class):** PascalCase — *Ví dụ:* `DataLoader`, `DataProcessor`
- **Tên phương thức & thuộc tính:** snake_case — *Ví dụ:* `load_data`, `process_input`
- **Tên biến cục bộ:** snake_case — *Ví dụ:* `temp_value`, `user_list`
- **Biến nội bộ/protected (quy ước):** `_var_name` — chỉ dùng trong class/module
- **Biến private (name mangling):** `__var_name` — hạn chế truy cập/override ngoài ý muốn

> ℹ️ Trong Python: `_` = protected/internal (quy ước), `__` = private (name mangling).

# 🧾 Chuẩn Docstring cho mỗi phương thức

Sử dụng định dạng Google-style hoặc NumPy-style. Ví dụ (Google-style):

```python
def method_name(param1: type1, param2: type2) -> return_type:
    """Mô tả ngắn gọn chức năng của phương thức.

    Args:
        param1 (type1): Mô tả ý nghĩa tham số.
        param2 (type2): Mô tả ý nghĩa tham số.

    Returns:
        return_type: Mô tả giá trị trả về.

    Raises:
        ValueError: Khi tham số không hợp lệ.
    """
    # Các bước xử lý quan trọng (ghi chú rõ ràng tại đây)
    ...
    return result
```

# 🧩 Bố cục chuẩn của một Class trong OOP Python

```python
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

class MyClass:
    """Mô tả ngắn: vai trò, phạm vi sử dụng, các thành phần chính.

    Attributes (public API):
        attr1 (type): Mô tả.
        attr2 (type): Mô tả.
    """

    # ==================== INTERNAL PRIVATE MEMBERS ====================
    __slots__ = (
        "_attr1",
        "_attr2",
        "_cache",
    )

    # -------------------- CONSTRUCTOR --------------------
    def __init__(self, attr1: int, attr2: str) -> None:
        """Khởi tạo đối tượng.
        
        Args:
            attr1 (int): Mô tả tham số 1.
            attr2 (str): Mô tả tham số 2.
        """
        # Gọi setter để áp điều kiện/validate
        self.attr1 = attr1
        self.attr2 = attr2
        self._cache = {}

    # -------------------- GETTER / SETTER --------------------
    @property
    def attr1(self) -> int:
        """Giá trị attr1 (đọc-only/public view)."""
        return self._attr1

    @attr1.setter
    def attr1(self, value: int) -> None:
        if not isinstance(value, int) or value < 0:
            raise ValueError("attr1 phải là số nguyên không âm.")
        self._attr1 = value

    @property
    def attr2(self) -> str:
        return self._attr2

    @attr2.setter
    def attr2(self, value: str) -> None:
        if not isinstance(value, str) or not value:
            raise ValueError("attr2 phải là chuỗi không rỗng.")
        self._attr2 = value

    # ==================== INTERNAL PRIVATE METHODS ====================
    def _precompute(self) -> None:
        """Xử lý nội bộ: chuẩn bị dữ liệu, cache kết quả."""
        # Ghi chú các bước xử lý quan trọng
        ...

    # ==================== PUBLIC METHODS (API) ====================
    def run(self, x: pd.DataFrame) -> pd.DataFrame:
        """Thực thi quy trình xử lý dữ liệu.

        Args:
            x (pd.DataFrame): Dữ liệu đầu vào.

        Returns:
            pd.DataFrame: Dữ liệu sau xử lý.
        """
        self._precompute()
        # Code xử lý dữ liệu chính
        ...
        return x

    # ==================== REPRESENTATION / UTILITIES ====================
    def __repr__(self) -> str:
        return f"<MyClass attr1={self._attr1} attr2='{self._attr2}'>"
```

# 🔒 Private vs Protected trong Python (Chuẩn xác)

- `_name` ➜ **Protected/Internal theo quy ước** (không chặn truy cập thật sự).
- `__name` ➜ **Private** (name-mangling thành `_ClassName__name`), hạn chế override & truy cập nhầm.
- Python **không có** `protected` “thực” như Java/C++; dùng quy ước `_` cho mục đích này.


# ✅ Checklist áp dụng trong dự án

- [ ] Mỗi class có docstring mô tả rõ vai trò & public API
- [ ] Dùng `__slots__` nếu muốn giới hạn thuộc tính & tiết kiệm bộ nhớ
- [ ] Mọi thuộc tính private: đặt tên `_name` và expose qua property
- [ ] Mọi phương thức public có docstring chuẩn (Args/Returns/Raises)
- [ ] Ghi chú rõ ràng các bước xử lý dữ liệu quan trọng
- [ ] Phân tách API public (method không gạch dưới) và logic nội bộ (`_method`)
- [ ] Tránh lặp logic I/O: tách DataLoader (read) và DataProcessor (transform)
- [ ] Viết `__repr__` gọn để dễ debug

