# Khai báo thư viện
from __future__ import annotations
from pathlib import Path
import pandas as pd

# Lớp để quản lý việc load dữ liệu
class DataLoader:
    """Load THPT datasets from a project folder.

    Attributes (public API):
        project_root (Path): Thư mục gốc của project. Có thể gán lại.

    Read-only properties (tự tính từ project_root):
        thpt2023_csv_path, thpt2024_csv_path, thpt2025_ct2006_xlsx_path, thpt2025_ct2018_xlsx_path
    """
    
    __slots__ = (
        "_project_root",
        "_dataset_dir",
        "_set2023",
        "_set2024",
        "_set2025",
        "_f_2023_ct2006",
        "_f_2024_ct2006",
        "_f_2025_ct2006",
        "_f_2025_ct2018",
    )

    # -------- Khởi tạo --------
    def __init__(self, project_root: Path | str | None = None):
        # --- “constructor” (public) ---
        here = Path(__file__).resolve().parent
        default_root = here.parent
        self._project_root: Path = Path(project_root) if project_root else default_root

        # tên thư mục/file chuẩn hóa để dễ đổi nếu cần
        self._dataset_dir = "Data"
        self._set2023 = "Data_Set_2023"
        self._set2024 = "Data_Set_2024"
        self._set2025 = "Data_Set_2025"

        self._f_2023_ct2006 = "diem_thi_thpt_2023.csv"
        self._f_2024_ct2006 = "diem_thi_thpt_2024.csv"
        self._f_2025_ct2006 = "diem_thi_thpt_2025-ct2006.xlsx"
        self._f_2025_ct2018 = "diem_thi_thpt_2025-ct2018a.xlsx"

    # -------- Getter/Setter kiểu Python cho cấu hình --------
    @property
    def project_root(self) -> Path:
        """Thư mục gốc của project (có thể gán lại)."""
        return self._project_root

    @project_root.setter
    def project_root(self, value: Path | str) -> None:
        p = Path(value).resolve()
        if not p.exists():
            raise FileNotFoundError(f"project_root không tồn tại: {p}")
        self._project_root = p
        # Không cần cập nhật đường dẫn lẻ vì chúng được tính động bên dưới.

    # -------- Xây dựng đường dẫn đến file: chỉ-đọc, tính động từ project_root --------
    @property
    def thpt2023_ct2006_csv_path(self) -> Path:
        return (
            self.project_root / self._dataset_dir / self._set2023 / self._f_2023_ct2006
        )

    @property
    def thpt2024_ct2006_csv_path(self) -> Path:
        return (
            self.project_root / self._dataset_dir / self._set2024 / self._f_2024_ct2006
        )

    @property
    def thpt2025_ct2006_xlsx_path(self) -> Path:
        return (
            self.project_root / self._dataset_dir / self._set2025 / self._f_2025_ct2006
        )

    @property
    def thpt2025_ct2018_xlsx_path(self) -> Path:
        return (
            self.project_root / self._dataset_dir / self._set2025 / self._f_2025_ct2018
        )

    # -------- Method: load dữ liệu --------
    def load_data(self):
        # kiểm tra tồn tại để báo lỗi sớm & rõ
        for p in (
            self.thpt2023_ct2006_csv_path,
            self.thpt2024_ct2006_csv_path,
            self.thpt2025_ct2006_xlsx_path,
            self.thpt2025_ct2018_xlsx_path,
        ):
            if not p.exists():
                raise FileNotFoundError(f"Không tìm thấy file: {p}")

        # đọc dữ liệu vào DataFrames
        df_2023_ct2006 = pd.read_csv(self.thpt2023_ct2006_csv_path, encoding="utf-8")
        df_2024_ct2006 = pd.read_csv(self.thpt2024_ct2006_csv_path, encoding="utf-8")
        df_2025_ct2006 = pd.read_excel(
            self.thpt2025_ct2006_xlsx_path, engine="openpyxl"
        )
        df_2025_ct2018 = pd.read_excel(
            self.thpt2025_ct2018_xlsx_path, engine="openpyxl"
        )

        # trả về 4 DataFrame
        return df_2023_ct2006, df_2024_ct2006, df_2025_ct2006, df_2025_ct2018
