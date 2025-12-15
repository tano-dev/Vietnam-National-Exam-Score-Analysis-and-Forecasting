# Khai báo thư viện
from __future__ import annotations
from importlib.resources import path
from pathlib import Path
import pandas as pd

# Lớp để quản lý việc load dữ liệu
class DataLoader:
    # ==================== INTERNAL PRIVATE METHODS: XỬ LÝ DỮ LIỆU =====================
    # ----------------------- Khai báo và thiết lập thuộc tính -------------------------
    """Load THPT datasets from a project folder.

    Attributes (public API):
        project_root (Path): Thư mục gốc của project. Có thể gán lại.

    Read-only properties (tự tính từ project_root):
        thpt2023_csv_path, thpt2024_csv_path, thpt2025_ct2006_xlsx_path, thpt2025_ct2018_xlsx_path
    """
    
    # Slots: Cố định các thuộc tính có thể sử dụng, để tiết kiệm bộ nhớ. Không thể thêm thuộc tính mới ngoài danh sách này.
    __slots__ = (
        "_project_root",                        # thư mục gốc của project
        "_dataset_dir",                         # tên thư mục chứa dữ liệu    
        "_set2023",                             # tên thư mục dữ liệu năm 2023
        "_set2024",                             # tên thư mục dữ liệu năm 2024
        "_set2025",                             # tên thư mục dữ liệu năm 2025
        "_f_2023_ct2006",                       # tên file dữ liệu năm 2023
        "_f_2024_ct2006",                       # tên file dữ liệu năm 2024
        "_f_2025_ct2006",                       # tên file dữ liệu năm 2025 CT2006
        "_f_2025_ct2018",                       # tên file dữ liệu năm 2025 CT2018  
    )   

    # -------- Khởi tạo --------
    def __init__(self, project_root: Path | str | None = None):
        # --- “constructor” (public) ---
        here = Path(__file__).resolve().parent
        default_root = here.parent
        self._project_root: Path = Path(project_root) if project_root else default_root

        # tên thư mục/file chuẩn hóa để dễ đổi nếu cần
        self._dataset_dir = "Raw_Data"
        
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

    # ====================
    # -------- Method: load dữ liệu --------
    def _load_data(self):
        """Load tất cả dữ liệu từ các file và trả về 4 DataFrame.
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
                Dữ liệu năm 2023, 2024, 2025 theo CT2006 và 2025 theo CT2018.
        """
        # Kiểm tra sự tồn tại của file
        paths = [
            self.thpt2023_ct2006_csv_path,
            self.thpt2024_ct2006_csv_path,
            self.thpt2025_ct2006_xlsx_path,
            self.thpt2025_ct2018_xlsx_path
        ]
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"File dữ liệu không tồn tại: {p}")
            
        # đọc dữ liệu vào DataFrames
        df_2023_ct2006 = pd.read_csv(self.thpt2023_ct2006_csv_path, encoding="utf-8")
        df_2024_ct2006 = pd.read_csv(self.thpt2024_ct2006_csv_path, encoding="utf-8")
        df_2025_ct2006 = pd.read_excel(
            self.thpt2025_ct2006_xlsx_path, engine="openpyxl"
        )
        df_2025_ct2018_s1 = pd.read_excel(
            self.thpt2025_ct2018_xlsx_path, engine="openpyxl", sheet_name="Sheet1"
        )
        df_2025_ct2018_s2 = pd.read_excel(
            self.thpt2025_ct2018_xlsx_path, engine="openpyxl", sheet_name="Sheet2"
        )
        df_2025_ct2018 = pd.concat([df_2025_ct2018_s1, df_2025_ct2018_s2], ignore_index=True)
        
        # trả về 4 DataFrame
        return df_2023_ct2006, df_2024_ct2006, df_2025_ct2006, df_2025_ct2018  
    
    # =============================================================================
# LỚP CleanDataLoader: Load dữ liệu CLEAN 2023–2025 (Block/Subject/Province)
# =============================================================================
class CleanDataLoader(DataLoader):
    """Quản lý việc load dữ liệu CLEAN phục vụ EDA, Change Point, Forecast,...

    Attributes (public API)
    -----------------------
    clean_dataset_root : Path
        Thư mục gốc chứa toàn bộ Clean_Data_2023-2025.
    """

    # Chỉ khai báo các thuộc tính MỚI so với DataLoader để tránh trùng slots.
    __slots__ = (
        "_dataset_dir",              # tên thư mục chứa dữ liệu CLEAN (ghi đè Raw_Data)
        "_block_data_dir",           # thư mục Block Data
        "_subject_data_dir",         # thư mục Subject Data
        "_province_data_dir",        # thư mục Province Data

        # Tiền tố thư mục: CleanData_<ten_khoi/mon/tinh>
        "_block_data_f_prefix",      # "CleanData"
        "_subject_data_f_prefix",    # "CleanData"
        "_province_data_f_prefix",   # "CleanData"

        # Tên file dữ liệu: Export_Analysis_<ten>.csv
        "_f_block_data_prefix",      # "Export_Analysis"
        "_f_subject_data_prefix",    # "Export_Analysis"
        "_f_province_data_prefix",   # "Export_Analysis"

        # Tên file dữ liệu: Export_Distribution_<ten>.csv
        "_d_block_data_prefix",      # "Export_Distribution"
        "_d_subject_data_prefix",    # "Export_Distribution"
        "_d_province_data_prefix",   # "Export_Distribution"
    )

    # ==================== Khởi tạo ====================
    def __init__(self, project_root: Path | str | None = None) -> None:
        """Khởi tạo CleanDataLoader."""
        # Gọi init của lớp cha để setup project_root
        super().__init__(project_root)

        # Ghi đè tên thư mục dữ liệu thành thư mục Clean
        self._dataset_dir = "Clean_Data_2023-2025"

        # Thư mục con
        self._block_data_dir = "Block_Data"
        self._subject_data_dir = "Subject_Data"
        self._province_data_dir = "Province_Data"

        # Tiền tố thư mục
        self._block_data_f_prefix = "CleanData"
        self._subject_data_f_prefix = "CleanData"
        self._province_data_f_prefix = "CleanData"

        # Tên file dữ liệu Analysis
        self._f_block_data_prefix = "Export_Analysis"
        self._f_subject_data_prefix = "Export_Analysis"
        self._f_province_data_prefix = "Export_Analysis"

        # Tên file dữ liệu Distribution
        self._d_block_data_prefix = "Export_Distribution"
        self._d_subject_data_prefix = "Export_Distribution"
        self._d_province_data_prefix = "Export_Distribution"

    # ==================== Getter ====================
    @property
    def clean_dataset_root(self) -> Path:
        """Thư mục gốc chứa toàn bộ Clean_Data_2023-2025."""
        return self.project_root / self._dataset_dir

    # ==================== INTERNAL PRIVATE METHODS ====================
    def _build_path(self, level: str, name: str, kind: str = "analysis") -> Path:
        """Tạo đường dẫn đầy đủ tới file Clean Data."""
        name = str(name).strip()

        # Bản đồ cấp dữ liệu tới thư mục con và tiền tố file
        level_map = {
            "block": (
                self._block_data_dir,
                self._block_data_f_prefix,
                self._f_block_data_prefix if kind == "analysis" else self._d_block_data_prefix,
            ),
            "subject": (
                self._subject_data_dir,
                self._subject_data_f_prefix,
                self._f_subject_data_prefix if kind == "analysis" else self._d_subject_data_prefix,
            ),
            "province": (
                self._province_data_dir,
                self._province_data_f_prefix,
                self._f_province_data_prefix if kind == "analysis" else self._d_province_data_prefix,
            ),
        }

        if level not in level_map:
            raise ValueError("level phải là 'block', 'subject' hoặc 'province'")

        sub_dir, clean_prefix, file_prefix = level_map[level]
        folder_name = f"{clean_prefix}_{name}"
        file_name = f"{file_prefix}_{name}.csv"

        return self.clean_dataset_root / sub_dir / folder_name / file_name

    # ==================== PUBLIC METHODS ====================
    def get_subject_data(self, subject: str, kind: str = "analysis") -> pd.DataFrame:
        path = self._build_path("subject", subject, kind)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {path}")
        return pd.read_csv(path)

    def get_block_data(self, block: str, kind: str = "analysis") -> pd.DataFrame:
        path = self._build_path("block", block, kind)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {path}")
        return pd.read_csv(path)

    def get_province_data(self, province: str, kind: str = "analysis") -> pd.DataFrame:
        path = self._build_path("province", province, kind)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {path}")
        return pd.read_csv(path)

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load tất cả dữ liệu từ các file và trả về dưới dạng tuple.
        
        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
                Dữ liệu năm 2023, 2024, 2025 theo CT2006 và 2025 theo CT2018.
        """
        return self._load_data()
