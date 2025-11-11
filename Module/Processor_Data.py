from Load_Data import DataLoader
from pathlib import Path
import pandas as pd
import numpy as np

class DataProcessor:
    # ------- Xử lý dữ liệu --------
    """Xử lý dữ liệu THPT đã load từ DataLoader.
    Attributes (public API):
        data_2023 (pd.DataFrame): Dữ liệu năm 2023.
        data_2024 (pd.DataFrame): Dữ liệu năm 2024.
        data_2025_ct2006 (pd.DataFrame): Dữ liệu năm 2025 theo CT2006.
        data_2025_ct2018 (pd.DataFrame): Dữ liệu năm 2025 theo CT2018.
    """
    __slots__ = (
        "_loader",
        "data_2023",
        "data_2024",
        "data_2025_ct2006",
        "data_2025_ct2018",
    )
    
    # -------- Khởi tạo --------
    def __init__(self, project_root: Path | str | None = None):
        self._loader = DataLoader(project_root)
        self.data_2023 = pd.DataFrame()
        self.data_2024 = pd.DataFrame()
        self.data_2025_ct2006 = pd.DataFrame()
        self.data_2025_ct2018 = pd.DataFrame()
        self.load_all_data()

    def load_all_data(self) -> None:
        """Load tất cả dữ liệu từ các file."""
        self.data_2023 = pd.read_csv(self._loader.thpt2023_ct2006_csv_path)
        self.data_2024 = pd.read_csv(self._loader.thpt2024_ct2006_csv_path)
        self.data_2025_ct2006 = pd.read_excel(self._loader.thpt2025_ct2006_xlsx_path)
        self.data_2025_ct2018 = pd.read_excel(self._loader.thpt2025_ct2018_xlsx_path)

    def preprocess_data(self) -> None:
        """Tiền xử lý dữ liệu (ví dụ: xử lý giá trị thiếu, chuẩn hóa)."""
        # Ví dụ đơn giản: loại bỏ hàng có giá trị thiếu
        self.data_2023.dropna(inplace=True)
        self.data_2024.dropna(inplace=True)
        self.data_2025_ct2006.dropna(inplace=True)
        self.data_2025_ct2018.dropna(inplace=True)
    
    # Nên xử lý dữ liệu như nào?    
    # Cần những thông số gì?
    # Cần những hàm gì?
    # Cần những thuộc tính gì?
    # Đổi quản lý sang thư viện NumPy như sao?
    
    