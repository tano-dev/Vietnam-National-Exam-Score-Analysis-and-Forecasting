# Khai báo thư viện
from __future__ import annotations
from importlib.resources import path
from pathlib import Path
from typing import Optional
import pandas as pd


# =============================================================================
# LỚP DataLoader: Load dữ liệu RAW 2023–2025
# =============================================================================
class DataLoader:
    """Quản lý việc load dữ liệu THPT RAW từ thư mục project.

    Attributes (public API)
    -----------------------
    project_root : Path
        Thư mục gốc của project. Có thể gán lại sau khi khởi tạo.

    Read-only properties (tính động từ project_root)
    ------------------------------------------------
    thpt2018_ct2006_csv_path : Path
        Đường dẫn tới file CSV điểm thi 2018 (CT2006).
    thpt2019_ct2006_csv_path : Path
        Đường dẫn tới file CSV điểm thi 2019 (CT2006).
    thpt2020_ct2006_csv_path : Path
        Đường dẫn tới file CSV điểm thi 2020 (CT2006).
    thpt2021_ct2006_csv_path : Path
        Đường dẫn tới file CSV điểm thi 2021 (CT2006).
    thpt2022_ct2006_csv_path : Path
        Đường dẫn tới file CSV điểm thi 2022 (CT2006).
    thpt2023_ct2006_csv_path : Path
        Đường dẫn tới file CSV điểm thi 2023 (CT2006).
    thpt2024_ct2006_csv_path : Path
        Đường dẫn tới file CSV điểm thi 2024 (CT2006).
    thpt2025_ct2006_xlsx_path : Path
        Đường dẫn tới file XLSX điểm thi 2025 (CT2006).
    thpt2025_ct2018_xlsx_path : Path
        Đường dẫn tới file XLSX điểm thi 2025 (CT2018).
    """

    # Slots: Cố định các thuộc tính có thể sử dụng, để tiết kiệm bộ nhớ.
    # Không thể thêm thuộc tính mới ngoài danh sách này.
    __slots__ = (
        "_project_root",                        # thư mục gốc của project
        "_dataset_dir",                         # tên thư mục chứa dữ liệu RAW
        
        # Tên thư mục/file chuẩn hóa để dễ đổi nếu cần
        "_set2018",                             # tên thư mục dữ liệu năm 2018
        "_set2019",                             # tên thư mục dữ liệu năm 2019   
        "_set2020",                             # tên thư mục dữ liệu năm 2020
        "_set2021",                             # tên thư mục dữ liệu năm 2021
        "_set2022",                             # tên thư mục dữ liệu năm 2022
        "_set2023",                             # tên thư mục dữ liệu năm 2023
        "_set2024",                             # tên thư mục dữ liệu năm 2024
        "_set2025",                             # tên thư mục dữ liệu năm 2025
        
        # tên file dữ liệu
        "_f_2018_ct2006",                       # tên file dữ liệu năm 2018
        "_f_2019_ct2006",                       # tên file dữ liệu năm 2019
        "_f_2020_ct2006",                       # tên file dữ liệu năm 2020
        "_f_2021_ct2006",                       # tên file dữ liệu năm 2021
        "_f_2022_ct2006",                       # tên file dữ liệu năm 2022
        "_f_2023_ct2006",                       # tên file dữ liệu năm 2023
        "_f_2024_ct2006",                       # tên file dữ liệu năm 2024
        "_f_2025_ct2006",                       # tên file dữ liệu năm 2025 CT2006
        "_f_2025_ct2018",                       # tên file dữ liệu năm 2025 CT2018
    )

    # ==================== Khởi tạo ====================
    def __init__(self, project_root: Path | str | None = None):
        """Khởi tạo DataLoader với project_root tuỳ chọn.

        Nếu không truyền project_root, mặc định lấy thư mục cha của file hiện tại.
        """
        here = Path(__file__).resolve().parent
        default_root = here.parent
        self._project_root: Path = Path(project_root) if project_root else default_root

        # tên thư mục/file chuẩn hóa để dễ đổi nếu cần
        self._dataset_dir = "Raw_Data"

        self._set2018 = "Data_Set_2018"
        self._set2019 = "Data_Set_2019"
        self._set2020 = "Data_Set_2020"
        self._set2021 = "Data_Set_2021"
        self._set2022 = "Data_Set_2022"
        self._set2023 = "Data_Set_2023"
        self._set2024 = "Data_Set_2024"
        self._set2025 = "Data_Set_2025"

        self._f_2018_ct2006 = "diem_thi_thpt_2018.csv"
        self._f_2019_ct2006 = "diem_thi_thpt_2019.csv"
        self._f_2020_ct2006 = "diem_thi_thpt_2020.csv"
        self._f_2021_ct2006 = "diem_thi_thpt_2021.csv"
        self._f_2022_ct2006 = "diem_thi_thpt_2022.csv"
        self._f_2023_ct2006 = "diem_thi_thpt_2023.csv"
        self._f_2024_ct2006 = "diem_thi_thpt_2024.csv"
        self._f_2025_ct2006 = "diem_thi_thpt_2025-ct2006.xlsx"
        self._f_2025_ct2018 = "diem_thi_thpt_2025-ct2018a.xlsx"

    # ==================== Getter / Setter ====================
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

    # ==================== Đường dẫn chỉ-đọc tới file RAW ====================
    @property
    def thpt2018_ct2006_csv_path(self) -> Path:
        """Đường dẫn tới dữ liệu THPT 2018 (CT2006)."""
        return (
            self.project_root / self._dataset_dir / self._set2018 / self._f_2018_ct2006
        )
    
    @property
    def thpt2019_ct2006_csv_path(self) -> Path:
        """Đường dẫn tới dữ liệu THPT 2019 (CT2006)."""
        return (
            self.project_root / self._dataset_dir / self._set2019 / self._f_2019_ct2006
        )
    
    @property
    def thpt2020_ct2006_csv_path(self) -> Path:
        """Đường dẫn tới dữ liệu THPT 2020 (CT2006)."""
        return (
            self.project_root / self._dataset_dir / self._set2020 / self._f_2020_ct2006
        )
    
    @property
    def thpt2021_ct2006_csv_path(self) -> Path:
        """Đường dẫn tới dữ liệu THPT 2021 (CT2006)."""
        return (
            self.project_root / self._dataset_dir / self._set2021 / self._f_2021_ct2006
        )
        
    @property
    def thpt2022_ct2006_csv_path(self) -> Path:
        """Đường dẫn tới dữ liệu THPT 2022 (CT2006)."""
        return (
            self.project_root / self._dataset_dir / self._set2022 / self._f_2022_ct2006
        )
    
    @property
    def thpt2023_ct2006_csv_path(self) -> Path:
        """Đường dẫn tới dữ liệu THPT 2023 (CT2006)."""
        return (
            self.project_root / self._dataset_dir / self._set2023 / self._f_2023_ct2006
        )

    @property
    def thpt2024_ct2006_csv_path(self) -> Path:
        """Đường dẫn tới dữ liệu THPT 2024 (CT2006)."""
        return (
            self.project_root / self._dataset_dir / self._set2024 / self._f_2024_ct2006
        )

    @property
    def thpt2025_ct2006_xlsx_path(self) -> Path:
        """Đường dẫn tới dữ liệu THPT 2025 (CT2006)."""
        return (
            self.project_root / self._dataset_dir / self._set2025 / self._f_2025_ct2006
        )

    @property
    def thpt2025_ct2018_xlsx_path(self) -> Path:
        """Đường dẫn tới dữ liệu THPT 2025 (CT2018)."""
        return (
            self.project_root / self._dataset_dir / self._set2025 / self._f_2025_ct2018
        )

    # ==================== INTERNAL PRIVATE METHODS ====================
    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load tất cả dữ liệu từ các file RAW và trả về 4 DataFrame.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
            (,df_2018_ct2006, df_2019_ct2006, df_2020_ct2006, df_2021_ct2006, df_2022_ct2006, df_2023_ct2006, df_2024_ct2006, df_2025_ct2006, df_2025_ct2018)
        """
        # Kiểm tra sự tồn tại của file
        paths = [
            self.thpt2018_ct2006_csv_path,
            self.thpt2019_ct2006_csv_path,
            self.thpt2020_ct2006_csv_path,
            self.thpt2021_ct2006_csv_path,
            self.thpt2022_ct2006_csv_path,
            self.thpt2023_ct2006_csv_path,
            self.thpt2024_ct2006_csv_path,
            self.thpt2025_ct2006_xlsx_path,
            self.thpt2025_ct2018_xlsx_path,
        ]
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"File dữ liệu không tồn tại: {p}")

        # đọc dữ liệu vào DataFrames
        df_2018_ct2006 = pd.read_csv(self.thpt2018_ct2006_csv_path, encoding="utf-8")
        df_2019_ct2006 = pd.read_csv(self.thpt2019_ct2006_csv_path, encoding="utf-8")
        df_2020_ct2006 = pd.read_csv(self.thpt2020_ct2006_csv_path, encoding="utf-8")
        df_2021_ct2006 = pd.read_csv(self.thpt2021_ct2006_csv_path, encoding="utf-8")
        df_2022_ct2006 = pd.read_csv(self.thpt2022_ct2006_csv_path, encoding="utf-8")
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
        
        # trả về 9 DataFrame
        return df_2018_ct2006, df_2019_ct2006, df_2020_ct2006, df_2021_ct2006, df_2022_ct2006, df_2023_ct2006, df_2024_ct2006, df_2025_ct2006, df_2025_ct2018  
    
    # ==================== PUBLIC API ====================
    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Public wrapper cho _load_data(), dùng cho DataProcessor.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
            (df_2018_ct2006, df_2019_ct2006, df_2020_ct2006, df_2021_ct2006, df_2022_ct2006, df_2023_ct2006, df_2024_ct2006, df_2025_ct2006, df_2025_ct2018)
        """
        return self._load_data()
    
# =============================================================================
# LỚP CleanDataLoader: Load dữ liệu CLEAN 2023–2025 (Block/Subject/Province)
# =============================================================================
class CleanDataLoader(DataLoader):
    """Quản lý việc load dữ liệu CLEAN phục vụ EDA, Change Point, Forecast,...

    Cấu trúc thư mục Clean Data (dạng chuẩn)
    ----------------------------------------
    Clean_Data_2023-2025/
        Block_Data/
            CleanData_A00/
                Export_Analysis_A00.csv
                Export_Distribution_A00.csv
        Subject_Data/
            CleanData_Toan/
                Export_Analysis_Toan.csv
                Export_Distribution_Toan.csv
        Province_Data/
            CleanData_HN/
                Export_Analysis_HN.csv
                Export_Distribution_HN.csv

    Attributes (public API)
    -----------------------
    project_root : Path
        Thư mục gốc của project. Có thể gán lại sau khi khởi tạo.

    Read-only properties (gợi ý sử dụng)
    ------------------------------------
    clean_dataset_root : Path
        Thư mục gốc chứa toàn bộ Clean_Data_2023-2025.
    block_data_root : Path
        Thư mục gốc chứa Block_Data.
    subject_data_root : Path
        Thư mục gốc chứa Subject_Data.
    province_data_root : Path
        Thư mục gốc chứa Province_Data.
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
        """Khởi tạo CleanDataLoader.

        Parameters
        ----------
        project_root : Path | str | None, optional
            Thư mục gốc của project. Nếu None, mặc định là thư mục cha của file hiện tại.
        """
        here = Path(__file__).resolve().parent
        default_root = here.parent
        # Thuộc tính _project_root đã được khai báo ở DataLoader
        self._project_root: Path = Path(project_root) if project_root else default_root

        # tên thư mục/file chuẩn hóa để dễ đổi nếu cần
        self._dataset_dir = "Clean_Data_2023-2025"

        # Thư mục con
        self._block_data_dir = "Block_Data"
        self._subject_data_dir = "Subject_Data"
        self._province_data_dir = "Province_Data"

        # Tiền tố thư mục: CleanData_<ten>
        # => ví dụ: CleanData_A00, CleanData_Toan, CleanData_HN
        self._block_data_f_prefix = "CleanData"
        self._subject_data_f_prefix = "CleanData"
        self._province_data_f_prefix = "CleanData"

        # Tên file dữ liệu: Export_Analysis_<ten>.csv
        # => ví dụ: Export_Analysis_A00.csv
        self._f_block_data_prefix = "Export_Analysis"
        self._f_subject_data_prefix = "Export_Analysis"
        self._f_province_data_prefix = "Export_Analysis"

        # Tên file dữ liệu: Export_Distribution_<ten>.csv
        # => ví dụ: Export_Distribution_A00.csv
        self._d_block_data_prefix = "Export_Distribution"
        self._d_subject_data_prefix = "Export_Distribution"
        self._d_province_data_prefix = "Export_Distribution"

    # ==================== Getter / Setter ====================
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

    # Một số property tiện dụng cho Clean Data
    @property
    def clean_dataset_root(self) -> Path:
        """Thư mục gốc chứa toàn bộ Clean_Data_2023-2025."""
        return self.project_root / self._dataset_dir

    @property
    def block_data_root(self) -> Path:
        """Thư mục gốc chứa Block_Data."""
        return self.clean_dataset_root / self._block_data_dir

    @property
    def subject_data_root(self) -> Path:
        """Thư mục gốc chứa Subject_Data."""
        return self.clean_dataset_root / self._subject_data_dir

    @property
    def province_data_root(self) -> Path:
        """Thư mục gốc chứa Province_Data."""
        return self.clean_dataset_root / self._province_data_dir

    # ==================== INTERNAL PRIVATE METHODS ====================
    def _build_path(self, level: str, name: str, kind: str = "analysis") -> Path:
        """Tạo đường dẫn đầy đủ tới file Clean Data.

        Parameters
        ----------
        level : {"block", "subject", "province"}
            Cấp dữ liệu muốn đọc:
            - "block"   : dữ liệu theo khối thi (A00, B00, D01, ...).
            - "subject" : dữ liệu theo môn (Toan, NguVan, NgoaiNgu, ...).
            - "province": dữ liệu theo tỉnh/thành (HN, HCM, DN, ...).
        name : str
            Tên khối/môn/tỉnh, ví dụ: "A00", "Toan", "HN".
        kind : {"analysis", "distribution"}, default "analysis"
            Loại file muốn đọc:
            - "analysis"     -> Export_Analysis_<name>.csv
            - "distribution" -> Export_Distribution_<name>.csv

        Returns
        -------
        Path
            Đường dẫn đầy đủ tới file CSV tương ứng.

        Raises
        ------
        ValueError
            Nếu level không hợp lệ hoặc kind không phải 'analysis' / 'distribution'.
        """
        name = str(name).strip()

        # Kiểm tra tham số đầu vào : Định dạng phân phối hay dataframe phân tích
        if kind not in {"analysis", "distribution"}:
            raise ValueError("kind phải là 'analysis' hoặc 'distribution'")

        # Bản đồ cấp dữ liệu tới thư mục con và tiền tố file
        level_map = {
            "block": (
                self._block_data_dir,
                self._block_data_f_prefix,         # "CleanData"
                self._f_block_data_prefix if kind == "analysis" else self._d_block_data_prefix,
            ),
            "subject": (
                self._subject_data_dir,
                self._subject_data_f_prefix,       # "CleanData"
                self._f_subject_data_prefix if kind == "analysis" else self._d_subject_data_prefix,
            ),
            "province": (
                self._province_data_dir,
                self._province_data_f_prefix,      # "CleanData"
                self._f_province_data_prefix if kind == "analysis" else self._d_province_data_prefix,
            ),
        }

        # Kiểm tra level hợp lệ: block / subject / province
        if level not in level_map:
            raise ValueError("level phải là 'block', 'subject' hoặc 'province'")

        # Lấy thông tin từ Map: thư mục con, tiền tố folder, tiền tố file
        sub_dir, clean_prefix, file_prefix = level_map[level]

        # ví dụ: CleanData_A00
        folder_name = f"{clean_prefix}_{name}"

        # ví dụ: Export_Analysis_A00.csv hoặc Export_Distribution_A00.csv
        file_name = f"{file_prefix}_{name}.csv"

        # Trả về đường dẫn đầy đủ: <root>/<sub_dir>/<folder_name>/<file_name>
        return (
            self.clean_dataset_root
            / sub_dir
            / folder_name
            / file_name
        )

    # ==================== PUBLIC METHODS ====================
    def get_total_students(self) -> pd.DataFrame:
        """Tải dữ liệu về số thí sinh tham gia của từng năm
        Returns:
            total_num (int): Tổng số thí sinh tham gia kỳ thi THPTQG các năm 2023-2025
        """
        path = self.project_root / "Clean_Data_2023-2025" / "Export_Yearly_Total_Students.csv"
        if not path.exists():
            raise FileNotFoundError(f"File dữ liệu tổng số thí sinh không tồn tại: {path}")
        df_total = pd.read_csv(path)
    
        return df_total
        
    def get_block_data(self, block: str, kind: str = "analysis") -> pd.DataFrame:
        """Đọc dữ liệu CLEAN theo khối thi (A00, B00, D01, ...).

        Parameters
        ----------
        block : str
            Mã khối, ví dụ "A00", "B00", "D01".
        kind : {"analysis", "distribution"}, default "analysis"
            Loại file muốn đọc.

        Returns
        -------
        pd.DataFrame
            DataFrame chứa dữ liệu đã được export cho khối tương ứng.
        """
        path = self._build_path("block", block, kind)
        if not path.exists():
            raise FileNotFoundError(f"File dữ liệu khối không tồn tại: {path}")
        return pd.read_csv(path)

    def get_subject_data(self, subject: str, kind: str = "analysis") -> pd.DataFrame:
        """Đọc dữ liệu CLEAN theo môn học (Toan, NguVan, NgoaiNgu, ...).

        Parameters
        ----------
        subject : str
            Tên môn, ví dụ "Toan", "NguVan", "NgoaiNgu".
        kind : {"analysis", "distribution"}, default "analysis"
            Loại file muốn đọc.

        Returns
        -------
        pd.DataFrame
            DataFrame chứa dữ liệu đã được export cho môn tương ứng.
        """
        path = self._build_path("subject", subject, kind)
        if not path.exists():
            raise FileNotFoundError(f"File dữ liệu môn không tồn tại: {path}")
        return pd.read_csv(path)

    def get_province_data(self, province: str, kind: str = "analysis") -> pd.DataFrame:
        """Đọc dữ liệu CLEAN theo tỉnh/thành (HN, HCM, DN, ...).

        Parameters
        ----------
        province : str
            Mã tỉnh/thành, ví dụ "HN", "HCM", "DN".
        kind : {"analysis", "distribution"}, default "analysis"
            Loại file muốn đọc.

        Returns
        -------
        pd.DataFrame
            DataFrame chứa dữ liệu đã được export cho tỉnh/thành tương ứng.
        """
        path = self._build_path("province", province, kind)
        if not path.exists():
            raise FileNotFoundError(f"File dữ liệu tỉnh/thành không tồn tại: {path}")
        return pd.read_csv(path)
    def list_subjects(self) -> list[str]:
        """Liệt kê tất cả các môn thi có trong dữ liệu CLEAN.

        Returns
        -------
        list[str]
            Danh sách các môn thi.
        """
        subject_dir = self.subject_data_root
        subjects = []
        for folder in subject_dir.iterdir():
            if folder.is_dir() and folder.name.startswith(self._subject_data_f_prefix + "_"):
                subject_name = folder.name.split("_", 1)[1]
                subjects.append(subject_name)
        return subjects
    def list_blocks(self) -> list[str]:
        """Liệt kê tất cả các khối thi có trong dữ liệu CLEAN.

        Returns
        -------
        list[str]
            Danh sách các khối thi (mã khối).
        """
        block_dir = self.block_data_root
        blocks = []
        for folder in block_dir.iterdir():
            if folder.is_dir() and folder.name.startswith(self._block_data_f_prefix + "_"):
                block_code = folder.name.split("_", 1)[1]
                blocks.append(block_code)
        return blocks
    def list_provinces(self) -> list[str]:
        """Liệt kê tất cả các tỉnh/thành có trong dữ liệu CLEAN.

        Returns
        -------
        list[str]
            Danh sách các tỉnh/thành (mã tỉnh/thành).
        """
        province_dir = self.province_data_root
        provinces = []
        for folder in province_dir.iterdir():
            if folder.is_dir() and folder.name.startswith(self._province_data_f_prefix + "_"):
                province_code = folder.name.split("_", 1)[1]
                provinces.append(province_code)
        return provinces
