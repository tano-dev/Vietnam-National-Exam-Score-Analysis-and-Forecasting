from Module.Load_Data import DataLoader
from pathlib import Path
import pandas as pd
import numpy as np

class DataProcessor:
    # ==================== INTERNAL PRIVATE METHODS: XỬ LÝ DỮ LIỆU =====================
    # ----------------------- Khai báo và thiết lập thuộc tính -------------------------
    """Tiền xử lý dữ liệu THPT đã load từ DataLoader.
    Attributes (public API):
        data_2023        (pd.DataFrame): Dữ liệu năm 2023 theo CT2006.
        data_2024        (pd.DataFrame): Dữ liệu năm 2024 theo CT2006.
        data_2025_ct2006 (pd.DataFrame): Dữ liệu năm 2025 theo CT2006.
        data_2025_ct2018 (pd.DataFrame): Dữ liệu năm 2025 theo CT2018.
    """
    
    # Slots: Cố định các thuộc tính có thể sử dụng, để tiết kiệm bộ nhớ. Không thể thêm thuộc tính mới ngoài danh sách này.
    __slots__ = (
        "_loader",                                    # instance của DataLoader để load dữ liệu
        "_data_2023",                                 # dữ liệu năm 2023
        "_data_2024",                                 # dữ liệu năm 2024
        "_data_2025_ct2006",                          # dữ liệu năm 2025 CT2006
        "_data_2025_ct2018",                          # dữ liệu năm 2025 CT2018
        "_combined_data"                              # dữ liệu tổng hợp từ các năm
    )
    
    # ------- Xây dựng setter & getter để xử lý các biến --------
    # Setter Getter: Loader
    @property
    def loader(self) -> DataLoader:
        """Trả về instance của DataLoader."""
        return self._loader
    
    @loader.setter
    def loader(self, value: DataLoader) -> None:
        if not isinstance(value, DataLoader):
            raise TypeError("loader phải là một instance của DataLoader.")
        self._loader = value
        
    # Setter Getter: DataFrames
    # ------- Data 2023 -------
    @property
    def data_2023(self) -> pd.DataFrame:
        return self._data_2023

    @data_2023.setter
    def data_2023(self, value: pd.DataFrame) -> None:
        # 1) Kiểm tra kiểu và đầu vào
        if not isinstance(value, pd.DataFrame):
            raise TypeError("data_2023 phải là pandas DataFrame.")
        if value.empty:
            # Nếu bạn muốn “luôn set dữ liệu thật”, cấm gán rỗng:
            raise ValueError("Giá trị gán cho data_2023 không được rỗng.")

        # 2) Kiểm tra TRẠNG THÁI LƯU TRỮ (_data_2023)
        # - None: chưa gán lần nào → cho gán
        # - DataFrame không rỗng: đã có dữ liệu → chặn
        # - DataFrame rỗng: nếu bạn coi là “đã có” thì chặn; nếu coi là “chưa có”, thì cho gán
        if self._data_2023 is None:
            self._data_2023 = value.copy()
        elif isinstance(self._data_2023, pd.DataFrame) and self._data_2023.empty:
            # Nếu bạn muốn “rỗng xem như chưa có”, cho gán:
            self._data_2023 = value.copy()
        else:
            # Đã có dữ liệu → không cho ghi đè
            raise AttributeError("data_2023 đã có dữ liệu, không cho phép gán lần nữa.")

    # ------- Data 2024 -------
    @property
    def data_2024(self) -> pd.DataFrame:
        return self._data_2024
    
    @data_2024.setter
    def data_2024(self, value: pd.DataFrame) -> None:
        # 1) Kiểm tra kiểu và đầu vào
        if not isinstance(value, pd.DataFrame):
            raise TypeError("data_2024  phải là pandas DataFrame.")
        if value.empty:
            # Nếu bạn muốn “luôn set dữ liệu thật”, cấm gán rỗng:
            raise ValueError("Giá trị gán cho data_2024 không được rỗng.")

        # 2) Kiểm tra TRẠNG THÁI LƯU TRỮ (_data_2024)
        # - None: chưa gán lần nào → cho gán
        # - DataFrame không rỗng: đã có dữ liệu → chặn
        # - DataFrame rỗng: nếu bạn coi là “đã có” thì chặn; nếu coi là “chưa có”, thì cho gán
        if self._data_2024 is None:
            self._data_2024 = value.copy()
        elif isinstance(self._data_2024, pd.DataFrame) and self._data_2024.empty:
            # Nếu bạn muốn “rỗng xem như chưa có”, cho gán:
            self._data_2024 = value.copy()
        else:
            # Đã có dữ liệu → không cho ghi đè
            raise AttributeError("data_2024 đã có dữ liệu, không cho phép gán lần nữa.")

    # ------- Data 2025 CT2006 -------
    @property
    def data_2025_ct2006(self) -> pd.DataFrame:
        return self._data_2025_ct2006
    
    @data_2025_ct2006.setter
    def data_2025_ct2006(self, value: pd.DataFrame) -> None:
        # 1) Kiểm tra kiểu và đầu vào
        if not isinstance(value, pd.DataFrame):
            raise TypeError("data_2025_ct2006 phải là pandas DataFrame.")
        if value.empty:
            # Nếu bạn muốn “luôn set dữ liệu thật”, cấm gán rỗng:
            raise ValueError("Giá trị gán cho data_2025_ct2006 không được rỗng.")

        # 2) Kiểm tra TRẠNG THÁI LƯU TRỮ (_data_2025_ct2006)
        # - None: chưa gán lần nào → cho gán
        # - DataFrame không rỗng: đã có dữ liệu → chặn
        # - DataFrame rỗng: nếu bạn coi là “đã có” thì chặn; nếu coi là “chưa có”, thì cho gán
        if self._data_2025_ct2006 is None:
            self._data_2025_ct2006 = value.copy()
        elif isinstance(self._data_2025_ct2006, pd.DataFrame) and self._data_2025_ct2006.empty:
            # Nếu bạn muốn “rỗng xem như chưa có”, cho gán:
            self._data_2025_ct2006 = value.copy()
        else:
            # Đã có dữ liệu → không cho ghi đè
            raise AttributeError("data_2025_ct2006 đã có dữ liệu, không cho phép gán lần nữa.")

    
    # ------- Data 2025 CT2018 -------
    @property
    def data_2025_ct2018(self) -> pd.DataFrame:
        return self._data_2025_ct2018
    
    @data_2025_ct2018.setter
    def data_2025_ct2018(self, value: pd.DataFrame) -> None:
        # 1) Kiểm tra kiểu và đầu vào
        if not isinstance(value, pd.DataFrame):
            raise TypeError("data_2025_ct2018 phải là pandas DataFrame.")
        if value.empty:
            # Nếu bạn muốn “luôn set dữ liệu thật”, cấm gán rỗng:
            raise ValueError("Giá trị gán cho data_2025_ct2018 không được rỗng.")

        # 2) Kiểm tra TRẠNG THÁI LƯU TRỮ (_data_2025_ct2018)
        # - None: chưa gán lần nào → cho gán
        # - DataFrame không rỗng: đã có dữ liệu → chặn
        # - DataFrame rỗng: nếu bạn coi là “đã có” thì chặn; nếu coi là “chưa có”, thì cho gán
        if self._data_2025_ct2018 is None:
            self._data_2025_ct2018 = value.copy()
        elif isinstance(self._data_2025_ct2018, pd.DataFrame) and self._data_2025_ct2018.empty:
            # Nếu bạn muốn “rỗng xem như chưa có”, cho gán:
            self._data_2025_ct2018 = value.copy()
        else:
            # Đã có dữ liệu → không cho ghi đè
            raise AttributeError("data_2025_ct2018 đã có dữ liệu, không cho phép gán lần nữa.")
     
    # -------- Combined Data --------
    @property
    def combined_data(self) -> pd.DataFrame:
        """Trả về DataFrame tổng hợp từ các năm."""
        return self._combined_data
    
    @combined_data.setter
    def combined_data(self, value: pd.DataFrame) -> None:
        if not isinstance(value, pd.DataFrame):
            raise TypeError("combined_data phải là một pandas DataFrame.")
        if value.empty:
            raise ValueError("Giá trị gán cho combined_data không được rỗng.")
        
        self._combined_data = value

    # -------- Khởi tạo và tải dữ liệu --------
    def __init__(self, project_root: Path | str | None = None):
        """ Khởi tạo DataProcessor với DataLoader bên trong.
        Args: 
            project_root (Path | str | None): Thư mục gốc của project. Nếu None, sử dụng thư mục hiện tại.
        """
        # Khởi tạo các thuộc tính bên trong
        self._loader = None
        self._data_2023 = None
        self._data_2024 = None
        self._data_2025_ct2006 = None
        self._data_2025_ct2018 = None
        self._combined_data = pd.DataFrame()
        
        # Khởi tạo DataLoader bên trong
        self.loader = DataLoader(project_root)
        self.load_all_data()

    def load_all_data(self) -> None:
        """Load tất cả dữ liệu từ DataLoader."""
        (
            self.data_2023,
            self.data_2024,
            self.data_2025_ct2006,
            self.data_2025_ct2018
        ) = self.loader.load_data()

    # ==================== INTERNAL PRIVATE METHODS: XỬ LÝ DỮ LIỆU =====================
    # ------- Method: Các hàm xử lý dữ liệu --------
    # Xứ lý giá trị thiếu của dữ liệu 
    def _preprocess_data(self) -> None:
        """Tiền xử lý dữ liệu (ví dụ: xử lý giá trị thiếu, chuẩn hóa)."""
        # Loại bỏ hàng có giá trị thiếu
        self.data_2023.dropna(how='all', inplace=True)
        self.data_2024.dropna(how='all', inplace=True)
        self.data_2025_ct2006.dropna(how='all', inplace=True)
        self.data_2025_ct2018.dropna(how='all', inplace=True)
    
    # Chuẩn hóa tên cột và cấu trúc các cột dữ liệu 
    def _normalize_columns(self) -> None:
        """ Chuẩn hóa tên cột(ví dụ: đổi tên cột để nhất quán giữa các năm). Thêm hoặc bớt cột nếu cần thiết."""
        
        # Trước khi rename
        self.data_2025_ct2006.drop(columns=["STT"], errors="ignore", inplace=True)
        self.data_2025_ct2018.drop(columns=["STT"], errors="ignore", inplace=True)
        
        # Xây dựng map chuẩn hóa tên môn học 
        # Map cho năm 2025 CT2006
        col_map_2025_ct2006 = {
            "SOBAODANH": "sbd",
            "Toán": "toan",
            "Văn": "ngu_van",
            "Lí": "vat_li",
            "Hóa": "hoa_hoc",
            "Sinh": "sinh_hoc",
            "Sử": "lich_su",
            "Địa": "dia_li",
            "Giáo dục công dân": "gdcd",
            "Ngoại ngữ": "ngoai_ngu",
            "Mã môn ngoại ngữ": "ma_ngoai_ngu"
        }
        # Áp dụng cho DataFrame:
        self.data_2025_ct2006.rename(columns=col_map_2025_ct2006, inplace=True)
        
        # Map cho năm 2025 CT2018
        col_map_2025_ct2018 = {
            "STT": None,  # hoặc bỏ cột này nếu không dùng
            "SOBAODANH": "sbd",
            "Toán": "toan",
            "Văn": "ngu_van",
            "Lí": "vat_li",
            "Hóa": "hoa_hoc",
            "Sinh": "sinh_hoc",
            "Tin học": "tin_hoc",
            "Công nghệ công nghiệp": "cn_cong_nghiep",
            "Công nghệ nông nghiệp": "cn_nong_nghiep",
            "Sử": "lich_su",
            "Địa": "dia_li",
            "Giáo dục kinh tế và pháp luật": "gdcd",
            "Ngoại ngữ": "ngoai_ngu",
            "Mã môn ngoại ngữ": "ma_ngoai_ngu"
        }
        # Áp dụng cho DataFrame:
        self.data_2025_ct2018.rename(columns=col_map_2025_ct2018, inplace=True)
        
        # Bổ sung các cột ở Data_2023, Data_2024 
        for df in [self.data_2023, self.data_2024, self.data_2025_ct2006]:
            df['cn_cong_nghiep'] = np.nan
            df['cn_nong_nghiep'] = np.nan
            df['tin_hoc'] = np.nan
            
        # Xây dựng cột thể hiện điểm theo năm học
        for df, year in [(self.data_2023, 2023), (self.data_2024, 2024), 
                         (self.data_2025_ct2006, 2025), (self.data_2025_ct2018, 2025)]:
            df['nam_hoc'] = year
    
    # ------- Xây dựng Data Tổng kết hợp dữ liệu từ các năm --------
    def _build_combined_data(self) -> pd.DataFrame:
        """ Kết hợp dữ liệu từ các năm thành một DataFrame duy nhất."""
        self._combined_data = pd.concat(
            [self.data_2023, self.data_2024, self.data_2025_ct2006, self.data_2025_ct2018],
            ignore_index=True
        )
        return self._combined_data
    
    # ===================== PUBLIC API: Hàm thực hiện toàn bộ quy trình xử lý =====================
    # ------- Xây dựng hàm để thực hiện toàn bộ quy trình xử lý --------
    def process_all(self) -> None:
        """Thực hiện toàn bộ quy trình xử lý dữ liệu."""
        self._preprocess_data()
        self._normalize_columns()
        self._build_combined_data()
    
    # ------- Hàm lấy dữ liệu đã được xử lý --------
    def get_processed_data(self) -> pd.DataFrame:
        """Trả về kết quả đã xử lý."""
        return self._combined_data