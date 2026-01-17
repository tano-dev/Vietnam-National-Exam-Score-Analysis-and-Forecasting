from Module.Load_Data import DataLoader
from pathlib import Path
import pandas as pd
import numpy as np

class DataProcessor:
    # ==================== INTERNAL PRIVATE METHODS: XỬ LÝ DỮ LIỆU =====================
    # ----------------------- Khai báo và thiết lập thuộc tính -------------------------
    """Tiền xử lý dữ liệu THPT đã load từ DataLoader.
    Attributes (public API):
        data_2018        (pd.DataFrame): Dữ liệu năm 2018 theo CT2006.
        data_2019        (pd.DataFrame): Dữ liệu năm 2019 theo CT2006.
        data_2020        (pd.DataFrame): Dữ liệu năm 2020 theo CT2006.
        data_2021        (pd.DataFrame): Dữ liệu năm 2021 theo CT2006.
        data_2022        (pd.DataFrame): Dữ liệu năm 2022 theo CT2006.
        data_2023        (pd.DataFrame): Dữ liệu năm 2023 theo CT2006.
        data_2024        (pd.DataFrame): Dữ liệu năm 2024 theo CT2006.
        data_2025_ct2006 (pd.DataFrame): Dữ liệu năm 2025 theo CT2006.
        data_2025_ct2018 (pd.DataFrame): Dữ liệu năm 2025 theo CT2018.
        combined_data    (pd.DataFrame): Dữ liệu tổng hợp từ các năm.
        
    Read-only properties (tự tính từ data_xxx):
        loader (DataLoader): Instance của DataLoader để load dữ liệu.
        combined_data (pd.DataFrame): Dữ liệu tổng hợp từ các năm.
    """
    
    # Slots: Cố định các thuộc tính có thể sử dụng, để tiết kiệm bộ nhớ. Không thể thêm thuộc tính mới ngoài danh sách này.
    __slots__ = (
        "_loader",                                    # instance của DataLoader để load dữ liệu
        "_data_2018",                                 # dữ liệu năm 2018
        "_data_2019",                                 # dữ liệu năm 2019
        "_data_2020",                                 # dữ liệu năm 2020
        "_data_2021",                                 # dữ liệu năm 2021
        "_data_2022",                                 # dữ liệu năm 2022
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
    # ------- Data 2018 -------
    @property
    def data_2018(self) -> pd.DataFrame:
        return self._data_2018
    
    @data_2018.setter
    def data_2018(self, value: pd.DataFrame) -> None:
        # 1) Kiểm tra kiểu và đầu vào
        if not isinstance(value, pd.DataFrame):
            raise TypeError("data_2018 phải là pandas DataFrame.")
        if value.empty:
            # Nếu bạn muốn “luôn set dữ liệu thật”, cấm gán rỗng:
            raise ValueError("Giá trị gán cho data_2018 không được rỗng.")

        # 2) Kiểm tra TRẠNG THÁI LƯU TRỮ (_data_2018)
        # - None: chưa gán lần nào → cho gán
        # - DataFrame không rỗng: đã có dữ liệu → chặn
        # - DataFrame rỗng: nếu bạn coi là “đã có” thì chặn; nếu coi là “chưa có”, thì cho gán
        if self._data_2018 is None:
            self._data_2018 = value.copy()
        elif isinstance(self._data_2018, pd.DataFrame) and self._data_2018.empty:
            # Nếu bạn muốn “rỗng xem như chưa có”, cho gán:
            self._data_2018 = value.copy()
        else:
            # Đã có dữ liệu → không cho ghi đè
            raise AttributeError("data_2018 đã có dữ liệu, không cho phép gán lần nữa.")
        
    # ------- Data 2019 -------
    @property
    def data_2019(self) -> pd.DataFrame:
        return self._data_2019
    
    @data_2019.setter
    def data_2019(self, value: pd.DataFrame) -> None:
        # 1) Kiểm tra kiểu và đầu vào
        if not isinstance(value, pd.DataFrame):
            raise TypeError("data_2019 phải là pandas DataFrame.")
        if value.empty:
            # Nếu bạn muốn “luôn set dữ liệu thật”, cấm gán rỗng:
            raise ValueError("Giá trị gán cho data_2019 không được rỗng.")

        # 2) Kiểm tra TRẠNG THÁI LƯU TRỮ (_data_2019)
        # - None: chưa gán lần nào → cho gán
        # - DataFrame không rỗng: đã có dữ liệu → chặn
        # - DataFrame rỗng: nếu bạn coi là “đã có” thì chặn; nếu coi là “chưa có”, thì cho gán
        if self._data_2019 is None:
            self._data_2019 = value.copy()
        elif isinstance(self._data_2019, pd.DataFrame) and self._data_2019.empty:
            # Nếu bạn muốn “rỗng xem như chưa có”, cho gán:
            self._data_2019 = value.copy()
        else:
            # Đã có dữ liệu → không cho ghi đè
            raise AttributeError("data_2019 đã có dữ liệu, không cho phép gán lần nữa.")
    
    # ------- Data 2020 -------
    @property
    def data_2020(self) -> pd.DataFrame:
        return self._data_2020
    
    @data_2020.setter
    def data_2020(self, value: pd.DataFrame) -> None:
        # 1) Kiểm tra kiểu và đầu vào
        if not isinstance(value, pd.DataFrame):
            raise TypeError("data_2020 phải là pandas DataFrame.")
        if value.empty:
            # Nếu bạn muốn “luôn set dữ liệu thật”, cấm gán rỗng:
            raise ValueError("Giá trị gán cho data_2020 không được rỗng.")

        # 2) Kiểm tra TRẠNG THÁI LƯU TRỮ (_data_2020)
        # - None: chưa gán lần nào → cho gán
        # - DataFrame không rỗng: đã có dữ liệu → chặn
        # - DataFrame rỗng: nếu bạn coi là “đã có” thì chặn; nếu coi là “chưa có”, thì cho gán
        if self._data_2020 is None:
            self._data_2020 = value.copy()
        elif isinstance(self._data_2020, pd.DataFrame) and self._data_2020.empty:
            # Nếu bạn muốn “rỗng xem như chưa có”, cho gán:
            self._data_2020 = value.copy()
        else:
            # Đã có dữ liệu → không cho ghi đè
            raise AttributeError("data_2020 đã có dữ liệu, không cho phép gán lần nữa.")
        
    # ------- Data 2021 -------
    @property
    def data_2021(self) -> pd.DataFrame:
        return self._data_2021
    
    @data_2021.setter
    def data_2021(self, value: pd.DataFrame) -> None:
        # 1) Kiểm tra kiểu và đầu vào
        if not isinstance(value, pd.DataFrame):
            raise TypeError("data_2021 phải là pandas DataFrame.")
        if value.empty:
            # Nếu bạn muốn “luôn set dữ liệu thật”, cấm gán rỗng:
            raise ValueError("Giá trị gán cho data_2021 không được rỗng.")

        # 2) Kiểm tra TRẠNG THÁI LƯU TRỮ (_data_2021)
        # - None: chưa gán lần nào → cho gán
        # - DataFrame không rỗng: đã có dữ liệu → chặn
        # - DataFrame rỗng: nếu bạn coi là “đã có” thì chặn; nếu coi là “chưa có”, thì cho gán
        if self._data_2021 is None:
            self._data_2021 = value.copy()
        elif isinstance(self._data_2021, pd.DataFrame) and self._data_2021.empty:
            # Nếu bạn muốn “rỗng xem như chưa có”, cho gán:
            self._data_2021 = value.copy()
        else:
            # Đã có dữ liệu → không cho ghi đè
            raise AttributeError("data_2021 đã có dữ liệu, không cho phép gán lần nữa.")
    
    # ------- Data 2022 -------
    @property
    def data_2022(self) -> pd.DataFrame:
        return self._data_2022
    
    @data_2022.setter
    def data_2022(self, value: pd.DataFrame) -> None:
        # 1) Kiểm tra kiểu và đầu vào
        if not isinstance(value, pd.DataFrame):
            raise TypeError("data_2022 phải là pandas DataFrame.")
        if value.empty:
            # Nếu bạn muốn “luôn set dữ liệu thật”, cấm gán rỗng:
            raise ValueError("Giá trị gán cho data_2022 không được rỗng.")

        # 2) Kiểm tra TRẠNG THÁI LƯU TRỮ (_data_2022)
        # - None: chưa gán lần nào → cho gán
        # - DataFrame không rỗng: đã có dữ liệu → chặn
        # - DataFrame rỗng: nếu bạn coi là “đã có” thì chặn; nếu coi là “chưa có”, thì cho gán
        if self._data_2022 is None:
            self._data_2022 = value.copy()
        elif isinstance(self._data_2022, pd.DataFrame) and self._data_2022.empty:
            # Nếu bạn muốn “rỗng xem như chưa có”, cho gán:
            self._data_2022 = value.copy()
        else:
            # Đã có dữ liệu → không cho ghi đè
            raise AttributeError("data_2022 đã có dữ liệu, không cho phép gán lần nữa.")
        
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
        self._data_2018 = None
        self._data_2019 = None
        self._data_2020 = None
        self._data_2021 = None
        self._data_2022 = None
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
            self.data_2018,
            self.data_2019,
            self.data_2020,
            self.data_2021,
            self.data_2022,
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
        cols_keep = ["sbd", "nam_hoc"]

        for df in [self.data_2018, self.data_2019, self.data_2020, self.data_2021, self.data_2022, 
                   self.data_2023, self.data_2024, self.data_2025_ct2006, self.data_2025_ct2018]:
            cols_check = [c for c in df.columns if c not in cols_keep]
            df.dropna(subset=cols_check, how="all", inplace=True)

        # Loại bỏ các hàng trùng lặp dựa trên cột 'SOBAODANH'
        self.data_2018.drop_duplicates(subset=["sbd"],keep='first', inplace=True)
        self.data_2019.drop_duplicates(subset=["sbd"],keep='first', inplace=True)
        self.data_2020.drop_duplicates(subset=["sbd"],keep='first', inplace=True)
        self.data_2021.drop_duplicates(subset=["sbd"],keep='first', inplace=True)
        self.data_2022.drop_duplicates(subset=["sbd"],keep='first', inplace=True)
        self.data_2023.drop_duplicates(subset=["sbd"],keep='first', inplace=True)
        self.data_2024.drop_duplicates(subset=["sbd"],keep='first', inplace=True)
        self.data_2025_ct2006.drop_duplicates(subset=["sbd"],keep='first', inplace=True)
        self.data_2025_ct2018.drop_duplicates(subset=["sbd"],keep='first', inplace=True)
    
    # Chuẩn hóa tên cột và cấu trúc các cột dữ liệu 
    def _normalize_columns(self) -> None:
        """ Chuẩn hóa tên cột(ví dụ: đổi tên cột để nhất quán giữa các năm). Thêm hoặc bớt cột nếu cần thiết."""
        
        # Trước khi rename
        self.data_2025_ct2006.drop(columns=["STT"], errors="ignore", inplace=True)
        self.data_2025_ct2018.drop(columns=["STT"], errors="ignore", inplace=True)
        
        # Xây dựng map chuẩn hóa tên môn học 
        # Map cho năm 2018 
        col_map_2018_ct2006 = {
            "SBD": "sbd",
            "Toan": "toan",
            "NguVan": "ngu_van",
            "NgoaiNgu": "ngoai_ngu",
            "VatLy": "vat_li",
            "HoaHoc": "hoa_hoc",
            "SinhHoc": "sinh_hoc",
            "LichSu": "lich_su",
            "DiaLy": "dia_li",
            "GDCD": "gdcd",
            "MaMonNgoaiNgu": "ma_ngoai_ngu",
        }
        # Áp dụng cho DataFrame:
        self.data_2018.rename(columns=col_map_2018_ct2006, inplace=True)
           
        # Map cho năm 2019
        col_map_2019_ct2006 = {
            "SBD": "sbd",
            "Toan": "toan",
            "NguVan": "ngu_van",
            "NgoaiNgu": "ngoai_ngu",
            "VatLy": "vat_li",
            "HoaHoc": "hoa_hoc",
            "SinhHoc": "sinh_hoc",
            "LichSu": "lich_su",
            "DiaLy": "dia_li",
            "GDCD": "gdcd",
            "MaMonNgoaiNgu": "ma_ngoai_ngu",
        }
        # Áp dụng cho DataFrame:
        self.data_2019.rename(columns=col_map_2019_ct2006, inplace=True)
        
        # Map cho năm 2020
        col_map_2020_ct2006 = {
            "SBD": "sbd",
            "Toan": "toan",
            "NguVan": "ngu_van",
            "NgoaiNgu": "ngoai_ngu",
            "VatLy": "vat_li",
            "HoaHoc": "hoa_hoc",
            "SinhHoc": "sinh_hoc",
            "LichSu": "lich_su",
            "DiaLy": "dia_li",
            "GDCD": "gdcd",
            "MaMonNgoaiNgu": "ma_ngoai_ngu",
        }
        # Áp dụng cho DataFrame:
        self.data_2020.rename(columns=col_map_2020_ct2006, inplace=True)

        # Map cho năm 2021: SBD,Toan,Ngu_Van,Ngoai_Ngu,Vat_Ly,Hoa_Hoc,Sinh_Hoc,Lich_Su,Dia_Ly,GDCD,Cum_Thi
        col_map_2021_ct2006 = {
            "SBD": "sbd",
            "Toan": "toan",
            "Ngu_Van": "ngu_van",
            "Ngoai_Ngu": "ngoai_ngu",
            "Vat_Ly": "vat_li",
            "Hoa_Hoc": "hoa_hoc",
            "Sinh_Hoc": "sinh_hoc",
            "Lich_Su": "lich_su",
            "Dia_Ly": "dia_li",
            "GDCD": "gdcd",
            "MaMonNgoaiNgu": "ma_ngoai_ngu",
        }
        # Áp dụng cho DataFrame:
        self.data_2021.rename(columns=col_map_2021_ct2006, inplace=True)    
        
        # Map cho năm 2022:sbd,toan,ngu_van,ngoai_ngu,vat_li,hoa_hoc,sinh_hoc,lich_su,dia_li,gdcd
            
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
        for df in [self.data_2018, self.data_2019, self.data_2020, self.data_2021, self.data_2022,
                   self.data_2023, self.data_2024, self.data_2025_ct2006]:
            df['cn_cong_nghiep'] = np.nan
            df['cn_nong_nghiep'] = np.nan
            df['tin_hoc'] = np.nan
            
        # Xây dựng cột thể hiện điểm theo năm học
        for df, year in [(self.data_2018, 2018), (self.data_2019, 2019), (self.data_2020, 2020), (self.data_2021, 2021), (self.data_2022, 2022),
                         (self.data_2023, 2023), (self.data_2024, 2024), 
                         (self.data_2025_ct2006, 2025), (self.data_2025_ct2018, 2025)]:
            df['nam_hoc'] = year
    
    # Xây dựng Target Schema cho tất cả các năm
    def _apply_target_schema_all_years(self) -> None:
        """
        Áp dụng Target Schema cho tất cả các DataFrame của các năm.
        Giữ lại chỉ các cột trong TARGET và loại bỏ các cột không cần thiết
        như 'ma_ngoai_ngu'.
        """    
    
        TARGET = [
            "sbd","toan","ngu_van","ngoai_ngu","vat_li","hoa_hoc","sinh_hoc",
            "lich_su","dia_li","gdcd","tin_hoc","cn_cong_nghiep","cn_nong_nghiep","nam_hoc"
        ]

        df_map = {
            "data_2018": self.data_2018,
            "data_2019": self.data_2019,
            "data_2020": self.data_2020,
            "data_2021": self.data_2021,
            "data_2022": self.data_2022,
            "data_2023": self.data_2023,
            "data_2024": self.data_2024,
            "data_2025_ct2006": self.data_2025_ct2006,
            "data_2025_ct2018": self.data_2025_ct2018,
        }

        for name, df in df_map.items():
            df = df.copy()
            df.drop(columns=["ma_ngoai_ngu"], errors="ignore", inplace=True)
            if "sbd" in df.columns:
                df["sbd"] = df["sbd"].astype("string").str.strip()

            df = df.reindex(columns=TARGET)   # <-- giờ mới thật sự giữ target

            setattr(self, f"_{name}", df)           # <-- gán lại vào self.*

    # Xây dựng Data Tổng kết hợp dữ liệu từ các năm 
    def _build_combined_data(self) -> pd.DataFrame:
        """ Kết hợp dữ liệu từ các năm thành một DataFrame duy nhất."""
        self._combined_data = pd.concat(
            [self.data_2018, self.data_2019, self.data_2020, self.data_2021, self.data_2022,
            self.data_2023, self.data_2024, self.data_2025_ct2006, self.data_2025_ct2018],
            ignore_index=True
        )
        return self._combined_data
    
    # Xây dựng hàm kiểm tra tính hợp lệ và ép kiểu dữ liệu đã xử lý
    def _validate_data(self) -> None:
        """Kiểm tra tính hợp lệ của self.combined_data và ép kiểu float.
        - Cho phép một số cột tùy chọn có thể toàn NaN (ví dụ môn không có ở CT2006).
        - Chỉ báo lỗi nếu các giá trị KHÔNG NaN nằm ngoài [0, 10].
        - Báo cáo số lượng giá trị bị ép (coerce) thành NaN khi to_numeric.
        """
        if not isinstance(self.combined_data, pd.DataFrame):
            raise TypeError("combined_data phải là DataFrame trước khi validate.")

        df = self.combined_data  # alias cho gọn

        required_cols = [
            'toan', 'ngu_van', 'vat_li', 'hoa_hoc', 'sinh_hoc',
            'lich_su', 'dia_li', 'gdcd', 'ngoai_ngu'
        ]
        optional_cols = ['cn_cong_nghiep', 'cn_nong_nghiep', 'tin_hoc']

        for col in required_cols + optional_cols:
            if col not in df.columns:
                # với dữ liệu của bạn hầu như đã tạo đủ cột; nếu thiếu thì raise rõ ràng
                raise ValueError(f"Thiếu cột bắt buộc: {col}")

            # Ép số; đếm số lượng bị coerce để giám sát chất lượng dữ liệu
            before_na = df[col].isna().sum()
            df[col] = pd.to_numeric(df[col], errors='coerce')
            after_na = df[col].isna().sum()
            coerced = max(0, after_na - before_na)

            # Chỉ xét các giá trị KHÔNG NaN
            notna_mask = df[col].notna()
            if notna_mask.any():
                out_range_mask = ~df.loc[notna_mask, col].between(0, 10)
                if out_range_mask.any():
                    bad_idx = df.loc[notna_mask].index[out_range_mask].tolist()[:10]  # show tối đa 10
                    raise ValueError(
                        f"Cột '{col}' có {out_range_mask.sum()} giá trị ngoài [0,10]. "
                        f"Ví dụ index: {bad_idx}"
                    )
            else:
                # Cả cột toàn NaN
                if col in required_cols:
                    raise ValueError(f"Cột bắt buộc '{col}' toàn NaN sau khi chuẩn hoá/ép kiểu.")
                # optional: cho phép toàn NaN, bỏ qua

            # Ép kiểu float rõ ràng
            df[col] = df[col].astype(float)

            # (Tùy chọn) log cảnh báo khi có giá trị bị coerce
            if coerced > 0 and hasattr(self, 'logger'):
                self.logger.warning(f"Cột '{col}': {coerced} giá trị không phải số đã bị coerce thành NaN.")

        # Lưu lại (không cần gán lại nếu df là alias, nhưng làm rõ ý đồ)
        self.combined_data = df        
    
    # Xây dựng hàm kiểm tra xung đột dữ liệu trước khi dedup
    def _check_conflicts_before_dedup(self) -> None:
        SCORE_COLS = [
            "toan","ngu_van","ngoai_ngu","vat_li","hoa_hoc","sinh_hoc",
            "lich_su","dia_li","gdcd","tin_hoc","cn_cong_nghiep","cn_nong_nghiep"
        ]

        df_map = {
            "2018": self.data_2018,
            "2019": self.data_2019,
            "2020": self.data_2020,
            "2021": self.data_2021,
            "2022": self.data_2022,
            "2023": self.data_2023,
            "2024": self.data_2024,
            "2025_ct2006": self.data_2025_ct2006,
            "2025_ct2018": self.data_2025_ct2018,
        }

        blank_msgs = []
        conflict_msgs = []
        dup_msgs = []  # chỉ warn

        for name, df in df_map.items():
            # 1) sbd blank
            s = df["sbd"].astype("string")
            blank = s.isna() | (s.str.strip() == "")
            if blank.any():
                blank_msgs.append(f"[{name}] {blank.sum()} dòng sbd rỗng/NaN")

            # 2) duplicate check
            dup_mask = df["sbd"].duplicated(keep=False)
            if not dup_mask.any():
                continue

            dup_df = df.loc[dup_mask, ["sbd"] + SCORE_COLS].copy()

            # ép numeric để tránh conflict giả (string vs float)
            for c in SCORE_COLS:
                dup_df[c] = pd.to_numeric(dup_df[c], errors="coerce")

            conflict_sbd = []
            for sbd, sub in dup_df.groupby("sbd", dropna=False):
                nunq = sub[SCORE_COLS].nunique(dropna=True)
                if (nunq > 1).any():
                    conflict_sbd.append(sbd)
            if conflict_sbd:
                conflict_msgs.append(
                    f"[{name}] {len(conflict_sbd)} SBD trùng nhưng ĐIỂM MÂU THUẪN. Ví dụ: {conflict_sbd[:5]}"
                )
            else:
                dup_msgs.append(f"[{name}] có trùng SBD nhưng điểm không mâu thuẫn (duplicate bản ghi).")

        # Raise chỉ khi thật sự nghiêm trọng
        if blank_msgs or conflict_msgs:
            msg = "PHÁT HIỆN LỖI DỮ LIỆU TRƯỚC DEDUP:\n- " + "\n- ".join(blank_msgs + conflict_msgs)
            raise ValueError(msg)

        # Nếu chỉ duplicate “lành” thì không chặn pipeline (tuỳ bạn in ra)
        if dup_msgs:
            print("CẢNH BÁO DUPLICATE (không mâu thuẫn):\n- " + "\n- ".join(dup_msgs))


    # ===================== PUBLIC API: Hàm thực hiện toàn bộ quy trình xử lý =====================
    # ------- Xây dựng hàm để thực hiện toàn bộ quy trình xử lý --------
    def process_all(self) -> None:
        """Thực hiện toàn bộ quy trình xử lý dữ liệu."""
        self._normalize_columns()
        self._apply_target_schema_all_years()
        
        self._check_conflicts_before_dedup()  
        
        self._preprocess_data()
        self._build_combined_data()
        self._validate_data()
    
    # ------- Hàm lấy dữ liệu đã được xử lý --------
    def get_processed_data(self) -> pd.DataFrame:
        """Trả về kết quả đã xử lý."""
        return self._combined_data