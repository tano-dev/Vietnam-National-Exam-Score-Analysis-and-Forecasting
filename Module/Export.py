from pathlib import Path
import unicodedata
import pandas as pd
import numpy as np
import os

from Module.Processor_Data import DataProcessor
from Module.Analysis import Analysis


class Export:
    """Export dữ liệu đã phân tích cuối pipeline sang CSV theo cấu trúc chuẩn.

    Mô tả:
        - Bước cuối của pipeline: Draw → Load → Processor → Analysis → Export → Clean Data
        - Tự động export toàn bộ dữ liệu thống kê theo Môn học, Khối thi & Tỉnh thành.
        - Không cần truyền tham số bên ngoài, toàn bộ domain được suy ra từ DataFrame.
        - Kết quả đầu ra là “Clean Data” dùng cho trực quan hoá (EDA), báo cáo, mô hình.

    Attributes (public API):
        processor (DataProcessor): Nguồn dữ liệu đã xử lý (data sạch).
        analysis  (Analysis): Module phân tích được đồng bộ với processor.
        root_path (str): Thư mục gốc lưu toàn bộ file CSV đã export.
    """

    # ==================== INTERNAL PRIVATE MEMBERS ====================
    __slots__ = (
        "_processor",      # Instance DataProcessor
        "_analysis",       # Instance Analysis (auto-sync theo processor)
        "_root_path",      # Đường dẫn thư mục gốc xuất dữ liệu
    )

    # -------------------- CONSTRUCTOR --------------------
    def __init__(self,
                 processor: DataProcessor,
                 root_path: str = "Clean_Data_2023-2025") -> None:
        """Khởi tạo Export, liên kết Analysis & tạo thư mục gốc nếu cần.

        Args:
            processor (DataProcessor): Đối tượng xử lý dữ liệu đầu vào.
            root_path (str): Thư mục gốc lưu clean data sau export.
        """
        # Gọi setter để đảm bảo validate & sync nội bộ
        self.processor = processor
        self.root_path = root_path

    # -------------------- GETTER / SETTER --------------------
    @property
    def processor(self) -> DataProcessor:
        """Đối tượng DataProcessor hiện tại (source của dữ liệu sạch)."""
        return self._processor

    @processor.setter
    def processor(self, value: DataProcessor) -> None:
        if not isinstance(value, DataProcessor):
            raise TypeError("processor phải là instance của DataProcessor.")
        self._processor = value
        # Mỗi lần thay processor, tạo mới Analysis tương ứng
        self._analysis = Analysis(value)

    @property
    def analysis(self) -> Analysis:
        """Đối tượng Analysis dùng nội bộ cho các phép phân tích."""
        return self._analysis

    @property
    def root_path(self) -> str:
        """Thư mục gốc để lưu toàn bộ Clean Data sau export."""
        return self._root_path

    @root_path.setter
    def root_path(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError("root_path phải là string.")
        self._root_path = value
        # Tự tạo thư mục gốc nếu chưa tồn tại
        Path(self._root_path).mkdir(parents=True, exist_ok=True)

    # ==================== INTERNAL PRIVATE METHODS ====================
    # ---------- Helpers: Detect domain từ dataframe ----------
    def _detect_subjects(self) -> list[str]:
        """Tự động lấy danh sách môn học từ dữ liệu sạch.

        Logic:
            - Chọn các cột dạng số (điểm thi).
            - Loại bỏ các cột không phải điểm: 'sbd', 'nam_hoc'.
        """
        df = self.processor.get_processed_data()
        return (
            df.select_dtypes(include="number")
              .columns.difference(["sbd", "nam_hoc"])
              .tolist()
        )

    def _detect_years(self) -> list[int]:
        """Lấy danh sách năm học trong dữ liệu (dùng cho phân tích, không gắn vào tên file)."""
        df = self.processor.get_processed_data()
        return sorted(df["nam_hoc"].unique().tolist())

    def _detect_provinces(self) -> list[str]:
        """Lấy danh sách tỉnh/thành thực tế có trong dữ liệu.

        Gọi qua Analysis.compare_by_region("ALL") để lấy phân phối,
        sau đó trích cột 'tinh'.
        """
        df = self.analysis.compare_by_region("ALL")
        return sorted(df["tinh"].unique().tolist())

    def _detect_blocks(self) -> list[str]:
        """Lấy danh sách khối thi có mặt trong dữ liệu.

        Thay vì truy cập map nội bộ, ta:
            - Gọi Analysis.analyze_scores_by_exam_block("All")
            - Lấy unique các giá trị cột 'khoi'.
        """
        df = self.analysis.analyze_scores_by_exam_block("All")
        if df.empty:
            return []
        return sorted(df["khoi"].unique().tolist())

    # ---------- Helpers: Chuẩn hoá tên folder/file ----------
    def _normalize_name(self, name: str) -> str:
        """Chuẩn hoá tên dùng cho folder/file.

        - Bỏ dấu tiếng Việt.
        - Xoá khoảng trắng & ký tự đặc biệt.
        - Giữ lại chỉ chữ cái, số, '_' để tránh lỗi path.

        Ví dụ:
            "Thành phố Hồ Chí Minh" -> "ThanhPhoHoChiMinh"
            "Hà Nội"                 -> "HaNoi"
        """
        # Tách accent
        normalized = unicodedata.normalize("NFD", name)
        # Bỏ các dấu (ký tự loại Mn = Mark, Nonspacing)
        without_accents = "".join(
            ch for ch in normalized if unicodedata.category(ch) != "Mn"
        )
        # Chỉ giữ a-zA-Z0-9 và '_'
        cleaned = "".join(
            ch for ch in without_accents if ch.isalnum() or ch == "_"
        )
        return cleaned

    def _build_path(self, category: str, name: str) -> str:
        """Tạo đường dẫn lưu file ANALYSIS (thống kê) theo cấu trúc chuẩn.

        Cấu trúc 4 lớp:
            root_path /
                {Subject_Data|Block_Data|Province_Data} /
                    CleanData_<Tên đã chuẩn hoá> /
                        Export_Analysis_<Tên đã chuẩn hoá>.csv
        """
        folder_map = {
            "subject": "Subject_Data",
            "block": "Block_Data",
            "province": "Province_Data",
        }

        safe_name = self._normalize_name(name)
        base_dir = (
            Path(self._root_path)
            / folder_map[category]
            / f"CleanData_{safe_name}"
        )
        base_dir.mkdir(parents=True, exist_ok=True)

        file_path = base_dir / f"Export_Analysis_{safe_name}.csv"
        return str(file_path)

    def _build_distribution_path(self, category: str, name: str) -> str:
        """Tạo đường dẫn lưu file DISTRIBUTION (DataFrame phân phối) cho EDA.

        Cùng cấu trúc thư mục với file thống kê, chỉ khác tên file:
            Export_Distribution_<Tên đã chuẩn hoá>.csv
        """
        folder_map = {
            "subject": "Subject_Data",
            "block": "Block_Data",
            "province": "Province_Data",
        }

        safe_name = self._normalize_name(name)
        base_dir = (
            Path(self._root_path)
            / folder_map[category]
            / f"CleanData_{safe_name}"
        )
        base_dir.mkdir(parents=True, exist_ok=True)

        file_path = base_dir / f"Export_Distribution_{safe_name}.csv"
        return str(file_path)

    # ---------- Export từng nhóm dữ liệu (internal only) ----------
    # ====== 1. Thống kê mô tả (dict → DataFrame) ======
    def _export_subject(self, subject: str) -> None:
        """Xuất CSV thống kê mô tả theo MÔN HỌC.

        Lấy từ:
            Analysis.get_statistics_by_subject(subject)
        Dữ liệu trong file:
            - Một dòng cho mỗi `nam_hoc`
            - Các cột: mean, median, mode, std, min, max
        """
        stats_dict = self.analysis.get_statistics_by_subject(subject)
        df = pd.DataFrame(stats_dict).T.reset_index().rename(columns={"index": "nam_hoc"})
        df.to_csv(self._build_path("subject", subject), index=False)

    def _export_block(self, block: str) -> None:
        """Xuất CSV thống kê mô tả theo KHỐI THI."""
        stats_dict = self.analysis.get_statistics_by_block(block)
        df = pd.DataFrame(stats_dict).T.reset_index().rename(columns={"index": "nam_hoc"})
        df.to_csv(self._build_path("block", block), index=False)

    def _export_province(self, province: str) -> None:
        """Xuất CSV thống kê mô tả theo TỈNH/THÀNH."""
        stats_dict = self.analysis.get_statistics_by_region(province)
        df = pd.DataFrame(stats_dict).T.reset_index().rename(columns={"index": "nam_hoc"})
        df.to_csv(self._build_path("province", province), index=False)

    # ====== 2. DataFrame phân phối cho EDA (từ Analysis) ======
    def _export_subject_distribution(self, subject: str) -> None:
        """Xuất CSV phân phối điểm theo MÔN HỌC để vẽ biểu đồ EDA.

        Lấy từ:
            Analysis.get_arregate_by_exam_subsections(subject)

        Cột output:
            ['nam_hoc', 'mon_hoc', 'diem', 'so_hoc_sinh']
        """
        df = self.analysis.get_arregate_by_exam_subsections(subject)
        df.to_csv(self._build_distribution_path("subject", subject), index=False)

    def _export_block_distribution(self, block: str) -> None:
        """Xuất CSV phân phối TỔNG ĐIỂM theo KHỐI THI để EDA.

        Lấy từ:
            Analysis.analyze_scores_by_exam_block(block)

        Cột output:
            ['khoi', 'nam_hoc', 'tong_diem', 'so_hoc_sinh']
        """
        df = self.analysis.analyze_scores_by_exam_block(block)
        df.to_csv(self._build_distribution_path("block", block), index=False)

    def _export_province_distribution(self, province: str) -> None:
        """Xuất CSV phân phối TỔNG ĐIỂM theo TỈNH/THÀNH để EDA.

        Lấy từ:
            Analysis.compare_by_region(province)

        Cột output:
            ['nam_hoc', 'tinh', 'tong_diem', 'so_hoc_sinh']
        """
        df = self.analysis.compare_by_region(province)
        df.to_csv(self._build_distribution_path("province", province), index=False)

    # ==================== PUBLIC METHODS (API) ====================
    def run_export_all(self) -> None:
        """Chạy toàn bộ quá trình export để sinh Clean Data cho EDA.

        Quy trình:
            1. Lấy danh sách môn học, khối thi, tỉnh/thành đang có trong data.
            2. Với mỗi nhóm, ghi ra HAI loại file:
                - Export_Analysis_<name>.csv      (thống kê mô tả)
                - Export_Distribution_<name>.csv  (DataFrame phân phối cho EDA)
            3. Tự động tạo đầy đủ thư mục & file CSV tương ứng.
        """
        subjects = self._detect_subjects()
        blocks = self._detect_blocks()
        provinces = self._detect_provinces()
        _ = self._detect_years()  # chỉ để đảm bảo có cột nam_hoc, không dùng vào tên file

        # Subject Data
        for s in subjects:
            self._export_subject(s)
            self._export_subject_distribution(s)

        # Block Data
        for b in blocks:
            self._export_block(b)
            self._export_block_distribution(b)

        # Province Data
        for p in provinces:
            self._export_province(p)
            self._export_province_distribution(p)

    # ==================== REPRESENTATION / UTILITIES ====================
    def __repr__(self) -> str:
        return f"<Export root='{self._root_path}' processor={type(self._processor).__name__}>"
