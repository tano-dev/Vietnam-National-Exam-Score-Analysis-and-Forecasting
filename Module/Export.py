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
        return df
    
    # ----------------------------- Internal Methods -----------------------------
    # Xuất dữ liệu phân tích điểm của một môn học cụ thể
    def _export_score_analysis(self, subject: str, filepath: str) -> pd.DataFrame:
        """Xuất dữ liệu phân tích điểm của một môn học cụ thể."""
        analysis = Analysis(self.Export)
        distribution = analysis.get_score_distribution(subject)
        distribution.to_csv(filepath, index=False)

    # Xuất dữ liệu phân tích điểm theo môn học.
    def _export_subject_analysis(self, subject: str, filepath: str) -> pd.DataFrame:
        """Xuất dữ liệu phân tích điểm theo môn học."""
        analysis = Analysis(self.Export)
        stats = analysis.get_arregate_by_exam_subsections(subject)
        stats.to_csv(filepath, index=False)
    
    # Xuất dữ liệu phân tích điểm theo khối thi.
    def _export_block_analysis(self, block: str, filepath: str) -> pd.DataFrame:
        """Xuất dữ liệu phân tích điểm theo khối thi."""
        analysis = Analysis(self.Export)
        stats = analysis.analyze_scores_by_exam_block(block)
        stats.to_csv(filepath, index=False)
    
    # Xuất dữ liệu phân tích điểm theo tỉnh thành.
    def _export_city_analysis(self, city: str, filepath: str) -> pd.DataFrame:
        """Xuất dữ liệu phân tích điểm theo tỉnh thành."""
        analysis = Analysis(self.Export)
        stats = analysis.compare_by_region(city)
        stats.to_csv(filepath, index=False)
        
        
    
    # Xuất dữ liệu điểm theo khối thi
    def _export_score_by_block(self, block: str) -> pd.DataFrame:
        """Xuất dữ liệu điểm theo khối thi."""
        analysis = Analysis(self.Export)
        stats = analysis.get_statistics_by_block(block)
        filename = f"Export_Score_Block_{block}.csv"
        stats.to_csv(filename, index=False)
    
    # Xuất dữ liệu điểm theo môn học
    def _export_score_by_subject(self, subject: str, filepath: str) -> pd.DataFrame:
        """Xuất dữ liệu điểm theo môn học."""
        analysis = Analysis(self.Export)
        distribution = analysis.get_statistics_by_subject(subject)
        distribution.to_csv(filepath, index=False)
    
    # Xuất dữ liệu điểm theo tỉnh thành
    def _export_score_by_city(self, city: str, filepath: str) -> pd.DataFrame:
        """Xuất dữ liệu điểm theo tỉnh thành."""
        analysis = Analysis(self.Export)
        comparison = analysis.get_statistics_by_region(city)
        comparison.to_csv(filepath, index=False)
    
    # ==================== PUBLIC API METHODS: XUẤT DỮ LIỆU ====================
    # Xuất dữ liệu điểm theo khối thi ra file CSV
    # def export_score_by_block(self, block: str, file_path: str) -> None:
    #     """Xuất dữ liệu điểm theo khối thi ra file CSV."""
    #     # Analysis object is now created once in __init__ and reused
    #     analysis = Analysis(self.Export)
    #     stats = analysis.get_statistics_by_block(block)
    #     stats.to_csv(file_path)
        
    # # Xuất dữ liệu điểm theo môn học ra file CSV
    # def export_score_by_subject(self, subject: str, file_path: str) -> None:
    #     """Xuất dữ liệu điểm theo môn học ra file CSV."""
    #     analysis = Analysis(self.Export)
    #     distribution = analysis.get_statistics_by_subject(subject)
    #     distribution.to_csv(file_path, header=['Score Distribution'])
    # # Xuất dữ liệu điểm theo tỉnh thành ra file CSV
    # def export_score_by_city(self, city: str, file_path: str) -> None:
    #     """Xuất dữ liệu điểm theo tỉnh thành ra file CSV."""
    #     analysis = Analysis(self.Export)
    #     comparison = analysis.get_statistics_by_region(city)
    #     comparison.to_csv(file_path)
        
    # # Xuất dữ liệu phân tích điểm của một môn học cụ thể
    # def export_score_analysis(self, subject: str, file_path: str) -> None:
    #     """Xuất dữ liệu phân tích điểm của một môn học cụ thể ra file CSV."""
    #     analysis = Analysis(self.Export)
    #     distribution = analysis.get_score_distribution(subject)
    #     distribution.to_csv(file_path, header=['Score Distribution'])
    # # Xuất dữ liệu phân tích điểm theo môn học.
    # def export_subject_analysis(self, subject: str, file_path: str) -> None:
    #     """Xuất dữ liệu phân tích điểm theo môn học ra file CSV."""
    #     analysis = Analysis(self.Export)
    #     stats = analysis.get_aggregate_by_exam_subsections(subject)
    #     stats.to_csv(file_path)
        
    # # Xuất dữ liệu phân tích điểm theo khối thi.
    # def export_block_analysis(self, block: str, file_path: str) -> None:
    #     """Xuất dữ liệu phân tích điểm theo khối thi ra file CSV."""
    #     analysis = Analysis(self.Export)
    #     stats = analysis.analyze_scores_by_exam_block(block)
    #     stats.to_csv(file_path)
        
    # # Xuất dữ liệu phân tích điểm theo tỉnh thành.  
    # def export_city_analysis(self, city: str, file_path: str) -> None:
    #     """Xuất dữ liệu phân tích điểm theo tỉnh thành ra file CSV."""
    #     analysis = Analysis(self.Export)
    #     stats = analysis.compare_by_region(city)
    #     stats.to_csv(file_path)
    
    # ==================== PUBLIC API METHODS: XUẤT DỮ LIỆU VỚI RETURN ====================
    # Xuất dữ liệu điểm theo khối thi ra DataFrame
    def export_score_by_block(self, block: str) -> pd.DataFrame:
        """Xuất dữ liệu điểm theo khối thi ra DataFrame."""
        return self._export_score_by_block(block)
    # Xuất dữ liệu điểm theo môn học ra DataFrame
    def export_score_by_subject(self, subject: str) -> pd.DataFrame:
        """Xuất dữ liệu điểm theo môn học ra DataFrame."""
        return self._export_score_by_subject(subject)
    # Xuất dữ liệu điểm theo tỉnh thành ra DataFrame
    def export_score_by_city(self, city: str) -> pd.DataFrame:
        """Xuất dữ liệu điểm theo tỉnh thành ra DataFrame."""
        return self._export_score_by_city(city)
    # Xuất dữ liệu phân tích điểm của một môn học cụ thể ra DataFrame
    def export_score_analysis(self, subject: str) -> pd.DataFrame:
        """Xuất dữ liệu phân tích điểm của một môn học cụ thể ra DataFrame."""
        return self._export_score_analysis(subject)
    # Xuất dữ liệu phân tích điểm theo môn học ra DataFrame
    def export_subject_analysis(self, subject: str) -> pd.DataFrame:
        """Xuất dữ liệu phân tích điểm theo môn học ra DataFrame."""
        return self._export_subject_analysis(subject)
    # Xuất dữ liệu phân tích điểm theo khối thi ra DataFrame
    def export_block_analysis(self, block: str) -> pd.DataFrame:
        """Xuất dữ liệu phân tích điểm theo khối thi ra DataFrame."""
        return self._export_block_analysis(block)
    # Xuất dữ liệu phân tích điểm theo tỉnh thành ra DataFrame
    def export_city_analysis(self, city: str) -> pd.DataFrame:
        """Xuất dữ liệu phân tích điểm theo tỉnh thành ra DataFrame."""
        return self._export_city_analysis(city)
