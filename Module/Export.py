from Module.Processor_Data import DataProcessor
from Module.Analysis import Analysis

import pandas as pd
import numpy as np

class Export:
    # =================== INTERNAL PRIVATE METHODS: XUẤT DỮ LIỆU ===================
    # ----------------------- Khai báo và thiết lập thuộc tính -------------------------
    """"Lớp để xuất dữ liệu đã được phân tích. Thiết lập nhóm thuộc tính qua setter với những yêu cầu cụ thể:
    - Theo môn học, xuất dữ liệu phân tích từ Processor_Data & Analysis.
    - Theo khối thi, xuất dữ liệu phân tích từ Processor_Data & Analysis.
    - Theo tỉnh thành, xuất dữ liệu phân tích từ Processor_Data & Analysis.
    
    Trả về các Data Frame theo:
    - Xuất phân phối điểm theo môn học.
    - Xuất thống kê điểm theo khối thi.
    - Xuất so sánh điểm theo tỉnh thành.
    - Các thông số thống kê khác.
    
    Tất cả lưu trong Data: Clean_Data_2023-2025 floder
    Tên file: Export_Analysis_<subject/block/city>_YYYY.csv
    """
    
    # Slots: Cố định các thuộc tính có thể sử dụng
    __slots__ = (
        "_Export",          # Đối tượng Export (DataProcessor) để lấy dữ liệu đã phân tích
    )
    
    # ------------------------ Setter và Getter -------------------------
    # Export Getter và Setter
    @property
    def Export(self) -> DataProcessor:
        """Đối tượng Export để lấy dữ liệu đã phân tích (instance của DataProcessor)."""
        return self._Export
    
    @Export.setter
    def Export(self, value: DataProcessor) -> None:
        if not isinstance(value, DataProcessor):
            raise TypeError("Export phải là một instance của DataProcessor.")
        self._Export = value
        
    # -------- Khởi tạo và thiết lập thuộc tính --------
    def __init__(self, Export: DataProcessor):
        """Khởi tạo lớp Export với một đối tượng DataProcessor đã xử lý dữ liệu."""
        self.Export = Export
    
    # ----------------------------- Internal Methods -----------------------------
    # Ở dưới, mình xây dựng các hàm nội bộ (_build_*) để chuẩn bị DataFrame,
    # các hàm public chỉ việc gọi và ghi ra file.
    
    def _build_score_by_block(self, block: str) -> pd.DataFrame:
        """
        Xây dựng DataFrame thống kê điểm theo khối thi từ Analysis.
        Dữ liệu gốc: dict {nam_hoc: {mean, median, mode, std, ...}}
        Trả về: DataFrame có cột 'nam_hoc' + các thống kê.
        """
        analysis = Analysis(self.Export)
        stats_dict = analysis.get_statistics_by_block(block)  # dict
        
        if not stats_dict:
            return pd.DataFrame()
        
        df = pd.DataFrame(stats_dict).T.reset_index().rename(columns={"index": "nam_hoc"})
        return df
    
    def _build_score_by_subject(self, subject: str) -> pd.DataFrame:
        """
        Xây dựng DataFrame thống kê điểm theo môn học từ Analysis.
        Dữ liệu gốc: dict {nam_hoc: {mean, median, mode, std, ...}}
        Trả về: DataFrame có cột 'nam_hoc' + các thống kê.
        """
        analysis = Analysis(self.Export)
        stats_dict = analysis.get_statistics_by_subject(subject)
        
        if not stats_dict:
            return pd.DataFrame()
        
        df = pd.DataFrame(stats_dict).T.reset_index().rename(columns={"index": "nam_hoc"})
        return df
    
    def _build_score_by_city(self, city: str) -> pd.DataFrame:
        """
        Xây dựng DataFrame thống kê điểm theo tỉnh/thành từ Analysis.
        Dữ liệu gốc: dict {nam_hoc: {mean, median, mode, std, ...}}
        Trả về: DataFrame có cột 'nam_hoc' + các thống kê.
        """
        analysis = Analysis(self.Export)
        stats_dict = analysis.get_statistics_by_region(city)
        
        if not stats_dict:
            return pd.DataFrame()
        
        df = pd.DataFrame(stats_dict).T.reset_index().rename(columns={"index": "nam_hoc"})
        return df
    
    # ==================== PUBLIC API METHODS: XUẤT DỮ LIỆU ====================
    # Xuất dữ liệu điểm theo khối thi ra file CSV
    def export_score_by_block(self, block: str, file_path: str) -> None:
        """Xuất dữ liệu điểm theo khối thi ra file CSV."""
        df = self._build_score_by_block(block)
        df.to_csv(file_path, index=False)
        
    # Xuất dữ liệu điểm theo môn học ra file CSV
    def export_score_by_subject(self, subject: str, file_path: str) -> None:
        """Xuất dữ liệu điểm theo môn học ra file CSV."""
        df = self._build_score_by_subject(subject)
        df.to_csv(file_path, index=False)
    
    # Xuất dữ liệu điểm theo tỉnh thành ra file CSV
    def export_score_by_city(self, city: str, file_path: str) -> None:
        """Xuất dữ liệu điểm theo tỉnh thành ra file CSV."""
        df = self._build_score_by_city(city)
        df.to_csv(file_path, index=False)
        
    # Xuất dữ liệu phân tích điểm của một môn học cụ thể (phân phối điểm)
    def export_score_analysis(self, subject: str, file_path: str) -> None:
        """Xuất dữ liệu phân tích điểm của một môn học cụ thể ra file CSV."""
        analysis = Analysis(self.Export)
        distribution = analysis.get_score_distribution(subject)  # Series
        distribution.to_csv(file_path, header=['Score Distribution'])
    
    # Xuất dữ liệu phân tích điểm của một môn học cụ thể
    def _export_score_analysis(self, subject: str) -> pd.DataFrame:
        """Xuất dữ liệu phân tích điểm của một môn học cụ thể."""
        analysis = Analysis(self.Export)
        distribution = analysis.get_score_distribution(subject)
        return distribution
    # Xuất dữ liệu phân tích điểm theo môn học.
    def _export_subject_analysis(self, subject: str) -> pd.DataFrame:
        """Xuất dữ liệu phân tích điểm theo môn học."""
        analysis = Analysis(self.Export)
        stats = analysis.get_arregate_by_exam_subsections(subject)  # DataFrame
        stats.to_csv(file_path, index=False)
        
        stats = analysis.get_aggregate_by_exam_subsections(subject)
        return stats
    
    # Xuất dữ liệu phân tích điểm theo khối thi.
    def _export_block_analysis(self, block: str) -> pd.DataFrame:
        """Xuất dữ liệu phân tích điểm theo khối thi."""
        analysis = Analysis(self.Export)
        stats = analysis.analyze_scores_by_exam_block(block)  # DataFrame
        stats.to_csv(file_path, index=False)
        
    # Xuất dữ liệu phân tích điểm theo tỉnh thành.  
    def export_city_analysis(self, city: str, file_path: str) -> None:
        """Xuất dữ liệu phân tích điểm theo tỉnh thành ra file CSV."""
        analysis = Analysis(self.Export)
        stats = analysis.compare_by_region(city)  # DataFrame
        stats.to_csv(file_path, index=False)
        stats = analysis.analyze_scores_by_exam_block(block)
        return stats
    
    # Xuất dữ liệu phân tích điểm theo tỉnh thành.
    def _export_city_analysis(self, city: str) -> pd.DataFrame:
        """Xuất dữ liệu phân tích điểm theo tỉnh thành."""
        analysis = Analysis(self.Export)
        stats = analysis.compare_by_region(city)
        return stats
    
    # Xuất dữ liệu điểm theo khối thi
    def _export_score_by_block(self, block: str) -> pd.DataFrame:
        """Xuất dữ liệu điểm theo khối thi."""
        analysis = Analysis(self.Export)
        stats = analysis.get_statistics_by_block(block)
        return stats
    
    # Xuất dữ liệu điểm theo môn học
    def _export_score_by_subject(self, subject: str) -> pd.DataFrame:
        """Xuất dữ liệu điểm theo môn học."""
        analysis = Analysis(self.Export)
        distribution = analysis.get_statistics_by_subject(subject)
        return distribution
    
    # Xuất dữ liệu điểm theo tỉnh thành
    def _export_score_by_city(self, city: str) -> pd.DataFrame:
        """Xuất dữ liệu điểm theo tỉnh thành."""
        analysis = Analysis(self.Export)
        comparison = analysis.get_statistics_by_region(city)
        return comparison
    
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
    
