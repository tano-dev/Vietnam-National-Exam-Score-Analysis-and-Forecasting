from Module.Processor_Data import Processor_Data
from PythonProject.Module import Analysis

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
        "_Export",          # Đối tượng Export để lấy dữ liệu đã phân tích
    )
    
    # ------------------------ Setter và Getter -------------------------
    # Export Getter và Setter
    @property
    def Export(self) -> Export:
        """Đối tượng Export để lấy dữ liệu đã phân tích."""
        return self._Export
    
    @Export.setter
    def Export(self, value: Export) -> None:
        self._Export = value
        
    # -------- Khởi tạo và thiết lập thuộc tính --------
    def __init__(self, Export: Export):
        self.Export = Export
    
    # ----------------------------- Internal Methods -----------------------------
    # Xuất dữ liệu phân tích điểm của một môn học cụ thể
    def _export_score_Export(self, subject: str, file_path: str) -> None:
        """Xuất dữ liệu phân tích điểm của một môn học cụ thể ra file CSV."""
        distribution = self.Export._analyze_score_distribution(subject)
        distribution.to_csv(file_path, header=['Score Distribution'])
        
    # ==================== PUBLIC API METHODS: XUẤT DỮ LIỆU ====================
    # Xuất dữ liệu điểm theo khối thi ra file CSV
    def export_score_by_block(self, block: str, file_path: str) -> None:
        """Xuất dữ liệu điểm theo khối thi ra file CSV."""
        analysis = Analysis(self.Export)
        stats = analysis._get_statistics_by_block(block)
        stats.to_csv(file_path)
        
    # Xuất dữ liệu điểm theo môn học ra file CSV
    def export_score_by_subject(self, subject: str, file_path: str) -> None:
        """Xuất dữ liệu điểm theo môn học ra file CSV."""
        analysis = Analysis(self.Export)
        distribution = analysis._get_statistics_by_subject(subject)
        distribution.to_csv(file_path, header=['Score Distribution'])
    # Xuất dữ liệu điểm theo tỉnh thành ra file CSV
    def export_score_by_city(self, city: str, file_path: str) -> None:
        """Xuất dữ liệu điểm theo tỉnh thành ra file CSV."""
        analysis = Analysis(self.Export)
        comparison = analysis._get_statistics_by_region(city)
        comparison.to_csv(file_path)
        
    # X
