from Module.Processor_Data import Processor_Data
import pandas as pd
import numpy as np

class Analysis:
    # =================== INTERNAL PRIVATE METHODS: PHÂN TÍCH DỮ LIỆU ===================
    # ----------------------- Khai báo và thiết lập thuộc tính -------------------------
    """"Lớp để phân tích dữ liệu đã được xử lý. Thiết lập nhóm thuộc tính qua setter với những yêu cầu cụ thể:
    - Theo môn học, lấy dữ liệu đã xử lý từ DataProcessor.
    - Theo khối thi, lấy dữ liệu đã xử lý từ DataProcessor.
    - Theo tỉnh thành, lấy dữ liệu đã xử lý từ DataProcessor.
    
    Xây dựng các phương thức để lấy dữ liệu thống kê stats:
    - Phân phối điểm theo môn học.
    - Thống kê điểm theo khối thi.
    - So sánh điểm theo tỉnh thành.
    
    Attributes (public API):
        processor (Processor_Data): Đối tượng DataProcessor để lấy dữ liệu đã xử lý.
    
    """
    # Slots: Cố định các thuộc tính có thể sử dụng
    __slots__ = (
        "_processor",          # Đối tượng DataProcessor để lấy dữ liệu đã xử lý
    )
    
    # ------------------------ Setter và Getter -------------------------
    @property
    def processor(self) -> Processor_Data:
        """Đối tượng DataProcessor để lấy dữ liệu đã xử lý."""
        return self._processor
    
    @processor.setter
    def processor(self, value: Processor_Data) -> None:
        self._processor = value
    
    # -------- Khởi tạo và thiết lập thuộc tính --------
    def __init__(self, processor: Processor_Data):
        self.processor = processor
    
    # ----------------------------- Internal Methods -----------------------------
    # Phân tích phân phối điểm của một môn học cụ thể
    def _analyze_score_distribution(self, subject: str) -> pd.Series:
        """Phân tích phân phối điểm của một môn học cụ thể."""
        df = self.processor.get_processed_data()
        if subject not in df.columns:
            raise ValueError(f"Môn học '{subject}' không tồn tại trong dữ liệu.")
        return df[subject].value_counts().sort_index()
    
    # ===== CÁC HÀM NHÓM PHÂN TÍCH DỮ LIỆU 
    # #Phân tích thống kê điểm theo môn học.
    # Requirements: Trả về DataFrame thống kê điểm theo môn học
    # def _aggregate_by_exam_subsections(self, block: str) -> pd.DataFrame:
    # Your Code start here
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Your Code end here 
    
    
    # # Phân tích điểm theo khối thi cụ thể
    # Requiremnts: Trả về DataFrame phân tích điểm theo khối thi sau khi được nhóm. Có xây dựng map khối thi -> cột điểm
    # def _analyze_scores_by_exam_block(self, block: str) -> pd.Data
    # Your Code start here
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Your Code end here
    
    # # Phân tích so sánh điểm theo tỉnh thành.
    # Requirements: Trả về DataFrame so sánh điểm theo tỉnh thành. Xây dựng map tỉnh thành ( sau đợt chuyển đổi, dùng cho dự báo 2026) -> cột điểm
    # def _compare_by_region(self, region: str) -> pd.DataFrame:
    # Your Code start here
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Your Code end here
    
    
    # ===== CÁC HÀM THỐNG KÊ DỮ LIỆU    
    # # def _get_statistics_by_subject(self, subject: str) -> dict:
    # # Requirements: Trả về dict thống kê điểm theo môn học( khối thi , tỉnh thành ), gồm: mean, median, mode, std, min, max
    # #     """Lấy thống kê điểm theo môn học."""
    # Your Code start here
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Your Code end here
    
    
    # ======================== PUBLIC METHODS: PHÂN TÍCH DỮ LIỆU =========================
    # ----------------------- Các hàm phân tích dữ liệu -------------------------
    def get_score_distribution(self, subject: str) -> pd.Series:
        """Lấy phân phối điểm của một môn học cụ thể."""
        return self._analyze_score_distribution(subject)