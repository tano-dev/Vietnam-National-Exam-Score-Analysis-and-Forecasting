from Module.Processor_Data import DataProcessor
import pandas as pd
import numpy as np

class Analysis:
    # Slots: Cố định các thuộc tính
    __slots__ = (
        "_processor",
        "_subject",
        "_block",
        "_region",
    )
    
    # ------------------------ Setter và Getter -------------------------
    @property
    def processor(self) -> DataProcessor: return self._processor
    @processor.setter
    def processor(self, value: DataProcessor) -> None: self._processor = value
    
    @property
    def subject(self) -> str: return self._subject
    @subject.setter
    def subject(self, value: str): self._subject = value

    @property
    def block(self) -> str: return self._block
    @block.setter
    def block(self, value: str): self._block = value

    @property
    def region(self) -> str: return self._region
    @region.setter
    def region(self, value: str): self._region = value
       
    # --- KHỞI TẠO ---
    def __init__(self, processor: DataProcessor):
        self.processor = processor
        self._subject = None
        self._block = None
        self._region = None
        
    # ----------------------------- Optimized Methods -----------------------------

    def _analyze_score_distribution(self, subject: str) -> pd.Series:
        df = self.processor.get_processed_data()
        if subject not in df.columns:
            return pd.Series()
        return df[subject].value_counts().sort_index()
    
    def _aggregate_by_exam_subsections(self, subject: str) -> pd.DataFrame:
        df = self.processor.get_processed_data()
        score_columns = df.select_dtypes(include='number').columns.difference(['sbd', 'nam_hoc'])
        
        target_cols = [subject] if subject != "All" else score_columns
        if subject != "All" and subject not in df.columns:
             return pd.DataFrame()

        results = []
        for col in target_cols:
            if col not in df.columns: continue
            temp_df = df.groupby(['nam_hoc', col]).size().reset_index(name='so_hoc_sinh')
            temp_df.rename(columns={col: 'diem'}, inplace=True)
            temp_df['mon_hoc'] = col
            results.append(temp_df)

        if not results:
            return pd.DataFrame(columns=['nam_hoc', 'mon_hoc', 'diem', 'so_hoc_sinh'])
            
        return pd.concat(results, ignore_index=True)[['nam_hoc', 'mon_hoc', 'diem', 'so_hoc_sinh']]

    # --- ĐÃ CẬP NHẬT: Thêm đầy đủ mã khối A00, B00... ---
    def _analyze_scores_by_exam_block(self, block: str) -> pd.DataFrame:
        df = self.processor.get_processed_data()
        
        # Map đầy đủ các khối thi phổ biến
        block_subjects_map = {
            # Khối A
            'A': ['toan', 'vat_li', 'hoa_hoc'],
            'A00': ['toan', 'vat_li', 'hoa_hoc'],
            'A01': ['toan', 'vat_li', 'ngoai_ngu'],
            # Khối B
            'B': ['toan', 'hoa_hoc', 'sinh_hoc'],
            'B00': ['toan', 'hoa_hoc', 'sinh_hoc'],
            # Khối C
            'C': ['ngu_van', 'lich_su', 'dia_li'],
            'C00': ['ngu_van', 'lich_su', 'dia_li'],
            # Khối D
            'D': ['toan', 'ngu_van', 'ngoai_ngu'],
            'D01': ['toan', 'ngu_van', 'ngoai_ngu'],
            # Khối Năng khiếu / Khác (nếu cần)
            'H00': ['ngu_van', 've_nt', 've_mt'], # Ví dụ
        }
        
        blocks_to_run = block_subjects_map.keys() if block == "All" else [block]
        results = []

        for b in blocks_to_run:
            subjects = block_subjects_map.get(b, [])
            if not subjects: continue
            
            # Kiểm tra các môn có tồn tại trong dữ liệu không
            valid_cols = [s for s in subjects if s in df.columns]
            
            # Nếu thiếu môn (ví dụ năm đó không thi môn Vẽ) -> Bỏ qua
            if len(valid_cols) < len(subjects): continue 

            # Lọc ra các thí sinh thi ĐỦ các môn trong khối (không bị NaN môn nào)
            # dropna rất quan trọng để tính tổng điểm chính xác
            df_b = df[['nam_hoc'] + valid_cols].dropna().copy()
            
            if df_b.empty: continue

            # Tính tổng điểm
            df_b['tong_diem'] = df_b[valid_cols].sum(axis=1)
            
            # Groupby đếm số lượng
            dist = df_b.groupby(['nam_hoc', 'tong_diem']).size().reset_index(name='so_hoc_sinh')
            dist['khoi'] = b
            results.append(dist)

        if not results:
            return pd.DataFrame(columns=['khoi', 'nam_hoc', 'tong_diem', 'so_hoc_sinh'])

        return pd.concat(results, ignore_index=True)

    def _compare_by_region(self, region: str) -> pd.DataFrame:
        df = self.processor.get_processed_data().copy()
        
        # Map mã tỉnh từ SBD
        if 'tinh' not in df.columns and 'sbd' in df.columns:
             df['ma_tinh'] = df['sbd'].astype(str).str.zfill(8).str[:2]
             df['tinh'] = df['ma_tinh'] 

        if region != "ALL":
            df = df[df['tinh'] == region]

        score_cols = df.select_dtypes(include='number').columns.difference(['nam_hoc'])
        # Tính tổng điểm các môn thí sinh đó thi
        df['tong_diem'] = df[score_cols].sum(axis=1)
        
        return df.groupby(['nam_hoc', 'tinh', 'tong_diem']).size().reset_index(name='so_hoc_sinh')

    # --- CÁC HÀM THỐNG KÊ ---
    def _get_statistics_by_subject(self, subject: str) -> dict:
        df_agg = self._aggregate_by_exam_subsections(subject)
        stats_dict = {}
        for year, grp in df_agg.groupby("nam_hoc"):
            values = np.repeat(grp['diem'].values, grp['so_hoc_sinh'].values)
            if len(values) == 0: continue
            stats_dict[year] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "mode": float(grp.loc[grp['so_hoc_sinh'].idxmax(), 'diem']),
                "std": float(np.std(values)),
                "min": float(values.min()),
                "max": float(values.max())
            }
        return stats_dict

    # --- PUBLIC API ---
    def get_score_distribution(self, subject: str) -> pd.Series:
        return self._analyze_score_distribution(subject)
    def get_arregate_by_exam_subsections(self, subject: str) -> pd.DataFrame:
        return self._aggregate_by_exam_subsections(subject)
    def analyze_scores_by_exam_block(self, block: str) -> pd.DataFrame:
        return self._analyze_scores_by_exam_block(block)
    def compare_by_region(self, region: str) -> pd.DataFrame:
        return self._compare_by_region(region)
    def get_statistics_by_subject(self, subject: str) -> dict:
        return self._get_statistics_by_subject(subject)
    def get_statistics_by_block(self, block: str) -> dict:
        return {} 
    def get_statistics_by_region(self, region: str) -> dict:
        return {}