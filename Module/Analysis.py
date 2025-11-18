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
    def _aggregate_by_exam_subsections(self, subject: str) -> pd.DataFrame:
        """
        Trả về DataFrame phân phối điểm theo môn học và theo từng năm.
        subject:
            - Tên môn cụ thể: 'toan', 'hoa_hoc', ...
            - 'All' → phân tích tất cả các môn.
        
        Output: ['nam_hoc', 'mon_hoc', 'diem', 'so_hoc_sinh']
        """
        df = self.processor.get_processed_data()

        # Các cột điểm (trừ sbd, nam_hoc)
        score_columns = df.select_dtypes(include='number').columns.difference(['sbd', 'nam_hoc'])

        records = []

        # Khi chỉ phân tích một môn
        if subject != "All":
            if subject not in score_columns:
                raise ValueError(f"Môn '{subject}' không tồn tại trong dữ liệu!")

            for year, df_year in df.groupby("nam_hoc"):
                counts = df_year[subject].value_counts().sort_index()
                for score, count in counts.items():
                    records.append({
                        "nam_hoc": year,
                        "mon_hoc": subject,
                        "diem": score,
                        "so_hoc_sinh": count
                    })

            return pd.DataFrame(records).sort_values(["nam_hoc", "mon_hoc", "diem"])

        # Nếu phân tích tất cả các môn
        for year, df_year in df.groupby("nam_hoc"):
            for col in score_columns:
                # Chỉ phân tích môn có thật trong năm đó
                if df_year[col].notna().sum() == 0:
                    continue

                counts = df_year[col].value_counts().sort_index()
                for score, count in counts.items():
                    records.append({
                        "nam_hoc": year,
                        "mon_hoc": col,
                        "diem": score,
                        "so_hoc_sinh": count
                    })

        result = pd.DataFrame(records)
        return result.sort_values(["nam_hoc", "mon_hoc", "diem"]).reset_index(drop=True)
    

    # Your Code end here 
    
    
    # # Phân tích điểm theo khối thi cụ thể
    # Requiremnts: Trả về DataFrame phân tích điểm theo khối thi sau khi được nhóm. Có xây dựng map khối thi -> cột điểm
    # def _analyze_scores_by_exam_block(self, block: str) -> pd.Data
    # Your Code start here
    def _analyze_scores_by_exam_block(self, block: str) -> pd.DataFrame:
        """
        Trả về phân phối tổng điểm theo khối thi và từng năm.
        block: 'A', 'B', 'C', 'D', 'Điểm gãy'.
        'All' -> phân tích tất cả các khối.
        Output: ['khoi', 'nam_hoc', 'tong_diem', 'so_hoc_sinh']
        """

        df = self.processor.get_processed_data()

        block_subjects_map = {
            'A': ['toan', 'vat_li', 'hoa_hoc'],
            'B': ['toan', 'hoa_hoc', 'sinh_hoc'],
            'C': ['ngu_van', 'lich_su', 'dia_li'],
            'D': ['toan', 'ngu_van', 'ngoai_ngu'],
            'Điểm gãy': ['tin_hoc', 'cn_cong_nghiep', 'cn_nong_nghiep', 'gdcd']
        }

        blocks_to_run = (
            block_subjects_map.keys() if block == "All" else [block]
        )

        results = []

        for b in blocks_to_run:
            subjects = block_subjects_map[b]
            valid_subjects = [s for s in subjects if s in df.columns]

            # Không có môn nào trong năm → bỏ qua
            if len(valid_subjects) == 0:
                continue

            df_b = df[['nam_hoc'] + valid_subjects].copy()
            df_b['so_mon_co_diem'] = df_b[valid_subjects].notna().sum(axis=1)

            # Chỉ nhận học sinh thi đủ môn
            df_b = df_b[df_b['so_mon_co_diem'] == len(valid_subjects)]

            if df_b.empty:
                continue

            df_b['tong_diem'] = df_b[valid_subjects].sum(axis=1)

            dist = (
                df_b.groupby(['nam_hoc', 'tong_diem'])
                    .size()
                    .reset_index(name='so_hoc_sinh')
                    .assign(khoi=b)
            )

            results.append(dist)

        if not results:
            return pd.DataFrame(columns=['khoi', 'nam_hoc', 'tong_diem', 'so_hoc_sinh'])

        out = pd.concat(results, ignore_index=True)
        return out[['khoi', 'nam_hoc', 'tong_diem', 'so_hoc_sinh']] \
                .sort_values(['khoi', 'nam_hoc', 'tong_diem'])
    
    

    # Your Code end here
    
    # # Phân tích so sánh điểm theo tỉnh thành.
    # Requirements: Trả về DataFrame so sánh điểm theo tỉnh thành. Xây dựng map tỉnh thành ( sau đợt chuyển đổi, dùng cho dự báo 2026) -> cột điểm
    # def _compare_by_region(self, region: str) -> pd.DataFrame:
    # Your Code start here
    def _compare_by_region(self, region: str) -> pd.DataFrame:
        """
        Phân phối tổng điểm theo tỉnh và theo từng năm.
        
        Parameters:
            region: Tên tỉnh cụ thể để phân tích hoặc "ALL" để phân tích tất cả tỉnh.
        
        Returns:
            DataFrame với cột ['nam_hoc', 'tinh', 'tong_diem', 'so_hoc_sinh']
        """
        df = self.processor.get_processed_data()

        # Map tỉnh theo nhiều mã sau đợt chuyển đổi
        region_map_raw = {
            "Hà Nội": ["01"],
            "Thành phố Hồ Chí Minh": ["02", "44", "52"],
            "Hải Phòng": ["03", "21"],
            "Đà Nẵng": ["04", "34"],
            "Huế": ["33"],
            "Cần Thơ": ["55", "59", "64"],
            "Tuyên Quang": ["05", "09"],
            "Cao Bằng": ["06"],
            "Lai Châu": ["07"],
            "Lào Cai": ["08", "13"],
            "Lạng Sơn": ["10"],
            "Thái Nguyên": ["11", "12"],
            "Sơn La": ["14"],
            "Phú Thọ": ["15", "16", "23"],
            "Bắc Ninh": ["18", "19"],
            "Quảng Ninh": ["17"],
            "Hưng Yên": ["22", "26"],
            "Ninh Bình": ["24", "25", "27"],
            "Điện Biên": ["62"],
            "Thanh Hóa": ["28"],
            "Nghệ An": ["29"],
            "Hà Tĩnh": ["30"],
            "Quảng Trị": ["31", "32"],
            "Quảng Ngãi": ["35", "36"],
            "Gia Lai": ["37", "38"],
            "Đắk Lắk": ["39", "40"],
            "Khánh Hòa": ["41", "45"],
            "Lâm Đồng": ["42", "47", "63"],
            "Đồng Nai": ["43", "48"],
            "Tây Ninh": ["46", "49"],
            "Đồng Tháp": ["50"],
            "Tiền Giang": ["53"],
            "An Giang": ["51", "54"],
            "Vĩnh Long": ["56", "57", "58"],
            "Cà Mau": ["60", "61"]
        }

        # Chuyển mã → tỉnh
        code_to_region = {code: region_name for region_name, codes in region_map_raw.items() for code in codes}

        # Chuẩn hóa SBD
        if 'sbd' not in df.columns:
            raise ValueError("Thiếu cột 'sbd'.")
        df['sbd'] = df['sbd'].astype(str).str.zfill(8)
        df['ma_tinh'] = df['sbd'].str[:2]
        df['tinh'] = df['ma_tinh'].map(code_to_region)
        df = df.dropna(subset=['tinh'])

        # Lọc theo tỉnh nếu user chỉ định
        if region != "ALL":
            if region not in region_map_raw:
                raise ValueError(f"Tỉnh '{region}' không hợp lệ.")
            df = df[df['tinh'] == region]

        # Danh sách môn học
        mon_hoc = [
            'toan', 'ngu_van', 'ngoai_ngu', 
            'vat_li', 'hoa_hoc', 'sinh_hoc',
            'lich_su', 'dia_li', 'gdcd',
            'cn_cong_nghiep', 'cn_nong_nghiep'
        ]
        score_cols = [c for c in mon_hoc if c in df.columns]

        # Xác định ai có ít nhất 1 môn có điểm
        df = df[df[score_cols].notna().any(axis=1)]

        # Tính tổng điểm
        df['tong_diem'] = df[score_cols].sum(axis=1, skipna=True)

        # === Phân phối điểm theo năm và tỉnh ===
        counts = (
            df.groupby(['nam_hoc', 'tinh', 'tong_diem'])
            .size()
            .reset_index(name='so_hoc_sinh')
            .sort_values(['nam_hoc', 'tinh', 'tong_diem'])
            .reset_index(drop=True)
        )

        return counts
    

# Your Code end here
    
    
    # ===== CÁC HÀM THỐNG KÊ DỮ LIỆU    
    # # def _get_statistics_by_subject(self, subject: str) -> dict:
    # # Requirements: Trả về dict thống kê điểm theo môn học( khối thi , tỉnh thành ), gồm: mean, median, mode, std, min, max
    # #     """Lấy thống kê điểm theo môn học."""
    # Your Code start here
    def _get_statistics_by_subject(self, subject: str) -> dict:
        """
        Trả về dict thống kê điểm theo môn học cho tất cả năm.
        key: nam_hoc, value: dict thống kê (mean, median, mode, std, min, max)
        """

        # Lấy DataFrame phân phối điểm
        df = self._aggregate_by_exam_subsections(subject)

        # Lọc chỉ môn cần phân tích
        if subject != "All":
            df = df[df["mon_hoc"] == subject]

        stats_dict = {}

        for year, df_year in df.groupby("nam_hoc"):
            # Mở rộng điểm theo số học sinh
            scores = df_year.loc[df_year.index.repeat(df_year["so_hoc_sinh"]), "diem"]

            if scores.empty:
                stats_dict[year] = None
                continue
            
            # Chuyển sang float cho dễ xem xét sau này
            mean = float(scores.mean())
            median = float(scores.median())
            mode = float(scores.mode().iloc[0]) if not scores.mode().empty else None
            std = float(scores.std())
            min_val = float(scores.min())
            max_val = float(scores.max())

            stats_dict[year] = {
                "mean": mean,
                "median": median,
                "mode": mode,
                "std": std,
                "min": min_val,
                "max": max_val
            }

        return stats_dict

    
    # Your Code end here
    
    # # def _get_statistics_by_block(self, block: str) -> dict:
    # # Requirements: Trả về dict thống kê điểm theo khối thi, gồm: mean, median, mode, std, min, max
    # #     """ Lấy thống kê điểm theo khối thi. """
    # Your code start here
    def _get_statistics_by_block(self, block: str) -> dict:
        """
        Trả về dict thống kê điểm theo khối thi cho tất cả năm.
        key: nam_hoc, value: dict thống kê (mean, median, mode, std, min, max)
        """

        # Lấy DataFrame phân phối tổng điểm theo khối
        df = self._analyze_scores_by_exam_block(block)

        stats_dict = {}

        for year, df_year in df.groupby("nam_hoc"):
            # Mở rộng tổng điểm theo số học sinh
            scores = df_year.loc[df_year.index.repeat(df_year["so_hoc_sinh"]), "tong_diem"]

            if scores.empty:
                stats_dict[year] = None
                continue

            mean = float(scores.mean())
            median = float(scores.median())
            mode_series = scores.mode()
            mode = float(mode_series.iloc[0]) if not mode_series.empty else None
            std = float(scores.std())
            min_val = float(scores.min())
            max_val = float(scores.max())

            stats_dict[year] = {
                "mean": mean,
                "median": median,
                "mode": mode,
                "std": std,
                "min": min_val,
                "max": max_val
            }

        return stats_dict

    
    # Your code end here

    
    # # def _get_statistics_by_region(self, region: str) -> dict:
    # # Requirements: Trả về dict thống kê điểm theo tỉnh thành, gồm: mean, median, mode, std, min, max
    # #     """ Lấy thống kê điểm theo tỉnh thành. """
    # Your code start here
    def _get_statistics_by_region(self, region: str) -> dict:
        """
        Trả về dict thống kê điểm cho một tỉnh.
        Kết quả: {nam_hoc: {mean, median, mode, std, min, max}}
        """
        # Lấy DataFrame phân phối tổng điểm theo tỉnh
        df = self._compare_by_region(region)

        stats_dict = {}

        for year, df_year in df.groupby("nam_hoc"):
            # Lấy dữ liệu của tỉnh đã chọn 
            df_prov = df_year[df_year["tinh"] == region]

            # Mở rộng điểm theo số học sinh
            scores = df_prov.loc[df_prov.index.repeat(df_prov["so_hoc_sinh"]), "tong_diem"]

            if scores.empty:
                stats_dict[year] = None
                continue

            mean = float(scores.mean())
            median = float(scores.median())
            mode_series = scores.mode()
            mode = float(mode_series.iloc[0]) if not mode_series.empty else None
            std = float(scores.std())
            min_val = float(scores.min())
            max_val = float(scores.max())

            stats_dict[year] = {
                "mean": mean,
                "median": median,
                "mode": mode,
                "std": std,
                "min": min_val,
                "max": max_val
            }

        return stats_dict
        
  
    # Your code end here

    
    # ======================== PUBLIC METHODS: PHÂN TÍCH DỮ LIỆU =========================
    # ----------------------- Các hàm phân tích dữ liệu -------------------------
    def get_score_distribution(self, subject: str) -> pd.Series:
        """Lấy phân phối điểm của một môn học cụ thể."""
        return self._analyze_score_distribution(subject)
        
    def get_arregate_by_exam_subsections(self, subject: str) -> pd.DataFrame:
        """Lấy dataframe thống kê điểm theo môn học."""
        return self._aggregate_by_exam_subsections(subject)
    
    def analyze_scores_by_exam_block(self, block: str) -> pd.DataFrame:
        """Lấy dataframe thống kê điểm theo khối."""
        return self._analyze_scores_by_exam_block(block)
    
    def compare_by_region(self, region: str) -> pd.DataFrame:
        """Lấy dataframe thống kê điểm theo tỉnh."""
        return self._compare_by_region(region)
    
    def get_statistics_by_subject(self, subject: str) ->dict:
        """Lấy dict thống kê điểm theo môn học, gồm: mean, median, mode, std, min, max"""
        return self._get_statistics_by_subject(subject)
    
    def get_statistics_by_block(self, block: str) ->dict:
        """Lấy dict thống kê điểm theo khối thi, gồm: mean, median, mode, std, min, max"""
        return self._get_statistics_by_block(block)    
    
    def get_statistics_by_region(self, region: str) ->dict:
        """Lấy dict thống kê điểm theo tỉnh thành, gồm: mean, median, mode, std, min, max"""
        return self._get_statistics_by_region(region)
