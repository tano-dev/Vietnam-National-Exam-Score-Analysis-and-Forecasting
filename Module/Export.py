from pathlib import Path
import unicodedata
import pandas as pd
import numpy as np
import os

from Module.Processor_Data import DataProcessor
from Module.Analysis import Analysis


# ================== MAP TỈNH CŨ (TRƯỚC KHI GỘP) ==================
# Sử dụng 2 ký tự đầu của SBD để suy ra tỉnh ban đầu
_PRE_REGION_MAP = {
    "Hà Nội": ["01"],
    "Thành phố Hồ Chí Minh": ["02"],
    "Hải Phòng": ["03"],
    "Đà Nẵng": ["04"],
    "Hà Giang": ["05"],
    "Cao Bằng": ["06"],
    "Lai Châu": ["07"],
    "Lào Cai": ["08"],
    "Tuyên Quang": ["09"],
    "Lạng Sơn": ["10"],
    "Bắc Kạn": ["11"],
    "Thái Nguyên": ["12"],
    "Yên Bái": ["13"],
    "Sơn La": ["14"],
    "Phú Thọ": ["15"],
    "Vĩnh Phúc": ["16"],
    "Quảng Ninh": ["17"],
    "Bắc Giang": ["18"],
    "Bắc Ninh": ["19"],
    "Hải Dương": ["21"],
    "Hưng Yên": ["22"],
    "Hoà Bình": ["23"],
    "Hà Nam": ["24"],
    "Nam Định": ["25"],
    "Thái Bình": ["26"],
    "Ninh Bình": ["27"],
    "Thanh Hoá": ["28"],
    "Nghệ An": ["29"],
    "Hà Tĩnh": ["30"],
    "Quảng Bình": ["31"],
    "Quảng Trị": ["32"],
    "Huế": ["33"],
    "Quảng Nam": ["34"],
    "Quảng Ngãi": ["35"],
    "Kon Tum": ["36"],
    "Bình Định": ["37"],
    "Gia Lai": ["38"],
    "Phú Yên": ["39"],
    "Đắk Lắk": ["40"],
    "Khánh Hoà": ["41"],
    "Lâm Đồng": ["42"],
    "Bình Phước": ["43"],
    "Bình Dương": ["44"],
    "Ninh Thuận": ["45"],
    "Tây Ninh": ["46"],
    "Bình Thuận": ["47"],
    "Đồng Nai": ["48"],
    "Long An": ["49"],
    "Đồng Tháp": ["50"],
    "An Giang": ["51"],
    "Vũng Tàu": ["52"],
    "Tiền Giang": ["53"],
    "Kiên Giang": ["54"],
    "Cần Thơ": ["55"],
    "Bến Tre": ["56"],
    "Vĩnh Long": ["57"],
    "Trà Vinh": ["58"],
    "Sóc Trăng": ["59"],
    "Bạc Liêu": ["60"],
    "Cà Mau": ["61"],
    "Điện Biên": ["62"],
    "Đăk Nông": ["63"],
    "Hậu Giang": ["64"],
}
# mã tỉnh (01–64) -> tên tỉnh cũ
_CODE_TO_OLD_REGION = {
    code: name for name, codes in _PRE_REGION_MAP.items() for code in codes
}


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
        """Lấy danh sách tỉnh/thành **cũ** thực tế có trong dữ liệu.

        Dựa trực tiếp vào 2 ký tự đầu SBD (mã tỉnh 01–64) và map `_PRE_REGION_MAP`.
        """
        df_dist = self._build_old_province_distribution()
        return sorted(df_dist["tinh"].unique().tolist())

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

    # ---------- Helper riêng cho TỈNH CŨ ----------
    def _build_old_province_distribution(self) -> pd.DataFrame:
        """Tạo phân phối tổng điểm theo **tỉnh cũ** và theo từng năm.

        Output:
            DataFrame với các cột:
                ['nam_hoc', 'tinh', 'tong_diem', 'so_hoc_sinh']
        """
        df = self.processor.get_processed_data().copy()

        if "sbd" not in df.columns:
            raise ValueError("Thiếu cột 'sbd' trong dữ liệu nguồn.")
        if "nam_hoc" not in df.columns:
            raise ValueError("Thiếu cột 'nam_hoc' trong dữ liệu nguồn.")

        # Lấy mã tỉnh từ 2 ký tự đầu SBD, map sang tên tỉnh cũ
        df["sbd"] = df["sbd"].astype(str).str.zfill(8)
        df["ma_tinh"] = df["sbd"].str[:2]
        df["tinh"] = df["ma_tinh"].map(_CODE_TO_OLD_REGION)
        df = df.dropna(subset=["tinh"])

        # Danh sách môn dùng để tính tổng điểm
        mon_hoc = [
            "toan", "ngu_van", "ngoai_ngu",
            "vat_li", "hoa_hoc", "sinh_hoc",
            "lich_su", "dia_li", "gdcd",
            "cn_cong_nghiep", "cn_nong_nghiep",
        ]
        score_cols = [c for c in mon_hoc if c in df.columns]

        if not score_cols:
            raise ValueError(
                "Không tìm thấy cột điểm nào để tính 'tong_diem' theo tỉnh."
            )

        # Giữ thí sinh có ít nhất 1 môn có điểm
        df = df[df[score_cols].notna().any(axis=1)]

        # Tổng điểm của tất cả môn (dùng cho phân tích theo tỉnh)
        df["tong_diem"] = df[score_cols].sum(axis=1, skipna=True)

        counts = (
            df.groupby(["nam_hoc", "tinh", "tong_diem"])
              .size()
              .reset_index(name="so_hoc_sinh")
              .sort_values(["nam_hoc", "tinh", "tong_diem"])
              .reset_index(drop=True)
        )

        return counts

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
        df = (
            pd.DataFrame(stats_dict)
            .T.reset_index()
            .rename(columns={"index": "nam_hoc"})
        )
        df.to_csv(self._build_path("subject", subject), index=False)

    def _export_block(self, block: str) -> None:
        """Xuất CSV thống kê mô tả theo KHỐI THI."""
        stats_dict = self.analysis.get_statistics_by_block(block)
        df = (
            pd.DataFrame(stats_dict)
            .T.reset_index()
            .rename(columns={"index": "nam_hoc"})
        )
        df.to_csv(self._build_path("block", block), index=False)

    def _export_province(self, province: str) -> None:
        """Xuất CSV thống kê mô tả theo TỈNH/THÀNH (tỉnh **cũ**).

        Tính thống kê trực tiếp từ phân phối
        `_build_old_province_distribution()` để không bị ảnh hưởng
        bởi map gộp tỉnh trong `Analysis.compare_by_region`.
        """
        dist = self._build_old_province_distribution()

        if province not in dist["tinh"].unique():
            raise ValueError(f"Tỉnh '{province}' không tồn tại trong dữ liệu (tỉnh cũ).")

        rows = []
        for year, df_year in dist[dist["tinh"] == province].groupby("nam_hoc"):
            scores = df_year.loc[
                df_year.index.repeat(df_year["so_hoc_sinh"]),
                "tong_diem",
            ]

            if scores.empty:
                continue

            mean = float(scores.mean())
            median = float(scores.median())
            mode_series = scores.mode()
            mode = float(mode_series.iloc[0]) if not mode_series.empty else None
            std = float(scores.std())
            min_val = float(scores.min())
            max_val = float(scores.max())

            rows.append(
                {
                    "nam_hoc": year,
                    "mean": mean,
                    "median": median,
                    "mode": mode,
                    "std": std,
                    "min": min_val,
                    "max": max_val,
                }
            )

        df_stats = pd.DataFrame(rows).sort_values("nam_hoc")
        df_stats.to_csv(self._build_path("province", province), index=False)

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

    # Xuất tổng học sinh tham gia theo năm:
    def _export_yearly_total_students(self) -> None:
        """
        Tính và lưu tổng số học sinh theo từng năm từ processed data gốc.

        File output:
            <root_path>/Export_Yearly_Total_Students.csv

        Cấu trúc:
            nam_hoc, total_students
            2023, ...
            2024, ...
            2025, ...
        """
        df = self.processor.get_processed_data()

        if "nam_hoc" not in df.columns:
            raise ValueError("Thiếu cột 'nam_hoc' trong dữ liệu nguồn.")
        if "sbd" not in df.columns:
            raise ValueError("Thiếu cột 'sbd' trong dữ liệu nguồn.")

        # Mỗi SBD được coi là một học sinh trong 1 năm
        yearly = (
            df.groupby("nam_hoc")["sbd"]
              .nunique()  # phòng trường hợp sbd trùng
              .reset_index(name="total_students")
              .sort_values("nam_hoc")
        )

        out_path = Path(self._root_path) / "Export_Yearly_Total_Students.csv"
        yearly.to_csv(out_path, index=False)
        print(f"[EXPORT] Đã lưu tổng số học sinh theo năm tại: {out_path}")
    
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

    # ==================== PUBLIC API: EXPORT TOÀN BỘ ====================
    def run_export_all(self) -> None:
        """
        Chạy full export:
        - Theo MÔN học: distribution + statistics
        - Theo KHỐI thi: distribution + statistics
        - Theo TỈNH CŨ: distribution + statistics

        Tất cả được lưu dưới thư mục root_path với cấu trúc:
            root_path/
                Subject_Data/CleanData_<mon>/Export_Analysis_*.csv
                Subject_Data/CleanData_<mon>/Export_Distribution_*.csv

                Block_Data/CleanData_<khoi>/Export_Analysis_*.csv
                Block_Data/CleanData_<khoi>/Export_Distribution_*.csv

                Province_Data/CleanData_<tinh_cu>/Export_Analysis_*.csv
                Province_Data/CleanData_<tinh_cu>/Export_Distribution_*.csv
        """
        # -------- 1. EXPORT THEO MÔN HỌC --------
        subjects = self._detect_subjects()
        for subj in subjects:
            # Distribution theo môn
            df_dist = self.analysis.get_arregate_by_exam_subsections(subj)
            dist_path = self._build_distribution_path("subject", subj)
            df_dist.to_csv(dist_path, index=False)

            # Statistics theo môn
            self._export_subject(subj)

        # -------- 2. EXPORT THEO KHỐI THI --------
        blocks = self._detect_blocks()
        for blk in blocks:
            # Distribution theo khối
            df_block = self.analysis.analyze_scores_by_exam_block(blk)
            if df_block is None or df_block.empty:
                continue
            dist_path = self._build_distribution_path("block", blk)
            df_block.to_csv(dist_path, index=False)

            # Statistics theo khối
            self._export_block(blk)

        # -------- 3. EXPORT THEO TỈNH CŨ --------
        # Lấy full distribution theo tỉnh cũ
        df_prov_all = self._build_old_province_distribution()
        provinces = sorted(df_prov_all["tinh"].unique().tolist())

        for prov in provinces:
            df_prov = df_prov_all[df_prov_all["tinh"] == prov].copy()
            if df_prov.empty:
                continue

            # Distribution theo tỉnh cũ
            dist_path = self._build_distribution_path("province", prov)
            df_prov.to_csv(dist_path, index=False)

            # Statistics theo tỉnh cũ
            self._export_province(prov)

        # -------- 4. EXPORT TỔNG SỐ HỌC SINH THEO NĂM --------
        self._export_yearly_total_students()
# ==================== END OF MODULE ====================