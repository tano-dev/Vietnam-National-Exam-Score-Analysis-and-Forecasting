import pandas as pd
import numpy as np
from scipy import stats

class ANOVA_ttest:
    """ Lớp để thực hiện phân tích điểm thi sử dụng 2 kiểm dịnh là ANOVA và T-test.
    
        Mô tả:
            - Lấy dữ liệu từ các tệp .csv trong phần Clean_Data_2023-2025 (cần chuyển về daaframe trước khi truyền vào).
            - Sử dụng kiểm định ANOVA và T-test để phân tích sự khác biệt điểm thi giữa các môn/khối giữa các năm
                hoặc giữa các tỉnh theo năm.
            - Kết quả trả về dưới dạng dict để dể dàng theo dõi.

        Attributes (public API):
            data (pd.DataFrame): Dữ liệu điểm thi được lấy từ Clean_Data_2023-2025 đã chuyển về DataFrame.
            group_col (str): Tên cột dùng để phân nhóm dữ liệu (mặc dịnh là "nam_hoc").
            score_col (str): Tên cột điểm số dùng để phân tích (mặc định là "tong_diem" nếu muốn thành điểm thì truyền "diem").
    """
    __slots__ = ("_data", # Dữ liệu điểm thi được lấy từ Clean_Data_2023-2025 đã chuyển về DataFrame.
                 "_group_col", # Tên cột dùng để phân nhóm dữ liệu (mặc dịnh là "nam_hoc").
                 "_score_col" # Tên cột điểm số dùng để phân tích (mặc định là "tong_diem" nếu muốn thành điểm thì truyền "diem"). 
                 ) 

    # -------- Khởi tạo và thiết lập thuộc tính --------
    def __init__(self, data: pd.DataFrame,
                 group_col: str = "nam_hoc",
                 score_col: str = "tong_diem"):
        self._data = data 
        self._group_col = group_col
        self._score_col = score_col

    
    # ------------------------
    # Getter / Setter
    # ------------------------
    @property
    def data(self) ->pd.DataFrame:
        return self._data

    @data.setter
    def data(self, value: pd.DataFrame) -> None:
        if not isinstance(value, pd.DataFrame):
            raise TypeError("Data phải là 1 pandas DataFrame.")
        self._data = value

    @property
    def group_col(self) -> str:
        return self._group_col

    @group_col.setter
    def group_col(self, value: str) -> None:
        if value not in self._data.columns: 
            raise ValueError(f"Column '{value}' không tồn tại trong DataFrame.")
        self._group_col = value

    @property
    def score_col(self) -> str:
        return self._score_col

    @score_col.setter
    def score_col(self, value: str) -> None:
        if value not in self._data.columns: 
            raise ValueError(f"Column '{value}' không tồn tại trong DataFrame.")
        self._score_col = value
    
    # ----------------------------- Internal Methods -----------------------------
    # =====================================================================
    # ------------------------- T-TEST ------------------------------------
    # =====================================================================

    # 1) T-test môn học giữa 2 năm
    def _t_test_subject_two_years(self, subject: str, year1: int, year2: int,
                                 one_tail: bool = False,
                                 alternative: str = "auto") -> dict:
        df = self.get_data()
        if "mon_hoc" not in df.columns or "diem" not in df.columns:
            raise ValueError("DataFrame không phải dữ liệu môn học (thiếu cột 'mon_hoc' hoặc 'diem').")

        df_sub = df[df["mon_hoc"] == subject]
        if df_sub.empty:
            raise ValueError(f"Môn '{subject}' không tồn tại trong dữ liệu.")

        g1 = np.repeat(df_sub[df_sub["nam_hoc"] == year1]["diem"], 
                       df_sub[df_sub["nam_hoc"] == year1]["so_hoc_sinh"]).astype(float)
        g2 = np.repeat(df_sub[df_sub["nam_hoc"] == year2]["diem"], 
                       df_sub[df_sub["nam_hoc"] == year2]["so_hoc_sinh"]).astype(float)

        if g1.size == 0 or g2.size == 0:
            raise ValueError("Một trong hai năm không có dữ liệu cho môn này.")

        t_stat, p_two_tailed = stats.ttest_ind(g1, g2, equal_var=False)

        if alternative == "auto":
            alternative = "less" if year2 > year1 else "greater"

        if one_tail:
            if alternative == "less":
                p_val = p_two_tailed / 2 if t_stat < 0 else 1 - p_two_tailed / 2
            elif alternative == "greater":
                p_val = p_two_tailed / 2 if t_stat > 0 else 1 - p_two_tailed / 2
            else:
                raise ValueError("alternative must be 'less', 'greater', or 'auto'.")
        else:
            p_val = p_two_tailed

        mean_diff = g1.mean() - g2.mean()
        pooled_sd = np.sqrt((g1.var(ddof=1) + g2.var(ddof=1)) / 2)
        cohens_d = mean_diff / pooled_sd if pooled_sd > 0 else 0

        if abs(cohens_d) < 0.2:
            strength = "Rất yếu"
        elif abs(cohens_d) < 0.5:
            strength = "Yếu"
        elif abs(cohens_d) < 0.8:
            strength = "Vừa"
        else:
            strength = "Mạnh"

        return {
            "subject": subject,
            "year1": year1,
            "year2": year2,
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "one_tail": one_tail,
            "alternative": alternative,
            "cohens_d": float(cohens_d),
            "effect_strength": strength,
            "interpretation": "Có sự khác biệt." if p_val < 0.05 else "Không có sự khác biệt thống kê."
        }

    # 2) T-test khối giữa 2 năm
    def _t_test_block_two_years(self, block: str, year1: int, year2: int,
                               one_tail: bool = False,
                               alternative: str = "auto") -> dict:
        df = self.get_data()
        if "khoi" not in df.columns or self._score_col not in df.columns:
            raise ValueError("DataFrame không phải dữ liệu khối (thiếu cột 'khoi' hoặc cột điểm).")

        df_blk = df[df["khoi"] == block]
        if df_blk.empty:
            raise ValueError(f"Khối '{block}' không tồn tại trong dữ liệu.")

        g1 = np.repeat(df_blk[df_blk["nam_hoc"] == year1][self._score_col],
                       df_blk[df_blk["nam_hoc"] == year1]["so_hoc_sinh"]).astype(float)
        g2 = np.repeat(df_blk[df_blk["nam_hoc"] == year2][self._score_col],
                       df_blk[df_blk["nam_hoc"] == year2]["so_hoc_sinh"]).astype(float)

        if g1.size == 0 or g2.size == 0:
            raise ValueError("Một trong hai năm không có dữ liệu của khối.")

        t_stat, p_two_tailed = stats.ttest_ind(g1, g2, equal_var=False)

        if alternative == "auto":
            alternative = "less" if year2 > year1 else "greater"

        if one_tail:
            if alternative == "less":
                p_val = p_two_tailed / 2 if t_stat < 0 else 1 - p_two_tailed / 2
            elif alternative == "greater":
                p_val = p_two_tailed / 2 if t_stat > 0 else 1 - p_two_tailed / 2
            else:
                raise ValueError("alternative must be 'less', 'greater', or 'auto'.")
        else:
            p_val = p_two_tailed

        mean_diff = g1.mean() - g2.mean()
        pooled_sd = np.sqrt((g1.var(ddof=1) + g2.var(ddof=1)) / 2)
        cohens_d = mean_diff / pooled_sd if pooled_sd > 0 else 0

        if abs(cohens_d) < 0.2:
            strength = "Rất yếu"
        elif abs(cohens_d) < 0.5:
            strength = "Yếu"
        elif abs(cohens_d) < 0.8:
            strength = "Vừa"
        else:
            strength = "Mạnh"

        return {
            "block": block,
            "year1": year1,
            "year2": year2,
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "one_tail": one_tail,
            "alternative": alternative,
            "cohens_d": float(cohens_d),
            "effect_strength": strength,
            "interpretation": "Có sự khác biệt." if p_val < 0.05 else "Không có sự khác biệt thống kê."
        }

    # 3) T-test 2 tỉnh trong 1 năm
    def _t_test_two_provinces(self, year: int, province1: str, province2: str,
                             one_tail: bool = False,
                             alternative: str = "auto") -> dict:
        df = self.get_data()
        if "tinh" not in df.columns or self._score_col not in df.columns:
            raise ValueError("DataFrame không phải dữ liệu tỉnh (thiếu cột 'tinh' hoặc cột điểm).")

        df_year = df[df["nam_hoc"] == year]
        if df_year.empty:
            raise ValueError(f"Năm {year} không có dữ liệu.")

        g1 = np.repeat(df_year[df_year["tinh"] == province1][self._score_col],
                       df_year[df_year["tinh"] == province1]["so_hoc_sinh"]).astype(float)
        g2 = np.repeat(df_year[df_year["tinh"] == province2][self._score_col],
                       df_year[df_year["tinh"] == province2]["so_hoc_sinh"]).astype(float)

        if g1.size == 0 or g2.size == 0:
            raise ValueError("Một trong hai tỉnh không có dữ liệu trong năm này.")

        if alternative == "auto":
            alternative = "less"  # giả định province2 < province1

        t_stat, p_two_tailed = stats.ttest_ind(g1, g2, equal_var=False)

        if one_tail:
            if alternative == "less":
                p_val = p_two_tailed / 2 if t_stat < 0 else 1 - p_two_tailed / 2
            elif alternative == "greater":
                p_val = p_two_tailed / 2 if t_stat > 0 else 1 - p_two_tailed / 2
            else:
                raise ValueError("alternative must be 'less' or 'greater'.")
        else:
            p_val = p_two_tailed

        mean_diff = g1.mean() - g2.mean()
        pooled_sd = np.sqrt((g1.var(ddof=1) + g2.var(ddof=1)) / 2)
        cohens_d = mean_diff / pooled_sd if pooled_sd > 0 else 0

        if abs(cohens_d) < 0.2:
            strength = "Rất yếu"
        elif abs(cohens_d) < 0.5:
            strength = "Yếu"
        elif abs(cohens_d) < 0.8:
            strength = "Vừa"
        else:
            strength = "Mạnh"

        return {
            "year": year,
            "province1": province1,
            "province2": province2,
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "one_tail": one_tail,
            "alternative": alternative,
            "cohens_d": float(cohens_d),
            "effect_strength": strength,
            "interpretation": "Có sự khác biệt." if p_val < 0.05 else "Không có sự khác biệt thống kê."
        }

    # =====================================================================
    # ------------------------- ANOVA -------------------------------------
    # =====================================================================

    # ANOVA môn học
    def _anova_subject(self, subject: str) -> dict:
        df = self.get_data()
        df_sub = df[df["mon_hoc"] == subject]
        if df_sub.empty:
            raise ValueError(f"Môn '{subject}' không tồn tại trong dữ liệu.")

        groups = [np.repeat(g["diem"], g["so_hoc_sinh"]).astype(float)
                  for _, g in df_sub.groupby("nam_hoc")]

        if len(groups) < 2:
            raise ValueError("Không đủ nhóm để chạy ANOVA (>=2 năm).")

        f_stat, p_val = stats.f_oneway(*groups)

        return {
            "subject": subject,
            "anova_f": float(f_stat),
            "anova_p": float(p_val),
            "interpretation": "Có khác biệt giữa các năm." if p_val < 0.05 else "Không có khác biệt đáng kể."
        }

    # ANOVA khối
    def _anova_block(self, block: str) -> dict:
        df = self.get_data()
        df_blk = df[df["khoi"] == block]
        if df_blk.empty:
            raise ValueError(f"Khối '{block}' không tồn tại trong dữ liệu.")

        groups = [np.repeat(g[self._score_col], g["so_hoc_sinh"]).astype(float)
                  for _, g in df_blk.groupby("nam_hoc")]

        if len(groups) < 2:
            raise ValueError("Không đủ nhóm để chạy ANOVA (>=2 năm).")

        f_stat, p_val = stats.f_oneway(*groups)

        return {
            "block": block,
            "anova_f": float(f_stat),
            "anova_p": float(p_val),
            "interpretation": "Có khác biệt giữa các năm." if p_val < 0.05 else "Không có khác biệt đáng kể."
        }

    # ANOVA tỉnh
    def _anova_province_one_year(self, year: int) -> dict:
        """
        ANOVA giữa >= 3 tỉnh trong 1 năm dựa trên self._data.
        DataFrame dạng: nam_hoc | tinh | tong_diem | so_hoc_sinh
        """

        df = self._data

        # ---- Kiểm tra cột ----
        required_cols = ["nam_hoc", "tinh", self._score_col, "so_hoc_sinh"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Thiếu cột '{col}' trong DataFrame.")

        # ---- Lọc 1 năm ----
        df_year = df[df["nam_hoc"] == year]
        if df_year.empty:
            raise ValueError(f"Không có dữ liệu cho năm {year}.")

        # ---- Gom theo tỉnh ----
        provinces = df_year["tinh"].unique()

        # BẮT BUỘC phải có ≥ 3 tỉnh
        if len(provinces) < 3:
            raise ValueError("ANOVA yêu cầu >= 3 tỉnh trong 1 năm. Dùng t-test cho 2 tỉnh.")

        groups = []

        for p in provinces:
            df_p = df_year[df_year["tinh"] == p]

            # Mở rộng dữ liệu theo số học sinh
            g = np.repeat(
                df_p[self._score_col].astype(float),
                df_p["so_hoc_sinh"].astype(int)
            )

            if len(g) > 0:
                groups.append(g)

        # ---- Kiểm tra đủ 3 nhóm dữ liệu ----
        if len(groups) < 3:
            raise ValueError("Không đủ dữ liệu để chạy ANOVA (cần >= 3 tỉnh có điểm).")

        # ---- ANOVA ----
        F, p = stats.f_oneway(*groups)

        return {
            "year": year,
            "num_provinces": len(groups),
            "F_statistic": float(F),
            "p_value": float(p),
            "interpretation": (
                "Có khác biệt điểm giữa các tỉnh."
                if p < 0.05 else
                "Không có khác biệt thống kê giữa các tỉnh."
            )
        }

    # ======================== PUBLIC METHODS=========================
    def get_data(self) -> pd.DataFrame:
        """Lấy dữ liệu điểm thi """
        return self._data
    
    def anova_subject(self, subject: str) -> dict:
        """Chạy ANOVA cho môn học."""
        return self._anova_subject(subject)

    def anova_block(self, block: str) -> dict:
        """Chạy ANOVA cho khối thi."""
        return self._anova_block(block)
    
    def anova_province(self, year: int) -> dict:
        """Chạy ANOVA cho tỉnh trong 1 năm."""
        return self._anova_province_one_year(year)
    
    def t_test_subject_two_years(self, subject: str, year1: int, year2: int,
                                 one_tail: bool = False,
                                 alternative: str = "auto") -> dict:
        """ Chạy t_test cho môn học giữa 2 năm."""
        return self._t_test_subject_two_years(subject, year1, year2, one_tail, alternative)

    def t_test_block_two_years(self, block: str, year1: int, year2: int,
                                 one_tail: bool = False,
                                 alternative: str = "auto") -> dict:
        """ Chạy t_test cho khối thi giữa 2 năm."""
        return self._t_test_block_two_years(block, year1, year2, one_tail, alternative)
    
    def t_test_two_provinces(self, year: int, province1: str, province2: str,
                             one_tail: bool = False,
                             alternative: str = "auto") -> dict:
        """ Chạy t_test cho  2 tỉnh thành trong 1 năm."""
        return self._t_test_two_provinces(year, province1, province2, one_tail, alternative)