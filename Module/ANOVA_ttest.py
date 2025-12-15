import pandas as pd
import numpy as np
from scipy import stats

class ANOVA_ttest:
    """Lớp để thực hiện phân tích điểm thi sử dụng 2 kiểm định ANOVA và T-test.

    Mô tả
    -----
    - Nhận dữ liệu từ các tệp .csv trong phần Clean_Data_2023-2025
      (thường là dạng Distribution đã đọc thành DataFrame).
    - Sử dụng kiểm định ANOVA và T-test để phân tích sự khác biệt điểm thi:
        * Giữa các năm cho một môn học / một khối.
        * Giữa các tỉnh trong một năm.
    - Kết quả trả về dưới dạng dict để dễ dàng theo dõi / log / chuyển thành DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        Dữ liệu điểm thi (thường là Distribution) đã đọc thành DataFrame.
        Cần có tối thiểu các cột:
            - group_col (mặc định 'nam_hoc'): cột phân nhóm (năm học).
            - score_col (ví dụ 'diem' hoặc 'tong_diem'): cột điểm.
            - 'so_hoc_sinh' : tần suất (số thí sinh đạt mức điểm đó).
        Tùy trường hợp có thêm:
            - 'mon_hoc'  : nếu phân tích nhiều môn trong cùng DataFrame.
            - 'khoi'     : nếu phân tích nhiều khối trong cùng DataFrame.
            - 'tinh'     : nếu phân tích nhiều tỉnh trong cùng DataFrame.
    group_col : str, default "nam_hoc"
        Tên cột dùng để phân nhóm (thường là năm học).
    score_col : str, default "tong_diem"
        Tên cột điểm số dùng để phân tích.
        - Với môn: thường sẽ truyền 'diem'.
        - Với khối/tỉnh: thường dùng 'tong_diem'.
    """

    __slots__ = (
        "_data",        # DataFrame dữ liệu distribution
        "_group_col",   # cột nhóm (thường là 'nam_hoc')
        "_score_col",   # cột điểm ('diem' hoặc 'tong_diem')
    )

    # -------- Khởi tạo và thiết lập thuộc tính --------
    def __init__(self, data: pd.DataFrame,
                 group_col: str = "nam_hoc",
                 score_col: str = "tong_diem"):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data phải là một pandas DataFrame.")
        self._data = data
        self._group_col = group_col
        self._score_col = score_col

    # ------------------------ Getter / Setter ------------------------
    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @data.setter
    def data(self, value: pd.DataFrame) -> None:
        if not isinstance(value, pd.DataFrame):
            raise TypeError("data phải là một pandas DataFrame.")
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

    # ------------------------ P-value helper ------------------------
    @staticmethod
    def _clip_p_value(p: float, min_p: float = 1e-16):
        """
        Chuẩn hóa p-value để:
        - Không bao giờ đúng bằng 0.0 (underflow).
        - Có thêm chuỗi p_value_text để báo cáo đẹp hơn.

        Returns
        -------
        (p_num, p_text)
            p_num  : giá trị số (>= min_p)
            p_text : chuỗi mô tả, ví dụ "< 1e-16", "1.23e-05", "0.0123"
        """
        if p <= 0 or np.isnan(p):
            return float(min_p), f"< {min_p:.0e}"
        if p < min_p:
            return float(min_p), f"< {min_p:.0e}"
        # p trong (min_p, 1)
        if p < 1e-3:
            return float(p), f"{p:.3e}"
        return float(p), f"{p:.4f}"

    # =====================================================================
    # ------------------------- T-TEST ------------------------------------
    # =====================================================================

    # 1) T-test môn học giữa 2 năm
    def _t_test_subject_two_years(
        self,
        subject: str,
        year1: int,
        year2: int,
        one_tail: bool = False,
        alternative: str = "auto",
    ) -> dict:
        """T-test điểm môn học giữa 2 năm.

        Dữ liệu self._data có thể:
        - Đã được lọc sẵn theo 1 môn (không có cột 'mon_hoc').
        - Hoặc chứa nhiều môn (có cột 'mon_hoc'), khi đó sẽ lọc theo `subject`.

        Trả về kết quả dưới dạng dict gồm: môn học, năm thứ 1, năm thứ 2, giá trị thống kê t,
        p-value, loại kiểm định, hướng kiểm dịnh, giá trị Cohen's d, mức độ khác biệt và kết luận.
        """
        # Lấy dữ liệu gốc
        df = self._data

        # Kiểm tra các cột bắt buộc có trong DataFrame không
        required = [self._group_col, self._score_col, "so_hoc_sinh"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Thiếu cột '{col}' trong DataFrame.")

        # Nếu có cột 'mon_hoc' → lọc theo tên môn; ngược lại, coi như đã lọc sẵn 1 môn
        if "mon_hoc" in df.columns:
            df_sub = df[df["mon_hoc"] == subject]   # Lọc môn học
            if df_sub.empty:
                raise ValueError(f"Môn '{subject}' không tồn tại trong dữ liệu.")
        else:
            df_sub = df

        # Lấy dữ liệu điểm thi cho 2 năm và mở rộng theo số học sinh
        g1 = np.repeat(
            df_sub[df_sub[self._group_col] == year1][self._score_col].values,
            df_sub[df_sub[self._group_col] == year1]["so_hoc_sinh"].values.astype(int),
        ).astype(float)
        g2 = np.repeat(
            df_sub[df_sub[self._group_col] == year2][self._score_col].values,
            df_sub[df_sub[self._group_col] == year2]["so_hoc_sinh"].values.astype(int),
        ).astype(float)

        # Kiểm tra môn đó có dữ liệu trong 2 năm không
        if g1.size == 0 or g2.size == 0:
            raise ValueError("Một trong hai năm không có dữ liệu cho môn này.")

        # Tính t-test hai mẫu độc lập (Welch)
        t_stat, p_two_tailed = stats.ttest_ind(g1, g2, equal_var=False)

        # Xác định alternative dùng để xác định kiểm định theo 2 phía hay 1 phía
        if alternative == "auto":
            # Giả định điểm năm sau thấp hơn năm trước nếu year2 > year1
            alternative = "less" if year2 > year1 else "greater"

        # Tính p-value theo 1 phía nếu muốn kiểm định 1 phía
        if one_tail:
            # Nếu giả thuyết: năm sau thấp hơn năm trước
            if alternative == "less":
                p_val_raw = p_two_tailed / 2 if t_stat < 0 else 1 - p_two_tailed / 2
            # Nếu giả thuyết: năm sau cao hơn năm trước
            elif alternative == "greater":
                p_val_raw = p_two_tailed / 2 if t_stat > 0 else 1 - p_two_tailed / 2
            # Báo lỗi nếu alternative không hợp lệ
            else:
                raise ValueError("alternative phải là 'less', 'greater' hoặc 'auto'.")
        # Nếu kiểm định 2 phía
        else:
            p_val_raw = p_two_tailed

        # Chuẩn hóa p-value (tránh 0 tuyệt đối, thêm p_value_text)
        p_val, p_text = self._clip_p_value(p_val_raw)

        # Tính effect size (Cohen's d)
        mean_diff = g1.mean() - g2.mean()  # Hiệu số trung bình giữa 2 năm
        pooled_sd = np.sqrt((g1.var(ddof=1) + g2.var(ddof=1)) / 2)  # Độ lệch chuẩn gộp
        cohens_d = mean_diff / pooled_sd if pooled_sd > 0 else 0.0  # Cohen's d

        # Xác định mức độ khác biệt dựa trên |Cohen's d|
        ad = abs(cohens_d)
        if ad < 0.2:
            strength = "Rất yếu"
        elif ad < 0.5:
            strength = "Yếu"
        elif ad < 0.8:
            strength = "Vừa"
        else:
            strength = "Mạnh"

        # Trả về kết quả dưới dạng dict
        return {
            "subject": subject,           # Tên môn học
            "year1": year1,              # Năm thứ nhất
            "year2": year2,              # Năm thứ hai
            "t_stat": float(t_stat),     # Giá trị thống kê t
            "p_value": p_val,            # p-value đã clip
            "p_value_text": p_text,      # Chuỗi mô tả p
            "one_tail": one_tail,        # Kiểm định 1 phía hay 2 phía
            "alternative": alternative,  # Hướng kiểm định: 'less' hoặc 'greater'
            "cohens_d": float(cohens_d), # Giá trị Cohen's d
            "effect_strength": strength, # Mức độ khác biệt theo Cohen's d
            "interpretation": (          # Kết luận
                "Có sự khác biệt."
                if p_val < 0.05 else
                "Không có sự khác biệt thống kê."
            ),
        }

    # 2) T-test khối giữa 2 năm
    def _t_test_block_two_years(
        self,
        block: str,
        year1: int,
        year2: int,
        one_tail: bool = False,
        alternative: str = "auto",
    ) -> dict:
        """T-test điểm khối thi giữa 2 năm.

        - Nếu DataFrame có cột 'khoi' → lọc theo block.
        - Nếu không có cột 'khoi' → coi như đã là dữ liệu của 1 khối.

        Trả về kết quả dưới dạng dict gồm: khối thi, năm thứ 1, năm thứ 2, giá trị thống kê t,
        p-value, loại kiểm định, hướng kiểm dịnh, giá trị Cohen's d, mức độ khác biệt và kết luận.
        """
        # Lấy dữ liệu gốc
        df = self._data

        # Kiểm tra các cột bắt buộc có trong DataFrame không
        required = [self._group_col, self._score_col, "so_hoc_sinh"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Thiếu cột '{col}' trong DataFrame.")

        # Lọc theo khối nếu có cột 'khoi', nếu không coi như df đã là dữ liệu 1 khối
        if "khoi" in df.columns:
            df_blk = df[df["khoi"] == block]
            if df_blk.empty:
                raise ValueError(f"Khối '{block}' không tồn tại trong dữ liệu.")
        else:
            df_blk = df

        # Lấy dữ liệu điểm thi cho 2 năm và mở rộng theo số học sinh
        g1 = np.repeat(
            df_blk[df_blk[self._group_col] == year1][self._score_col].values,
            df_blk[df_blk[self._group_col] == year1]["so_hoc_sinh"].values.astype(int),
        ).astype(float)

        g2 = np.repeat(
            df_blk[df_blk[self._group_col] == year2][self._score_col].values,
            df_blk[df_blk[self._group_col] == year2]["so_hoc_sinh"].values.astype(int),
        ).astype(float)

        # Kiểm tra khối đó có dữ liệu trong 2 năm không
        if g1.size == 0 or g2.size == 0:
            raise ValueError("Một trong hai năm không có dữ liệu cho khối này.")

        # Tính t-test hai mẫu độc lập (Welch)
        t_stat, p_two_tailed = stats.ttest_ind(g1, g2, equal_var=False)

        # Xác định alternative dùng để xác định kiểm định theo 2 phía hay 1 phía
        if alternative == "auto":
            alternative = "less" if year2 > year1 else "greater"  # Giả định năm sau thấp hơn

        # Tính p-value theo 1 phía nếu muốn tính 1 phía
        if one_tail:
            # Nếu giả thuyết: năm sau thấp hơn năm trước
            if alternative == "less":
                p_val_raw = p_two_tailed / 2 if t_stat < 0 else 1 - p_two_tailed / 2
            # Nếu giả thuyết: năm sau cao hơn năm trước
            elif alternative == "greater":
                p_val_raw = p_two_tailed / 2 if t_stat > 0 else 1 - p_two_tailed / 2
            # Báo lỗi nếu alternative không hợp lệ
            else:
                raise ValueError("alternative must be 'less', 'greater', or 'auto'.")
        # Nếu kiểm định 2 phía
        else:
            p_val_raw = p_two_tailed

        # Chuẩn hóa p-value
        p_val, p_text = self._clip_p_value(p_val_raw)

        # Tính effect size (Cohen's d)
        mean_diff = g1.mean() - g2.mean()  # Hiệu số trung bình giữa 2 năm
        pooled_sd = np.sqrt((g1.var(ddof=1) + g2.var(ddof=1)) / 2)  # Độ lệch chuẩn gộp
        cohens_d = mean_diff / pooled_sd if pooled_sd > 0 else 0.0  # Cohen's d

        # Xác định mức độ khác biệt dựa trên |Cohen's d|
        ad = abs(cohens_d)
        if ad < 0.2:
            strength = "Rất yếu"
        elif ad < 0.5:
            strength = "Yếu"
        elif ad < 0.8:
            strength = "Vừa"
        else:
            strength = "Mạnh"

        # Trả về kết quả dưới dạng dict    
        return {
            "block": block,               # Tên khối thi
            "year1": year1,               # Năm thứ nhất
            "year2": year2,               # Năm thứ hai
            "t_stat": float(t_stat),      # Giá trị thống kê t
            "p_value": p_val,             # p-value đã clip
            "p_value_text": p_text,       # Chuỗi mô tả p
            "one_tail": one_tail,         # Kiểm định 1 phía hay 2 phía
            "alternative": alternative,   # Hướng kiểm định
            "cohens_d": float(cohens_d),  # Giá trị Cohen's d
            "effect_strength": strength,  # Mức độ khác biệt
            "interpretation": (           # Kết luận
                "Có sự khác biệt."
                if p_val < 0.05 else
                "Không có sự khác biệt thống kê."
            ),
        }
                                   
    # 3) T-test 2 tỉnh trong 1 năm
    def _t_test_two_provinces(
        self,
        year: int,
        province1: str,
        province2: str,
        one_tail: bool = False,
        alternative: str = "auto",
    ) -> dict:
        """T-test điểm giữa 2 tỉnh trong 1 năm.

        Trả về kết quả dưới dạng dict gồm: năm, tỉnh thứ 1, tỉnh thứ 2, giá trị thống kê t,
        p-value, loại kiểm định, hướng kiểm dịnh, giá trị Cohen's d, mức độ khác biệt và kết luận.
        """
        # Lấy dữ liệu gốc
        df = self._data

        # Kiểm tra các cột bắt buộc
        required = [self._group_col, "tinh", self._score_col, "so_hoc_sinh"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Thiếu cột '{col}' trong DataFrame.")

        # Lọc theo năm
        df_year = df[df[self._group_col] == year]
        if df_year.empty:
            raise ValueError(f"Năm {year} không có dữ liệu.")

        # Lấy dữ liệu điểm cho 2 tỉnh và mở rộng theo số học sinh
        g1 = np.repeat(
            df_year[df_year["tinh"] == province1][self._score_col].values,
            df_year[df_year["tinh"] == province1]["so_hoc_sinh"].values.astype(int),
        ).astype(float)
        g2 = np.repeat(
            df_year[df_year["tinh"] == province2][self._score_col].values,
            df_year[df_year["tinh"] == province2]["so_hoc_sinh"].values.astype(int),
        ).astype(float)

        # Kiểm tra tỉnh đó có dữ liệu không
        if g1.size == 0 or g2.size == 0:
            raise ValueError("Một trong hai tỉnh không có dữ liệu trong năm này.")

        # Xác định alternative (hướng kiểm định) nếu để auto
        if alternative == "auto":
            alternative = "less"  # giả định province2 < province1 (có thể chỉnh tay)

        # Tính t-test
        t_stat, p_two_tailed = stats.ttest_ind(g1, g2, equal_var=False)

        # Tính p-value theo 1 phía nếu muốn kiểm định 1 phía
        if one_tail:
            # Nếu giả thuyết: tỉnh 2 thấp hơn tỉnh 1
            if alternative == "less":
                p_val_raw = p_two_tailed / 2 if t_stat < 0 else 1 - p_two_tailed / 2
            # Nếu giả thuyết: tỉnh 2 cao hơn tỉnh 1
            elif alternative == "greater":
                p_val_raw = p_two_tailed / 2 if t_stat > 0 else 1 - p_two_tailed / 2
            # Báo lỗi nếu alternative không hợp lệ
            else:
                raise ValueError("alternative must be 'less' or 'greater'.")
        # Nếu kiểm định 2 phía
        else:
            p_val_raw = p_two_tailed

        # Chuẩn hóa p-value
        p_val, p_text = self._clip_p_value(p_val_raw)

        # Tính effect size (Cohen's d)
        mean_diff = g1.mean() - g2.mean()  # Hiệu số trung bình giữa 2 tỉnh
        pooled_sd = np.sqrt((g1.var(ddof=1) + g2.var(ddof=1)) / 2)  # Độ lệch chuẩn gộp
        cohens_d = mean_diff / pooled_sd if pooled_sd > 0 else 0.0  # Cohen's d

        # Xác định mức độ khác biệt
        ad = abs(cohens_d)
        if ad < 0.2:
            strength = "Rất yếu"
        elif ad < 0.5:
            strength = "Yếu"
        elif ad < 0.8:
            strength = "Vừa"
        else:
            strength = "Mạnh"

        # Trả về kết quả dưới dạng dict    
        return {
            "year": year,                 # Năm cần kiểm định
            "province1": province1,       # Tỉnh thứ nhất
            "province2": province2,       # Tỉnh thứ hai
            "t_stat": float(t_stat),      # Giá trị thống kê t
            "p_value": p_val,             # p-value đã clip
            "p_value_text": p_text,       # Chuỗi mô tả p
            "one_tail": one_tail,         # Kiểm định 1 phía hay 2 phía
            "alternative": alternative,   # Hướng kiểm định
            "cohens_d": float(cohens_d),  # Giá trị Cohen's d
            "effect_strength": strength,  # Mức độ khác biệt
            "interpretation": (           # Kết luận
                "Có sự khác biệt."
                if p_val < 0.05 else
                "Không có sự khác biệt thống kê."
            ),
        }

    # =====================================================================
    # ------------------------- ANOVA -------------------------------------
    # =====================================================================

    # ANOVA môn học
    def _anova_subject(self, subject: str) -> dict:
        """
        Thực hiện kiểm định ANOVA cho môn học

        Trả về dict gồm môn học, giá trị thống kê anova, giá trị p-value, kết luận
        """
        # Lấy dữ liệu gốc
        df = self._data

        # Kiểm tra cột bắt buộc
        required = [self._group_col, self._score_col, "so_hoc_sinh"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Thiếu cột '{col}' trong DataFrame.")

        # Nếu có 'mon_hoc' → lọc; nếu không, dùng toàn bộ (đã là 1 môn)
        if "mon_hoc" in df.columns:
            df_sub = df[df["mon_hoc"] == subject]  # Lọc môn học        
            if df_sub.empty:
                raise ValueError(f"Môn '{subject}' không tồn tại trong dữ liệu.")
        else:
            df_sub = df

        # Tạo nhóm lưu số điểm của học sinh theo năm học
        groups = []
        for _, g in df_sub.groupby(self._group_col):
            arr = np.repeat(
                g[self._score_col].astype(float),
                g["so_hoc_sinh"].astype(int)   # Mở rộng dữ liệu theo số học sinh
            )
            if arr.size > 0:
                groups.append(arr)

        # Nếu có ít hơn 2 năm thì không chạy được ANOVA
        if len(groups) < 2:
            raise ValueError("Không đủ nhóm để chạy ANOVA (>=2 năm).")

        # Tính ANOVA
        f_stat, p_raw = stats.f_oneway(*groups)
        anova_p, anova_p_text = self._clip_p_value(p_raw)

        # Trả về kết quả
        return {
            "subject": subject,           # Tên môn
            "anova_f": float(f_stat),     # Giá trị thống kê ANOVA
            "anova_p": anova_p,           # p-value đã clip
            "anova_p_text": anova_p_text, # Chuỗi mô tả p-value
            "interpretation": (           # Kết luận
                "Có khác biệt giữa các năm."
                if anova_p < 0.05 else
                "Không có khác biệt đáng kể."
            ),
        }

    # ANOVA khối
    def _anova_block(self, block: str) -> dict:
        """
        Thực hiện kiểm định ANOVA cho khối thi

        Trả về dict gồm khối, giá trị thống kê anova, giá trị p-value, kết luận
        """
        # Lấy dữ liệu gốc
        df = self._data

        # Kiểm tra cột bắt buộc
        required = [self._group_col, self._score_col, "so_hoc_sinh"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Thiếu cột '{col}' trong DataFrame.")

        # Lọc theo khối nếu có cột 'khoi'
        if "khoi" in df.columns:
            df_blk = df[df["khoi"] == block]
            if df_blk.empty:
                raise ValueError(f"Khối '{block}' không tồn tại trong dữ liệu.")
        else:
            df_blk = df

        # Gom điểm theo năm cho khối đó
        groups = []
        for _, g in df_blk.groupby(self._group_col):
            arr = np.repeat(
                g[self._score_col].astype(float),
                g["so_hoc_sinh"].astype(int)   # Mở rộng theo số học sinh
            )
            if arr.size > 0:
                groups.append(arr)

        # Nếu có ít hơn 2 năm
        if len(groups) < 2:
            raise ValueError("Không đủ nhóm để chạy ANOVA (>=2 năm).")

        # Tính ANOVA
        f_stat, p_raw = stats.f_oneway(*groups)
        anova_p, anova_p_text = self._clip_p_value(p_raw)

        # Trả về kết quả
        return {
            "block": block,               # Tên khối
            "anova_f": float(f_stat),     # Giá trị thống kê ANOVA
            "anova_p": anova_p,           # p-value đã clip
            "anova_p_text": anova_p_text, # Chuỗi mô tả p-value
            "interpretation": (           # Kết luận
                "Có khác biệt giữa các năm."
                if anova_p < 0.05 else
                "Không có khác biệt đáng kể."
            ),
        }

    # ANOVA tỉnh
    def _anova_province_one_year(self, year: int) -> dict:
        """ANOVA giữa >= 3 tỉnh trong 1 năm.

        DataFrame dạng tối thiểu:
            group_col | tinh | score_col | so_hoc_sinh

        Trả về dict gồm năm, số tỉnh, giá trị thống kê anova, giá trị p-value, kết luận
        """
        # Lấy dữ liệu gốc
        df = self._data

        # Kiểm tra các cột bắt buộc
        required = [self._group_col, "tinh", self._score_col, "so_hoc_sinh"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Thiếu cột '{col}' trong DataFrame.")

        # Lọc dữ liệu theo năm
        df_year = df[df[self._group_col] == year]
        if df_year.empty:
            raise ValueError(f"Không có dữ liệu cho năm {year}.")

        # Lấy danh sách các tỉnh trong năm đó
        provinces = df_year["tinh"].unique()

        # BẮT BUỘC phải có ≥ 3 tỉnh để chạy ANOVA
        if len(provinces) < 3:
            raise ValueError("ANOVA yêu cầu >= 3 tỉnh trong 1 năm. Dùng t-test cho 2 tỉnh.")

        # Tạo nhóm lưu số điểm của học sinh theo từng tỉnh
        groups = []
        for p in provinces:
            df_p = df_year[df_year["tinh"] == p]
            g = np.repeat(
                df_p[self._score_col].astype(float),
                df_p["so_hoc_sinh"].astype(int)   # Mở rộng theo số học sinh
            )
            if g.size > 0:
                groups.append(g)

        # Kiểm tra đủ 3 nhóm dữ liệu có điểm
        if len(groups) < 3:
            raise ValueError("Không đủ dữ liệu để chạy ANOVA (cần >= 3 tỉnh có điểm).")

        # Tính ANOVA
        F, p_raw = stats.f_oneway(*groups)
        p_val, p_text = self._clip_p_value(p_raw)

        # Trả về kết quả
        return {
            "year": year,                    # Năm cần kiểm định
            "num_provinces": len(groups),    # Số tỉnh tham gia kiểm định
            "F_statistic": float(F),         # Giá trị thống kê ANOVA
            "p_value": p_val,                # p-value đã clip
            "p_value_text": p_text,          # Chuỗi mô tả p-value
            "interpretation": (              # Kết luận
                "Có khác biệt điểm giữa các tỉnh."
                if p_val < 0.05 else
                "Không có khác biệt thống kê giữa các tỉnh."
            ),
        }

    # ======================== PUBLIC METHODS =========================
    def get_data(self) -> pd.DataFrame:
        """Lấy dữ liệu điểm thi hiện tại."""
        return self._data

    def anova_subject(self, subject: str) -> dict:
        """Chạy ANOVA cho môn học."""
        return self._anova_subject(subject)

    def anova_block(self, block: str) -> dict:
        """Chạy ANOVA cho khối thi."""
        return self._anova_block(block)

    def anova_province(self, year: int) -> dict:
        """Chạy ANOVA giữa các tỉnh trong 1 năm."""
        return self._anova_province_one_year(year)

    def t_test_subject_two_years(
        self,
        subject: str,
        year1: int,
        year2: int,
        one_tail: bool = False,
        alternative: str = "auto",
    ) -> dict:
        """Chạy t-test cho môn học giữa 2 năm."""
        return self._t_test_subject_two_years(subject, year1, year2, one_tail, alternative)

    def t_test_block_two_years(
        self,
        block: str,
        year1: int,
        year2: int,
        one_tail: bool = False,
        alternative: str = "auto",
    ) -> dict:
        """Chạy t-test cho khối thi giữa 2 năm."""
        return self._t_test_block_two_years(block, year1, year2, one_tail, alternative)

    def t_test_two_provinces(
        self,
        year: int,
        province1: str,
        province2: str,
        one_tail: bool = False,
        alternative: str = "auto",
    ) -> dict:
        """Chạy t-test cho 2 tỉnh/thành trong 1 năm."""
        return self._t_test_two_provinces(year, province1, province2, one_tail, alternative)
