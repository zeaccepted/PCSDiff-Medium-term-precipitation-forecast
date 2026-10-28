import os
import glob
import netCDF4 as nc
import numpy as np
from datetime import datetime, timedelta
from torch.utils.data import Dataset
import torch
from scipy.interpolate import RegularGridInterpolator

# 导入配置文件
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config_server import truth_scale_factor
import config_server as config

# ---------------------- 路径配置 ----------------------
DATA_ROOT = config.data_config["data_path"]  # RVD_DATA_ROOT：预处理数据根目录，见 README.md
TRUTH_DIR = os.path.join(DATA_ROOT, "CRA_truth_nc")  # CRA 日降水真值
FORECAST_DIR = os.path.join(DATA_ROOT, "ecmwf_tp_daily_noensmean")  # ECMWF TP 集合预报（含所需成员）
# 预报时效：从config_server读取（默认10天，可通过环境变量RVD_FORECAST_LEAD_TIME设置）
try:
    import config_server as _config
    FORECAST_LEAD_TIME = getattr(_config, 'forecast_lead_time', 10)
except Exception:
    FORECAST_LEAD_TIME = 10

# ---------------------- 辅助函数：提取日期 ----------------------
def extract_date_from_filename(filename):
    """
    从文件名中提取日期字符串
    支持格式:
    - s2s_ecmf_20040101_tp_tp_daily.nc → "20040101"
    - CRA_truth_20040101.nc → "20040101"
    """
    import re
    match = re.search(r'(\d{8})', filename)
    if match:
        return match.group(1)
    return filename

# ---------------------- 辅助函数：获取经纬度变量 ----------------------
def get_lat_lon_vars(f):
    """从netCDF文件中获取经纬度变量名和数据"""
    lat_var_names = ["latitude", "lat", "LAT", "LATITUDE"]
    lon_var_names = ["longitude", "lon", "LON", "LONGITUDE"]
    
    lats = None
    lons = None
    lat_name = None
    lon_name = None
    
    for name in lat_var_names:
        if name in f.variables:
            lats = f.variables[name][:]
            lat_name = name
            break
    
    for name in lon_var_names:
        if name in f.variables:
            lons = f.variables[name][:]
            lon_name = name
            break
    
    return lats, lons, lat_name, lon_name


def get_var_units(var):
    """Return a normalized units string for a netCDF variable."""
    for attr in ("units", "unit"):
        if attr in var.ncattrs():
            return str(var.getncattr(attr)).strip().lower()
    return ""


def convert_accumulated_precip_to_mm(data, units="", log_prefix="", var_label="降水", truth_scale_factor=None):
    """Convert accumulated precipitation to millimeters.

    Accumulated precipitation in kg m-2 is numerically equal to millimeters of
    liquid-water depth, so it is returned unchanged.
    """
    units = (units or "").lower().replace(" ", "")
    data = np.asarray(data, dtype=np.float32)

    # 打印原始数据统计，便于调试
    if log_prefix:
        print(f"{log_prefix} {var_label}原始数据统计 - 最小值: {np.nanmin(data):.8f}, 最大值: {np.nanmax(data):.8f}, 均值: {np.nanmean(data):.8f}")

    data_max = np.nanmax(data)
    
    # 对于未知单位，根据数据范围判断
    if units in {"unknown", ""}:
        # 如果提供了真值缩放因子，使用配置的缩放因子
        if truth_scale_factor is not None and truth_scale_factor != 1.0:
            if log_prefix:
                print(f"{log_prefix} {var_label}单位为'{units}'，使用配置的缩放因子{truth_scale_factor}")
            return data * truth_scale_factor
        
        # 从数据范围来看，真值数据需要乘以约10000才能和forecast一个量级
        # 这可能是因为：
        # 1. 数据存储时使用了某种缩放因子
        # 2. 单位确实是某种微小的单位（如0.1 mm的千分之一）
        # 3. 数据是平均降水率，需要转换为累积值
        if data_max > 0 and data_max < 0.01:
            # 乘以 10000 使数据与forecast在同一个量级
            if log_prefix:
                print(f"{log_prefix} {var_label}单位为'{units}'，数据最大值{data_max:.8f}非常小，乘以10000转换为mm")
            return data * 10000.0
        # 如果数据最大值很小（< 0.1），很可能是米单位，转换为毫米
        elif data_max > 0 and data_max < 0.1:
            if log_prefix:
                print(f"{log_prefix} {var_label}单位为'{units}'，数据最大值{data_max:.8f}很小，按米单位处理，转换为mm")
            return data * 1000.0
        else:
            if log_prefix:
                print(f"{log_prefix} {var_label}单位为'{units}'，按kg/m²累积降水处理（数值等同于mm），不做单位缩放")
            return data
    
    if units in {"kgm-2", "kg/m2", "kgm**-2", "kgm^-2", "kg/m^2", "kgm−2", "kg m-2"}:
        if log_prefix:
            print(f"{log_prefix} {var_label}单位为kg/m²累积降水，数值等同于mm，不做单位缩放")
        return data

    if units in {"m", "meter", "meters"}:
        if log_prefix:
            print(f"{log_prefix} {var_label}单位为m，转换为mm")
        return data * 1000.0

    if units in {"mm", "millimeter", "millimeters"}:
        if log_prefix:
            print(f"{log_prefix} {var_label}单位为mm，不做单位缩放")
        return data

    if any(token in units for token in ("s-1", "s**-1", "s^-1", "s−1", "/s", "sec-1", "second-1")):
        raise ValueError(
            f"{var_label} variable appears to be a rate ({units}), but this dataset expects accumulated precipitation."
        )

    # 对于其他未知单位，按kg/m²处理
    if log_prefix:
        print(f"{log_prefix} 未识别{var_label}单位'{units}'，按kg/m²累积降水处理，不做单位缩放")
    return data

# ---------------------- 辅助函数：计算区域索引 ----------------------
def compute_region_indices(lats, lons, lat_range, lon_range):
    """计算给定经纬度范围内的索引（与test目录一致，使用精确范围裁剪）"""
    lats_arr = np.asarray(lats, dtype=np.float64)
    lons_arr = np.asarray(lons, dtype=np.float64)
    
    # 处理纬度（可能是从北到南或从南到北）
    if len(lats_arr) > 1 and lats_arr[1] < lats_arr[0]:
        # 纬度从北到南递减
        lat_indices = np.where((lats_arr <= lat_range[1]) & (lats_arr >= lat_range[0]))[0]
    else:
        # 纬度从南到北递增
        lat_indices = np.where((lats_arr >= lat_range[0]) & (lats_arr <= lat_range[1]))[0]
    
    # 处理经度
    lon_indices = np.where((lons_arr >= lon_range[0]) & (lons_arr <= lon_range[1]))[0]
    
    if len(lat_indices) == 0 or len(lon_indices) == 0:
        return None, None, None, None
    
    # 与test目录一致：使用精确的lat_indices和lon_indices
    return lat_indices[0], lat_indices[-1] + 1, lon_indices[0], lon_indices[-1] + 1

# ---------------------- 构建有效样本：每个预报文件对应完整的46天真值序列 ----------------------
def find_ensemble_members(base_filename):
    """
    查找同一日期的所有11个集合预报成员文件
    返回: 11个成员文件的列表（按成员编号排序）
    """
    # 假设文件名格式: s2s_ecmf_YYYYMMDD_tp_daily.nc 或
    #                 s2s_ecmf_YYYYMMDD_member001_tp_daily.nc
    import re
    
    # 尝试不同的匹配模式
    # 模式1: 直接使用base_filename（可能已经包含所有成员在一个文件中）
    all_files = glob.glob(os.path.join(FORECAST_DIR, "s2s_ecmf_*.nc"))
    
    # 提取日期
    date_pattern = r"s2s_ecmf_(\d{8})"
    match = re.search(date_pattern, base_filename)
    if not match:
        return [base_filename]
    
    date_str = match.group(1)
    
    # 尝试找到所有成员文件
    # 模式: s2s_ecmf_YYYYMMDD_*_tp_daily.nc
    member_patterns = [
        f"s2s_ecmf_{date_str}_member*_tp_daily.nc",
        f"s2s_ecmf_{date_str}_*_tp_daily.nc",
    ]
    
    member_files = []
    for pattern in member_patterns:
        member_files = sorted(glob.glob(os.path.join(FORECAST_DIR, pattern)))
        if len(member_files) == 11:
            break
    
    # 如果找到11个成员，返回排序后的列表
    if len(member_files) == 11:
        # 按文件名排序（确保成员顺序一致）
        return sorted(member_files)
    elif len(member_files) > 0:
        print(f"[警告] 找到 {len(member_files)} 个成员文件，期望11个")
        return sorted(member_files)
    else:
        # 如果没找到成员文件，可能所有数据在一个文件中
        return [base_filename]

def build_valid_samples():
    """
    构建有效样本列表
    返回: valid_samples列表，每个元素是：预报文件 + 46天真值文件列表 + 起报日期
    """
    # 收集所有真值文件，建立日期→文件的映射
    truth_files = sorted(glob.glob(os.path.join(TRUTH_DIR, "CRA_truth_*.nc")))
    truth_date_to_file = {}
    for f in truth_files:
        date_str = extract_date_from_filename(os.path.basename(f))
        truth_date_to_file[date_str] = f
    print(f"找到 {len(truth_files)} 个真值文件")

    # 收集所有预报文件
    forecast_files_all = sorted(glob.glob(os.path.join(FORECAST_DIR, "s2s_ecmf_*.nc")))
    print(f"找到 {len(forecast_files_all)} 个预报文件")
    
    valid_samples = []
    processed_dates = set()  # 用于避免重复处理同一日期
    missing_truth_count = 0  # 记录因缺少真值而跳过的样本数
    
    for forecast_file in forecast_files_all:
        # 提取预报文件的起报日期（使用正则匹配8位日期）
        init_date_str = extract_date_from_filename(os.path.basename(forecast_file))
        
        # 如果已经处理过这个日期，跳过
        if init_date_str in processed_dates:
            continue
        processed_dates.add(init_date_str)
        
        init_date = datetime.strptime(init_date_str, "%Y%m%d")
        
        # 收集该预报文件对应的10天真值文件
        truth_files_list = []
        target_dates_list = []
        valid = True
        
        for lead_day in range(0, FORECAST_LEAD_TIME):
            target_date = init_date + timedelta(days=lead_day)
            target_date_str = target_date.strftime("%Y%m%d")
            
            # 检查该天的真值是否存在
            if target_date_str not in truth_date_to_file:
                if missing_truth_count < 3:  # 只打印前3个
                    print(f"[跳过] 日期 {init_date_str}: 缺少第{lead_day}天真值文件 {target_date_str}")
                valid = False
                missing_truth_count += 1
                break
            
            truth_files_list.append(truth_date_to_file[target_date_str])
            target_dates_list.append(target_date_str)
        
        # 只有FORECAST_LEAD_TIME天真值都存在时，才加入有效样本
        if valid:
            # 查找该日期的所有11个成员文件
            ensemble_members = find_ensemble_members(forecast_file)
            
            valid_samples.append({
                "forecast_file": forecast_file,  # 基础预报文件
                "ensemble_members": ensemble_members,  # 11个成员文件列表
                "truth_files": truth_files_list,
                "init_date": init_date_str,
                "target_dates": target_dates_list
            })

    print(f"找到 {len(valid_samples)} 个有效预报文件（{FORECAST_LEAD_TIME}天真值均存在）")
    if missing_truth_count > 0:
        print(f"因缺少真值文件跳过了 {missing_truth_count} 个预报日期")
    
    # 打印前10个样本的日期
    if len(valid_samples) > 0:
        print(f"前10个样本日期: {[s['init_date'] for s in valid_samples[:10]]}")
    
    return valid_samples

# ---------------------- 气象数据集类 ----------------------
class PrecipitationDataset(Dataset):
    def __init__(self, valid_samples, mode="train", train_ratio=0.8, region="global",
                 lat_range=None, lon_range=None, use_land_mask=True, super_resolution=False,
                 target_resolution=(0.25, 0.25), train_end_year=2021, train_end_date=None,
                 use_july_only=False,
                 use_anomaly=False, anomaly_scale="month", anomaly_max_samples_per_group=15):
        """
        Args:
            valid_samples: 有效样本列表
            mode: "train"、"val" 或 "test"（当前主流程使用 train/test）
            train_ratio: 训练集比例（按年份划分时使用train_end_year）
            region: "global" 或 "china"
            lat_range: 纬度范围 (min_lat, max_lat)
            lon_range: 经度范围 (min_lon, max_lon)
            use_land_mask: 是否使用陆地掩码（处理NaN值）
            super_resolution: 是否启用超分模式
            target_resolution: 超分目标分辨率 (lat_step, lon_step)，默认为0.25°×0.25°
            train_end_year: 训练集结束年份（含），之后的年份为测试集，默认2021
            train_end_date: 严格训练截止日期 YYYYMMDD。训练样本的最后一个
                目标日期不得晚于此日期，防止训练标签与测试期观测重叠。
            use_july_only: 是否只使用7月份样本（与test目录一致），默认False
            use_anomaly: 是否启用距平模式（输入减气候态，推理加回），默认False
            anomaly_scale: 气候态尺度 "month" 或 "pentad"
            anomaly_max_samples_per_group: 每个月/侯最多读取的样本数(用于计算气候态)
        """
        # 按起报日期排序
        valid_samples = sorted(valid_samples, key=lambda x: x["init_date"])
        if train_end_date is None:
            train_end_date = f"{train_end_year}1231"
        self.train_end_date = str(train_end_date)
        
        # 保留该选项仅供旧实验复现；主训练/测试流程不应启用它。
        if use_july_only:
            july_samples = []
            for sample in valid_samples:
                month = int(sample["init_date"][4:6])
                if month == 7:
                    july_samples.append(sample)
            print(f"筛选7月份样本: {len(july_samples)} 个")
            # use_july_only 时，不按年份划分，所有 7 月样本都作为数据集。
            self.valid_samples = july_samples
        else:
            # 严格按时间划分。训练样本的所有标签必须留在训练期内；
            # 测试样本则从切分日的下一天起报。
            train_samples = []
            test_samples = []
            for sample in valid_samples:
                init_date = sample["init_date"]
                target_dates = sample.get("target_dates", [])
                last_target_date = target_dates[-1] if target_dates else init_date
                if last_target_date <= self.train_end_date:
                    train_samples.append(sample)
                elif init_date > self.train_end_date:
                    test_samples.append(sample)
            
            if mode == "train":
                self.valid_samples = train_samples
            else:
                self.valid_samples = test_samples

            if mode == "train":
                print(f"[时间划分] 训练截止目标日期: {self.train_end_date}; "
                      f"最后一个目标日越界的年末起报样本已排除")
            else:
                print(f"[时间划分] 测试起报日期: > {self.train_end_date}; 使用全部月份")
        
        # 区域设置
        self.region = region
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.use_land_mask = use_land_mask
        
        # 超分设置
        self.super_resolution = super_resolution
        self.target_lat_step = target_resolution[0]
        self.target_lon_step = target_resolution[1]
        
        # 分别存储预报值和真值的区域索引（因为分辨率不同）
        self.forecast_lat_start = None
        self.forecast_lat_end = None
        self.forecast_lon_start = None
        self.forecast_lon_end = None
        self.truth_lat_start = None
        self.truth_lat_end = None
        self.truth_lon_start = None
        self.truth_lon_end = None
        self.forecast_lats_region = None
        self.forecast_lons_region = None
        self.truth_lats_region = None
        self.truth_lons_region = None
        
        # 超分目标网格
        self.sr_target_lats = None
        self.sr_target_lons = None
        
        # 陆地掩码（基于真值数据）
        self.land_mask = None
        
        # 归一化参数
        self.forecast_min = None
        self.forecast_max = None
        self.truth_min = None
        self.truth_max = None

        # 距平(Anomaly)配置
        self.use_anomaly = use_anomaly
        self.anomaly_scale = anomaly_scale
        self.anomaly_max_samples_per_group = anomaly_max_samples_per_group
        self.clim_forecast = {}  # {month/pentad_key: (H, W)} 预报气候态
        self.clim_truth = {}     # {month/pentad_key: (H, W)} 真值气候态

        if region == "china" and lat_range is not None and lon_range is not None:
            self._setup_region_indices(valid_samples)
        elif region == "global":
            # 全球模式也需要设置预报/真值的经纬度区域（用于重网格化）
            self._setup_global_region_indices(valid_samples)

        # 距平模式：计算气候态（基于训练样本，按月/侯分组）
        if self.use_anomaly:
            self._compute_climatology(valid_samples)

        # 计算归一化参数
        if mode == "train":
            self._compute_normalization_params()
        
        print(f"{mode} 集样本数: {len(self.valid_samples)}")
        if self.super_resolution:
            print(f"超分模式: 目标分辨率 {self.target_lat_step}°×{self.target_lon_step}°")

    def _setup_region_indices(self, valid_samples):
        """设置预报值和真值的区域索引"""
        if len(valid_samples) == 0:
            return
        
        # 1. 获取预报值的经纬度坐标
        forecast_file = valid_samples[0]["forecast_file"]
        with nc.Dataset(forecast_file, "r") as f:
            print(f"\n预报文件变量列表: {list(f.variables.keys())}")
            forecast_lats, forecast_lons, fc_lat_name, fc_lon_name = get_lat_lon_vars(f)
            
            if forecast_lats is not None and forecast_lons is not None:
                print(f"预报值使用经纬度变量: {fc_lat_name}, {fc_lon_name}")
                print(f"预报值纬度范围: [{forecast_lats.min():.2f}, {forecast_lats.max():.2f}]")
                print(f"预报值经度范围: [{forecast_lons.min():.2f}, {forecast_lons.max():.2f}]")
                
                # 计算预报值区域索引
                indices = compute_region_indices(forecast_lats, forecast_lons, 
                                                self.lat_range, self.lon_range)
                if indices[0] is not None:
                    self.forecast_lat_start, self.forecast_lat_end, \
                    self.forecast_lon_start, self.forecast_lon_end = indices
                    self.forecast_lats_region = np.asarray(
                        forecast_lats[self.forecast_lat_start:self.forecast_lat_end],
                        dtype=np.float64,
                    )
                    self.forecast_lons_region = np.asarray(
                        forecast_lons[self.forecast_lon_start:self.forecast_lon_end],
                        dtype=np.float64,
                    )
                    print(f"预报值中国区域索引: lat[{self.forecast_lat_start}:{self.forecast_lat_end}], "
                          f"lon[{self.forecast_lon_start}:{self.forecast_lon_end}]")
                else:
                    print("警告: 未在预报值中找到符合范围的经纬度")
            else:
                print("警告: 未找到预报值经纬度变量")
        
        # 2. 获取真值的经纬度坐标
        truth_file = valid_samples[0]["truth_files"][0]
        with nc.Dataset(truth_file, "r") as f:
            print(f"\n真值文件变量列表: {list(f.variables.keys())}")
            truth_lats, truth_lons, tr_lat_name, tr_lon_name = get_lat_lon_vars(f)
            
            if truth_lats is not None and truth_lons is not None:
                print(f"真值使用经纬度变量: {tr_lat_name}, {tr_lon_name}")
                print(f"真值纬度范围: [{truth_lats.min():.2f}, {truth_lats.max():.2f}]")
                print(f"真值经度范围: [{truth_lons.min():.2f}, {truth_lons.max():.2f}]")
                
                # 计算真值区域索引
                indices = compute_region_indices(truth_lats, truth_lons,
                                                self.lat_range, self.lon_range)
                if indices[0] is not None:
                    self.truth_lat_start, self.truth_lat_end, \
                    self.truth_lon_start, self.truth_lon_end = indices
                    self.truth_lats_region = np.asarray(
                        truth_lats[self.truth_lat_start:self.truth_lat_end],
                        dtype=np.float64,
                    )
                    self.truth_lons_region = np.asarray(
                        truth_lons[self.truth_lon_start:self.truth_lon_end],
                        dtype=np.float64,
                    )
                    print(f"真值中国区域索引: lat[{self.truth_lat_start}:{self.truth_lat_end}], "
                          f"lon[{self.truth_lon_start}:{self.truth_lon_end}]")
                else:
                    print("警告: 未在真值中找到符合范围的经纬度")
                
                # 3. 创建超分目标网格（0.25°×0.25°分辨率）
                if self.super_resolution:
                    self._create_sr_target_grid(truth_lats, truth_lons)
                
                # 4. 创建陆地掩码（基于真值数据的第一个时间步）
                if self.use_land_mask:
                    self._create_land_mask(f, truth_lats, truth_lons)
            else:
                print("警告: 未找到真值经纬度变量")

    def _setup_global_region_indices(self, valid_samples):
        """全球模式下设置预报/真值的全局经纬度索引（不裁剪，仅设置完整的经纬度数组）"""
        if len(valid_samples) == 0:
            return

        forecast_file = valid_samples[0]["forecast_file"]
        with nc.Dataset(forecast_file, "r") as f:
            forecast_lats, forecast_lons, _, _ = get_lat_lon_vars(f)
            if forecast_lats is not None and forecast_lons is not None:
                self.forecast_lats_region = np.asarray(forecast_lats[:], dtype=np.float64)
                self.forecast_lons_region = np.asarray(forecast_lons[:], dtype=np.float64)
                # 全局范围：不裁剪，start=0, end=length
                self.forecast_lat_start = 0
                self.forecast_lat_end = len(self.forecast_lats_region)
                self.forecast_lon_start = 0
                self.forecast_lon_end = len(self.forecast_lons_region)
                print(f"[Global] 预报网格: lat[{len(self.forecast_lats_region)}], lon[{len(self.forecast_lons_region)}], "
                      f"range=[{self.forecast_lats_region.min():.1f},{self.forecast_lats_region.max():.1f}] × "
                      f"[{self.forecast_lons_region.min():.1f},{self.forecast_lons_region.max():.1f}]")
            else:
                print("[Global] 警告: 未找到预报值经纬度变量")

        truth_file = valid_samples[0]["truth_files"][0]
        with nc.Dataset(truth_file, "r") as f:
            truth_lats, truth_lons, _, _ = get_lat_lon_vars(f)
            if truth_lats is not None and truth_lons is not None:
                self.truth_lats_region = np.asarray(truth_lats[:], dtype=np.float64)
                self.truth_lons_region = np.asarray(truth_lons[:], dtype=np.float64)
                self.truth_lat_start = 0
                self.truth_lat_end = len(self.truth_lats_region)
                self.truth_lon_start = 0
                self.truth_lon_end = len(self.truth_lons_region)
                print(f"[Global] 真值网格: lat[{len(self.truth_lats_region)}], lon[{len(self.truth_lons_region)}], "
                      f"range=[{self.truth_lats_region.min():.1f},{self.truth_lats_region.max():.1f}] × "
                      f"[{self.truth_lons_region.min():.1f},{self.truth_lons_region.max():.1f}]")
                if self.use_land_mask:
                    self._create_land_mask(f, truth_lats, truth_lons)
            else:
                print("[Global] 警告: 未找到真值经纬度变量")

    def _create_sr_target_grid(self, truth_lats, truth_lons):
        """创建超分目标网格（0.25°×0.25°分辨率）"""
        if self.truth_lats_region is None or self.truth_lons_region is None:
            print("警告: 未设置真值区域，无法创建超分目标网格")
            return
        
        # 基于真值区域创建高分辨率网格
        lat_min = self.truth_lats_region.min()
        lat_max = self.truth_lats_region.max()
        lon_min = self.truth_lons_region.min()
        lon_max = self.truth_lons_region.max()
        
        # 创建目标分辨率的经纬度网格
        # 确保覆盖整个真值区域
        self.sr_target_lats = np.arange(lat_max, lat_min - self.target_lat_step, 
                                        -self.target_lat_step, dtype=np.float64)
        self.sr_target_lons = np.arange(lon_min, lon_max + self.target_lon_step, 
                                        self.target_lon_step, dtype=np.float64)
        
        # 确保网格单调递减（与真值数据一致）
        if self.sr_target_lats[0] < self.sr_target_lats[-1]:
            self.sr_target_lats = self.sr_target_lats[::-1]
        
        print(f"\n超分目标网格创建成功:")
        print(f"  纬度: {len(self.sr_target_lats)} 个点, 范围 [{self.sr_target_lats.min():.4f}, {self.sr_target_lats.max():.4f}]")
        print(f"  经度: {len(self.sr_target_lons)} 个点, 范围 [{self.sr_target_lons.min():.4f}, {self.sr_target_lons.max():.4f}]")
        print(f"  分辨率: {self.target_lat_step}° × {self.target_lon_step}°")
        
    def _create_land_mask(self, f, lats, lons):
        """创建陆地掩码 — 与test目录一致：基于真值数据tp的NaN值生成

        CRA真值数据中，海洋区域的tp值为NaN，陆地区域为有效值。
        因此 land_mask = ~np.isnan(tp_data)，即"tp不是NaN的地方就是陆地"。
        这确保了主工程与test目录使用完全相同的land mask。

        在超分模式下，会将掩码上采样到超分目标网格。
        """
        try:
            tp_var = f.variables.get("tp")
            if tp_var is None:
                self.land_mask = None
                return

            # 根据tp变量的shape提取数据（与test目录一致）
            if len(tp_var.shape) == 2:
                if self.region == "china" and self.truth_lat_start is not None:
                    sample_data = tp_var[self.truth_lat_start:self.truth_lat_end,
                                        self.truth_lon_start:self.truth_lon_end]
                else:
                    sample_data = tp_var[:, :]
            elif len(tp_var.shape) == 3:
                if self.region == "china" and self.truth_lat_start is not None:
                    sample_data = tp_var[0,
                                        self.truth_lat_start:self.truth_lat_end,
                                        self.truth_lon_start:self.truth_lon_end]
                else:
                    sample_data = tp_var[0, :, :]
            elif len(tp_var.shape) == 4:
                if self.region == "china" and self.truth_lat_start is not None:
                    sample_data = tp_var[0, 0,
                                        self.truth_lat_start:self.truth_lat_end,
                                        self.truth_lon_start:self.truth_lon_end]
                else:
                    sample_data = tp_var[0, 0, :, :]
            elif len(tp_var.shape) == 5:
                if self.region == "china" and self.truth_lat_start is not None:
                    sample_data = tp_var[0, 0, 0,
                                        self.truth_lat_start:self.truth_lat_end,
                                        self.truth_lon_start:self.truth_lon_end]
                else:
                    sample_data = tp_var[0, 0, 0, :, :]
            else:
                sample_data = tp_var[0]

            sample_data = np.array(sample_data, dtype=np.float32)

            # 关键：与test目录一致，基于NaN值生成land mask
            # CRA真值数据中，海洋区域的tp值为NaN
            self.land_mask = ~np.isnan(sample_data)

            land_ratio = np.mean(self.land_mask)
            print(f"[Land Mask] Created from tp NaN: shape={self.land_mask.shape}, land ratio: {land_ratio:.2%}")
            print(f"[Land Mask] 陆地像素数: {np.sum(self.land_mask)}, 海洋像素数: {self.land_mask.size - np.sum(self.land_mask)}")

        except Exception as e:
            print(f"创建陆地掩码时出错: {e}")
            import traceback
            traceback.print_exc()
            self.land_mask = None

    def _create_geographic_land_mask(self, lats, lons):
        """基于固定地理信息创建中国区域陆地掩码
        
        使用简化的中国陆地轮廓多边形判断，不依赖数据NaN值。
        陆地=True，海洋=False。
        """
        from matplotlib.path import Path
        
        # 创建经纬度网格
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        points = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))
        
        # 中国陆地轮廓简化多边形（顺时针）
        # 覆盖中国大陆主要陆地区域，排除海洋
        china_land_polygon = [
            (70, 40),     # 西部边界起点
            (73, 45),
            (78, 48),
            (82, 49),
            (88, 48),
            (92, 50),
            (96, 52),
            (105, 53),   # 北部边界
            (115, 53),
            (120, 53),
            (125, 53),
            (130, 48),
            (134, 48),
            (135, 45),
            (131, 43),
            (130, 40),
            (124, 40),
            (122, 39),
            (121, 38),
            (120, 37),
            (121, 35),
            (120, 32),
            (121, 30),
            (120, 27),
            (118, 24),
            (115, 22),
            (110, 21),
            (108, 21),
            (108, 18),
            (105, 22),
            (97, 28),
            (95, 29),
            (90, 28),
            (85, 28),
            (80, 30),
            (75, 37),
            (70, 40),     # 闭合
        ]
        
        path = Path(china_land_polygon)
        mask = path.contains_points(points)
        land_mask = mask.reshape(len(lats), len(lons))
        
        return land_mask

    def _regrid_truth_to_forecast_grid(self, truth):
        """将真值重网格到预报网格（与test目录一致，使用scipy.ndimage.zoom双线性插值）

        简单可靠，避免conservative remapping导致的网格对齐问题。
        """
        if self.super_resolution and self.sr_target_lats is not None and self.sr_target_lons is not None:
            return self._regrid_truth_to_sr_grid(truth)

        if (
            self.forecast_lats_region is None
            or self.forecast_lons_region is None
            or self.truth_lats_region is None
            or self.truth_lons_region is None
        ):
            return truth

        truth = np.asarray(truth, dtype=np.float32)
        if truth.ndim == 2:
            truth = truth[None, ...]
        if truth.ndim != 3:
            raise ValueError(f"Expected truth shape (time, lat, lon), got {truth.shape}")

        n_src_lat, n_src_lon = truth.shape[-2], truth.shape[-1]
        n_tgt_lat = len(self.forecast_lats_region)
        n_tgt_lon = len(self.forecast_lons_region)

        if n_src_lat == n_tgt_lat and n_src_lon == n_tgt_lon:
            return truth

        from scipy.ndimage import zoom

        lat_ratio = n_tgt_lat / n_src_lat
        lon_ratio = n_tgt_lon / n_src_lon

        regridded = zoom(truth, (1, lat_ratio, lon_ratio), order=1)

        if regridded.shape[-2] != n_tgt_lat or regridded.shape[-1] != n_tgt_lon:
            regridded = regridded[:, :n_tgt_lat, :n_tgt_lon]

        regridded = np.nan_to_num(regridded, nan=0.0)
        return np.maximum(regridded, 0.0)
    
    def _upsample_forecast_to_sr_grid(self, forecast):
        """将低分辨率预报数据上采样到超分目标网格（0.25°×0.25°）
        
        使用双线性插值进行上采样，这是超分任务中常用的方法。
        """
        if not self.super_resolution or self.sr_target_lats is None or self.sr_target_lons is None:
            return forecast
            
        if (
            self.forecast_lats_region is None
            or self.forecast_lons_region is None
        ):
            return forecast

        forecast = np.asarray(forecast, dtype=np.float32)
        
        original_shape = forecast.shape
        if len(original_shape) == 4:
            num_members, num_times, h, w = original_shape
            forecast = forecast.reshape(-1, h, w)
        elif len(original_shape) == 3:
            num_times, h, w = original_shape
        elif len(original_shape) == 2:
            forecast = forecast[None, ...]
        else:
            raise ValueError(f"Expected forecast shape (members, time, lat, lon) or (time, lat, lon) or (lat, lon), got {original_shape}")

        src_lats = np.asarray(self.forecast_lats_region, dtype=np.float64)
        src_lons = np.asarray(self.forecast_lons_region, dtype=np.float64)
        tgt_lats = np.asarray(self.sr_target_lats, dtype=np.float64)
        tgt_lons = np.asarray(self.sr_target_lons, dtype=np.float64)

        forecast_work = forecast
        if src_lats[0] > src_lats[-1]:
            src_lats = src_lats[::-1]
            forecast_work = forecast_work[:, ::-1, :]
        if src_lons[0] > src_lons[-1]:
            src_lons = src_lons[::-1]
            forecast_work = forecast_work[:, :, ::-1]

        if tgt_lats[0] > tgt_lats[-1]:
            tgt_lats_working = tgt_lats[::-1]
            reverse_lat = True
        else:
            tgt_lats_working = tgt_lats
            reverse_lat = False

        upsampled = np.empty((forecast_work.shape[0], len(tgt_lats), len(tgt_lons)), dtype=np.float32)

        for t in range(forecast_work.shape[0]):
            interp = RegularGridInterpolator(
                (src_lats, src_lons),
                forecast_work[t],
                method='linear',
                bounds_error=False,
                fill_value=0.0
            )
            
            tgt_points = np.array(np.meshgrid(tgt_lats_working, tgt_lons, indexing='ij')).T.reshape(-1, 2)
            result = interp(tgt_points).reshape(len(tgt_lats), len(tgt_lons))
            
            if reverse_lat:
                result = result[::-1, :]
            
            upsampled[t] = result

        upsampled = np.nan_to_num(upsampled, nan=0.0, posinf=0.0, neginf=0.0)
        upsampled = np.maximum(upsampled, 0.0)

        if len(original_shape) == 4:
            upsampled = upsampled.reshape(num_members, num_times, len(tgt_lats), len(tgt_lons))

        print(f"[超分] 预报数据上采样完成: {original_shape} -> {upsampled.shape}")
        
        return upsampled
    
    def _regrid_truth_to_sr_grid(self, truth):
        """将真值数据插值到超分目标网格（0.25°×0.25°）
        
        使用双线性插值进行重网格，保持降水数据的空间分布特征。
        """
        if not self.super_resolution or self.sr_target_lats is None or self.sr_target_lons is None:
            return truth
            
        if (
            self.truth_lats_region is None
            or self.truth_lons_region is None
        ):
            return truth

        truth = np.asarray(truth, dtype=np.float32)
        
        original_shape = truth.shape
        if len(original_shape) == 3:
            num_times, h, w = original_shape
        elif len(original_shape) == 2:
            truth = truth[None, ...]
            num_times, h, w = truth.shape
        else:
            raise ValueError(f"Expected truth shape (time, lat, lon) or (lat, lon), got {original_shape}")

        src_lats = np.asarray(self.truth_lats_region, dtype=np.float64)
        src_lons = np.asarray(self.truth_lons_region, dtype=np.float64)
        tgt_lats = np.asarray(self.sr_target_lats, dtype=np.float64)
        tgt_lons = np.asarray(self.sr_target_lons, dtype=np.float64)

        truth_work = truth
        if src_lats[0] > src_lats[-1]:
            src_lats = src_lats[::-1]
            truth_work = truth_work[:, ::-1, :]
        if src_lons[0] > src_lons[-1]:
            src_lons = src_lons[::-1]
            truth_work = truth_work[:, :, ::-1]

        if tgt_lats[0] > tgt_lats[-1]:
            tgt_lats_working = tgt_lats[::-1]
            reverse_lat = True
        else:
            tgt_lats_working = tgt_lats
            reverse_lat = False

        regridded = np.empty((truth_work.shape[0], len(tgt_lats), len(tgt_lons)), dtype=np.float32)

        for t in range(truth_work.shape[0]):
            interp = RegularGridInterpolator(
                (src_lats, src_lons),
                truth_work[t],
                method='linear',
                bounds_error=False,
                fill_value=0.0
            )
            
            tgt_points = np.array(np.meshgrid(tgt_lats_working, tgt_lons, indexing='ij')).T.reshape(-1, 2)
            result = interp(tgt_points).reshape(len(tgt_lats), len(tgt_lons))
            
            if reverse_lat:
                result = result[::-1, :]
            
            regridded[t] = result

        regridded = np.nan_to_num(regridded, nan=0.0, posinf=0.0, neginf=0.0)
        regridded = np.maximum(regridded, 0.0)

        if len(original_shape) == 2:
            regridded = regridded[0]

        print(f"[超分] 真值数据重网格完成: {original_shape} -> {regridded.shape}")

        return regridded

    def _get_clim_key(self, init_date_str):
        """根据起报日期和anomaly_scale返回气候态分组key"""
        if self.anomaly_scale == "month":
            return int(init_date_str[4:6])  # 月份 1-12
        else:  # pentad (侯尺度, 5天)
            dt = datetime.strptime(init_date_str, "%Y%m%d")
            day_of_year = dt.timetuple().tm_yday
            return (day_of_year - 1) // 5 + 1  # 侯 1-73

    def _read_sample_forecast_truth_day0(self, sample):
        """读取单个样本的第一天forecast(tp)和truth，用于气候态计算
        返回: (forecast_day0 (H,W), truth_day0 (H,W)) 或 None
        """
        try:
            # 读取forecast第一天tp
            with nc.Dataset(sample["forecast_file"], "r") as f:
                tp_var = f.variables.get("tp")
                if tp_var is None:
                    return None
                if self.region == "china" and self.forecast_lat_start is not None:
                    fc_data = tp_var[0, self.forecast_lat_start:self.forecast_lat_end,
                                     self.forecast_lon_start:self.forecast_lon_end]
                else:
                    fc_data = tp_var[0]
                fc_units = get_var_units(tp_var)
                fc_data = np.array(fc_data, dtype=np.float32)
                fc_data = convert_accumulated_precip_to_mm(fc_data, fc_units)
                fc_data = np.nan_to_num(fc_data, nan=0.0)

            # 读取truth第一天
            with nc.Dataset(sample["truth_files"][0], "r") as f:
                tp_var = f.variables["tp"]
                tr_units = get_var_units(tp_var)
                if len(tp_var.shape) == 2:
                    if self.region == "china" and self.truth_lat_start is not None:
                        tr_data = tp_var[self.truth_lat_start:self.truth_lat_end,
                                         self.truth_lon_start:self.truth_lon_end]
                    else:
                        tr_data = tp_var[:, :]
                elif len(tp_var.shape) == 3:
                    if self.region == "china" and self.truth_lat_start is not None:
                        tr_data = tp_var[0, self.truth_lat_start:self.truth_lat_end,
                                         self.truth_lon_start:self.truth_lon_end]
                    else:
                        tr_data = tp_var[0, :, :]
                else:
                    tr_data = tp_var[0]
                tr_data = np.array(tr_data, dtype=np.float32)
                tr_data = convert_accumulated_precip_to_mm(
                    tr_data, tr_units, truth_scale_factor=truth_scale_factor)
                tr_data = np.nan_to_num(tr_data, nan=0.0)
                # 重网格到预报网格
                tr_data = self._regrid_truth_to_forecast_grid(tr_data)
                if tr_data.ndim == 3:
                    tr_data = tr_data[0]

            return fc_data, tr_data
        except Exception as e:
            print(f"[气候态] 读取样本 {sample.get('init_date', '?')} 失败: {e}")
            return None

    def _compute_climatology(self, all_samples):
        """计算气候态（按月或侯分组，基于训练样本）
        clim_forecast[key] / clim_truth[key]: (H, W) 空间气候态
        """
        print(f"\n[气候态] 开始计算 (scale={self.anomaly_scale})...")

        # 按月/侯分组训练样本。与数据划分一致，任何目标日期跨入测试期的
        # 年末样本都不能参与气候态估计。
        groups = {}
        for sample in all_samples:
            init_date = sample["init_date"]
            target_dates = sample.get("target_dates", [])
            last_target_date = target_dates[-1] if target_dates else init_date
            if last_target_date > self.train_end_date:
                continue
            key = self._get_clim_key(sample["init_date"])
            if key not in groups:
                groups[key] = []
            groups[key].append(sample)

        print(f"[气候态] 共 {len(groups)} 个分组: {sorted(groups.keys())}")

        for key, samples_in_group in sorted(groups.items()):
            # 限制每组读取的样本数
            max_n = min(self.anomaly_max_samples_per_group, len(samples_in_group))
            # 均匀采样
            if len(samples_in_group) > max_n:
                indices = np.linspace(0, len(samples_in_group) - 1, max_n).astype(int)
                selected = [samples_in_group[i] for i in indices]
            else:
                selected = samples_in_group

            fc_list = []
            tr_list = []
            for s in selected:
                result = self._read_sample_forecast_truth_day0(s)
                if result is not None:
                    fc_list.append(result[0])
                    tr_list.append(result[1])

            if len(fc_list) > 0:
                self.clim_forecast[key] = np.mean(fc_list, axis=0).astype(np.float32)
                self.clim_truth[key] = np.mean(tr_list, axis=0).astype(np.float32)
                print(f"[气候态] 分组 {key}: {len(fc_list)} 个样本, "
                      f"fc均值={np.mean(self.clim_forecast[key]):.4f}, "
                      f"tr均值={np.mean(self.clim_truth[key]):.4f}")

        print(f"[气候态] 计算完成: {len(self.clim_forecast)} 个分组")

    def _get_clim_forecast(self, init_date_str):
        """获取指定日期的预报气候态 (H, W)"""
        key = self._get_clim_key(init_date_str)
        return self.clim_forecast.get(key)

    def _get_clim_truth(self, init_date_str):
        """获取指定日期的真值气候态 (H, W)"""
        key = self._get_clim_key(init_date_str)
        return self.clim_truth.get(key)

    def _compute_normalization_params(self):
        """计算线性归一化参数（与test目录完全一致）
        关键：只读tp变量(1个成员)，不转单位，truth读10天
        
        距平模式下：先减气候态，再计算分位数，确保归一化范围适配距平值分布
        """
        print("\n[归一化参数] 开始计算（与test完全一致）...")

        all_forecast = []
        all_truth = []
        sample_count = min(80, len(self.valid_samples))

        for i in range(sample_count):
            sample = self.valid_samples[i]
            init_date = sample["init_date"]

            # 读取预报数据：只读tp变量（与test一致，不读11个成员）
            with nc.Dataset(sample["forecast_file"], "r") as f:
                tp_var = f.variables.get("tp")
                if tp_var is not None:
                    forecast_units = get_var_units(tp_var)
                    if self.region == "china" and self.forecast_lat_start is not None:
                        data = tp_var[:, self.forecast_lat_start:self.forecast_lat_end,
                                    self.forecast_lon_start:self.forecast_lon_end]
                    else:
                        data = tp_var[:]
                    data = np.array(data, dtype=np.float32)
                    if data.shape[0] > FORECAST_LEAD_TIME:
                        data = data[:FORECAST_LEAD_TIME]
                    # 关键修复：与__getitem__一致，先转mm再计算分位数
                    data = convert_accumulated_precip_to_mm(data, forecast_units)
                    data = np.nan_to_num(data, nan=0.0)
                    
                    # 预报裁剪：与__getitem__保持一致（clip在单位转换之后）
                    clip_max = getattr(config, 'forecast_clip_max', 0.0)
                    if clip_max and clip_max > 0:
                        data = np.clip(data, 0.0, clip_max)
                    
                    # 距平模式：预报数据先减气候态（分辨率匹配，都是预报网格）
                    if self.use_anomaly:
                        clim_fc = self._get_clim_forecast(init_date)
                        if clim_fc is not None:
                            data = data - clim_fc[np.newaxis, :, :]
                    
                    all_forecast.append(data)

            # 读取真值数据：前10天（与test一致，不是5天）
            for j, truth_file in enumerate(sample["truth_files"]):
                if j >= FORECAST_LEAD_TIME:
                    break
                with nc.Dataset(truth_file, "r") as f:
                    tp_var = f.variables["tp"]
                    if tp_var is not None:
                        truth_units = get_var_units(tp_var)

                        if len(tp_var.shape) == 2:
                            if self.region == "china" and self.truth_lat_start is not None:
                                truth_day = tp_var[self.truth_lat_start:self.truth_lat_end,
                                                 self.truth_lon_start:self.truth_lon_end]
                            else:
                                truth_day = tp_var[:, :]
                        elif len(tp_var.shape) == 3:
                            if self.region == "china" and self.truth_lat_start is not None:
                                truth_day = tp_var[0, self.truth_lat_start:self.truth_lat_end,
                                                 self.truth_lon_start:self.truth_lon_end]
                            else:
                                truth_day = tp_var[0, :, :]
                        elif len(tp_var.shape) == 4:
                            if self.region == "china" and self.truth_lat_start is not None:
                                truth_day = tp_var[0, 0, self.truth_lat_start:self.truth_lat_end,
                                                 self.truth_lon_start:self.truth_lon_end]
                            else:
                                truth_day = tp_var[0, 0, :, :]
                        elif len(tp_var.shape) == 5:
                            if self.region == "china" and self.truth_lat_start is not None:
                                truth_day = tp_var[0, 0, 0, self.truth_lat_start:self.truth_lat_end,
                                                 self.truth_lon_start:self.truth_lon_end]
                            else:
                                truth_day = tp_var[0, 0, 0, :, :]
                        else:
                            truth_day = tp_var[0]

                        truth_day = np.array(truth_day, dtype=np.float32)
                        truth_day = convert_accumulated_precip_to_mm(truth_day, truth_units)
                        truth_day = np.nan_to_num(truth_day, nan=0.0)
                        
                        # 距平模式：真值数据先重网格，再减气候态
                        if self.use_anomaly:
                            clim_tr = self._get_clim_truth(init_date)
                            if clim_tr is not None:
                                # 重网格真值到预报网格分辨率
                                truth_regrid = self._regrid_truth_to_forecast_grid(truth_day)
                                if truth_regrid.ndim == 3:
                                    truth_regrid = truth_regrid[0]
                                truth_day = truth_regrid - clim_tr
                        
                        all_truth.append(truth_day)

        # 线性归一化：使用1%和99%分位数（平衡极端值处理和数据分布）
        if len(all_forecast) > 0:
            all_forecast = np.concatenate(all_forecast)
            self.forecast_min = float(np.percentile(all_forecast, 1.0))
            self.forecast_max = float(np.percentile(all_forecast, 99.0))
            if self.use_anomaly:
                print(f"[归一化参数-距平] 预报距平范围: [{self.forecast_min:.4f}, {self.forecast_max:.4f}]")
            else:
                print(f"[归一化参数] 预报值范围: [{self.forecast_min:.4f}, {self.forecast_max:.4f}]")
        else:
            self.forecast_min = 0.0
            self.forecast_max = 100.0

        if len(all_truth) > 0:
            all_truth = np.concatenate(all_truth)
            self.truth_min = float(np.percentile(all_truth, 1.0))
            self.truth_max = float(np.percentile(all_truth, 99.0))
            if self.use_anomaly:
                print(f"[归一化参数-距平] 真值距平范围: [{self.truth_min:.4f}, {self.truth_max:.4f}]")
            else:
                print(f"[归一化参数] 真值范围: [{self.truth_min:.4f}, {self.truth_max:.4f}]")
        else:
            self.truth_min = 0.0
            self.truth_max = 100.0
    
    def normalize(self, data, data_type):
        """线性归一化到[-1, 1]（与test目录一致，不使用对数变换）

        forecast和truth都使用forecast的min/max归一化到同一空间，
        这样残差 = truth_norm - forecast_norm 是简单的加性校正。
        """
        if self.forecast_min is None or self.forecast_max is None:
            return data

        data = np.asarray(data, dtype=np.float32)
        # 统一使用forecast的min/max归一化（与test一致）
        data_clipped = np.clip(data, self.forecast_min, self.forecast_max)
        normalized = 2.0 * (data_clipped - self.forecast_min) / (self.forecast_max - self.forecast_min + 1e-8) - 1.0
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)
    
    def forecast_to_truth_space(self, forecast):
        """归一化空间转换 - 线性归一化下forecast和truth已在同一空间，直接返回"""
        return forecast

    def build_correction_target(self, forecast, truth, residual_clip_value=0.5):
        """Residual correction target in normalized truth space."""
        forecast_truth_space = self.forecast_to_truth_space(forecast)
        residual = np.clip(truth - forecast_truth_space, -residual_clip_value, residual_clip_value)
        return np.clip(residual / residual_clip_value, -1.0, 1.0)

    def denormalize(self, data, data_type):
        """线性反归一化
        normalize中forecast和truth统一用forecast的min/max归一化（保证残差空间一致），
        denormalize也统一用forecast的min/max反变换（与test一致）。
        
        距平模式下：不clip≥0（距平可以为负），由调用方在恢复物理空间后再clip。
        非距平模式下：clip≥0（降水量不能为负）。
        """
        if np.isnan(data).any():
            data = np.nan_to_num(data, nan=0.0)

        data = np.asarray(data, dtype=np.float32)

        # 统一用forecast的min/max反变换（与normalize对应）
        if self.forecast_min is not None:
            min_val = self.forecast_min
            max_val = self.forecast_max
        else:
            return np.maximum(data, 0.0)

        # 与test一致：直接反变换
        original = (data + 1.0) * 0.5 * (max_val - min_val) + min_val
        original = np.nan_to_num(original, nan=0.0)
        
        # 距平模式不clip（允许负距平），非距平模式clip≥0
        if not self.use_anomaly:
            original = np.maximum(original, 0.0)
        return original

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        sample = self.valid_samples[idx]

        # 读取预报数据：tp单独作为raw_forecast(确定性预报)，tp_2到tp_11作为forecast(10个集合成员)
        forecast_file = sample["forecast_file"]
        forecast_members = []
        forecast_units = None
        tp_data = None

        with nc.Dataset(forecast_file, "r") as f:
            # 读取tp(确定性预报)单独作为raw_forecast
            tp_var = f.variables.get("tp")
            if tp_var is not None:
                forecast_units = get_var_units(tp_var)
                if self.region == "china" and self.forecast_lat_start is not None:
                    tp_data = tp_var[:, self.forecast_lat_start:self.forecast_lat_end,
                                     self.forecast_lon_start:self.forecast_lon_end]
                else:
                    tp_data = tp_var[:]
                tp_data = np.array(tp_data, dtype=np.float32)
                if tp_data.shape[0] > FORECAST_LEAD_TIME:
                    tp_data = tp_data[:FORECAST_LEAD_TIME]
            else:
                tp_data = np.zeros((FORECAST_LEAD_TIME, len(self.forecast_lats), len(self.forecast_lons)), dtype=np.float32)
                forecast_units = ""

            # 检查预报数据的时间维度（调试）
            if "forecast_day" in f.variables:
                forecast_day_var = f.variables["forecast_day"][:]
                if not hasattr(self, '_forecast_day_printed'):
                    self._forecast_day_printed = True
                    print(f"[DEBUG] forecast_day 长度: {len(forecast_day_var)}")
                    print(f"[DEBUG] forecast_day 第一个值: {forecast_day_var[0]}")
                    print(f"[DEBUG] forecast_day 最后一个值: {forecast_day_var[-1]}")
                    print(f"[DEBUG] forecast_day 前5个值: {forecast_day_var[:5]}")
                    print(f"[DEBUG] forecast_day 后5个值: {forecast_day_var[-5:]}")

            # 读取tp_2到tp_11(10个集合成员)作为forecast
            ensemble_vars = sorted([v for v in f.variables if v.startswith('tp_')],
                                   key=lambda x: int(x.split('_')[1]))
            for var_name in ensemble_vars[:10]:  # 最多10个集合成员
                if var_name in f.variables:
                    forecast_var = f.variables[var_name]
                    if self.region == "china" and self.forecast_lat_start is not None:
                        member_data = forecast_var[:, self.forecast_lat_start:self.forecast_lat_end,
                                                   self.forecast_lon_start:self.forecast_lon_end]
                    else:
                        member_data = forecast_var[:]
                    member_data = np.array(member_data, dtype=np.float32)
                    if member_data.shape[0] > FORECAST_LEAD_TIME:
                        member_data = member_data[:FORECAST_LEAD_TIME]
                    forecast_members.append(member_data)

        # tp单位转换(确定性预报)
        tp_data = convert_accumulated_precip_to_mm(
            tp_data,
            forecast_units,
            log_prefix="[单位转换]" if not hasattr(self, '_tp_unit_converted') else "",
            var_label="tp确定性预报",
        )
        self._tp_unit_converted = True

        # 堆叠成 shape=(10, FORECAST_LEAD_TIME, lat, lon)
        forecast = np.stack(forecast_members, axis=0)

        # 打印原始预报数据（单位转换前）
        if not hasattr(self, '_forecast_data_printed'):
            self._forecast_data_printed = True
            print(f"[DEBUG] 原始forecast数据(10成员) - 均值: {np.nanmean(forecast):.6f}, 标准差: {np.nanstd(forecast):.6f}, 最小值: {np.nanmin(forecast):.6f}, 最大值: {np.nanmax(forecast):.6f}")

        forecast = convert_accumulated_precip_to_mm(
            forecast,
            forecast_units,
            log_prefix="[单位转换]" if not hasattr(self, '_forecast_unit_converted') else "",
            var_label="预报",
        )
        self._forecast_unit_converted = True
        
        # 2. 读取46天真值数据，拼接成shape=(46, lat, lon)
        truth_list = []
        # 打印预报文件信息（调试时间对齐）
        print(f"[时间对齐] 预报文件: {os.path.basename(sample['forecast_file'])}")
        print(f"[时间对齐] 起报日期: {sample['init_date']}")
        
        for i, truth_file in enumerate(sample["truth_files"]):
            truth_date = os.path.basename(truth_file).split('_')[-1].replace('.nc', '')
            # 打印真值文件信息（调试时间对齐）
            if i < 3 or i > 43:  # 只打印前3天和后3天
                print(f"[时间对齐] 第{i}天(lead={i}): {os.path.basename(truth_file)}")
            # 检查时间对齐是否正确
            if i == 0:
                expected_date = sample["init_date"]
                if truth_date != expected_date:
                    print(f"[WARN] 时间对齐可能有问题！期望第0天为 {expected_date}，实际为 {truth_date}")
            
            with nc.Dataset(truth_file, "r") as f:
                tp_var = f.variables["tp"]
                truth_units = get_var_units(tp_var)
                # 打印tp变量的维度信息，便于调试
                if i == 0:
                    print(f"[真值读取] tp变量维度: {tp_var.shape}")
                    # 打印变量的属性信息
                    print(f"[真值读取] tp变量属性:")
                    for attr in tp_var.ncattrs():
                        print(f"[真值读取]   {attr}: {tp_var.getncattr(attr)}")
                
                # 根据tp变量的维度读取数据
                if len(tp_var.shape) == 2:
                    # 直接是 (lat, lon)
                    if self.region == "china" and self.truth_lat_start is not None:
                        truth_day = tp_var[self.truth_lat_start:self.truth_lat_end,
                                         self.truth_lon_start:self.truth_lon_end]
                    else:
                        truth_day = tp_var[:, :]
                elif len(tp_var.shape) == 3:
                    # 可能是 (time, lat, lon)
                    if self.region == "china" and self.truth_lat_start is not None:
                        truth_day = tp_var[0, self.truth_lat_start:self.truth_lat_end,
                                         self.truth_lon_start:self.truth_lon_end]
                    else:
                        truth_day = tp_var[0, :, :]
                elif len(tp_var.shape) == 4:
                    # 可能是 (time, step, surface, lat, lon)
                    if self.region == "china" and self.truth_lat_start is not None:
                        truth_day = tp_var[0, 0, 0, self.truth_lat_start:self.truth_lat_end,
                                         self.truth_lon_start:self.truth_lon_end]
                    else:
                        truth_day = tp_var[0, 0, 0, :, :]
                elif len(tp_var.shape) == 5:
                    # 可能是 (time, step, surface, lat, lon)
                    if self.region == "china" and self.truth_lat_start is not None:
                        truth_day = tp_var[0, 0, 0, self.truth_lat_start:self.truth_lat_end,
                                         self.truth_lon_start:self.truth_lon_end]
                    else:
                        truth_day = tp_var[0, 0, 0, :, :]
                else:
                    # 其他维度情况，取第一个时间步
                    truth_day = tp_var[0]
                
                truth_day = np.array(truth_day, dtype=np.float32)
                truth_day = convert_accumulated_precip_to_mm(
                    truth_day,
                    truth_units,
                    log_prefix="[单位转换]" if i == 0 else "",
                    var_label="真值",
                    truth_scale_factor=truth_scale_factor,
                )
                truth_list.append(truth_day)
        truth = np.stack(truth_list, axis=0)
        
        # 打印原始truth数据（单位转换前）
        if not hasattr(self, '_truth_data_printed'):
            self._truth_data_printed = True
            print(f"[DEBUG] 原始truth数据 - 均值: {np.nanmean(truth):.6f}, 标准差: {np.nanstd(truth):.6f}, 最小值: {np.nanmin(truth):.6f}, 最大值: {np.nanmax(truth):.6f}")
        
        # 打印真值数据的统计信息，便于调试（仅首次）
        if not hasattr(self, '_first_item_processed'):
            self._first_item_processed = True
            print(f"[真值处理] 原始真值数据统计 - 均值: {np.nanmean(truth):.6f}, 标准差: {np.nanstd(truth):.6f}, 最小值: {np.nanmin(truth):.6f}, 最大值: {np.nanmax(truth):.6f}")
        
        # build_valid_samples already pairs the forecast file with the 46 target
        # days after initialization, so no additional temporal shift is applied.
        
        # 3. 处理NaN值（将NaN替换为0）
        forecast = np.nan_to_num(forecast, nan=0.0)
        tp_data = np.nan_to_num(tp_data, nan=0.0)
        truth = np.nan_to_num(truth, nan=0.0)

        # 4. 超分模式：将预报数据上采样到高分辨率网格
        if self.super_resolution:
            forecast = self._upsample_forecast_to_sr_grid(forecast)
            tp_data = self._upsample_forecast_to_sr_grid(tp_data)

        # 5. 将truth重网格到forecast坐标
        #    超分模式下：将真值插值到超分目标网格
        #    非超分模式下：将真值重网格到预报网格
        truth = self._regrid_truth_to_forecast_grid(truth)
        # 检查维度匹配：forecast是(10, 46, H, W)，truth是(46, H, W)
        if len(forecast.shape) == 4:  # 10个成员
            if truth.shape != forecast.shape[1:]:
                raise ValueError(
                    f"Regridded truth shape {truth.shape} does not match forecast shape {forecast.shape}"
                )
        else:  # 1个成员（均值）
            if truth.shape != forecast.shape:
                raise ValueError(
                    f"Regridded truth shape {truth.shape} does not match forecast shape {forecast.shape}"
                )

        # 时间错位检查：计算forecast与truth不同时间的相关系数（在插值之后执行）
        if not hasattr(self, '_time_shift_checked'):
            self._time_shift_checked = True
            if truth.shape[0] > 1:
                # forecast: (10, 46, 31, 51) 或 (46, 31, 51)
                # truth: (46, 31, 51)
                # 取forecast第一天（第一个成员或均值）
                if len(forecast.shape) == 4:  # 10个成员
                    f_day0 = forecast[0, 0].flatten()  # 第一个成员第一天
                else:  # 1个成员
                    f_day0 = forecast[0].flatten()
                
                t_day0 = truth[0].flatten()
                t_day1 = truth[1].flatten()
                
                valid0 = ~np.isnan(f_day0) & ~np.isnan(t_day0)
                valid1 = ~np.isnan(f_day0) & ~np.isnan(t_day1)
                
                if valid0.sum() > 100:
                    corr0 = np.corrcoef(f_day0[valid0], t_day0[valid0])[0, 1]
                else:
                    corr0 = np.nan
                
                if valid1.sum() > 100:
                    corr1 = np.corrcoef(f_day0[valid1], t_day1[valid1])[0, 1]
                else:
                    corr1 = np.nan
                
                print(f"[时间错位检查] forecast第0天 与 truth[0] 相关系数: {corr0:.4f}")
                print(f"[时间错位检查] forecast第0天 与 truth[1] 相关系数: {corr1:.4f}")
                
                if corr1 > corr0 + 0.1:
                    print(f"[WARN] 可能存在时间错位！forecast第0天 与 truth[1] 更相关")

        # 打印处理后的数据统计信息，便于调试（仅首次）
        if hasattr(self, '_first_item_processed') and self._first_item_processed:
            print(f"[真值处理] 处理后真值数据统计 - 均值: {np.mean(truth):.6f}, 标准差: {np.std(truth):.6f}, 最小值: {np.min(truth):.6f}, 最大值: {np.max(truth):.6f}")

        # 5. 保存原始数据副本（未经归一化，用于nc文件保存和CRPS计算，与test目录一致）
        # 红线相关：raw_forecast/raw_truth/raw_ensemble 保持原始值不变
        raw_forecast = tp_data.copy()    # (T, H, W) - tp确定性预报
        raw_ensemble = forecast.copy()   # (10, T, H, W) - 10个集合成员
        raw_truth = truth.copy()        # (46, H, W)

        # 5.5 预报裁剪：仅裁剪模型输入(forecast/tp_data)，红线raw_*不变
        clip_max = getattr(config, 'forecast_clip_max', 0.0)
        if clip_max and clip_max > 0:
            forecast = np.clip(forecast, 0.0, clip_max)
            tp_data = np.clip(tp_data, 0.0, clip_max)
            if not hasattr(self, '_clip_printed'):
                self._clip_printed = True
                print(f"[预报裁剪] 已将forecast中>{clip_max}mm的值裁剪到{clip_max}mm（红线raw_*保持不变）")

        # 6. 距平变换：输入减气候态，真值减气候态（红线raw_*不变）
        clim_truth_2d = None
        if self.use_anomaly:
            clim_fc_2d = self._get_clim_forecast(sample["init_date"])  # (H, W)
            clim_truth_2d = self._get_clim_truth(sample["init_date"])  # (H, W)
            if clim_fc_2d is not None and clim_truth_2d is not None:
                # forecast: (10, T, H, W) - clim (H,W) 广播
                forecast = forecast - clim_fc_2d[np.newaxis, np.newaxis, :, :]
                # truth: (T, H, W) - clim (H,W) 广播
                truth = truth - clim_truth_2d[np.newaxis, :, :]
                if not hasattr(self, '_anomaly_printed'):
                    self._anomaly_printed = True
                    print(f"[距平] 已减气候态: fc均值={np.mean(clim_fc_2d):.4f}, tr均值={np.mean(clim_truth_2d):.4f}")
            else:
                print(f"[距平警告] 样本 {sample['init_date']} 无气候态，跳过距平变换")

        # 7. 归一化处理（距平模式下对距平值归一化）
        forecast = self.normalize(forecast, "forecast")
        truth = self.normalize(truth, "truth")

        # 打印归一化后的数据统计信息，便于调试（仅首次）
        if hasattr(self, '_first_item_processed') and self._first_item_processed:
            print(f"[真值处理] 归一化后真值数据统计 - 均值: {np.mean(truth):.6f}, 标准差: {np.std(truth):.6f}, 最小值: {np.min(truth):.6f}, 最大值: {np.max(truth):.6f}")

        # 5. 调整维度顺序：(10, T, lat, lon) -> (T, 10, lat, lon)，时间为第一维，成员作为通道
        # forecast: (10, T, H, W) -> (T, 10, H, W)，truth: (T, H, W) -> (T, 1, H, W)
        if len(forecast.shape) == 4:  # 10个成员
            T_steps = forecast.shape[1]
            forecast = np.transpose(forecast, (1, 0, 2, 3))  # (T, 10, H, W)
            raw_ensemble = np.transpose(raw_ensemble, (1, 0, 2, 3))  # (T, 10, H, W)
            raw_forecast = np.expand_dims(raw_forecast, axis=1)  # (T, 1, H, W)
            truth = np.expand_dims(truth, axis=1)  # (T, 1, H, W)
            raw_truth = np.expand_dims(raw_truth, axis=1)  # (T, 1, H, W)
        else:  # 1个成员（均值）
            T_steps = forecast.shape[0]
            forecast = np.expand_dims(forecast, axis=1)  # (T, 1, H, W)
            raw_forecast = np.expand_dims(raw_forecast, axis=1)  # (T, 1, H, W)
            raw_ensemble = np.expand_dims(raw_ensemble, axis=1)  # (T, 1, H, W)
            truth = np.expand_dims(truth, axis=1)  # (T, 1, H, W)
            raw_truth = np.expand_dims(raw_truth, axis=1)  # (T, 1, H, W)

        # 距平模式：构建clim_truth张量 (T, 1, H, W)，用于推理时加回
        clim_truth_tensor = None
        if self.use_anomaly and clim_truth_2d is not None:
            clim_truth_arr = np.broadcast_to(
                clim_truth_2d, (T_steps, 1, clim_truth_2d.shape[0], clim_truth_2d.shape[1])
            ).astype(np.float32).copy()
            clim_truth_tensor = torch.from_numpy(clim_truth_arr)

        # 6. 转为Tensor
        forecast = torch.from_numpy(forecast)
        truth = torch.from_numpy(truth)
        raw_forecast = torch.from_numpy(raw_forecast)
        raw_truth = torch.from_numpy(raw_truth)
        raw_ensemble = torch.from_numpy(raw_ensemble)

        # 5. 处理陆地掩码
        land_mask_tensor = None
        if self.land_mask is not None:
            land_mask_tensor = torch.from_numpy(self.land_mask.astype(np.float32))

        result = {
            "forecast": forecast,
            "raw_forecast": raw_forecast,
            "raw_ensemble": raw_ensemble,
            "truth": truth,
            "raw_truth": raw_truth,
            "init_date": sample["init_date"],
            "target_dates": sample["target_dates"]
        }

        if clim_truth_tensor is not None:
            result["clim_truth"] = clim_truth_tensor
        
        if land_mask_tensor is not None:
            result["land_mask"] = land_mask_tensor
        
        return result
