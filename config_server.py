import os

# training config
n_step = 5000  # 训练步数
scheduler_checkpoint_step = 3000  # 学习率调度起点延迟到3000步，前3000步保持高lr
log_checkpoint_step = 5000  # 步数结束时保存结果
gradient_accumulate_every = 4  # 梯度累积

train_lr = 1e-4  # 学习率提高到1e-4，加速收敛
lr = train_lr  # 保持向后兼容
decay = 0.9  # 保持衰减率
train_minf = 0.2
minf = train_minf  # 保持向后兼容
ema_decay = 0.995  # 保持EMA平滑度
optimizer = "adamw"  # 使用AdamW优化器
ema_step = 10  # 保持EMA更新频率
ema_start_step = 500  # 提前开始EMA

# load
load_model = False
load_step = False

# ===== Step 2: SimpleUNet + PIMD 配置 =====
# 架构：4层UNet编码器 + PIMD多分支解码器
# 输入：10通道 ensemble forecast（tp_2-tp_11）
# PIMD创新：3个专精分支（轻雨/中雨/强雨）+ 温度缩放软门控融合
# 注意：保持训练端纯MSE，熵正则损失在deterministic_correction.py中添加

# model config (SimpleUNet + PIMD)
loss_type = "l2"  # MSE纯损失
huber_delta = 1.0  # 保持（未使用）

context_dim_factor = 1.0  # 兼容旧接口
init_num_of_frame = 1
pred_mode = "pred_true"
clip_noise = True
transform_mode = None
val_num_of_batch = 1
backbone = "simple_unet_pimd"  # 标识当前用SimpleUNet + PIMD
aux_loss = False

# ===== 模型输入：10通道完整集合成员 =====
model_input_channels = 10
use_ensemble_spread = False

# ===== PIMD配置：降水强度感知多分支解码 =====
use_pimd = True  # 启用PIMD模块
pimd_num_branches = 3  # 分支数：light(轻雨)/moderate(中雨)/heavy(强雨)
pimd_temperature = 1.0  # softmax温度系数（控制门控尖锐程度）
pimd_entropy_loss_weight = 0.01  # 熵正则损失权重（防止门控塌缩到单分支）

# ===== 残差/订正配置（SimpleUNet基线：全部关闭训练端clip/加权）=====
# 训练端完全自由学习；推理端门控在外部numpy层施加
residual_clip_value = 1.0
positive_residual_clip_value = 1.0   # 训练端不clip正残差
negative_residual_clip_value = 1.0   # 训练端不clip负残差
max_abs_residual_norm = 1.0
residual_blend_weight = 1.0

# ===== 损失：纯MSE单损失（与test目录一致），关闭所有复杂分量 =====
correction_loss_weight = 0.0
correction_target_weight = 0.0
fss_loss_weight = 0.0
use_fss_loss = False
physical_space_loss_weight = 0.0
residual_spatial_loss_weight = 0.0
heavy_precip_weight = 1.0
negative_residual_loss_weight = 1.0  # Step1：平衡正负残差，不偏向负残差

# ===== TP Loss 配置：可切换的可微TP损失（直接优化评估指标）=====
# 设置为True启用TP损失，设置为False回退到纯MSE
# 可通过环境变量 RVD_USE_TP_LOSS=0 或 RVD_USE_TP_LOSS=1 覆盖。
_use_tp_loss_env = os.environ.get("RVD_USE_TP_LOSS", "")
use_tp_loss = True  # 默认启用
if _use_tp_loss_env == "0":
    use_tp_loss = False
elif _use_tp_loss_env == "1":
    use_tp_loss = True
# 权重匹配: 归一化MSE量级≈0.001~0.01，(1-TP)量级≈0.2~0.5，所以tp_loss_weight≈0.01可使两项贡献相当
# tp_loss_weight=0.5（旧值）会让TP loss强度是MSE的10~50倍，训练失衡导致效果下降
tp_loss_weight = 0.01  # TP损失权重: total_loss = mse + tp_loss_weight * (1 - TP)
tp_loss_temperature = 10.0  # soft阈值的sigmoid温度参数（越大越接近硬阈值）
tp_loss_thresholds = [0.1, 1.0, 5.0]  # TP计算的多阈值(mm)
tp_loss_lat_min = 15.0  # TP加权的纬度下限
tp_loss_lat_max = 60.0  # TP加权的纬度上限
tp_loss_blend_with_mse = True  # True: mse + tp_loss, False: 纯tp_loss

# ===== FSS Loss 配置：可切换的可微FSS损失（直接优化FSS评估指标）=====
# 设置为True启用FSS Loss，设置为False回退到纯MSE
# 可通过环境变量 RVD_USE_FSS_LOSS=0 或 RVD_USE_FSS_LOSS=1 覆盖。
_use_fss_loss_env = os.environ.get("RVD_USE_FSS_LOSS", "")
use_fss_loss = False  # 默认不启用，需要时通过环境变量开启
if _use_fss_loss_env == "0":
    use_fss_loss = False
elif _use_fss_loss_env == "1":
    use_fss_loss = True
# FSS Loss参数：窗口大小=3，阈值=1.0mm
fss_loss_weight = 0.01  # FSS损失权重: total_loss = mse + fss_loss_weight * (1 - FSS)
fss_loss_window_size = 3  # 邻域窗口大小（奇数），用于计算FSS
fss_loss_threshold = 1.0  # FSS阈值(mm)，用于定义降水事件
fss_loss_temperature = 10.0  # soft阈值的sigmoid温度参数
fss_loss_blend_with_mse = True  # True: mse + fss_loss, False: 纯fss_loss

# 关闭所有额外干预机制，保持推理最简
preserve_strong_forecast = False
remove_negative_residual_bias = False
forecast_truth_space_clip_min = -1.0
forecast_truth_space_clip_max = 1.0
prediction_norm_clip_min = -1.0
prediction_norm_clip_max = 1.0

# 距平(Anomaly)订正配置
# 输入用EC减气候态，真值也减气候态算loss，推理时加回真值气候态
# 红线(raw_forecast)保持原始tp不变，只有蓝线(corrected)受影响
use_anomaly = True  # 是否启用距平模式
anomaly_scale = "pentad"  # 气候态尺度: "month"(月尺度) 或 "pentad"(侯尺度,5天)
anomaly_max_samples_per_group = 8  # 侯尺度每侯最多读取样本数(5天×8年=40个样本足够估计，避免OOM)

# 预报数据裁剪配置
# 将forecast中超过该值的降水裁剪到此值，用于去除极端异常值
# 仅影响模型输入(蓝线)，红线(raw_forecast/raw_ensemble)保持原始值不变
forecast_clip_max = 100.0  # 预报裁剪上限(mm)，设为0则不裁剪

# 预报时效配置（支持10天/40天等不同预报时效）
# 默认10天，可通过环境变量 RVD_FORECAST_LEAD_TIME 设置
forecast_lead_time = int(os.environ.get("RVD_FORECAST_LEAD_TIME", "10"))

# 时效动态权重配置（用于解决长时效CSI交叉问题）
# 模型在长时效推理时，校正可靠性下降，需要随时效衰减校正强度
# 格式：[(天数阈值, 权重), ...]，按天数从小到大排序
# 例如：[(10, 1.0), (20, 0.7), (40, 0.4)] 表示：
#   Day 1-10: 权重1.0（全力校正）
#   Day 11-20: 权重0.7（逐渐减弱）
#   Day 21-40: 权重0.4（保守校正）
lead_time_weights = [
    (10, 1.0),   # 前10天：全力校正（模型训练过的区间）
    (20, 0.7),   # 11-20天：逐渐减弱
    (40, 0.4),   # 21-40天：保守校正
    (100, 0.3),  # 41-100天：极弱校正（气候态区间）
]

# FSS Loss 参数配置
fss_thresholds = [0.2, 0.5, 1.0]  # 更有区分度的降水阈值
fss_scales = [5, 11]  # 中+大尺度窗口，避免过平滑

additional_note = "-pimd-branch3-entropy0.01-linear_norm-blend1.0-simple_loss-bilinear_regrid"

# 真值数据缩放因子（用于单位转换）
# 真值数据已乘以86400秒/天，转换为日累积降水量
# 所以这里设置为1.0，不再额外乘以86400
truth_scale_factor = 1.0  # 真值已经处理过，不需要再乘86400

# data config
data_config = {
    "dataset_name": "precipitation",
    # 设置 RVD_DATA_ROOT 为预处理数据根目录，目录结构见 README.md。
    "data_path": os.environ.get("RVD_DATA_ROOT", ""),
    "sequence_length": 1,  # 恢复序列长度，确保维度匹配
    "img_size": 32,  # 保持图像尺寸
    "img_channel": 10,  # 10个集合预报成员(tp_2到tp_11)
    "output_channel": 1,  # 残差输出通道数（对应真值，只有1个）
    "add_noise": True,  # 添加噪声增强
    "img_hz_flip": True,  # 水平翻转增强
    "region": "china",  # 数据区域：global 或 china
    "lat_range": (15, 55),  # 中国区域纬度范围
    "lon_range": (70, 140),  # 中国区域经度范围
    "use_land_mask": True,  # 是否使用陆地掩码（处理海洋NaN值）
    "super_resolution": False,  # 是否启用超分模式
    "target_resolution": (0.25, 0.25),  # 超分目标分辨率 (lat_step, lon_step)
}

# Stage 2: 2-D Conv-ResNet regression prior + conditional denoising diffusion.
# The target shape is read from the native CRA grid; no hard-coded 3x factor is used.
_sr_project_root = os.path.abspath(os.environ.get("RVD_PROJECT_ROOT", os.path.dirname(__file__)))
_sr_result_dir = os.path.join(_sr_project_root, "results", "super_resolution")
sr_config = {
    "train_correction_file": os.path.join(_sr_result_dir, "correction_input", "train", "global_corrected.nc"),
    "test_correction_file": os.path.join(_sr_result_dir, "correction_input", "test", "global_corrected.nc"),
    "truth_dir": os.path.join(data_config["data_path"], "CRA_truth_nc"),
    "result_dir": _sr_result_dir,
    "lat_range": (15.0, 55.0),
    "lon_range": (70.0, 140.0),
    "validation_fraction": 0.1,  # only the chronological tail of the pre-2022 data
    "normalization_max_mm": 250.0,
    "batch_size": 4,
    "evaluation_batch_size": 2,
    "num_workers": 4,
    "regression_channels": 64,
    "regression_blocks": 8,
    "regression_residual_scale": 0.5,
    "diffusion_channels": 48,
    "diffusion_residual_clip": 4.0,  # clip in training-residual standard deviations
    # Phase 2 learns all fine scales plus a controlled part of the broad residual.
    # Strong residuals may use more coarse correction to recover intense rain cores.
    "coarse_residual_fraction": 0.35,
    "coarse_residual_max_fraction": 0.80,
    "phase2_gate_min": 0.10,
    "phase2_gate_max": 0.95,
    "timesteps": 100,
    "ddim_sampling_steps": 25,
    "reuse_phase1": True,
    "phase1_epochs": 40,
    "phase2_epochs": 40,
    "phase1_lr": 2e-4,
    "phase2_lr": 1e-4,
    "diffusion_val_batches": 32,
    "end_to_end_val_samples": 64,
    "residual_stat_batches": 512,
    "phase2_patience": 8,
    "sr_x0_loss_weight": 1.0,
    "sr_reconstruction_loss_weight": 4.0,
    "sr_consistency_loss_weight": 0.1,
    "sr_gradient_loss_weight": 0.15,
    "sr_structure_loss_weight": 0.8,
    "sr_heavy_rain_weight": 6.0,
    "sr_fss_loss_weight": 1.0,
    "sr_gate_loss_weight": 0.5,
    "sampling_seed": 2026,
    "max_visualizations": 20,
}

# ===== Step 1: SimpleUNet Baseline 的 embed_dim 和 dim_mults（硬编码）=====
# SimpleUNet 4层UNet深度=4（与test目录一致：64→128→256→512→512）
# 下采样倍数 = 2^4 = 16，保证中国区32×32/全球121×240等奇数尺寸也能被padding后正确对齐
embed_dim = 64            # SimpleUNet base_dim = 64
dim_mults = (1, 2, 4, 8)  # 4层，每层通道倍增（与simple_unet_backbone.py完全对齐）
batch_size = 1

model_name = f"{backbone}-{optimizer}-{pred_mode}-{loss_type}-{data_config['dataset_name']}-d{embed_dim}-deterministic-{transform_mode}-al{aux_loss}{additional_note}"

# ===== 结果目录统一管理：所有输出（模型/NC/可视化/TensorBoard）都放在results下 =====
# 原问题：tensorboard_logs单独在项目根、global_correction_results单独在根，造成"可视化结果不在results文件夹"
# 新结构：
#   result_root/<run_number>/
#       ├── <model_name>/              模型checkpoint + training_loss.png + nc_results/（trainer保存）
#       ├── tensorboard_logs/          TensorBoard日志（统一收敛到results下）
#       └── global_correction_results/ infer_global推理NC + 后续plot_metrics图（对齐test目录结构）
# 从当前代码文件位置计算工程根目录。这样工程改名、移动或复制后，结果仍会
# 写入这份工程自身的 results/，而不会在旧的硬编码路径另建同名工程目录。
project_root = os.path.abspath(os.environ.get("RVD_PROJECT_ROOT", os.path.dirname(__file__)))
result_root = os.path.join(project_root, "results")

# 创建自动编号的结果目录
def get_next_run_number(root_dir):
    """获取下一个运行编号"""
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
        return 1

    # 查找现有编号目录
    run_numbers = []
    for item in os.listdir(root_dir):
        if item.isdigit() and os.path.isdir(os.path.join(root_dir, item)):
            run_numbers.append(int(item))

    if not run_numbers:
        return 1
    else:
        return max(run_numbers) + 1


def _resolve_run_number():
    """统一解析run_number：
    优先级 1) 环境变量 RVD_RUN_NUMBER（提交脚本分配，保证同一次运行共用）
    优先级 2) current_run_dir下的 .run_number 持久化文件（防环境变量丢失的第二道保险）
    优先级 3) 调用get_next_run_number()新建，并写入持久化文件
    这样同一个submit任务中，4个Python脚本不会各申请一个新的run目录！
    """
    # 1) 环境变量优先
    env_run = os.environ.get("RVD_RUN_NUMBER")
    if env_run is not None:
        try:
            rn = int(env_run)
            persist_file = os.path.join(result_root, str(rn), ".run_number")
            if not os.path.exists(persist_file):
                os.makedirs(os.path.dirname(persist_file), exist_ok=True)
                with open(persist_file, "w") as pf:
                    pf.write(str(rn))
            return rn
        except ValueError:
            pass

    # 2) 找最新存在且包含 .run_number 的 run 目录（一次任务中的后续脚本，环境变量丢失的fallback）
    if os.path.exists(result_root):
        existing_dirs = []
        for item in os.listdir(result_root):
            if item.isdigit() and os.path.isdir(os.path.join(result_root, item)):
                pf = os.path.join(result_root, item, ".run_number")
                if os.path.exists(pf):
                    existing_dirs.append(int(item))
        if existing_dirs:
            return max(existing_dirs)

    # 3) 否则新建，并持久化
    rn = get_next_run_number(result_root)
    run_dir_temp = os.path.join(result_root, str(rn))
    os.makedirs(run_dir_temp, exist_ok=True)
    persist_file = os.path.join(run_dir_temp, ".run_number")
    with open(persist_file, "w") as pf:
        pf.write(str(rn))
    # 同步写入环境变量，后续import同一进程内不再重复分配
    os.environ["RVD_RUN_NUMBER"] = str(rn)
    return rn


# 获取当前运行编号（同一个submit任务内全部脚本共用同一个run_number）
run_number = _resolve_run_number()
# 允许通过 RVD_RESULT_SUBDIR 环境变量指定子目录（用于基线模型训练，避免创建独立的run_number目录）
# 例: RVD_RESULT_SUBDIR=baseline → current_run_dir = results/<run>/baseline/
_subdir = os.environ.get("RVD_RESULT_SUBDIR", "").strip()
if _subdir:
    current_run_dir = os.path.join(result_root, str(run_number), _subdir)
else:
    current_run_dir = os.path.join(result_root, str(run_number))

# ==================== 简洁的目录结构 ====================
# 目标：results/<run_number>/
#   ├── models/                    ← 模型checkpoint
#   ├── tensorboard_logs/          ← TensorBoard日志
#   ├── nc_china/                  ← 中国区域推理NC（原 global_correction_results）
#   ├── nc_global/                 ← 全球区域推理NC（原 global_correction_results_global）
#   └── plots/                     ← 所有可视化图表（集中一处，不再分散！）
# =================================================================
tensorboard_run_dir = os.path.join(current_run_dir, "tensorboard_logs")
global_output_dir = os.path.join(current_run_dir, "nc_china")        # 中国区域 → nc_china/
plots_dir = os.path.join(current_run_dir, "plots")                  # 所有图 → plots/

# 确保目录存在
os.makedirs(current_run_dir, exist_ok=True)
os.makedirs(tensorboard_run_dir, exist_ok=True)
os.makedirs(global_output_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# 【关键】所有配置信息输出到 stderr，而非 stdout
# 原因：提交脚本使用 $(python -c "import config_server; print(...)") 捕获 stdout 获取 run_number
# 如果配置信息也输出到 stdout，会污染捕获结果导致 bash 正则匹配失败
import sys as _sys
print(f"当前运行编号: {run_number}", file=_sys.stderr)
print(f"结果根目录: {current_run_dir}", file=_sys.stderr)
print(f"  ├─ 模型:           {current_run_dir}/{model_name}/  (后续迁移到 models/)", file=_sys.stderr)
print(f"  ├─ TensorBoard:    {tensorboard_run_dir}", file=_sys.stderr)
print(f"  ├─ 中国区域NC:     {global_output_dir}", file=_sys.stderr)
print(f"  ├─ 全球区域NC:     {current_run_dir}/nc_global/", file=_sys.stderr)
print(f"  └─ 可视化图表:     {plots_dir}  ← 所有PNG/CSV集中于此，不再分散！", file=_sys.stderr)
