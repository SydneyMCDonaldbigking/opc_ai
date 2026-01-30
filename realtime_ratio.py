import pandas as pd
import numpy as np
import time
import os
import sys
from collections import deque

# 尝试导入 PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    print("错误: 需要安装 PyTorch (pip install torch)")
    HAS_TORCH = False

# ==================== 配置区 ====================
CSV_FILENAME = "opc_ratio.csv"
TAG_GLRQLL = "PLC1.xt_apc.glrqll" # 输入1: 高炉煤气
TAG_ZLRQLL = "PLC1.xt_apc.zlrqll" # 输入2: 转炉煤气
TAG_ZQLL   = "PLC1.xt_apc.zqll"   # 目标: 蒸汽产量

UPDATE_INTERVAL = 1   
BUFFER_CAPACITY = 2000 # 增大缓冲区，看更远的历史
TRAIN_BATCH_SIZE = 128 # 增大 Batch，让梯度更稳定
# 噪声门控：当输入变化太小时，样本不参与训练
# 原 0.002 -> 降低到 0.0005，捕捉微小波动
DELTA_GATE_FRAC = 0.0005   
DELTA_GATE_ABS_GL = 0.2    # 降低绝对门限
DELTA_GATE_ABS_ZL = 0.2   
# 平滑去噪
ENABLE_EMA_SMOOTH = True
EMA_ALPHA = 0.2  # 0.1~0.3 越小越平滑
# 物理比例先验（软约束）
ENABLE_RATIO_PRIOR = True
RATIO_TARGET = 1.8
PHYSICS_LOSS_WEIGHT = 50.0 # 强力纠偏：既然GL波动多为噪声，必须用超强物理约束对抗统计衰减
# 按老师思路：两点解方程
USE_TWO_POINT_SOLVE = False
TWO_POINT_LOOKBACK = 800
TWO_POINT_STEP = 5
TWO_POINT_GAP = 50
DET_THRESHOLD_FRAC = 1e-4
DELTA_Y_MIN = 2.0
DELTA_X_MIN_FRAC = 0.0015
DELTA_GATE_ABS_Y = 1.0
TWO_POINT_MIN_PAIRS = 20
TWO_POINT_MAX_PAIRS = 400
# 按老师思路：找“独立变化”段来解
USE_INDEPENDENT_SEGMENTS = True
INDEPENDENT_STABLE_FRAC = 0.6
MIN_INDEPENDENT_SAMPLES = 10
# 方案A：滑动窗差分辨识
USE_SLIDING_DIFF = True
DIFF_WINDOW = 400
MIN_KEEP_DIFF = 60
# 方案B：防塌缩下限（不指定比例，只防止 k_gl 变成 0）
ENABLE_FLOOR_LOSS = True
FLOOR_RATIO_TAU = 0.08
FLOOR_LOSS_WEIGHT = 0.5
# === 初始物理滞后猜测 (秒) ===
# 稍后会由 auto_tune_lag 动态调整
DEFAULT_LAG_GL = 300
DEFAULT_LAG_ZL = 300
AUTO_TUNE_LAG = True  # 开启自动搜索，让模型自己找最佳滞后时间


# LSTM 配置
SEQ_LEN = 15           
HIDDEN_SIZE = 64
LR = 0.005             # 提速：原0.001 -> 0.005，加快适应强约束
# ================================================

if HAS_TORCH:
    class LSTMTwoInputModel(nn.Module):
        def __init__(self):
            super(LSTMTwoInputModel, self).__init__()
            # input_size=2 (高炉, 转炉)
            self.lstm = nn.LSTM(input_size=2, hidden_size=HIDDEN_SIZE, num_layers=1, batch_first=True)
            
            # 约束层: 强制权重为正的物理约束层
            self.fc_energy = nn.Sequential(
                nn.Linear(HIDDEN_SIZE, 16),
                nn.ReLU(),
                nn.Linear(16, 2), # 输出 [k_gl, k_zl]
                nn.Softplus()     # 强制 > 0
            )
            
            # === 改进1: 引入可学习的 Bias (截距) ===
            # 吸收固定底座能量，让 k 专注于解释变化量
            self.bias = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            # x: (Batch, Seq, 2)
            out, _ = self.lstm(x)
            features = out[:, -1, :] 
            
            # 系数预测
            coeffs = self.fc_energy(features) # (Batch, 2)
            k_gl = coeffs[:, 0:1]
            k_zl = coeffs[:, 1:2]
            
            # 物理公式: y = k1*x1 + k2*x2 + Bias
            current_inputs = x[:, -1, :]
            gl_in = current_inputs[:, 0:1]
            zl_in = current_inputs[:, 1:2]
            
            pred_y = k_gl * gl_in + k_zl * zl_in + self.bias
            return pred_y, k_gl, k_zl, self.bias
else:
    class LSTMTwoInputModel: pass

class LSTMRatioLearner:
    def __init__(self):
        if not HAS_TORCH: return
        self.device = torch.device("cpu")
        self.model = LSTMTwoInputModel().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR, weight_decay=1e-5) # 降低正则化：原1e-3 -> 1e-5, 允许更大的 k 值
        self.criterion = nn.MSELoss()
        
        # 归一化参数 (改为 MaxScaling，保留零点物理意义)
        self.max_in = np.ones(2) 
        self.max_out = 1.0

        self.t = 0
        self.loss_history = deque(maxlen=20)
        
        # 结果平滑器 (EMA)
        self.smooth_ratio = None
        # 移除硬编码猜测，初始化为None，第一次计算时直接赋值
        self.smooth_k_gl = None
        self.smooth_k_zl = None
        
        # 全局统计锚点 (初始假设 ZL 热值约为 GL 的 1.8 倍)
        self.base_k_gl_norm = 0.3 
        self.base_k_zl_norm = 0.54
        
        # 动态滞后参数 (初始化为默认值)
        self.lag_gl = DEFAULT_LAG_GL
        self.lag_zl = DEFAULT_LAG_ZL
        # 独立变化段计数（调试用）
        self.last_indep_gl = 0
        self.last_indep_zl = 0

    def _ema_smooth(self, arr, alpha):
        if not arr:
            return arr
        out = np.zeros(len(arr))
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
        return out

    def auto_tune_lag(self, gl_arr, zl_arr, zq_arr):
        # === 改进3: 微步滞后调整 (Micro-Adjustment) ===
        # 抛弃全局搜索，每次只允许微调 ±1s
        # 频率提高到每 5 次迭代一次，因为每次只动一点点
        if self.t % 5 != 0: return 
        
        n = len(zq_arr)
        if n < 1000: return 
        
        # 窗口
        scan_window = 900 
        g_recent = np.array(gl_arr)[-scan_window:]
        z_recent = np.array(zl_arr)[-scan_window:]
        q_recent = np.array(zq_arr)[-scan_window:]
        
        # --- GL 微调 ---
        best_corr_gl = -2 # Correlation range -1 to 1
        best_lag_gl = self.lag_gl
        
        # 只看 [lag-1, lag, lag+1]
        candidates = [self.lag_gl - 1, self.lag_gl, self.lag_gl + 1]
        
        for lag in candidates:
            if lag < 0: continue
            min_len = min(len(q_recent)-lag, len(g_recent)-lag)
            if min_len < 100: continue
            
            gs = g_recent[:min_len]
            qs = q_recent[lag:lag+min_len]
            
            # 使用增量相关性而不是绝对值相关性！
            # 增量相关性对滞后更敏感
            dgs = np.diff(gs)
            dqs = np.diff(qs)
            
            if np.std(dgs) > 1e-6 and np.std(dqs) > 1e-6:
                c_gl = np.corrcoef(dgs, dqs)[0,1]
                if c_gl > best_corr_gl:
                     best_corr_gl = c_gl
                     best_lag_gl = lag

        # --- ZL 微调 ---
        best_corr_zl = -2 
        best_lag_zl = self.lag_zl
        
        candidates_z = [self.lag_zl - 1, self.lag_zl, self.lag_zl + 1]

        for lag in candidates_z:
            if lag < 0: continue
            min_len = min(len(q_recent)-lag, len(z_recent)-lag)
            if min_len < 100: continue
            
            zs = z_recent[:min_len]
            qs = q_recent[lag:lag+min_len]

            dzs = np.diff(zs)
            dqs = np.diff(qs)

            if np.std(dzs) > 1e-6 and np.std(dqs) > 1e-6:
                c_zl = np.corrcoef(dzs, dqs)[0,1]
                if c_zl > best_corr_zl:
                     best_corr_zl = c_zl
                     best_lag_zl = lag

        # 直接更新，不需要平滑，因为每次只动1秒
        self.lag_gl = best_lag_gl
        self.lag_zl = best_lag_zl

    def update_stats(self, gl_data, zl_data, zq_data):
        # 0. 尝试自动校准滞后时间 (Micro Mode)
        if AUTO_TUNE_LAG:
            self.auto_tune_lag(gl_data, zl_data, zq_data)
    
        # 使用绝对值最大值进行归一化，保留 y=0, x=0 的物理原点
        # 这样 y = kx 的公式才成立
        if ENABLE_EMA_SMOOTH:
            gl_data = self._ema_smooth(gl_data, EMA_ALPHA)
            zl_data = self._ema_smooth(zl_data, EMA_ALPHA)
            zq_data = self._ema_smooth(zq_data, EMA_ALPHA)

        in_data = np.column_stack((gl_data, zl_data))
        current_max_in = np.max(np.abs(in_data), axis=0)
        current_max_out = np.max(np.abs(zq_data))
        
        # 缓慢更新最大值，防止波动剧烈
        self.max_in = np.maximum(self.max_in, current_max_in)
        self.max_out = max(self.max_out, current_max_out)
        
        # 2. 计算全局线性回归系数
        # 必须对齐数据！！！
        max_lag = max(self.lag_gl, self.lag_zl)
        if len(gl_data) > max_lag + 10:
            # 简单的对齐切片进行回归
            g_arr = np.array(gl_data)
            z_arr = np.array(zl_data)
            q_arr = np.array(zq_data)
            
            # 对齐: q[t] ~ g[t-lag_gl], z[t-lag_zl]

            limit = len(q_arr)
            Y = q_arr[max_lag : limit]
            
            # 切片截止点
            end_g = limit - self.lag_gl
            if self.lag_gl == 0: end_g = limit
            X_gl = g_arr[max_lag - self.lag_gl : end_g]
            
            end_z = limit - self.lag_zl
            if self.lag_zl == 0: end_z = limit
            X_zl = z_arr[max_lag - self.lag_zl : end_z]
            
            # 再次确认长度一致
            min_len = min(len(X_gl), len(X_zl), len(Y))
            g_n = X_gl[:min_len] / self.max_in[0]
            z_n = X_zl[:min_len] / self.max_in[1]
            q_n = Y[:min_len]    / self.max_out
            
            try:
                should_update_anchor = False
                if USE_TWO_POINT_SOLVE:
                    # 按老师意见：选两条方程直接解（多对取中位数，避免坏点）
                    thr_gl = max(self.max_in[0] * max(DELTA_GATE_FRAC, DELTA_X_MIN_FRAC), DELTA_GATE_ABS_GL)
                    thr_zl = max(self.max_in[1] * max(DELTA_GATE_FRAC, DELTA_X_MIN_FRAC), DELTA_GATE_ABS_ZL)
                    det_threshold = (self.max_in[0] * self.max_in[1]) * DET_THRESHOLD_FRAC

                    k1_list = []
                    k2_list = []
                    start_idx = max(0, min_len - TWO_POINT_LOOKBACK)
                    for i in range(start_idx, min_len - TWO_POINT_GAP, TWO_POINT_STEP):
                        g1, z1, y1 = X_gl[i], X_zl[i], Y[i]
                        for j in range(i + TWO_POINT_GAP, min_len, TWO_POINT_STEP * 2):
                            g2, z2, y2 = X_gl[j], X_zl[j], Y[j]
                            if abs(g1 - g2) < thr_gl and abs(z1 - z2) < thr_zl:
                                continue
                            if abs(y1 - y2) < DELTA_Y_MIN:
                                continue
                            det = g1 * z2 - g2 * z1
                            if abs(det) < det_threshold:
                                continue
                            k = np.linalg.solve(np.array([[g1, z1], [g2, z2]]), np.array([y1, y2]))
                            k1_raw, k2_raw = k[0], k[1]
                            if np.isfinite(k1_raw) and np.isfinite(k2_raw):
                                k1_list.append(k1_raw)
                                k2_list.append(k2_raw)
                                if len(k1_list) >= TWO_POINT_MAX_PAIRS:
                                    break
                        if len(k1_list) >= TWO_POINT_MAX_PAIRS:
                            break

                    if len(k1_list) < TWO_POINT_MIN_PAIRS:
                        # 不可辨识，保持原锚点
                        return

                    # 取中位数，抗异常值
                    k1_raw = float(np.median(k1_list))
                    k2_raw = float(np.median(k2_list))
                    k1_safe = max(0.001, k1_raw * (self.max_in[0] / self.max_out))
                    k2_safe = max(0.001, k2_raw * (self.max_in[1] / self.max_out))
                    should_update_anchor = True
                else:
                    # 回退：差分回归（滑动窗）
                    if USE_SLIDING_DIFF:
                        start = max(0, min_len - DIFF_WINDOW)
                        Xg_win = X_gl[start:min_len]
                        Xz_win = X_zl[start:min_len]
                        Y_win = Y[start:min_len]
                    else:
                        Xg_win = X_gl[:min_len]
                        Xz_win = X_zl[:min_len]
                        Y_win = Y[:min_len]

                    d_g = np.diff(Xg_win)
                    d_z = np.diff(Xz_win)
                    d_y = np.diff(Y_win)
                    thr_gl = max(self.max_in[0] * DELTA_GATE_FRAC, DELTA_GATE_ABS_GL)
                    thr_zl = max(self.max_in[1] * DELTA_GATE_FRAC, DELTA_GATE_ABS_ZL)
                    thr_y = max(self.max_out * DELTA_GATE_FRAC, DELTA_GATE_ABS_Y, DELTA_Y_MIN)
                    if USE_INDEPENDENT_SEGMENTS:
                        # 用“GL动、ZL稳”的段估 k_gl，用“ZL动、GL稳”的段估 k_zl
                        stable_z = np.abs(d_z) <= thr_zl * INDEPENDENT_STABLE_FRAC
                        stable_g = np.abs(d_g) <= thr_gl * INDEPENDENT_STABLE_FRAC
                        move_g = (np.abs(d_g) >= thr_gl) & (np.abs(d_y) >= thr_y)
                        move_z = (np.abs(d_z) >= thr_zl) & (np.abs(d_y) >= thr_y)

                        idx_gl = np.where(move_g & stable_z)[0]
                        idx_zl = np.where(move_z & stable_g)[0]
                        self.last_indep_gl = len(idx_gl)
                        self.last_indep_zl = len(idx_zl)

                        if len(idx_gl) >= MIN_INDEPENDENT_SAMPLES and len(idx_zl) >= MIN_INDEPENDENT_SAMPLES:
                            k1_raw = np.median(d_y[idx_gl] / (d_g[idx_gl] + 1e-9))
                            k2_raw = np.median(d_y[idx_zl] / (d_z[idx_zl] + 1e-9))
                            if np.isfinite(k1_raw) and np.isfinite(k2_raw):
                                k1_safe = max(0.001, k1_raw * (self.max_in[0] / self.max_out))
                                k2_safe = max(0.001, k2_raw * (self.max_in[1] / self.max_out))
                                should_update_anchor = True
                            else:
                                return
                        else:
                            return
                    else:
                        keep = ((np.abs(d_g) >= thr_gl) | (np.abs(d_z) >= thr_zl)) & (np.abs(d_y) >= thr_y)
                        if np.sum(keep) < MIN_KEEP_DIFF:
                            return
                        d_gn = d_g[keep] / self.max_in[0]
                        d_zn = d_z[keep] / self.max_in[1]
                        d_yn = d_y[keep] / self.max_out
                        Xd = np.vstack([d_gn, d_zn]).T
                        n_features = Xd.shape[1]
                        lambda_ridge = 0.5
                        I = np.eye(n_features)
                        XTX = Xd.T @ Xd
                        XTy = Xd.T @ d_yn
                        beta = np.linalg.inv(XTX + lambda_ridge * I) @ XTy
                        k1_safe = max(0.001, beta[0])
                        k2_safe = max(0.001, beta[1])
                        y_hat = Xd @ beta
                        mse = np.mean((y_hat - d_yn) ** 2)
                        if np.isfinite(mse) and mse < 0.02 and Xd.shape[0] >= 200:
                            should_update_anchor = True
                        else:
                            return
                
                # === 物理修正 (Physical Correction) ===
                # 如果统计出来的系数比例太离谱 (例如 k_zl / k_gl > 10)，
                # 说明数据共线性严重导致归因错误。
                # 我们强制把它们拉回到合理的物理比例附近 (k_zl_real ≈ 2 * k_gl_real)
                
                # 转换到真实空间
                w1_real = k1_safe * (self.max_out / self.max_in[0])
                w2_real = k2_safe * (self.max_out / self.max_in[1])
                
                # 检查比例 (允许 1.0 ~ 4.0 之间的波动，中心值 2.0)
                # 如果 w2/w1 太大，说明高炉被低估，转炉被高估
                if w1_real > 1e-9:
                    ratio = w2_real / w1_real
                    if ratio > 4.0: 
                        # 强行修正: 保持总能量贡献近似不变，重新分配 k
                        # Current: E = k1*g + k2*z
                        # Target:  E = k1'*g + k2'*z, subject to k2'/k1' = 2.5 (保守一点)
                        # 这是一个简化处理
                        w2_real = 2.5 * w1_real 
                        # 重新归一化回去会很麻烦，这里简单以平滑的方式拉回锚点
                        # 或者是直接限制 k2_safe 不准太大
                        pass
                
                # 这里的修正比较复杂，简单一点：
                # 如果 update_stats 算出来的结果极其不合理，就不要大幅更新锚点
                # 或者直接在这里应用 ratio 约束的 "期望值"
                
            except:
                # 极少数情况矩阵不可逆，保持原值
                k1_safe = self.base_k_gl_norm
                k2_safe = self.base_k_zl_norm

            if should_update_anchor:
                # 锚点更新: 降低信任度 0.05 -> 0.01，防止错误的数据统计带偏物理模型
                # 只有当数据统计非常确定且长期一致时，才慢慢移动锚点
                self.base_k_gl_norm = 0.99 * self.base_k_gl_norm + 0.01 * k1_safe
                self.base_k_zl_norm = 0.99 * self.base_k_zl_norm + 0.01 * k2_safe

    def train_step(self, gl_arr, zl_arr, zq_arr):
        # 确保数据够长，能覆盖最大的滞后
        max_lag = max(self.lag_gl, self.lag_zl)
        if len(gl_arr) < max_lag + SEQ_LEN + 10: return None
        
        # 对齐数据: Target(t) 对应 GL(t - lag_gl) 和 ZL(t - lag_zl)
        # 我们从 buffer 的尾部向前回溯
        # 假设 arr 最后一位是当前时刻 T
        # 我们可以使用的有效 target 范围是 [max_lag + SEQ_LEN, T]
        
        total_len = len(gl_arr)
        valid_targets_start = max_lag + SEQ_LEN
        valid_targets_end = total_len
        
        if valid_targets_end <= valid_targets_start: return None
        
        inputs = []
        targets = []
        
        # 随机采样训练
        sample_range = range(valid_targets_start, valid_targets_end)
        sample_count = min(len(sample_range), TRAIN_BATCH_SIZE)
        if sample_count <= 0: return None
        sample_idxs = np.random.choice(sample_range, size=sample_count, replace=False)
        
        # 预先转换为 numpy 方便切片
        if ENABLE_EMA_SMOOTH:
            gl_arr = self._ema_smooth(gl_arr, EMA_ALPHA)
            zl_arr = self._ema_smooth(zl_arr, EMA_ALPHA)
            zq_arr = self._ema_smooth(zq_arr, EMA_ALPHA)

        g_full_raw = np.array(gl_arr)
        z_full_raw = np.array(zl_arr)
        g_full = g_full_raw / self.max_in[0]
        z_full = z_full_raw / self.max_in[1]
        q_full = np.array(zq_arr) / self.max_out

        # 噪声门控阈值（动态量程 + 绝对门限）
        thr_gl = max(self.max_in[0] * DELTA_GATE_FRAC, DELTA_GATE_ABS_GL)
        thr_zl = max(self.max_in[1] * DELTA_GATE_FRAC, DELTA_GATE_ABS_ZL)
        
        for t in sample_idxs:
            # 构造 t 时刻的输入特征序列 (SEQ_LEN)
            # t 是 Target 的时间点
            
            # GL 序列结束点: t - lag_gl
            t_gl_end = t - self.lag_gl
            if t_gl_end < SEQ_LEN: continue 
            
            # ZL 序列结束点: t - lag_zl
            t_zl_end = t - self.lag_zl
            if t_zl_end < SEQ_LEN: continue
            
            # 噪声门控：若最近一跳变化过小，跳过该样本
            d_gl = g_full_raw[t_gl_end - 1] - g_full_raw[t_gl_end - 2]
            d_zl = z_full_raw[t_zl_end - 1] - z_full_raw[t_zl_end - 2]
            if abs(d_gl) < thr_gl and abs(d_zl) < thr_zl:
                continue

            seq_gl = g_full[t_gl_end - SEQ_LEN : t_gl_end]
            seq_zl = z_full[t_zl_end - SEQ_LEN : t_zl_end]
            
            # 拼接
            seq_xq = np.column_stack((seq_gl, seq_zl))
            
            target_val = q_full[t] # Target 就是 t 时刻的蒸汽
            target_prev = q_full[t - 1]
            
            inputs.append(seq_xq)
            targets.append((target_val, target_prev))
            
        if not inputs: return None

        t_inputs = torch.FloatTensor(np.array(inputs)).to(self.device)
        t_targets = torch.FloatTensor(np.array(targets)).to(self.device)
        
        self.model.train()
        self.optimizer.zero_grad()
        
        # Unpack: pred_y, k_gl(normalized), k_zl(normalized), bias
        pred_y, k_gl, k_zl, bias = self.model(t_inputs)
        
        y_t = t_targets[:, 0:1]
        y_prev = t_targets[:, 1:2]

        # 1. MSE Loss (总量拟合)
        loss_mse = self.criterion(pred_y, y_t)
        
        # === 改进2: 差分约束 (Differential Loss) ===
        # 强迫物理公式对“变化量”负责: Delta Y ≈ k * Delta X
        # 我们利用 t_inputs 里的时间序列信息
        # t_inputs shape: (Batch, Seq, 2)
        # 取 Seq 最后两步计算 Delta
        x_t   = t_inputs[:, -1, :]
        x_prev= t_inputs[:, -2, :]
        delta_x = x_t - x_prev # (Batch, 2)
        
        # 物理预测的差分 (Bias 被抵消)
        pred_delta_y = k_gl * delta_x[:, 0:1] + k_zl * delta_x[:, 1:2]
        true_delta_y = y_t - y_prev
        loss_diff = self.criterion(pred_delta_y, true_delta_y)
        
        # 真实差分
        # 注意: 我们随机采样时没有取 target 的前一时刻，
        # 但可以在 batch 内部近似，或者更严格地应该在采样时多采一个点。
        # 鉴于代码结构，我们利用输入序列的连续性，
        # 假设 y 的变化也是连续的。这里稍微需要 hack 一下：
        # 我们目前只有 t 时刻的 targets。
        # 为了不改动太多采样逻辑，我们弱化这个约束，只计算 "Trend Consistency"
        # 或者我们假设短时间内 k 不变，那么 pred_y(t) - pred_y(t-1) 应该等于 target(t) - target(t-1)
        # 但我们没有 target(t-1)。
        
        # --- 补丁: 既然拿不到 target(t-1)，我们用 Gradient Loss ---
        # 我们希望 pred_y 对 x 的梯度接近 k
        # 这是一个显式的物理约束。
        # 但 PyTorch 自动微分已经处理了这个。
        
        # --- 修正方案: 还是加上 Reg Loss ---
        
        # 2. 物理约束与稳定性 Loss
        
        # A. 锚点约束 (Anchor Constraint)
        # 适度信任统计学计算出的全局系数，防止偏离太远
        loss_anchor = torch.mean((k_gl - self.base_k_gl_norm)**2 + (k_zl - self.base_k_zl_norm)**2)
        
        # B. 波动约束 (Variance Constraint)
        loss_var = torch.var(k_gl) + torch.var(k_zl)
        
        # C. Bias 约束 (防止 Bias 过大吃掉所有能量)
        # 强力压制 Bias，迫使 k_gl 承担基座能量
        loss_bias = torch.mean(bias**2)

        # D. 防塌缩下限（避免 k_gl 被压到 0）
        loss_floor = 0.0
        if ENABLE_FLOOR_LOSS:
            eps = 1e-8
            ratio_floor = k_gl / (k_zl + eps)
            loss_floor = torch.mean(torch.relu(FLOOR_RATIO_TAU - ratio_floor) ** 2)

        # F. 物理比例先验 (Physical Ratio Prior)
        loss_prior = 0.0
        if ENABLE_RATIO_PRIOR:
            # 希望 k_zl / k_gl ≈ RATIO_TARGET
            # 也就意味着 k_zl ≈ RATIO_TARGET * k_gl
            # 在归一化空间需要转换一下： ratio = (k_zl/k_gl) * (max_in_g/max_in_z) 
            scale_factor = self.max_in[1] / (self.max_in[0] + 1e-5)
            target_norm_ratio = RATIO_TARGET * scale_factor
            
            # 使用 MSE 约束： k_zl_norm ≈ target * k_gl_norm
            loss_prior = torch.mean((k_zl - target_norm_ratio * k_gl)**2)

        # E. 变化率约束 (Gradient Consistency) - 新增
        # 防止 k 值突变，不仅要方差小，还要相邻时刻接近 (Smoothness)
        # 利用 batch 内的样本大概率是乱序的，这个很难在随机batch做。
        # 依靠 variance loss 已经够了。

        # 组合 Loss
        # 0.5 * loss_anchor: 锚点拉住均值
        # 5.0 * loss_bias: 大幅提高惩罚(原0.2)，强迫模型归因于输入
        loss = loss_mse + 0.5 * loss_anchor + 2.0 * loss_var + 5.0 * loss_bias + 0.5 * loss_diff + FLOOR_LOSS_WEIGHT * loss_floor + PHYSICS_LOSS_WEIGHT * loss_prior
        
        loss.backward()
        self.optimizer.step()
        
        self.loss_history.append(loss.item())
        self.t += 1
        return loss.item()

    def calculate_sensitivity_ratio(self, current_gl, current_zl):
        self.model.eval()
        
        # Max Scaling
        gl_n = current_gl / self.max_in[0]
        zl_n = current_zl / self.max_in[1]
        
        base_seq = np.zeros((1, SEQ_LEN, 2))
        base_seq[:, :, 0] = gl_n
        base_seq[:, :, 1] = zl_n
        t_base = torch.FloatTensor(base_seq).to(self.device)
        
        with torch.no_grad():
            out, _ = self.model.lstm(t_base)
            feat = out[:, -1, :]
            coeffs = self.model.fc_energy(feat).numpy()[0] # [k_gl_norm, k_zl_norm]
            bias = self.model.bias.item()
            
        k_gl_norm, k_zl_norm = coeffs[0], coeffs[1]
        
        # 还原到物理空间
        w_gl_real = k_gl_norm * (self.max_out / self.max_in[0])
        w_zl_real = k_zl_norm * (self.max_out / self.max_in[1])
        real_bias = bias * self.max_out # Bias 也要反归一化 (假设它也是针对归一化后的y)

        
        if abs(w_gl_real) < 1e-9: w_gl_real = 1e-9
        
        # === 核心改进: 指数平滑 (EMA) ===
        if self.smooth_k_gl is None:
            self.smooth_k_gl = w_gl_real
            self.smooth_k_zl = w_zl_real
            self.smooth_ratio = w_zl_real / (w_gl_real + 1e-9)
        else:
            # 调大 alpha，让它敢于跳动
            alpha = 0.3 # 70% 相信历史，30% 接受新值 (反应速度提升6倍)
            self.smooth_k_gl = self.smooth_k_gl * (1-alpha) + w_gl_real * alpha
            self.smooth_k_zl = self.smooth_k_zl * (1-alpha) + w_zl_real * alpha
            
            current_smooth_ratio = self.smooth_k_zl / (self.smooth_k_gl + 1e-9)
            if self.smooth_ratio is None:
                self.smooth_ratio = current_smooth_ratio
            self.smooth_ratio = self.smooth_ratio * (1-alpha) + current_smooth_ratio * alpha

        return self.smooth_k_gl, self.smooth_k_zl, self.smooth_ratio

    
    def predict(self, gl, zl):
        self.model.eval()
        # Max Scaling
        gl_n = gl / self.max_in[0]
        zl_n = zl / self.max_in[1]
        
        seq = np.zeros((1, SEQ_LEN, 2))
        seq[:, :, 0] = gl_n
        seq[:, :, 1] = zl_n
        
        with torch.no_grad():
            # forward 返回 (pred_y, k_gl, k_zl)，只取第一个
            pred_tuple = self.model(torch.FloatTensor(seq))
            pred_norm = pred_tuple[0].item()
            
        return pred_norm * self.max_out

# 历史数据池 (Ring Buffer)
dq_gl = deque(maxlen=BUFFER_CAPACITY)
dq_zl = deque(maxlen=BUFFER_CAPACITY)
dq_zq = deque(maxlen=BUFFER_CAPACITY)


def calculate_ratio():
    CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), CSV_FILENAME)
    
    learner = LSTMRatioLearner()
    if not HAS_TORCH: return
    
    print("=" * 60)
    print(f"🔥 LSTM 深度热值辨识系统 v3.3 (Lag-Aware) 🔥 (PyTorch内核)")
    print(f"初始滞后: GL={DEFAULT_LAG_GL}s, ZL={DEFAULT_LAG_ZL}s (固定) | 记忆长度: {SEQ_LEN}步")
    print(f"模式: 实时监听及更新 ({CSV_FILENAME})")
    print("=" * 60)

    last_read_lines = 0
    print(f"开始监听 {CSV_PATH} ...")

    while True:
        if not os.path.exists(CSV_PATH):
            print(f"\r等待文件... ", end="")
            time.sleep(2)
            continue
            
        try:
            try:
                # 实时读取：如果文件较大，需考虑优化
                df_all = pd.read_csv(CSV_PATH)
                current_lines = len(df_all)
            except Exception:
                time.sleep(0.1)
                continue
            
            if current_lines > last_read_lines:
                new_rows = df_all.iloc[last_read_lines:]
                # print(f"\r新增数据: {len(new_rows)} 条", end="")
                
                for index, last_row in new_rows.iterrows():
                    try:
                        # 获取数据
                        v_gl = pd.to_numeric(last_row.get(TAG_GLRQLL, 0), errors='coerce')
                        v_zl = pd.to_numeric(last_row.get(TAG_ZLRQLL, 0), errors='coerce')
                        v_zq = pd.to_numeric(last_row.get(TAG_ZQLL, 0), errors='coerce')

                        if v_gl > 10 and v_zq > 1: # 运行中
                            dq_gl.append(v_gl)
                            dq_zl.append(v_zl)
                            dq_zq.append(v_zq)
                        
                        # 当数据足够滞后训练时
                        max_lag_steps = max(learner.lag_gl, learner.lag_zl)
                        
                        if len(dq_gl) > max_lag_steps + SEQ_LEN + 10:
                            
                            # 1. 更新统计量
                            learner.update_stats(list(dq_gl), list(dq_zl), list(dq_zq))
                            
                            # 2. 训练一步 (从历史Buffer随机采样)
                            loss = learner.train_step(list(dq_gl), list(dq_zl), list(dq_zq))
                            
                            # 3. 提取对齐后的实时数据 (用于验证/推理)
                            idx_gl = -1 - learner.lag_gl
                            idx_zl = -1 - learner.lag_zl
                            
                            val_gl_aligned = dq_gl[idx_gl]
                            val_zl_aligned = dq_zl[idx_zl]
                            if ENABLE_EMA_SMOOTH and len(dq_gl) >= max_lag_steps + 2:
                                val_gl_aligned = (1 - EMA_ALPHA) * dq_gl[idx_gl - 1] + EMA_ALPHA * dq_gl[idx_gl]
                                val_zl_aligned = (1 - EMA_ALPHA) * dq_zl[idx_zl - 1] + EMA_ALPHA * dq_zl[idx_zl]
                            
                            # 3. 计算灵敏度 (Ratios)
                            w1, w2, ratio = learner.calculate_sensitivity_ratio(val_gl_aligned, val_zl_aligned)
                            
                            # 4. 预测验证
                            pred = learner.predict(val_gl_aligned, val_zl_aligned)
                            err_rate = abs(pred - v_zq) / (v_zq + 1e-5) * 100
                            acc = max(0, 100 - err_rate)

                            # 5. 显示
                            if index == new_rows.index[-1]: 
                                # ANSI转义: 光标归位 + 清除屏幕剩余 (无闪烁)
                                print("\033[H\033[J", end="")
                                
                                c1 = w1 * val_gl_aligned
                                c2 = w2 * val_zl_aligned
                                
                                print(f"\n[实时 Iter: {learner.t}] Loss: {loss:.5f} | Acc: {acc:.1f}%")
                                print(f"  滞后: GL={learner.lag_gl}s, ZL={learner.lag_zl}s")
                                print(f"  预测: {pred:.1f} (实{v_zq:.1f}) | 误差: {pred-v_zq:+.1f}")
                                #显示 Bias 有助于调试
                                bias_val = learner.model.bias.item() * learner.max_out
                                
                                # 计算 Ratio
                                if w1 > 1e-9:
                                     current_ratio = w2 / w1
                                else:
                                     current_ratio = 0.0
                                     
                                print(f"Ratio={current_ratio:.2f}")
                                print(f"  范围: MaxGL={learner.max_in[0]:.1f}, MaxZL={learner.max_in[1]:.1f}, MaxZQ={learner.max_out:.1f}")
                                print(f"  归一化系数: k_gl_norm={w1 / (learner.max_out/learner.max_in[0]):.4f}, k_zl_norm={w2 / (learner.max_out/learner.max_in[1]):.4f}")
                                print(f"  物理系数: k_gl={w1:.5f}, k_zl={w2:.5f}")
                                # print(f"  系数: k_gl={w1:.4f}, k_zl={w2:.4f} (Ratio={current_ratio:.2f})")
                                # print(f"  偏差: Bias={bias_val:.1f}")
                                if USE_INDEPENDENT_SEGMENTS:
                                    # 计算一下 buffer 里的波动标准差，看看是不是死数据
                                    std_gl = np.std(dq_gl) if len(dq_gl) > 0 else 0
                                    std_zl = np.std(dq_zl) if len(dq_zl) > 0 else 0
                                    print(f"  数据波动(Std): GL={std_gl:.2f}, ZL={std_zl:.2f} (有效段: {learner.last_indep_gl}/{learner.last_indep_zl})")
                                print("-" * 40)

                        else:
                            if learner.t % 100 == 0:
                               print(f"\r正在缓冲... {len(dq_gl)}/{max_lag_steps + SEQ_LEN}", end="")
                    
                    except Exception:
                        pass
                
                last_read_lines = current_lines
            
            time.sleep(UPDATE_INTERVAL)
            
        except KeyboardInterrupt:
            print("\n用户中断")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)


if __name__ == "__main__":
    calculate_ratio()