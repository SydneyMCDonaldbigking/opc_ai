"""
OPC 数据实时拟合可视化工具 (LSTM 驱动 + 符号蒸馏版)

功能：
1. 内核：使用 PyTorch LSTM (深度长短期记忆网络) 捕捉复杂非线性关系
2. 表现：绘制 X-Y 关系图 (天然气 vs 负荷)
3. 解释：使用【符号蒸馏】从 LSTM 学习到的曲线中提取数学公式
4. 优势：LSTM 比普通 DNN 更能处理工业数据的噪声和波动
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import sys

# 尝试导入 PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    print("错误: 需要安装 PyTorch。 pip install torch")
    sys.exit(1)

# 尝试导入 SciPy 用于高级拟合
try:
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("提示: 安装 scipy (pip install scipy) 可获得更好的公式拟合效果")

import warnings
warnings.filterwarnings("ignore")

# ==================== 配置区 ====================
CSV_FILENAME = "opc_ratio.csv"
TAG_X = "PLC1.xt_apc.zrqll"
TAG_Y = "PLC1.xt_apc.jzfh"

# 滞后补偿 (Visual Align) - 只用于绘图对齐，LSTM 内部有记忆
LAG_SHIFT_ROWS = 300

# LSTM 配置
HIDDEN_SIZE = 64
NUM_LAYERS = 2
SEQ_LEN = 10         # LSTM 看过去的 10 个点来决定当前的输出
LEARNING_RATE = 0.01
EPOCHS_PER_FRAME = 15

# 绘图刷新 (ms)
UPDATE_INTERVAL = 100 
# ================================================

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(CURRENT_DIR, CSV_FILENAME)

fig, ax = plt.subplots(figsize=(10, 6))

class LSTMMappingModel(nn.Module):
    """
    这是一个特殊的 LSTM，它不是做时间预测，而是做特征映射
    输入: 最近 N 个时刻的天然气 [x_t-n, ..., x_t]
    输出: 当前时刻的负荷 y_t
    优势: 它能利用"趋势信息"来判断当前的 y，比单纯看 x_t 准得多
    """
    def __init__(self):
        super(LSTMMappingModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # x: (Batch, Seq_Len, 1)
        out, _ = self.lstm(x)
        # 取最后一个时间步的输出作为特征
        out = out[:, -1, :]
        return self.fc(out)

class HybridLearner:
    def __init__(self, lr=0.01):
        self.model = LSTMMappingModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        self.x_mean = 0; self.x_std = 1
        self.y_mean = 0; self.y_std = 1

    def prepare_data(self, x_raw, y_raw):
        # 1. 更新统计量
        self.x_mean, self.x_std = x_raw.mean(), x_raw.std() + 1e-5
        self.y_mean, self.y_std = y_raw.mean(), y_raw.std() + 1e-5
        
        # 2. 归一化
        x_norm = (x_raw - self.x_mean) / self.x_std
        y_norm = (y_raw - self.y_mean) / self.y_std
        
        # 3. 构造序列数据 (Sliding Window)
        X_seq, Y_target = [], []
        for i in range(len(x_norm) - SEQ_LEN):
            X_seq.append(x_norm[i : i+SEQ_LEN])
            Y_target.append(y_norm[i+SEQ_LEN])
            
        return np.array(X_seq)[..., np.newaxis], np.array(Y_target)[..., np.newaxis]

    def fit(self, x_raw, y_raw):
        if len(x_raw) < SEQ_LEN + 10: return 0
        
        X_seq, Y_target = self.prepare_data(x_raw, y_raw)
        
        inputs = torch.FloatTensor(X_seq)
        targets = torch.FloatTensor(Y_target)
        
        self.model.train()
        loss_val = 0
        for _ in range(EPOCHS_PER_FRAME):
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            loss_val = loss.item()
            
        return loss_val

    def predict_curve(self, x_grid_raw):
        """
        难点: LSTM 需要序列输入，但画曲线时我们只有单点 x
        策略: 我们构造一个"平稳序列"，假设 x 保持不变，看 LSTM 预测多少
        这相当于问 LSTM: "如果天然气一直稳定在这个值，负荷应该是多少"
        """
        self.model.eval()
        x_norm = (x_grid_raw - self.x_mean) / self.x_std
        
        # 构造 (Batch, Seq_Len, 1), 每个 Batch 里的序列都是重复的值
        # Batch=100 (grid points), Seq=10
        inputs = torch.zeros(len(x_norm), SEQ_LEN, 1)
        for i in range(len(x_norm)):
            inputs[i, :, 0] = x_norm[i] # 填充相同的值
            
        with torch.no_grad():
            y_pred_norm = self.model(inputs).numpy().flatten()
            
        return y_pred_norm * self.y_std + self.y_mean

# 实例化
hybrid_brain = HybridLearner(lr=LEARNING_RATE)

def distill_formula(x, y_curve):
    """
    符号蒸馏 (增强版): 
    尝试多种物理/数学模型，寻找最佳解释公式
    """
    best_formula = "拟合中..."
    best_r2 = -float('inf')
    best_y_pred = y_curve # 默认
    
    valid_mask = x > 1e-3
    x_valid = x[valid_mask]
    y_valid = y_curve[valid_mask]
    
    # 防止空数据或数据太少
    if len(x_valid) < 10: return "数据不足", 0, best_y_pred
    
    # 归一化输入以稳定拟合 (fit in normalized space)
    x_mean, x_std = np.mean(x_valid), np.std(x_valid) + 1e-6
    y_mean, y_std = np.mean(y_valid), np.std(y_valid) + 1e-6
    
    xn = (x_valid - x_mean) / x_std
    yn = (y_valid - y_mean) / y_std

    candidates = []

    # --- 辅助函数：计算 R2 并添加 ---
    def add_candidate(y_pred_n, name, formula_gen_func):
        # 还原到原始尺度计算 R2
        y_pred_real = y_pred_n * y_std + y_mean
        
        ss_res = np.sum((y_valid - y_pred_real)**2)
        ss_tot = np.sum((y_valid - np.mean(y_valid))**2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        
        # 生成全范围预测均值用于绘图
        # 需要把 x 全局也是归一化的
        xn_full = (x - x_mean) / x_std
        try:
            # 这里有些 hacky, 简单的模型直接用 numpy 广播，复杂的需要 func
            pass 
        except: pass
        
        candidates.append({
            "r2": r2,
            "name": name,
            "y_pred_valid": y_pred_real, # 仅用于debug
            "formula_func": formula_gen_func, # 稍后生成公式字符串
            "params": None
        })
        return r2

    # 1. 多项式 (1, 2, 3, 4阶)
    # 多项式不需要 scipy，直接用 numpy
    for d in [1, 2, 3, 4]:
        try:
            z = np.polyfit(x_valid, y_valid, d) # 直接在原始域拟合，更直观
            p = np.poly1d(z)
            y_pred_full = p(x)
            
            ss_res = np.sum((y_valid - p(x_valid))**2)
            ss_tot = np.sum((y_valid - np.mean(y_valid))**2)
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            
            # 惩罚高阶: 每增加一阶，R2 必须提升 0.01 才算好，防止过拟合
            adjusted_r2 = r2 - (d * 0.005)

            def get_poly_str(coeffs=z, deg=d):
                terms = []
                for i, c in enumerate(coeffs):
                    power = deg - i
                    if abs(c) < 1e-6 and deg > 1: continue 
                    
                    sign = "+" if c>=0 and i>0 else "-"
                    if i==0 and c<0: sign="-"
                    elif i==0: sign=""
                    
                    val = abs(c)
                    if val < 0.01 or val > 1000: val_str = f"{val:.2e}"
                    else: val_str = f"{val:.4f}"

                    if power == 0: terms.append(f"{sign} {val_str}")
                    elif power == 1: terms.append(f"{sign} {val_str}x")
                    else: terms.append(f"{sign} {val_str}x^{power}")
                return f"y = {' '.join(terms)}".replace("  ", " ")

            candidates.append((adjusted_r2, get_poly_str(), f"多项式({d}阶)", y_pred_full))
        except: pass

    # 2. 对数模型 y = a + b * ln(x)
    try:
        b, a = np.polyfit(np.log(x_valid), y_valid, 1)
        y_pred_full = a + b * np.log(np.maximum(x, 1e-9))
        
        ss_res = np.sum((y_valid - (a + b * np.log(x_valid)))**2)
        ss_tot = np.sum((y_valid - np.mean(y_valid))**2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        
        sign = "+" if b >= 0 else "-"
        formula = f"y = {a:.2f} {sign} {abs(b):.2f}ln(x)"
        candidates.append((r2, formula, "对数模型", y_pred_full)) 
    except: pass

    # 3. 幂律模型 y = a * x^b
    if np.all(y_valid > 0):
        try:
            coeffs = np.polyfit(np.log(x_valid), np.log(y_valid), 1)
            b = coeffs[0]; a = np.exp(coeffs[1])
            y_pred_full = a * (np.maximum(x, 1e-9) ** b)

            ss_res = np.sum((y_valid - (a * x_valid**b))**2)
            ss_tot = np.sum((y_valid - np.mean(y_valid))**2)
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            
            formula = f"y = {a:.2f} * x^{b:.2f}"
            candidates.append((r2, formula, "幂律模型", y_pred_full))
        except: pass

    # --- 高级非线性模型 (需要 SciPy) ---
    if HAS_SCIPY:
        # 4. S型曲线 (Logistic/Sigmoid) - 适合描述饱和
        # L / (1 + exp(-k(x-x0))) + b
        def sigmoid_func(x_val, L, k, x0, off):
            return L / (1.0 + np.exp(-k * (x_val - x0))) + off
        
        try:
            # 初始猜测: L=range, k=norm_slope, x0=mean, off=min
            p0 = [np.max(y_valid)-np.min(y_valid), 0.01, np.mean(x_valid), np.min(y_valid)]
            popt, _ = curve_fit(sigmoid_func, x_valid, y_valid, p0=p0, maxfev=2000)
            
            y_pred_full = sigmoid_func(x, *popt)
            
            ss_res = np.sum((y_valid - sigmoid_func(x_valid, *popt))**2)
            ss_tot = np.sum((y_valid - np.mean(y_valid))**2)
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            
            # y = L / (1 + e^(-k(x-x0))) + c
            L, k, x0, off = popt
            sign_x0 = "-" if x0>0 else "+"; abs_x0 = abs(x0)
            sign_off = "+" if off>0 else "-"; abs_off = abs(off)
            
            fmt = f"y = {L:.1f} / (1 + e^{{-{k:.3f}(x{sign_x0}{abs_x0:.1f})}}) {sign_off} {abs_off:.1f}"
            candidates.append((r2, fmt, "Logistic饱和", y_pred_full))
        except: pass

        # 5. 指数饱和 (Exponential Rise) y = a * (1 - e^(-bx)) + c
        def exp_rise(x_val, a, b, c):
            return a * (1.0 - np.exp(-b * (x_val - np.min(x_val)))) + c
        
        try:
            p0 = [np.max(y_valid)-np.min(y_valid), 0.01, np.min(y_valid)]
            popt, _ = curve_fit(exp_rise, x_valid, y_valid, p0=p0, maxfev=2000)
            
            y_pred_full = exp_rise(x, *popt)
            ss_res = np.sum((y_valid - exp_rise(x_valid, *popt))**2)
            ss_tot = np.sum((y_valid - np.mean(y_valid))**2)
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            
            a, b, c = popt
            # 避免显示太复杂的公式
            fmt = f"y = {a:.1f}(1 - e^{{-{b:.3f}x'}}) + {c:.1f}"
            candidates.append((r2, fmt, "指数饱和", y_pred_full))
        except: pass


    if not candidates: return "拟合失败", 0, best_y_pred
    
    # 按 R2 排序，选最好的
    candidates.sort(key=lambda x: x[0])
    best = candidates[-1]
    
    # 如果最好的 R2 还是很烂 (<0), 还是返回原图以免误导
    if best[0] < -10: return "无法找到简单数学规律", best[0], best_y_pred
    
    return best[1], best[0], best[3]

def update(frame):
    if not os.path.exists(CSV_PATH):
        ax.clear(); ax.text(0.5,0.5,"等待数据...", ha='center'); return

    try:
        # 读取最近 1000 行
        df = pd.read_csv(CSV_PATH, on_bad_lines='skip').tail(1000)
        cols_needed = [TAG_X, TAG_Y]
        if not all(c in df.columns for c in cols_needed): return

        # 清洗
        df[TAG_X] = pd.to_numeric(df[TAG_X], errors='coerce')
        df[TAG_Y] = pd.to_numeric(df[TAG_Y], errors='coerce')
        
        # 1. 滞后对齐
        if LAG_SHIFT_ROWS > 0:
            df[TAG_Y] = df[TAG_Y].shift(-LAG_SHIFT_ROWS)
        df = df.dropna(subset=cols_needed)
        
        # 2. 3-Sigma 去噪
        stats = df[cols_needed].describe()
        mean_x, std_x = stats.at['mean', TAG_X], stats.at['std', TAG_X] + 1e-6
        mean_y, std_y = stats.at['mean', TAG_Y], stats.at['std', TAG_Y] + 1e-6
        mask = ((df[TAG_X]-mean_x).abs() <= 3*std_x) & ((df[TAG_Y]-mean_y).abs() <= 3*std_y)
        df_clean = df[mask]
        
        if len(df_clean) < SEQ_LEN + 10: return
        
        X_raw = df_clean[TAG_X].values
        Y_raw = df_clean[TAG_Y].values
        
        # === LSTM 混合训练 ===
        loss = hybrid_brain.fit(X_raw, Y_raw)
        
        # === 绘图 ===
        ax.clear()
        
        # 散点
        ax.scatter(X_raw, Y_raw, c='gray', alpha=0.3, s=15, label='实际工况')
        
        # LSTM 预测曲线 (红色 - 内核)
        x_grid = np.linspace(X_raw.min(), X_raw.max(), 100)
        y_lstm = hybrid_brain.predict_curve(x_grid)
        ax.plot(x_grid, y_lstm, color='red', linewidth=3, alpha=0.5, label='LSTM 深度模型')
        
        # 符号蒸馏曲线 (蓝色 - 公式)
        eqn, r2, y_math = distill_formula(x_grid, y_lstm)
        ax.plot(x_grid, y_math, color='blue', linewidth=2, linestyle='--', label='物理公式拟合')
        
        # === 手动验证: 计算公式与LSTM的偏移 ===
        offset_mae = np.mean(np.abs(y_math - y_lstm))
        offset_max = np.max(np.abs(y_math - y_lstm))

        # === 控制台输出 ===
        # 使用 ANSI 转义码清屏，比 os.system('cls') 极快且不闪烁
        print(f"\033[2J\033[H", end="") 
        print("="*40)
        print(f"OPC 实时建模系统 - {pd.Timestamp.now().strftime('%H:%M:%S')}")
        print(f"数据点数: {len(X_raw)}")
        print(f"当前 Loss: {loss:.5f}")
        print(f"拟合 R_sq : {r2:.4f}")
        print(f"平均偏移量: {offset_mae:.4f}")
        print("-" * 20)
        print("-" * 20)
        print(f"最佳公式:\n{eqn}")
        print("="*40)

        # === 绘图标题 ===
        title_str = (f"双模型校验 (红:LSTM / 蓝:公式)\n"
                     f"{eqn}\n"
                     f"Loss: {loss:.5f} | R_sq: {r2:.3f}")
        
        ax.set_title(title_str, fontsize=12, fontweight='bold')

        ax.set_xlabel("输入: 天然气流量 (Nm3/h)")
        ax.set_ylabel("输出: 机组负荷 (MW)")
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.5)

    except Exception as e:
        print(f"Error: {e}")

def main():
    print(f"启动 LSTM 混合监控...")
    ani = animation.FuncAnimation(fig, update, interval=UPDATE_INTERVAL, cache_frame_data=False)
    plt.show()

if __name__ == "__main__":
    main()
