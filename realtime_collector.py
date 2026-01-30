import win32com.client
import time
import csv
from datetime import datetime
import os

# ==================== 配置区 ====================
OPC_SERVER = "CoDeSys.OPC.DA"
SAMPLING_INTERVAL = 1
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(CURRENT_DIR, "opc_ratio.csv")

# 滚动覆盖配置
MAX_ROWS = 7200         # 最大保留行数（约2小时数据）
CLEANUP_THRESHOLD = 1.2 # 当行数超过 MAX_ROWS 的 1.2 倍时触发清理
KEEP_RATIO = 0.7        # 清理时保留最新 70% 的数据

TAGS = [
    "PLC1.xt_apc.zrqll",
    "PLC1.xt_apc.jzfh",
    "PLC1.xt_apc.glrqll",
    "PLC1.xt_apc.zlrqll",
    "PLC1.xt_apc.zqll",
]
# ================================================

class OPCDataCollector:
    def __init__(self, server_name, tags):
        self.server_name = server_name
        self.tags = tags
        self.opc = None
        self.group = None
        self.items = {}
        
    def connect(self):
        print("正在连接到 OPC 服务器...")
        self.opc = win32com.client.Dispatch('OPC.Automation')
        self.opc.Connect(self.server_name)
        groups = self.opc.OPCGroups
        self.group = groups.Add("DataCollectorGroup")
        self.group.IsActive = True
        self.group.IsSubscribed = True
        self.group.UpdateRate = 1000
        print(f"✓ 已连接并创建组")
        
    def add_tags(self):
        print(f"\n正在添加标签...")
        for tag_name in self.tags:
            try:
                item = self.group.OPCItems.AddItem(tag_name, 0)
                self.items[tag_name] = item
                print(f"  ✓ {tag_name}")
            except Exception as e:
                print(f"  ✗ {tag_name} - 失败: {e}")
        
    def read_all(self):
        data = {}
        timestamp = datetime.now()
        for tag_name, item in self.items.items():
            try:
                full_data = item.Read(1)
                data[tag_name] = {
                    'value': full_data[0],
                    'quality': item.Quality,
                    'timestamp': timestamp
                }
            except:
                data[tag_name] = {'value': None, 'quality': 0, 'timestamp': timestamp}
        return data
    
    def disconnect(self):
        if self.opc:
            self.opc.Disconnect()
            print("\n✓ 已断开连接")

def save_to_csv(data, filename):
    """保存数据并实现滚动覆盖"""
    file_exists = os.path.exists(filename)
    
    # 1. 正常写入数据
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        if not data: return
        fieldnames = ['时间戳'] + list(data.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        
        row = {'时间戳': list(data.values())[0]['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
        for tag, info in data.items():
            row[tag] = info['value']
        writer.writerow(row)

    # 2. 滚动清理逻辑（每100条数据检查一次，避免频繁IO）
    if not hasattr(save_to_csv, "counter"):
        save_to_csv.counter = 0
    
    save_to_csv.counter += 1
    if save_to_csv.counter >= 100:
        save_to_csv.counter = 0
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 如果行数超过阈值 (MAX_ROWS * 1.2)
            if len(lines) > MAX_ROWS * CLEANUP_THRESHOLD:
                header = lines[0]
                # 计算需要保留的行数
                keep_count = int(MAX_ROWS * KEEP_RATIO)
                new_lines = [header] + lines[-keep_count:]
                
                with open(filename, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)
                print(f"\n[滚动覆盖] 已清理旧数据，当前保留行数: {len(new_lines)}")
        except Exception as e:
            print(f"\n[清理失败] {e}")

def main():
    print("=" * 60)
    print("OPC 实时数据采集器 (滚动覆盖版)")
    print("=" * 60)
    
    collector = OPCDataCollector(OPC_SERVER, TAGS)
    
    try:
        collector.connect()
        collector.add_tags()
        
        if not collector.items: return

        print(f"\n开始采集... 滚动窗口设定: {MAX_ROWS} 行")
        count = 0
        while True:
            data = collector.read_all()
            count += 1
            
            # 简易控制台打印
            ts = datetime.now().strftime('%H:%M:%S')
            zq = data.get("PLC1.xt_apc.zqll", {}).get('value', 'N/A')
            print(f"\r[{count}] {ts} | 蒸汽流量: {zq} | 状态: 运行中...", end="")
            
            save_to_csv(data, CSV_FILE)
            time.sleep(SAMPLING_INTERVAL)
            
    except KeyboardInterrupt:
        print(f"\n停止采集。总记录: {count}")
    finally:
        collector.disconnect()

if __name__ == "__main__":
    main()