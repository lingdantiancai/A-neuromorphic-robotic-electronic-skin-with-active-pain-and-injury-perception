"""
获取脉冲的位置信息
用于购买的数据采集卡的数据处理
"""

import numpy as np
import matplotlib.pyplot as plt

threshold = 0.1
def zero_base_line(data):
    sample_Rate = 200
    # time_data, pulse_data = np.loadtxt('AM_6.2k-PW_3k-PS_1000nf.txt', delimiter='\t', unpack=True, skiprows = 6)
    pulse_data = data
    pain_condition = 0
    has_greater_than_20 = any(i < -2.0 for i in pulse_data)
    if has_greater_than_20:
        pain_condition = 1
    else:
        pain_condition = 0
    time_data = np.arange(len(pulse_data)) * (1 / (sample_Rate * 1000))

    pulse_data = (pulse_data - np.average(pulse_data))*-1 # 减去基线，并且将波形调正

    # print("The based line is:",np.average(pulse_data))
    return time_data, pulse_data, pain_condition

def non_zero_base_line(data):
    sample_Rate = 200
    # time_data, pulse_data = np.loadtxt('AM_6.2k-PW_3k-PS_1000nf.txt', delimiter='\t', unpack=True, skiprows = 6)
    pulse_data = data
    time_data = np.arange(len(pulse_data)) * (1 / (sample_Rate * 1000))
    pulse_data = pulse_data*-1
    return time_data, pulse_data

def divide_pulse(data):

    time_data, pulse_data, pain_condition =zero_base_line(data)
    # time_data0, pulse_data0 = non_zero_base_line(data)

    # pain_condition = 0
    # amplitude_value = np.average(pulse_data)
    # has_greater_than_39 = any(i > 2.5 for i in pulse_data)
    # max_pulse_value = max(pulse_data)
    # min_pulse_value = min(pulse_data)
    # amplitude_value = max_pulse_value - min_pulse_value
    # if has_greater_than_39:
    #     pain_condition = 1
    # elif amplitude_value > 2.5:
    #     pain_condition = 1
    

    # if pulse_data0:
    #     max_pulse_value0 = max(pulse_data0)
    #     min_pulse_value0 = min(pulse_data0)
    #     amplitude_value0 = max_pulse_value0 - min_pulse_value0
    #     average_pulse = np.average(pulse_data)
    # # has_greater_than_39 = any(i > 3.0 for i in pulse_data)
    # # if has_greater_than_39:
    # #     pain_condition = 1
    #     if amplitude_value0 > 2.5:
    #         pain_condition = 1
    #     elif amplitude_value > 2.5:
    #         pain_condition = 1
    #     elif has_greater_than_39:
    #         pain_condition = 1
    #     elif average_pulse > 2.5:
    #         pain_condition = 1
        
        # return pain_condition
    
    try:
        # 找到所有上升沿的位置
        rising_edges = np.where((pulse_data[:-1] <= threshold) & (pulse_data[1:] > threshold))[0]
        # 找到所有下降沿的位置
        falling_edges = np.where((pulse_data[:-1] > threshold) & (pulse_data[1:] <= threshold))[0]

        # 确保第一个上升沿在第一个下降沿之前
        if falling_edges[0] < rising_edges[0]:
            falling_edges = falling_edges[1:]
        # 确保上升沿和下降沿的数量相同
        if len(rising_edges) > len(falling_edges):
            rising_edges = rising_edges[:len(falling_edges)]
    except:
        # print(f"No pulse is detected in this segment")
        return None

    # 分离脉冲，并检查每个脉冲的时间宽度
    pulses = []
    pulse_times = []
    extension = 100
    for start, end in zip(rising_edges, falling_edges):
        # 向前后各扩展50个点，以保证脉冲的完整性
        start = max(0, start-extension)
        end = min(end+extension, len(pulse_data))
        pulse_time = time_data[start:end]
        pulse = pulse_data[start:end]
        # 只保留那些时间宽度大于0.001的脉冲，剔除掉噪声
        if pulse_time[-1] - pulse_time[0] > 0.001:
            pulse_times.append(pulse_time)
            pulses.append(pulse)

    return pulse_times, pulses, pain_condition



if __name__ == '__main__':

    
    pulse_times, pulses = divide_pulse(filename)
    print(len(pulse_times))

    for pulse, pulse_time in zip(pulses, pulse_times):
        plt.figure()
        # 绘制 pulse_time vs pulse
        plt.plot(pulse_time, pulse)
        # plt.plot(time_data, Data_fft_filter)
        # 设置图形的标题和轴标签
        plt.title('Pulse over Time')
        plt.xlabel('Time')
        plt.ylabel('Pulse')
        # 显示图形
        plt.show()