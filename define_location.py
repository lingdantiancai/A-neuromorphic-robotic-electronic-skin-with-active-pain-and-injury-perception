"""
获取脉冲的位置信息,脉宽，脉冲的最大值，脉冲的斜率
"""
#*******************看脉冲一个非常重要的参数，截至电压

import numpy as np
import locate_pulse__unify_base_line as lp
import os
from scipy.signal import savgol_filter
from scipy import signal

# filename = 'PS-1000nf-PW-1.8k-AM-2.7k_20240605193739_200K.txt'
threshold = 0.3
unfilted_curve = []

def locate_the_pulse(am,pw,ps,FQ):
    amplitude = am# 计算脉冲幅度的代码
    pulse_width = pw# 计算脉冲宽度的代码
    pulse_slope = ps# 计算脉冲斜率的代码
    characteristic_frequency = FQ#计算特征频率的代码
    # 根据特征频率返回不同的结果
    if 4000 <= characteristic_frequency <= 5500:
        frame = 3
    if 5500 <= characteristic_frequency <= 7000:
        frame = 2
    if characteristic_frequency >= 7000:
        frame = 1
    else:
        frame = 0
    # 根据脉冲幅度返回不同的结果
    if 0 <= amplitude <= 0.7:
        channel = 1
    elif 0.7 <= amplitude <= 1.1:
        channel = 2
    elif 1.1 <= amplitude <= 1.6:
        channel = 3
    elif 1.6 <= amplitude <= 2.2:
        channel = 4
    else:
        channel = 0  # 如果脉冲幅度不在任何一个范围内，返回0
    # 根据脉冲宽度返回不同的结果
    if 0.9 <= pulse_width <= 1.1:
        section = 1
    elif 1.2 <= pulse_width <= 2.3:
        section = 2
    elif 2.5<= pulse_width <= 3.1:
        section = 3
    elif 3.1 <= pulse_width <= 3.9:
        section = 4
    else:
        section = 0  # 如果脉冲幅度不在任何一个范围内，返回0
    # 根据脉冲斜率返回不同的结果
        # 根据脉冲宽度返回不同的结果
    if 14100 <= pulse_slope <= 20000:
        area = 1
    elif 11000 <= pulse_slope <= 14100:
        area = 2
    elif 8400<= pulse_slope <= 11000:
        area = 3
    elif 3000 <= pulse_slope <= 8400:
        area = 4
    else:
        area = 0  # 如果脉冲幅度不在任何一个范围内，返回0
    return  area, section, channel, frame 

def get_pulse_width(x, y):
    # 输入一个完整的脉冲，返回脉冲的脉宽
    start = None
    y = np.abs(y)
    # 找到脉冲开始和结束的位置
    start = np.where(y > threshold)[0][0]
    try:
        end = np.where(y[start:] <= threshold)[0][0] + start
    except IndexError:
        end = start
    
    return (x[end] - x[start])*1000   #返回脉宽的时间,单位是ms


def get_peak_frequency(x,y):
    count = 0
    start = None
    start_real = None
    y = np.abs(y)
    average = np.mean(y)
    # 找到脉冲开始和结束的位置
    start = np.where(y > 0.6*average)[0][0]
    try:
        end = np.where(y[start:] <= 0.4*average)[0][0] + start
    except IndexError:
        end = start 
    # for i in range(start,end+1):
    #     b,a = signal.butter(2, [4000/100000,11000/100000], btype = 'band')#高通滤波
    #     y = signal.filtfilt(b,a,y)
    b,a = signal.butter(2, [4000/100000,11000/100000], btype = 'band')#高通滤波
    for i in range(start,end+1):
        unfilted_curve.append(y[i])
    padlen = min(len(unfilted_curve) - 1, 3 * max(len(b), len(a)))
    filted_curve = signal.filtfilt(b, a, unfilted_curve, padlen=padlen)        
    # filted_curve = signal.filtfilt(b,a,unfilted_curve, padlen=len(unfilted_curve)-1) 
    filted_x = x[start:end]
    # final_curve = y[:start].tolist() + filted_curve.tolist() + y[end:].tolist()
    # final_curve = np.array(final_curve)
    # average = np.mean(filted_curve) 


    # length = len(y) 
    # y_f = y[int(0.3*length):int(0.6*length)]
    # aver =  np.mean(y[int(0.3*length):int(0.6*length)])
    # filted_x =  filted_x[int(0.3*length):int(0.6*length)]   


    for i in range (len(filted_curve)-1):
        if filted_curve[i] <= 0 and filted_curve[i+1] > 0:
            count += 1
            # if count == 2:
            #     start_real = filted_curve[i]
    if count <= 2:
        count = 0
    # else:
    #     count = count - 1
    #     start = int(start_real)
    #     filted_x = filted_x[start:]
    #     filted_curve = filted_curve[start:]



    # edges = []
    # start1 = None
    # end1 = None
    # for i in range(len(y_f)):
    #     if y_f[i] < 0:    #如果为负值，把翻转过来
    #         y_f[i] = y_f[i]*-1
            
    #     if y_f[i] > aver and start1 is None:
    #         # 脉冲开始
    #         start1 = x[i]
    #         #print("Got one")
    #     elif y_f[i] <= aver and start1 is not None:
    #         # 脉冲结束
    #         end1 = x[i]
    #         edges.append((start1, end1))
            
    #         start1 = None
    #         end1 = None
    # print("Average_f: ",len(edges)/(filted_x[-1]-filted_x[0]))
    # for i in range(start,end+1):
    #     if y[i] <= 0.05 and y[i+1] > 0.05:
    #         count += 1 

    # count = count - 2
    peak_frequency = count/(x[end] - x[start])
    # count = 0
    unfilted_curve.clear()
    return peak_frequency
    # return peak_frequency, count, filted_x, filted_curve


 
# def get_peak_frequency(x,y):
#     start = None
#     y = np.abs(y)
#     # 找到脉冲开始和结束的位置
#     start = np.where(y > threshold)[0][0]
#     try:
#         end = np.where(y[start:] <= threshold)[0][0] + start
#     except IndexError:
#         end = start
#     b,a = signal.butter(8,0.1,'highpass')#高通滤波
#     y1 = signal.filtfilt(b,a,y)
#     for i in range(len(y)):
#         if i in range(start,end+1):
#             y[i] = y1[i]
#     return y

def get_pulse_amplitude(x,y):
    pulse = np.array(y)
    return pulse.max() #返回脉冲的最大值

def get_gradient(x,y):
    
    dy = np.gradient(y,x)
    return dy.max()    

def unify_pulse(x,y,amplitude):
    if 0 <= amplitude <= 0.7:
        channel = 6.2
    elif 0.7 <= amplitude <= 1.1:
        channel = 3.9
    elif 1.1 <= amplitude <= 1.6:
        channel = 2.7
    elif 1.6 <= amplitude <= 2.2:
        channel = 2
    y = y*(2/amplitude)  #将脉冲的幅度统一到2V
    # y[y < 0.5] *= (amplitude/2)  # 将小于0.5的值乘以0.5
    return x,y

def smooth_and_derivative(data, window_size, poly_order):
    # 使用Savitzky-Golay滤波器来平滑数据
    smoothed_data = savgol_filter(data, window_size, poly_order)
    # 使用Savitzky-Golay滤波器来计算一阶导数
    derivative = savgol_filter(data, window_size, poly_order, deriv=1)
    return smoothed_data, derivative

def smooth_data(data, window_size):
    # 创建一个均匀窗口
    window = np.ones(window_size) / window_size
    # 使用np.convolve函数来平滑数据
    smoothed_data = np.convolve(data, window, mode='same')

    smoothed_data, derivative = smooth_and_derivative(smoothed_data, 50, 2)  # 使用窗口大小为5，多项式阶数为2的Savitzky-Golay滤波器来平滑数据并计算导数
    return smoothed_data
#*******************输入数据****************************************
location = {
    'section': '',
    'area': '',
    'channel': '',
}

if __name__ == '__main__':
    filenames = [filename for filename in os. listdir('./') if filename.endswith('.txt')]

    for filename in filenames:
        pulse_times, pulses = lp.divide_pulse(filename)
        print(filename)
        for pulse, pulse_time in zip(pulses, pulse_times):

            am = get_pulse_amplitude(pulse_time,pulse)
            x,y = unify_pulse(pulse_time,pulse,am)
            pw = get_pulse_width(x,y)
            ps = get_gradient(x,y)
            FQ = get_peak_frequency(x,y)
            print(f"ps:{ps}, pw:{pw}, am:{am}, FQ:{FQ}",locate_the_pulse(am,pw,ps,FQ))




    