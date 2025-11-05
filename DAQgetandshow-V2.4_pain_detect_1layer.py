import ctypes
import time
import os
from enum import IntEnum
from typing import List, Tuple
import numpy as np
from scipy.signal import savgol_filter
import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib.animation as animation  
import sys  
import locate_pulse__unify_base_line as LP
import define_location as DL
import SVM_prediction as SP
import multiprocessing
import matplotlib.image as mpimg  
from queue import Empty

class DAQVoltage(IntEnum):
    Voltage5V = 6
    Voltage10V = 7


class DAQSampleRate(IntEnum):
    SampleRate100 = 100
    SampleRate500 = 500
    SampleRate1K = 1000
    SampleRate5K = 5000
    SampleRate10K = 10000
    SampleRate50K = 50000
    SampleRate100K = 100000
    SampleRate200K = 200000


class DAQADCChannel(IntEnum):
    NoChannel = 0b00000000
    AIN1 = 0b00000001
    AIN2 = 0b00000001 << 1
    AIN3 = 0b00000001 << 2
    AIN4 = 0b00000001 << 3
    AIN5 = 0b00000001 << 4
    AIN6 = 0b00000001 << 5
    AIN7 = 0b00000001 << 6
    AIN8 = 0b00000001 << 7
    AIN_ALL = 0b11111111


class DAQ122:
    """
    A class to interface with the DAQ122 data acquisition system.

    Attributes:
        dll_path (str): The path to the DAQ122 DLL.
    """

    def __init__(self, dll_path: str = None):
        #获取当前脚本的路径  
        current_file_path = __file__  
        # 获取当前脚本所在的目录  
        current_dir = os.path.dirname(current_file_path)  
        # 如果需要，可以更改工作目录  
        os.chdir(current_dir)
        #打开并配置DAQ122的数据采集文件
        dll_path='./lib/Windows/libdaq/lib/x64/libdaq-2.0.0.dll'
        self.dll = ctypes.CDLL(dll_path)
        self._setup_function_prototypes()
        self.obj = None

    def _setup_function_prototypes(self):
        # Set up function prototypes according to the actual DLL functions
        self.dll.DAQ122_New.restype = ctypes.POINTER(ctypes.c_uint32)
        self.dll.DAQ122_Delete.argtypes = [ctypes.POINTER(ctypes.c_uint32)]

        self.dll.DAQ122_InitializeDevice.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
        self.dll.DAQ122_InitializeDevice.restype = ctypes.c_bool

        self.dll.DAQ122_ConnectedDevice.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
        self.dll.DAQ122_ConnectedDevice.restype = ctypes.c_bool

        self.dll.DAQ122_ConfigureSamplingParameters.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint64,
                                                                ctypes.c_uint64]
        self.dll.DAQ122_ConfigureSamplingParameters.restype = ctypes.c_bool

        self.dll.DAQ122_ConfigADCChannel.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint64]
        self.dll.DAQ122_ConfigADCChannel.restype = ctypes.c_bool

        self.dll.DAQ122_StartCollection.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
        self.dll.DAQ122_StartCollection.restype = ctypes.c_bool

        self.dll.DAQ122_StopCollection.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
        self.dll.DAQ122_StopCollection.restype = ctypes.c_bool

        self.dll.DAQ122_TryReadData.argtypes = [ctypes.POINTER(ctypes.c_uint32),
                                                ctypes.c_uint32,  # channel
                                                ctypes.POINTER(ctypes.c_double),  # data
                                                ctypes.c_uint32,  # read size
                                                ctypes.c_uint32]  # timeout, 默认值需通过 functools.partial 或者 lambda 实现
        self.dll.DAQ122_TryReadData.restype = ctypes.c_bool

        self.dll.DAQ122_ADCDataBufferIsValid.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32]
        self.dll.DAQ122_ADCDataBufferIsValid.restype = ctypes.c_bool

    def __enter__(self) -> "DAQ122":
        self.create_device()
        if not self.initialize_device():
            raise Exception("Failed to initialize the DAQ device.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_collection()
        self.delete_device()

    def create_device(self):
        if self.obj is None:
            self.obj = self.dll.DAQ122_New()
        if not self.obj:
            raise Exception("Failed to create DAQ122 object")

    def delete_device(self):
        if self.obj:
            self.dll.DAQ122_Delete(self.obj)
            self.obj = None

    def initialize_device(self) -> bool:
        if not self.dll.DAQ122_InitializeDevice(self.obj):
            raise RuntimeError("Device initialization failed.")
        return True

    def is_connected(self) -> bool:
        if not self.dll.DAQ122_ConnectedDevice(self.obj):
            raise RuntimeError("Device connection failed.")
        return True

    def configure_sampling_parameters(self, voltage: DAQVoltage, sample_rate: DAQSampleRate) -> bool:
        if not self.dll.DAQ122_ConfigureSamplingParameters(self.obj, voltage.value, sample_rate.value):
            raise RuntimeError("Failed to configure sampling parameters.")
        return True

    def config_adc_channel(self, channel: DAQADCChannel) -> bool:
        if not self.dll.DAQ122_ConfigADCChannel(self.obj, channel.value):
            raise RuntimeError("Failed to configure ADC channel.")
        return True

    def start_collection(self):
        if not self.dll.DAQ122_StartCollection(self.obj):
            raise RuntimeError("Failed to start data collection.")

    def stop_collection(self):
        if not self.dll.DAQ122_StopCollection(self.obj):
            raise RuntimeError("Failed to stop data collection.")

    def read_data(self, read_elements_count: int = 1000, channel_number: int = 0) -> Tuple[bool, List[float]]:
        if read_elements_count > 1000:
            raise ValueError("read_elements_count must not exceed 1000.")
        data_buffer = (ctypes.c_double * read_elements_count)()
        success = self.dll.DAQ122_TryReadData(self.obj, channel_number, data_buffer, read_elements_count, 250)
        return success, list(data_buffer)

T_initial = time.time()
Time_duration_array=np.ones((8,8))
np.set_printoptions(precision=1)
Pulse_time_cache_array= np.zeros((8,8))+T_initial

def map_value(value, from_min, from_max, to_min, to_max):
    # 计算比例
    if value > 100:
        value = 50
    if value < 0:
        value = 0
    scale = (to_max - to_min) / (from_max - from_min)
    # 映射值
    return to_min + (value - from_min) * scale

def Get_DAQ_data(queue):
    with DAQ122() as daq:
        if daq.is_connected():
            print("Device is connected")

        if daq.configure_sampling_parameters(DAQVoltage.Voltage10V, DAQSampleRate.SampleRate200K):
            print("Sampling parameters configured")

        if daq.config_adc_channel(DAQADCChannel.AIN1):
            daq.start_collection()
            time.sleep(0.5)  # Wait for data to accumulate

            n = 0
            one_second_data = []
            while True:
                n += 1  
                
                success, data = daq.read_data()
                if success:
                    one_second_data.extend(data)
                    if n % 10 == 0:
                        # 一秒钟的数据采集完毕
                        result = LP.divide_pulse(one_second_data)                          
                        if result:
                            Pulse_time, Pulse_data, pain_condition = result
                            # print(f"Number of pulses detected: {len(Pulse_time)}")
                            print(f"Pain condition: {pain_condition}")

                            for pulse, pulse_time in zip(Pulse_data, Pulse_time):
                                am = DL.get_pulse_amplitude(pulse_time,pulse)
                                x,y = DL.unify_pulse(pulse_time,pulse,am)
                                y_smooth  = DL.smooth_data(y,20)#一维卷积平滑 + 滤波
                                ps = DL.get_gradient(x,y_smooth)
                                pw = DL.get_pulse_width(x,y)

                                if pw > 0.1:
                                    a,b,c = SP.prediction([[ps]],[[pw]],[[am]])
                                    #获得受压位置的坐标
                                    print(SP.convert_location_to_coordinates(a,b,c))                                    
                                    Dx,Dy=SP.convert_location_to_coordinates(a,b,c)
                                    #将压力分布阵列输出到队列queue
                                    queue.put((Pulse_time_cache_array, pain_condition))  

                        one_second_data = []  # 清空列表以便重新开始累计

def plot_data(queue):  
    global pressure_matrix  # 使用全局变量来更新压力矩阵     
    global cache_pressure_matrix
    """从队列中获取二维数据并实时显示色度图"""  
    # 创建一个图形和轴
    # # 加载Log图片（确保你有这个文件，并放在正确的路径） 
    current_file_path = __file__  
    # 获取当前脚本所在的目录  
    current_dir = os.path.dirname(current_file_path)  
    # 如果需要，可以更改工作目录  
    os.chdir(current_dir)   
    log_img_path = 'log.jpg'  # 修改为你的图片路径  
    log_img = mpimg.imread(log_img_path) 
    logo_extent = [0, 8.0, 10, 18]  # 根据你的图表和数据范围调整  
    fig, ax = plt.subplots()  
    #这里产生的随机阵列

    pressure_matrix=np.random.rand(8, 8)*255
    pressure_matrix = np.clip(pressure_matrix, 0, 255)
    cache_pressure_matrix = pressure_matrix
    # 初始化图像显示   
    im = ax.imshow(pressure_matrix, cmap='plasma', interpolation='nearest') 

    # 更新函数，每次调用时都会更新压力矩阵并重新绘制图像  
    def update(frame):  
        global pressure_matrix  # 使用全局变量来更新压力矩阵     
        global cache_pressure_matrix        

        #获取queue队列里的数据
        try:
            item = queue.get(block=False)
            Pulse_time_cache_time, pain_condition = item

            Time_duration_array = time.time()  - Pulse_time_cache_time
            pressure_matrix = (1/Time_duration_array)*200
            pressure_matrix = np.clip(pressure_matrix, 0, 127)
            pressure_matrix_np = np.array(pressure_matrix)
            # print(f"The pain condition is {pain_condition}, Data coming !!!!")
            if pain_condition == 1:
                max_position = np.unravel_index(np.argmax(pressure_matrix_np), pressure_matrix_np.shape)
                # 修改最大值位置的值，例如将其设置为255
                print("Pain detected!", max_position)
                pressure_matrix[max_position] = 255
            
            im.set_array(pressure_matrix)
            cache_pressure_matrix = Pulse_time_cache_time
        except Empty:
            # print('No data in queue.')
            Time_duration_array = time.time()  - cache_pressure_matrix
            pressure_matrix = (1/Time_duration_array)*100
            pressure_matrix = np.clip(pressure_matrix, 0, 127)  # 确保压力值在0-255之间
            im.set_array(pressure_matrix)
            # pressure_matrix = cache_pressure_matrix
            pass

        # # Apply red color to pixels with value less than 0.5
        # colored_pressure_matrix = np.zeros((3, 8, 8, 3))  # Create a color matrix
        # for (i, j), val in np.ndenumerate(pressure_matrix[0]):
        #     if val < 0.1:# Show red when less than 0.3
        #         colored_pressure_matrix[0][i, j] = [1, 0, 0]  # Red color
        #     else:
        #         gray_value = val / 255.0
        #         colored_pressure_matrix[0][i, j] = [gray_value, gray_value, gray_value]  # Gray color

        # for (i, j), val in np.ndenumerate(pressure_matrix[1]):
        #     if val < 0.1:# Show red when less than 0.3
        #         colored_pressure_matrix[1][i, j] = [1, 0, 0]  # Red color
        #     else:
        #         gray_value = val / 255.0
        #         colored_pressure_matrix[0][i, j] = [gray_value, gray_value, gray_value]  # Gray color

        # for (i, j), val in np.ndenumerate(pressure_matrix[2]):
        #     if val < 0.1:# Show red when less than 0.3
        #         colored_pressure_matrix[2][i, j] = [1, 0, 0]  # Red color
        #     else:
        #         gray_value = val / 255.0
        #         colored_pressure_matrix[0][i, j] = [gray_value, gray_value, gray_value]  # Gray color


        # colored_pressure_matrix = pressure_matrix  # Create a color matrix
        # for layer in range(3):
        #     for (i, j), val in np.ndenumerate(pressure_matrix[layer]):
        #         if val < 0.1:  # Show red when less than 0.1
        #             colored_pressure_matrix[layer][i, j] = 200  # Red color


        # im1.set_data(colored_pressure_matrix[0])  # 更新图像数据 
        # im2.set_data(colored_pressure_matrix[1])
        # im3.set_data(colored_pressure_matrix[2])

        return im,
 # 监听关闭事件  
    def on_close(event):  
        print('Closing figure... Exiting program.')  
        sys.exit()  
    
    # 绑定关闭事件  
    fig.canvas.mpl_connect('close_event', on_close)  
    # 创建动画，这里设置interval为1000毫秒（1秒）  
    ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 5), interval=1, blit=True)  
    
    # 展示图形  
    plt.show()  

# Example usage with context management
if __name__ == "__main__":

    # 创建队列  
    queue = multiprocessing.Queue()  
    
    # 创建进程  
    p1 = multiprocessing.Process(target=Get_DAQ_data, args=(queue,))  

    p2 = multiprocessing.Process(target=plot_data, args=(queue,))  
  
    # 启动进程  
    p1.start()  
    p2.start()  

    # 假设我们让这两个进程运行一段时间 10s 
    time.sleep(1200)  
    
    # 注意：这里简单通过强制终止进程来结束，实际使用中应该考虑更优雅的停止方式  
    p1.terminate()  
    p2.terminate()  

    # 关闭队列  
    queue.close()  
    queue.join_thread()

           



                
                      

