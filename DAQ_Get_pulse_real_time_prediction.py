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
         
        current_file_path = __file__  
          
        current_dir = os.path.dirname(current_file_path)  
          
        os.chdir(current_dir)
        
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



# Example usage with context management
if __name__ == "__main__":
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
            number_of_pulses = 0
            quene_data=np.zeros((3,8,8))
            while True:
                n += 1  
                
                success, data = daq.read_data()
                if success:
                    one_second_data.extend(data)
                    if n % 10 == 0:
                        result = LP.divide_pulse(one_second_data)

                        if result:
                            Pulse_time, Pulse_data = result
                            print(f"Number of pulses detected: {number_of_pulses}")

                            for pulse, pulse_time in zip(Pulse_data, Pulse_time):
                                am = DL.get_pulse_amplitude(pulse_time,pulse)
                                
                                x,y = DL.unify_pulse(pulse_time,pulse,am)
                                y_smooth  = DL.smooth_data(y,20)#一维卷积平滑 + 滤波
                                ps = DL.get_gradient(x,y_smooth)
                                ps_origin = DL.get_gradient(pulse_time,pulse)
                                pw = DL.get_pulse_width(x,y)
                                FQ = DL.get_peak_frequency(x,y)
 
                                if pw > 0.5:
                                    #print(f"am:{am:.3f}, pw:{pw:.3f}, ps:{ps:.3f}, ps_origin:{ps_origin:.3f}",DL.locate_the_pulse(am,pw,ps))
                                    a,b,c,d = SP.prediction([[ps]],[[pw]],[[am]],[[FQ]])
                                    
                                    print(SP.convert_location_to_coordinates(a,b,c,d))
                                    Dx,Dy,Da=SP.convert_location_to_coordinates(a,b,c,d)
                                    quene_data[Da-1][Dx-1][Dy-1]=pw
                                    
                            print(quene_data)



                        one_second_data = []  



                

                    #通过机器学习函数将波形函数更新到一个8*8的阵列 Pressure_map=f(PS,PW,AM）

                    # 监听关闭事件  
                    # def on_close(event):  
                    #     print('Closing figure... Exiting program.')  
                    #     sys.exit()  
                    
                    # 绑定关闭事件  
                    # fig.canvas.mpl_connect('close_event', on_close)  
                    
                    # 创建动画，这里设置interval为1000毫秒（1秒）  
                    # ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 100), interval=1, blit=True)  
                    
                    # 展示图形  
                    # plt.show()  
                



                
                      

