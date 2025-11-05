import joblib  # 更新的导入方式
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np

# def prediction(PS, PW, AM,FQ):
def prediction(PS, PW, AM):
    # 现在，我们将加载这个模型
    PS_Mode = joblib.load('PS_svm_model_PCB.joblib')  # 加载模型
    PW_Mode = joblib.load('PW_svm_model_PCB.joblib')  # 加载模型
    AM_Mode = joblib.load('AM_svm_model_PCB.joblib')  # 加载模型
    FQ_Mode = joblib.load('FQ_svm_model.joblib')  # 加载模型

    # 使用加载的模型进行预测
    # 假设你有一些新的数据需要进行预测
    # new_features = np.array([...])
    PS_location = PS_Mode.predict(PS)
    AM_location = AM_Mode.predict(AM)
    PW_location = PW_Mode.predict(PW)
    # FQ_location = FQ_Mode.predict(FQ)
    # return PS_location[0], PW_location[0], AM_location[0], FQ_location[0]
    return PS_location[0], PW_location[0], AM_location[0]

# def convert_location_to_coordinates(a, b, c, d):
def convert_location_to_coordinates(a, b, c):
    quadrant_x = 0
    sub_quadrant_x = 0
    position_x = 0

    quadrant_y = 0
    sub_quadrant_y = 0
    position_y = 0

    # 确定4x4区域的起始坐标
    if a == 1:
        quadrant_x = 0
        quadrant_y = 0
    if a == 2:
        quadrant_x = 1
        quadrant_y = 0
    if a == 3:
        quadrant_x = 0
        quadrant_y = 1
    if a == 4:
        quadrant_x = 1
        quadrant_y = 1
    
    # 确定2x2子区域的起始坐标
    if b == 1:
        sub_quadrant_x = 0
        sub_quadrant_y = 0
    if b == 2:
        sub_quadrant_x = 0
        sub_quadrant_y = 1
    if b == 3:
        sub_quadrant_x = 1
        sub_quadrant_y = 0
    if b == 4:
        sub_quadrant_x = 1
        sub_quadrant_y = 1      
    
    # 确定在2x2子区域内的具体位置
    if c == 1:
        position_x = 1
        position_y = 2
    if c == 2:
        position_x = 1
        position_y = 1
    if c == 3:
        position_x = 2
        position_y = 1
    if c == 4:
        position_x = 2
        position_y = 2
    
    # 计算最终坐标
    x = quadrant_x*4 + sub_quadrant_x*2 + position_x
    y = quadrant_y*4 + sub_quadrant_y*2 + position_y

    # return x, y, d #(x,y,frame)
    return x, y #(x,y)

# 打印预测结果
# print(predictions)
# if __name__ == '__main__':

#     am = 1.364
#     pw = 1.210
#     ps = 8467.269
#     FQ = 4400.15
#     print(prediction([[ps]],[[pw]],[[am]],[[FQ]]))

#     a,b,c,d = prediction([[ps]],[[pw]],[[am]],[[FQ]])

#     print(convert_location_to_coordinates(a,b,c,d))
