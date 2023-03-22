import pandas as pd
import numpy as np
import torch

def data_process(path_data, path_datameta):
    # 读取所有类别的x，y位置和速度数据
    data = pd.read_csv(path_data, usecols=['trackId', 'xCenter', 'yCenter', 'xVelocity', 'yVelocity'])

    # 读取所有id对应的类别
    data_class = pd.read_csv(path_datameta, usecols=['trackId', 'class'])
    data_class_car = data_class[data_class['class'] == 'car']
    # data_class_pedestrian = data_class[data_class['class'] == 'pedestrian']
    # data_class_bicycle = data_class[data_class['class'] == 'bicycle']
    # data_class_truck_bus = data_class[data_class['class'] == 'truck_bus']

    # 读取car类别的x，y位置和速度数据
    data_car = data[data['trackId'].isin(data_class_car['trackId'])]
    data_car = data_car.groupby('trackId')
    data_car_group = []
    data_car_id = []
    for data_car_i in list(data_car):
        data_car_i_id = data_car_i[1]['trackId'].reset_index(drop=True)
        data_car_i = data_car_i[1][['xCenter', 'yCenter', 'xVelocity', 'yVelocity']].reset_index(drop=True)

        # 排除静止车辆，选取起始位置为右边的车辆
        if ((data_car_i.xVelocity == 0).sum() > 100) | (data_car_i['xCenter'][0] < 60):
            continue
        else:
            data_car_group.append(data_car_i.values)
            data_car_id.append(data_car_i_id.values[0])

    return data_car_group, data_car_id

def slide_window(data_group, step_in, step_out, slide_step):
    '''
    :param data_group: 输入已根据ID分组的数据
    :param step_in: 输入步长
    :param step_out: 输出步长
    :param slide_step: 窗口滑动步长
    :return: 处理后的该组数据的输入和输出(numpy.array)
    '''
    data_group_input = []
    data_group_output = []
    for i in range(0, len(data_group) - (step_in + step_out), slide_step):
        data_group_input.append(data_group[i: i + step_in])
        data_group_output.append(data_group[i + step_in: i + step_in + step_out])
    data_group_input = np.array(data_group_input, dtype='float32')
    data_group_output = np.array(data_group_output, dtype='float32')
    return data_group_input, data_group_output

def data2tensor(data_group, data_group_id):
    tensor_all = {}
    for i in range(len(data_group)):
        tensor_all['car' + str(data_group_id[i])] = torch.tensor(data_group[i])
    return tensor_all

# data_car_group, data_car_id = data_process('10_tracks.csv', '10_tracksMeta.csv')
# data_group_input, data_group_output = slide_window(data_car_group[0], 5, 3, slide_step=3)
#
# data_input_ts = torch.from_numpy(data_group_input)
# data_output_ts = torch.from_numpy(data_group_output)
#
# print('test')


