import keras
import numpy as np
from keras.models import load_model
import csv

window_size=32
start_1=33


test_xs= []
test_x_id= []


file = open('Submission_1.csv') #数据目录存放路径
reader = csv.reader(file)
test_1=list(reader)

for i in range(1,118):
    test_x_tmp=np.load('./test/%s.npz' % test_1[i][0]) #测试数据存放路径
    test_x_tmp_1=np.array(test_x_tmp['voxel']*np.array(test_x_tmp['seg']))
    test_xs.append(test_x_tmp_1[start_1:start_1+window_size,start_1:start_1+window_size,start_1:start_1+window_size])
    test_x_id.append(test_1[i][0])

test_x=np.array(test_xs)
test_x = test_x.reshape(test_x.shape[0], window_size, window_size, window_size,1)
test_x = test_x.astype('float32') / 255



model = load_model('model')  #模型存放路径
test_y=np.array(model.predict(test_x))

result_id = test_x_id
result = []

for i in range(0,117):
    result_tmp=[result_id[i],test_y[i][0]]
    result.append(result_tmp)

f = open('Submission.csv','w')
writer = csv.writer(f,lineterminator='\n')
writer.writerow(['Id','Predicted'])
for i in range(0,117):
    writer.writerow(result[i])
f.close()