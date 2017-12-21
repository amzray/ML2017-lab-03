import os
#定义返回指定路径path的所有文件的地址的数组
def get_path(path):
    return [os.path.join(path,f) for f in os.listdir(path)]

from PIL import Image
from pylab import *
import numpy as np
#将face文件夹里的500张图的路径存到face_pth
face_pth = get_path('/datasets/original/face')
#定义一个链表通过对face_pth的遍历将每一张图转化为24*24的灰度图并存到face链表里
face=[]
for i in face_pth:
    im = Image.open(i).convert('L')
    im=np.array(im.resize((24,24)))
    face.append(im)
    
from PIL import Image
from pylab import *
import numpy as np
#将nonface文件夹里的500张图的路径存到non_pth
non_pth = get_path('/datasets/original/nonface')
#定义一个链表通过对non_pth的遍历将每一张图转化为24*24的灰度图并存到non_face链表里
non_face=[]
for i in non_pth:
    im = Image.open(i).convert('L')
    im=np.array(im.resize((24,24)))
    non_face.append(im)

#定义一个face_f链表并对face进行遍历存储利用NPDFeature类的extract方法提取到的每个灰度图的特征
face_f=[]
for i in face:
    NPDf=NPDFeature(i)
    face_f.append((np.array(NPDf.extract())).reshape((165600,1)))
#定义一个non_face_f链表并对non_face进行遍历存储利用NPDFeature类的extract方法提取到的每个灰度图的特征
non_face_f=[]
for i in non_face:
    NPDf=NPDFeature(i)
    non_face_f.append((np.array(NPDf.extract())).reshape((165600,1)))
    
#调用AdaBoostClassifier类的静态方法save将上面提取到的人脸图的特征存到文件feature.txt中
AdaBoostClassifier.save(face_f,'feature.txt')

#调用AdaBoostClassifier类的静态方法save将上面提取到的非人脸图的特征存到文件feature.txt中
AdaBoostClassifier.save(non_face_f,'non_f.txt')

#调用AdaBoostClassifier类的静态方法load方法从上面缓存的数据中提取特征
fff=AdaBoostClassifier.load('feature.txt')
nnn=AdaBoostClassifier.load('non_f.txt')

#构造一个由300个人脸图和300个非人脸图组成的训练集
X_train=np.zeros((600,165600))
y_train=np.vstack((ones((300,1)),-ones((300,1))))
for i in range(300):
    X_train[i,:]=fff[i].reshape(165600,)
for i in range(300):
    X_train[i+300,:]=nnn[i].reshape(165600,)
#把样本和label合起来，方便下面进行随机打乱
Xy_train=np.hstack((X_train,y_train))

#将样本进行随机打乱
np.random.shuffle(Xy_train)

#将打乱的样本进行分割得到特征集和label集
X_train=Xy_train[:,0:165600]
y_train=Xy_train[:,165600].reshape(600,1)

#构造一个由200个人脸图和200个非人脸图组成的测试集
X_test=np.zeros((400,165600))
y_test=np.vstack((ones((200,1)),-ones((200,1))))
for i in range(200):
    X_test[i,:]=fff[i+300].reshape(165600,)
for i in range(200):
    X_test[i+200,:]=nnn[i+300].reshape(165600,)
    
#训练
from sklearn import tree
T = 20 #设置基分类器的个数
h_list=np.zeros((T,400)) #创建数组以储存各个基分类器的结果
alpha=np.zeros((T,1)) #创建数组以储存各个基分类器对应的Alpha
w=np.ones((600,1))/600 #初始化所有训练样本的权值
mode=tree.DecisionTreeClassifier(max_depth=2) #生成一个初始的分类方法
for i in range(T):
    mode.fit(X_train,y_train,sample_weight=w.reshape(600,)) #生成单个基分类器
    epsilon=1-mode.score(X_train,y_train) #计算分类的错误率epsilon
    if epsilon>0.5: 
        break
    alpha[i]=0.5*np.log((1-epsilon)/epsilon) #计算Alpha
    #更新权重
    w=w*np.exp(-alpha[i]*y_train*((mode.predict(X_train)).reshape(600,1)))
    Zm = np.sum(w)
    w = w/Zm
    
    h_list[i]=mode.predict(X_test)#储存基分类器在测试集上的结果
    #计算并储存各基分类器Boost的结果
y_label=(h_list*alpha).sum(axis = 0)
y_label[y_label>=0]=1
y_label[y_label<0]=-1
#计算在测试集上的准确率
right=0
for i in range(400):
    if y_label[i]==y_test[i]:
        right=right+1
print(right/400.0)

import codecs
from sklearn import metrics
#用sklearn.metrics库的classification_report()函数将预测结果写入report.txt中
f=codecs.open('/report.txt','w')
f.write(metrics.classification_report(y_test, y_label))