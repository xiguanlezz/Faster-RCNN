# Faster-RCNN
pytorch实现的Faster-RCNN模型，**参考了许多人写的代码积累起来的**。

环境：
pytorch版本为**1.5**   &ensp;&ensp;
python版本为**python3.7**(只要是3问题不大)  &ensp;&ensp;
内存最好**32G**, 数据集的那个类用了空间换时间的思想, 本来需要频繁IO装载图片张量, 我写的是直接一次性全拉到内存, IO次数大大减少, 缩短了训练单张图片的时间。

<br>

数据集约定格式：  <br>
训练所需要的数据集格式为VOC格式。 <br>

VOCx  <br>
--Annotations <br>
&ensp;&ensp;  --*.xml(存放总训练集中各个图片的标注, 图片名和标注文件名一一对应)  <br>
--ImageSets <br>
&ensp;&ensp;  --Main  <br>
&ensp;&ensp;&ensp;&ensp;    --train.txt(总训练集中抽出一部分作为训练集的txt)  <br>
&ensp;&ensp;&ensp;&ensp;    --val.txt(总训练集中抽出一部分作为验证集的txt)  <br>
&ensp;&ensp;&ensp;&ensp;    --trainval.txt(总训练集中抽出一部分作为训练和验证集的txt)  <br>
&ensp;&ensp;&ensp;&ensp;    --train_test.txt(总训练集中抽出一部分作为测试集的txt) <br>
&ensp;&ensp;&ensp;&ensp;    --test.txt(真正测试集的txt) <br>
--JPEGImages  <br>
&ensp;&ensp;  --resize_test(真正测试集中将最小边放缩到600px之后的图片路径)  <br>
&ensp;&ensp;  --resize_train_test(总训练集中抽出一部分作为测试集中将最小边放缩到600px之后的图片路径)  <br>
&ensp;&ensp;  --resize_trainval(总训练集中抽出一部分作为训练和验证集中将最小边放缩到600px之后的图片路径) <br>
&ensp;&ensp;  --testing(原始的真正测试集图片路径) <br>
&ensp;&ensp;  --training(原始的总训练集图片路径) <br>
  
<br>
  
步骤： <br>
1、写入训练的txt文件  <br>
修改configs包下面的config文件中三个属性，如下图： <br>
![image](https://github.com/xiguanlezz/Faster-RCNN/blob/master/img_for_readme/img1.png)


然后根据自己的数据集的标注文件是怎么个形式选择执行data包下面process_data.py文件中的两个方法。  <br>

我这里写了两个方法：  <br>
&ensp;&ensp; ① 方法后缀名为_byTXT是根据txt标注生成txt文件同时生成xml标注, 这里注意要修改train_label_path为自己数据集中txt标注文件的绝对路径(process_data.py文件的最上方); <br> 
&ensp;&ensp; ② 后缀名为_byXML是根据xml标注生成txt文件。  <br>

2、修改配置文件中两个代表类名属性(一个是元组，一个是列表)以及class_num(总类别数, 而且对应类别的下标都是从1开始, 0表示背景), 都改为自己数据集中的类别即可。 <br>
![image](https://github.com/xiguanlezz/Faster-RCNN/blob/master/img_for_readme/img2.png)

3、根据机子来修改下面的两个配置, 一个是vgg16预训练权重的路径, 另一个是device(再GPU上跑还是CPU上跑)。  <br>
![image](https://github.com/xiguanlezz/Faster-RCNN/blob/master/img_for_readme/img3.png)

4、先根据下图修改配置, 再点击nets包下面的train.py文件即开始训练。  <br>
![image](https://github.com/xiguanlezz/Faster-RCNN/blob/master/img_for_readme/img4.png)


5、训练完成之后, 先根据下图修改配置, 再点击根目录下面的show_result.py文件即开始批量保存预测后的结果到show_result的文件夹下面。  <br>
![image](https://github.com/xiguanlezz/Faster-RCNN/blob/master/img_for_readme/img5.png)










  
  
