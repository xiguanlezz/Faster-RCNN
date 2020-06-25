# Faster-RCNN
pytorch实现的Faster-RCNN模型，参考了许多人写的代码积累起来的。

环境：
pytorch版本为1.5
python版本为python3.7(只要是3问题不大)

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

然后根据自己的数据集的标注文件是怎么个形式选择执行data包下面process_data.py文件中的两个方法。  <br>

我这里写了两个方法：  <br>
&ensp;&ensp; ① 方法后缀名为_byTXT是根据txt标注生成txt文件同时生成xml标注, 这里注意要修改train_label_path为自己数据集中txt标注文件的绝对路径(process_data.py文件的最上方); <br> 
&ensp;&ensp; ② 后缀名为_byXML是根据xml标注生成txt文件。  <br>

2、
  
  
