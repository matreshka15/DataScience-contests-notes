
-----------------------------
### Chapter 1:关于机器学习的冥想
#### Machine Learning Mindset  
  [Step]
  1. Set the research goal.	(I want to predict how heavy traffic will be on a given day.
  2. Make a hypothesis.	(I think the weather forecast is an informative signal.
  3. Collect the data.	(Collect historical traffic data and weather on each day.
  4. Test your hypothesis.	(Train a model using this data.
  5. Analyze your results.	(Is this model better than existing systems?
  6. Reach a conclusion.	(I should (not) use this model to make predictions, because of X, Y, and Z.
  7. Refine hypothesis and repeat.	(Time of year could be a helpful signal.

#### Golden Sentences
- “Garbage in, garbage out”

- Keep in mind Goodhart's law , "When a measure becomes a target, it ceases to be a good measure."  

- Start with the problem, not the solution. Make sure you aren't treating ML as a hammer for your problems.  

- As a rough rule of thumb, your model should train on at least an order of magnitude more examples than trainable parameters. Simple models on large data sets generally beat fancy models on small data sets.  

--------------------------
### Chapter 2:数据准备 Data Preparation
  如何处理数据集？
#### 失衡数据 Imbalanced data
  - Imbalanced Data中有Majority Class与Minority Class。而Minority Class所占比重极小。

            |Minority Class Percentage|Degree of imbalance|
            |-------------------------|-------------------|
            |Mild	                    | 20-40% of the data set|
            |Moderate	                | 1-20% of the data set|
            |Extreme	                | <1% of the data set|
  - 解决办法：
    - 首先直接在Imbalanced数据集中训练。
    - 如果效果不好，使用下采样(DownSampling)与上权重(Upweighting)
    - DownSampling:例如，数据集中有200个反例，10个正例。首先将反例以因子20进行下采样:拿出10%的数据
    - Upweighting:由于对数据进行了下采样，需要对其样本权重进行补偿——由于取出了10%的数据，因此取出的数据的样本权重为10倍的原始权重。即：当计算loss时，模型看待这些样本的比重为10倍的原始数据。

#### 数据集可靠性 Reliability of dataset
  Below is a list of possible causes of unreliable dataset.
  - Omitted values. For instance, a person forgot to enter a value for a house's age.
  - Duplicate examples. For example, a server mistakenly uploaded the same logs twice.
  - Bad labels. For instance, a person mislabeled a picture of an oak tree as a maple.
  - Bad feature values. For example, someone typed an extra digit, or a thermometer was left out in the sun.

#### Data Transformation 数据变形（归一化）
  - ##### Scaling to a range(缩放至范围内) :
    $$x'=\frac{x-x_{min}}{x_{max}-x_{min}}$$

    - 注意，使用这种缩放满足以下两个先验条件时效果较好：  
     (1) 数据分布的上界、下界大概值可以确认，且没有outliers  
     (2) 数据在此范围内大体满足均匀分布
  - ##### 特征裁剪 Feature Clipping
    - 在应用其他Normalization之前使用特征裁剪。  
    当存在outlier时：  
    (1)设置最大最小值区间Min-Max,把此区间外的值剔除。   
    (2)Z-score(Z分数)裁剪法：将数据的Z分数限制在+-Nσ区间内。其中N通常取3.  
    (3)log缩放：$x'=log(x)$ ，当数据近似满足指数分布，且数据分布较稀疏时可以使用。  
    - 补充：Z score保证了数据为均值为0，方差为1的分布。当数据中存在少量outlier,但又不是严重到必须使用特征裁剪时，可以只使用Z score归一化数据。
    $$x'=\frac{x-μ}{σ}$$
  - ####Bucketing 装桶  
    - 当输入特征为浮点数，而输出与该特征非线性关系时，可将其装桶（打包）。例如：把经度平均划分为1度一个区间。  
     The boundaries are fixed and encompass the same range (for example, 0-4 degrees, 5-9 degrees...)  
    - 分位数打包(Quantile bucketing)：当数据集中分布在某一区间，而在另一区间内数据较少时，可采用分位数打包。  
    Each bucket has the same number of points. The boundaries are not fixed and could encompass a narrow or wide span of values.
--------------------------
### Chapter 3:机器学习知识点Recap
#### 基于树的方法 Tree-based Methods
  树的方法在很多Data Competition中很有用如Decision Tree,Random Forest,GBDT等。下面一一介绍。
##### 决策树Decision tree
  决策树基于简单的分而治之(Divide-and-conquer)策略。一个决策树通常包含一个根结点、若干的内部结点和若干个叶结点。叶结点对应于决策结果，其他节点对应于属性测试。    
  注意划分决策树的最优划分属性时有多种算法，如信息熵(ID3决策树)、增益率(C4.5决策树)、基尼系数（CART决策树）等。    
  - 剪枝处理（pruning）:决策树对付过拟合的主要手段-通过主动去掉一些分支来降低过拟合的风险。
    - 预剪枝（pre-pruning）：首先将数据集划分为训练集、验证集，在划分每个结点时先计算不划分结点时的准确率，然后计算划分结点后的准确率；若后者大于前者，则确认在此处划分结点。  
    *注意：可能带来欠拟合风险*
    - 后剪枝(post-pruning)：先生成一颗完整的决策树，然后自下至上，对每个结点的子树分析——若将子树替换为叶节点，判断其准确率有无提高；若提高或相同（奥卡姆剃刀准则），则进行剪枝。    
     *注意：后剪枝的决策树通常比预剪枝的决策树有更多的分支，因此欠拟合的风险较小，泛化性能通常较优；但训练时间较长。*
##### 随机森林 Random forest
  随机森林相当于一组决策树；根据”少数服从多数”的原则进行预测。   
  “随机”的含义可从两个层面解释：假设给定数据集X,以每一行为一个样本，每一列为一个特征。
  - 从行的角度：每个决策树只根据训练集中随机选取的一小部分（比如10%）的数据进行训练，因此每个决策树预测结果将会不同。
  - 从列的角度：投喂给每个决策树用于训练的特征也仅为所有特征中随机选取的某一部分（比如10%）。    

  注意在以下几种情况下，随机森林不是一个好的算法：    
  - 当训练集很小时，随机森林表现很差。（甚至还不如线性算法）
  - 随机森林的可解释性很差。随机森林是一个预测工具，而不是一个描述工具。
  - 用来训练多个决策树的时间有时会很长很长。对于一个n类的类别变量，随机森林要分裂$2^n-1$个节点。
  - 对于回归问题，随机森林能给出的预测值只能为训练集中的数据已经出现了的范围内。它不能像线性回归一样给出不在训练集中的值的范围。
##### 集成方法(Ensembling)
  集成学习先产生一组个体学习器，再用某种策略将其结合起来。个体学习器可以是同质也可以是异质的。通过将多个学习器进行结合，通常可以获得比单一学习器显著优越的泛化性能。*此效应对弱学习器更为明显（弱学习器指泛化性能略优于随机猜测的学习器）*    
  要使集成学习获得更好的性能，需要保证个体学习器应有一定的准确性，且学习器之间要有一定*差异*——好而不同。但实际上准确性与差异之间是一对矛盾，因为学习器是为解决同一问题而训练出来的。    

  根据个体学习器的生成方式，集成方法大致可分为两类
  - 个体学习器之间存在强依赖关系，必须串行生成的序列化方法——Boosting
  - 个体学习器之间不存在强依赖关系，可以并行生成——Bagging与随机森林

  ###### Boosting ######
  Boosting是一族可将弱学习器提升为强学习器的算法。其工作原理大体为：先从初始训练集训练出基学习器，再根据基学习器的表现对训练样本分布进行调整，使得先前基学习器表现不好的样本得到更多关注，基于调整后的样本分布来训练下一个基学习器。反复进行直到基学习器数目达到指定值T。最终将这T个基学习器进行加权结合。    
  典型代表：AdaBoost算法 （西瓜书p173）    
  注：Boosting方法主要关注于降低偏差，因此Boosting能基于泛化性能相当弱的学习器构建出很强的集成。

  ###### Bagging（Bootsrap aggregating的缩写） ######
  为了使个体学习器的差异尽量大，一种做法是将训练样本进行采样，产生若干的不同的子集用于训练基学习器；然而这样会降低实际用的训练样本数目，导致准确率降低。因此实际上可以考虑相互有交叠的采样子集。    
  Bagging基于自助采样法：给定m个样本的数据集，先随即取出一个样本放入采样集中，再把该样本放回初始数据集，使得下次采样时仍可能选中该样本。这样，我们可得到T个含m个样本的采样集，然后基于每个采样集训练出一个基学习器，再将这些基学习器进行结合——对分类任务，使用简单投票法；对回归任务，使用简单平均法。    
  由于使用了自助采样法，初始训练集中约有63.2%的样本被使用于训练，还剩下36.8%的数据可用于*包外估计(out-of-bag estimate)*    
  注：Bagging主要关注于降低方差，因此它在不剪枝的决策树、神经网络等容易受样本扰动干扰的学习器上效用更明显。

  ##### 结合策略  #####
  下面是几种对多个学习器$h_{i}(x)$进行结合的常见策略：（西瓜书p181）
  - 平均法：适用于数值型输出
    - 简单平均法
    - 加权平均法：学习出的权重不完全可靠，且可能出现过拟合。因此实际效果未必优于简单平均法。
  - 投票法：适用于分类问题
    - 绝对多数投票法
    - 相对多数投票法
    - 加权投票法
  - 学习法：通过另一个学习器来进行结合。将个体学习器称为初级学习器，用于结合的学习器为次级学习器或元学习器(meta-learner)。典型代表：Stacking

--------------------------
### Chapter 4:Tensorflow Keras知识点

- 使用Callback终止模型训练：  
  - 定义Callback
  ```python
  class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,log={}):
        if(log['acc']>DESIRED_ACCURACY):
            print('Training terminated.')
            self.model.stop_training = True      
  callbacks = myCallback()
  ```
  - 在model.fit中使用callback
  ```python
  # 使用fit_generator举例.
  history = model.fit_generator(
    train_generator,
    steps_per_epoch = 5,
    epochs = 20,
    verbose=1,
    callbacks=[callbacks])
    ```

- 使用卷积层与MaxPooling

      ```python
        tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(150,150,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512,activation='relu'),
        tf.keras.layers.Dense(1,activation='sigmoid')])
      ```


- 使用卷积层与MaxPooling

    ```python
      model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(150,150,3)),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(512,activation='relu'),
      tf.keras.layers.Dense(1,activation='sigmoid')])
     ```

- 使用ImageDataGenerator读取image Data
  ```python
  from tensorflow.keras.preprocessing.image import ImageDataGenerator
  #归一化数据
  train_datagen = ImageDataGenerator(rescale=1./255.)
  #直接从文件夹中读取图片数据
  train_generator = train_datagen.flow_from_directory(
    "/tmp/h-or-s", # 文件目录
    target_size=(150,150),  # 希望统一化的像素
    batch_size=10,     #Batch大小
    class_mode='binary')  #类型  也可以是catorgorical
  ```

- 使用ImageDataGenerator 进行图像增强（数据增强）
  ```python
  train_datagen = ImageDataGenerator(
      rotation_range=40,  #40°以内随机倾斜
      width_shift_range=0.2,  #横座标平移20%
      height_shift_range=0.2,  #纵坐标平移20%
      shear_range=0.2,    #倾斜20%
      zoom_range=0.2,     #放大20%
      horizontal_flip=True, #水平翻转
      fill_mode='nearest')  #填充空缺像素的模式
  ```

  - 迁移学习
    - 加载模型、处理权重
    ```python
    #加载预训练模型
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    #从本地加载权重，而不要使用自带的权重
    local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

    pre_trained_model = InceptionV3(input_shape = (150, 150, 3),
                                    include_top = False,  #去掉最后的几层
                                    weights = None)   # 不使用自带的权重

    pre_trained_model.load_weights(local_weights_file)  #加载本地权重

    for layer in pre_trained_model.layers:
      layer.trainable = False  #制定某些层，锁定权重
    ```
    - 取出其中某层作为最后一层：
    ```python
    last_layer = pre_trained_model.get_layer('mixed7')
    print('last layer output shape: ', last_layer.output_shape)
    last_output = last_layer.output
    # Flatten the output layer to 1 dimension
    x = layers.Flatten()(last_output)
    # Add a fully connected layer with 1,024 hidden units and ReLU activation
    x = layers.Dense(1024, activation='relu')(x)
    # Add a dropout rate of 0.2
    x = layers.Dropout(0.2)(x)                  
    # Add a final sigmoid layer for classification
    x = layers.Dense  (1, activation='sigmoid')(x)    
    ```
-------------------------------------------------------
