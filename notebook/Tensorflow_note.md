--------------------------
## Golden Sentences.
- “Garbage in, garbage out”

- Keep in mind Goodhart's law , "When a measure becomes a target, it ceases to be a good measure."  

- Start with the problem, not the solution. Make sure you aren't treating ML as a hammer for your problems.  

- As a rough rule of thumb, your model should train on at least an order of magnitude more examples than trainable parameters. Simple models on large data sets generally beat fancy models on small data sets.  

--------------------------
### 数据准备 Data Preparation
#### Imbalanced data
- Imbalanced Data中有Majority Class与Minority Class。而Minority Class所占比重极小。

|Minority Class Percentage|Degree of imbalance|
|-------------------------|-------------------|
|Mild	    | 20-40% of the data set|
|Moderate	| 1-20% of the data set|
|Extreme	| <1% of the data set|
-解决办法：
  - 首先直接在Imbalanced数据集中训练。
  - 如果效果不好，使用下采样(DownSampling)与上权重(Upweighting)
  - DownSampling:例如，数据集中有200个反例，10个正例。首先将反例以因子20进行下采样:拿出10%的数据
  - Upweighting:由于对数据进行了下采样，需要对其样本权重进行补偿——由于取出了10%的数据，因此取出的数据的样本权重为10倍的原始权重。即：当计算loss时，模型看待这些样本的比重为10倍的原始数据。

#### Reliability of dataset
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
### Machine Learning Mindset  
  [Step]
  1. Set the research goal.	(I want to predict how heavy traffic will be on a given day.
  2. Make a hypothesis.	(I think the weather forecast is an informative signal.
  3. Collect the data.	(Collect historical traffic data and weather on each day.
  4. Test your hypothesis.	(Train a model using this data.
  5. Analyze your results.	(Is this model better than existing systems?
  6. Reach a conclusion.	(I should (not) use this model to make predictions, because of X, Y, and Z.
  7. Refine hypothesis and repeat.	(Time of year could be a helpful signal.
------------------------
### Tensorflow Keras知识点

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

- 使用ImageDataGenerator
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
