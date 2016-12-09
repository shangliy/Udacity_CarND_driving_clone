# driving_clone

**Use Deep Learning to Clone Driving Behavior**
===================


This is project to use udacity self-driving car simulator to do clone driving training.

----------


**Model Architecture and Training Strategy**
-------------
The basic **structure** of my network inspired by the idea from the reference paper [[link]](chrome-extension://ecnphlgnajanjnkcmbpancdjoidceilk/content/web/viewer.html?file=http%3A%2F%2Fimages.nvidia.com%2Fcontent%2Ftegra%2Fautomotive%2Fimages%2F2016%2Fsolutions%2Fpdf%2Fend-to-end-dl-using-px.pdf). 
The key differences include three parts:
 >* I use **RGB** value rather than Y value as input;
 >* Rather than take the steering angle as output. I separate the output into two part. 
 >One for **direction**, and another for the **absolute angle value**. The first one is taken as 3-classed classification problem (0-turn left, 1-go straight, 2-turn right); The second one as regression problem.
 >* Bring in activation,normalization and dropout layer.

Detail of the model explained below.
Another key of this project is how to **gather and pre-process** training data. I implemented several technologies  to improve training performance,  including **smoothing** and **refining**.
Besides, some methods were also used to reduce the effect of **overfitting**. 

> **Note:**

> - There are mainly four python scripts for this project:
	* **model.py** :  training keras based deep learning model 
	* **drive.py** :  run simulator based on trained model
	*  **plot_training_data.py**  : Transform input file type, smooth and visualize training data
	* **visulize_output.py** : Visualize and refine test predictions
> - There are another two files related with model:
	* **model.json** :  model structure
	* **model.h5** :  model weights
> - You can check the simulator result video from youtube.[ [link ](https://www.youtube.com/watch?v=dZbjTP7d100&feature=youtu.be)]


#### **Graph Structure**

The graph contains convolution layer, activation layer, normalization layer, full connected layer and dropout layer. The activation layer bring in the nonlinearity . The normalization layer normalize the data.

To visualize the graph:
	![enter image description here](https://github.com/shangliy/driving_clone/blob/master/model.png?raw=true)

####  **Overfitting Reduce**

To reduce the effect of overfitting. Two methods applied in this project.
>*  Training data is separated into training and **validation** data. 
	Number of training data: Number of validation data = **9:1**
>* The **dropout** layer is used with **0.5** percent of kept units.

####  **model parameters** 
* Batch-size: 32 
* Learning rate: Using **Adam** method
* Optimizer: **categorical_crossentropy** for classification problem and **mse** for regression problem
 * **final_model.compile(optimizer='adam', loss=['categorical_crossentropy','mse'], loss_weights=[1,1])**

####  **Training Data**
* **Data Generation:**

	I use the simulator to generate the driving_log.csv file containing the center image path and steering angle. I then use plot_training_data.py to load the driving_log.csv and slit them into training dataset and validation dataset. Then I save them into train.p and test.p. The total number of data is about 40000.

>Example of training data: 
|                  | NUMBER                        |   ANGLE |
 ----------------- | ---------------------------- || ---------------------------- |
| image_1  |  	![enter image description here](https://github.com/shangliy/driving_clone/blob/master/sample_data/0.jpg?raw=true)   | 0 |
| image_2          | ![enter image description here](https://github.com/shangliy/driving_clone/blob/master/sample_data/905.jpg?raw=true)|-0.2|
|  image_3             |![enter image description here](https://github.com/shangliy/driving_clone/blob/master/sample_data/2155.jpg?raw=true)| 0.2|



	
*  **Data Preprocessing**:
 * **Normalization** : Normalize input image data from 0~255 to 0~1.
 * ** Smoothing**: Considering the sensitivity of the simulator. The training data contains lots of noisy . Thus, it is better to smoothing the training data. 
 I use **pyasl.smooth** to smooth the training data. Below is the comparison of before and after smoothing . Detail in **plot_training_data.py**.
 
 	**Before Smoothing**
 	![Before Smoothing](https://github.com/shangliy/driving_clone/blob/master/figure_1.png?raw=true)
 	**After Smoothing**
 	![After Smoothing](https://github.com/shangliy/driving_clone/blob/master/figure_2-1.png?raw=true)

	 * **Refining**: 
	 There are still lots of zeros-value angle and wrong training data. 
	 
	  Thus, I firstly remove half of zero value training data, then I built an apito refine the data by visualize the input image and target value.Use this api to refine the training data. use this api, we can increase or decrease the target value based on the image. It on the one hand, help to remove the wrong training data, on the other hand, it help to remove the noisy in training data. By the way, it also help to see what the car do wrong in the test. Detail in visulize_output.py.
	  **Refining UI**
	  ![Refining UI](https://github.com/shangliy/driving_clone/blob/master/imageedit_20_3610796380.jpg?raw=true)


**Architecture and Training Documentation**
-------------
#### **solution design**
  The initial idea for this project is an regression problem.  We need to use convolution layer to extract feature, set normalization layer to improve performance and  use activation layer to bring none nonlinearity. Detail of the graph is shown above.
At first, I use several full connected layer to get final one unit, which the prediction of the steering angle. But, considering the high noisy in training data, the accuracy of the data is low. Thus, I separate the output into two problem, one classification, and one for regression. 
The classification is for the direction. It is a 3-classes problem. 0 for turn left, 1 for go straight, 2 for turn right. The reason to do that is because the direction is more robust. For example, when we turn right, the angle varies much considering the simulator sensitivity or different situation, however, the direction is still same which has high confidence 
The regression problem is for absolute steering angle. Thus, in the test, if direct == 1, steering_angle = 0, if direct == 0: steering_angle = (-1)*value, elif direct == 2,  steering_angle = value.
So, I use two output for the graph.

#### **Architecture Detail**

|                  | NUMBER                        | 
 ----------------- | ---------------------------- |
| Convolution2D  |5           | 
| Full connected           | 5           |
| BatchNormalization           | 7 | 
| Activation layer  |    9           | 
| Dropout layer           | 1 | 
| Total          | 27 | 




* **Structure detail**
>
     -- **Convolution2D layer**, window_size **(5x5) **,  stride **(2x2) **, depth:  **32 **
     -- **Activation layer**, 'relu'
     --  **BatchNormalization layer**
>
   -- **Convolution2D layer**, window_size **(5x5) **, stride **(2x2 **), depth:  **64 **
     -- **Activation layer**,'relu'
     -- **BatchNormalization layer**
>
   -- **Convolution2D layer**, window_size **(5x5) **, stride **(2x2) **, depth:  **128 **
     -- **Activation layer**,'relu'
     -- **BatchNormalization layer**
>
   -- **Convolution2D layer**, window_size **(3x3) **, stride **(1x1) **, depth:  **128 **
     -- **Activation layer**,'relu'
     -- **BatchNormalization layer**
>
   -- **Convolution2D layer**, window_size **(3x3 **), stride **(1x1) **, depth:  **128 **
     -- **Activation layer**,'relu'
     -- **BatchNormalization layer**
>
   -- **Flaten layer**
     -- **Dropout layer**  **(0.5) **
>
   -- **Full connected layer** : ->  **100 **
     -- **Activation layer**, *'relu' 
     -- **BatchNormalization layer**
>
   -- **Full connected layer** :  **100 -> 50 **
     -- **Activation layer**,' relu' 
     -- **BatchNormalization layer**
>
   -- **Full connected layer** :  **50 -> 3 **
      -- **Activation layer**, 'softmax''
>
   -- **Full connected layer** :  **50 -> 10 **
       -- **Activation layer**,'relu''
       -- **Full connected layer** :  **10 -> 1 **
      
#### **Training Process**
> 1: Training data detail described above;
> 2: Training Process, use **Adam**, **categorical_crossentropy** and **mse** for training. 
>  **Screen shot of training process**
	  ![Refining UI](https://github.com/shangliy/driving_clone/blob/master/train_process.jpg?raw=true)



**Simulation**
-------------

>You can check the simulator result video from youtube.[ [link ]](https://www.youtube.com/watch?v=dZbjTP7d100&feature=youtu.be)



