<!--
 * @Descripttion: 
 * @version: 
 * @Author: Cai Weichao
 * @Date: 2022-03-12 20:23:23
 * @LastEditors: Cai Weichao
 * @LastEditTime: 2022-03-13 13:10:05
-->
# Readme
This is a personally created pytorch based code template, which will be updated from time to time 

## Structure 
In this template, there are the following folders and files:
+ checkpoint : save checkpoint file
+ config : Store config.yaml 
  + config.yaml : Configure parameters related to the model, including the optimizer, whether to use checkpoint **(There are only a few commonly used ones, you can add them according to the needs of the task)**
+ data : You can put the dataset here 
  + dataset.py : here, You can configure your own torch.utils.data.Dataset based on your dataset
+ model : This folder stores the files that define the model 
  + layers.py : Various modules that can be used in the model can be defined here 
  + model.py : Assemble modules as models 
  + **Note : It's just my own personal habit**
+ utils : tools to store 
  + metrics.py : Various evaluation indicators can be stored here **(still just an empty file)**
  + parser.py : various hyperparameters, **(There are only a few commonly used ones, you can add them according to the needs of the task)**
  + txtlogger.py : import logging here and initialize 
+ main.py : Call main.py, the program starts to execute, For specific parameters, see ./utils/parser.py
  + ```python
       python main.py [argument]
    ```
+ train.py : Defines classes related to training models 











