# HSN-LSTM
This code is the implementation of our paper (A Hybrid Spiking Neurons Embedded LSTM Network for Multivariate Time Series Learning under Concept-drift Environment).

# Environment version
PyTorch = 1.6

# Usage
Example:
You can run the pm25_HSN_LSTM.py file directly to implement the PM2.5 Air Quality forecasting task.  
Then, you can execute the following command:  

```
python pm25_HSN_LSTM.py
```
Note that since this is experimental code, you will need to manually set some parameters in the code, such as the time window and the size of the forecast horizons.

# References
If you are interested, please cite this paper.  

@ARTICLE{HSN_LSTM,
  author={Zheng, Wendong and Zhao, Putian and Chen, Gang and Zhou, Huihui and Tian, Yonghong},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={A Hybrid Spiking Neurons Embedded LSTM Network for Multivariate Time Series Learning under Concept-drift Environment}, 
  year={2022},
  pages={1-14},
  doi={10.1109/TKDE.2022.3178176}}
