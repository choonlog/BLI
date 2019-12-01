# BLI

Order of implementation<br />
1. Multi-process.py<br />
- I created this file to train each model in parallel. There is only one thing to note when running, which is the path setting.<br />
2. TrainingCPU.py or TrainingGPU.py<br />
- There is a CPU version and a GPU version, so you can use it according to the situation. When you decide which version to use, you also need to change the path setting of Multi-process.py. The learning results are all stored in the CheckPoint directory.<br />
3. Inference.py<br />
- Because the structure of the learning model is the same and only the parameters are different, you can run this file to use the pre-trained model. Boolean logic will be inferred from this file.<br />
*Preprocessing.py<br />
- I have already completed the preprocessing work and it was saved as processedData.csv file in Data directory.<br />
<br />
Thank you!
