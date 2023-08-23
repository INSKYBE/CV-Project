# CV-Project
针对自动驾驶障碍物识别这个主题要求，首先需要对可能的障碍物进行准确的识别和跟踪。在本次报告中，我分别选择了YOLOv5以及基于Transformer的DETR算法，以探究它们在自动驾驶障碍物识别任务中的性能和优劣。

为了有效地进行障碍物识别，我选择了YOLOv5和DETR两种不同类型的算法。这两种算法在目标检测领域都有着显著的影响力，但采用了不同的方法来解决目标检测问题。简单来说，YOLO系列是一种基于深度卷积神经网络的实时目标检测算法，以其高效的检测速度和精准的检测结果而著称。相较于传统的目标检测算法，YOLO系列将目标检测问题转化为一个回归问题，通过将图像划分为网格并预测每网格中的目标的边界框和类别概率，从而实现了实时性和准确性的平衡。DETR算法这是基于Transformer架构的目标检测方法，通过将目标检测问题转化为一个序列到序列的转换任务，引入了注意力机制来捕捉目标之间的关系。能够在一次性推理中同时处理多个目标，具有良好的拓展性和准确性。之后会分别对两个算法进行更加详细的阐述。
