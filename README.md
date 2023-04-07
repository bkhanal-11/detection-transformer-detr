# Detection Transformer (DETR)

Implementation of Detection Transformer (DETR) from scratch for object detection.

DETR (DEtection TRansformer) is a state-of-the-art object detection model that uses the transformer architecture to detect objects in an image. It was introduced in 2020 by Facebook AI Research.

DETR works by first encoding an input image using a convolutional neural network (CNN) to generate a set of feature maps. Then, it feeds these feature maps into a transformer encoder, which generates a set of encoded feature vectors representing the image.



Next, DETR generates a fixed number of object queries, which are learnable vectors that are used to query the encoded feature vectors to detect objects. Each object query is responsible for detecting a specific object in the image.

DETR uses bipartite matching between the object queries and ground truth objects to assign the object queries to the actual objects in the image. Once the object queries are matched to ground truth objects, DETR predicts the class and location of each object.

The losses in DETR are calculated using a combination of classification and regression losses. The classification loss is a binary cross-entropy loss that penalizes incorrect class predictions, and the regression loss is a smooth L1 loss that penalizes incorrect bounding box predictions. The losses are then combined using a weighted sum to generate a final loss function, which is optimized during training using stochastic gradient descent (SGD) or a variant like Adam.