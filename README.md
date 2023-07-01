# ImageCaptionGenerator
The repo contains a machine-learning model to generate captions for an image. The model is deployed on the web using Streamlit and ngrok.
The code of the model can be found in **model_code.ipynb** while the deployment code code is in **deploy_code.ipynb**.
Details of the dataset and Model Architecture are as follows

# Dataset
The model is trained on the flickr8k dataset, which can be found here: https://www.kaggle.com/datasets/adityajn105/flickr8k?select=Images
The dataset consists of around 8000 images with an average of 5 captions per image.

# Model Architecture
I am using a Pre-trained VGG16 model to compute features present in an image. Skipping the last softmax activation layer of VGG16, the previous layer's output is directly fed into the model below.   
![model](https://github.com/ABHISHEKgauti25/ImageCaptionGenerator/assets/109408129/b69c931d-e6b1-493e-9767-b56220b510a6)

**Note**: I have uploaded a trained version of the above model architecture which can be found here:https://github.com/ABHISHEKgauti25/Webapp-imgCap/tree/main/trained%20models Trained for nearly 50 epochs, this was able to achieve a Bleu score of 0.51.
