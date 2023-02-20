from fastai.learner import load_learner
import gradio as gr
import requests


model_url = 'https://openmmlab-open.oss-cn-shanghai.aliyuncs.com/model-center/checkpoints/139430/model_cat_dog.pkl'
filename = "model_cat_dog.pkl"
response = requests.get(model_url)
with open(filename, 'wb') as f:
  f.write(response.content)

learn = load_learner(filename)

categories = ('Cat','Dog','Lion','None','Tiger','Wolf')

def classify_img(image):
  pred,idx,probs = learn.predict(image)
  return (dict(zip(categories,map(float,probs))))


image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()
examples = ['cat.jpg','food.jpg','tiger.jpg','cat_dog.jpg']
title = 'Simple classifier'
description = 'The model classifies input images into Dog, Cat, Lion, Tiger, Wolf classes. If the input image is not in target class, it is classified as None'
intf = gr.Interface(fn=classify_img, inputs=image, outputs=label,title=title,description=description,examples=examples)
intf.launch(inline=False)
