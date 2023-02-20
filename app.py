from fastai.learner import load_learner
import gradio as gr


learn = load_learner('https://openmmlab-open.oss-cn-shanghai.aliyuncs.com/model-center/checkpoints/139430/model_cat_dog.pkl')

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
