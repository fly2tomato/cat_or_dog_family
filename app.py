from fastai.vision.all import *
import gradio as gr


learn = load_learner('modelcat_dog.pkl')

categories = ('Cat','Dog','Lion','None','Tiger','Wolf')

def classify_img(image):
  pred,idx,probs = learn.predict(image)
  return (dict(zip(categories,map(float,probs))))


image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()
examples = ['cat.jpg','food.jpg','husky.jpg','tiger.jpg','cat_dog.jpg']
intf = gr.Interface(fn=classify_img, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)