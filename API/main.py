from fastapi import FastAPI,File,UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
app = FastAPI()
TOMATO_MODEL=tf.keras.models.load_model("../models/3")

CLASS_NAMES=['Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato__Target_Spot',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']

@app.get("/first")
async def check():
    return "its working"
def read_file_as_image(data)->np.ndarray:
    image=np.array(Image.open(BytesIO(data)))
    return image
@app.post("/pic")
async def predict(
        file: UploadFile = File(...)
):
    image =read_file_as_image(await file.read())
    batch1=np.expand_dims(image,0)
    prediction=TOMATO_MODEL.predict(batch1)
    pred_dis=CLASS_NAMES[np.argmax(prediction[0])]
    conf=np.max(prediction[0])
    return{
        'class':pred_dis,
        'confidence':float(conf)
    }
if __name__=="__main__":
    uvicorn.run(app, host='localhost', port=8000)