import numpy as np
import onnxruntime
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from dl import cfg

# Load the ONNX model
sess = onnxruntime.InferenceSession(cfg.onnx.path)
in_name = sess.get_inputs()[0].name
in_shape = sess.get_inputs()[0].shape
out_shape = sess.get_outputs()[0].shape


class Body(BaseModel):
    input: list[list[float]] = Field(..., description=f'Model batch input with shape {in_shape}', example=np.zeros(in_shape).tolist())


class Response(BaseModel):
    output: list[list[float]] = Field(..., description=f'Model batch output with shape {out_shape}', example=np.zeros(out_shape).tolist())


app = FastAPI()


@app.get('/')
async def root():
    return RedirectResponse(url='/docs')


@app.post('/predict', response_model=Response)
async def predict(body: Body):
    output = sess.run(None, {in_name: body.input})[0]
    return {'output': output.tolist()}
