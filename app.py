from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# Load the Real-ESRGAN model
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
    scale=4,
    model_path='weights/RealESRGAN_x4plus.pth',
    model=model,
    tile=0,  # 0 for no tiling
    tile_overlap=0,
    pre_pad=0,
    half=False  # Use FP32 precision
)

app = FastAPI()

@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    # Enhance the image
    output, _ = upsampler.enhance(np.array(img), outscale=4)

    # Convert enhanced image to bytes
    pil_image = Image.fromarray(output)
    byte_io = io.BytesIO()
    pil_image.save(byte_io, format="JPEG")
    byte_io.seek(0)

    return StreamingResponse(byte_io, media_type="image/jpeg")
