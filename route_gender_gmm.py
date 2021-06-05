from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from model import VoiceModelGMM
import librosa

router = APIRouter(prefix='/gender_gmm')
model = VoiceModelGMM()

class Response(BaseModel):
    class_value: int
    class_name: str

@router.post("/", response_model=Response)
async def gender(file: UploadFile = File(...)):
    data, sr = librosa.load(file.file)
    class_value = model.predict(data, sr)
    class_name = model.get_target_name(class_value)
    return {"class_value": class_value, "class_name": class_name}