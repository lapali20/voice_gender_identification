from fastapi import APIRouter, UploadFile, File , FastAPI, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from model import VoiceModelSVM
import librosa

router = APIRouter(prefix='/gender_svm') 
templates = Jinja2Templates(directory="templates")
app = FastAPI()

model = VoiceModelSVM()

class Response(BaseModel):
    class_value: int
    class_name: str

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/route_gender_svm", response_model=Response)
async def gender(file: UploadFile = File(...)):
    data, sr = librosa.load(file.file)
    class_value = model.predict(data, sr)
    class_name = model.get_target_name(class_value)
    return {"class_value": class_value, "class_name": class_name}