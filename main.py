from fastapi import FastAPI

import route_gender_gmm
import route_gender_svm

app = FastAPI()

@app.get("/ping")
async def pong():
    return {"message": "pong"}

app.include_router(route_gender_svm.router)
app.include_router(route_gender_gmm.router)