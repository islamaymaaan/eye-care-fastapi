import os
import shutil
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        from my_model import Integrated_Model
        result, confidence = Integrated_Model(temp_file_path)
        os.remove(temp_file_path)

        return JSONResponse(content={
            "DiagnosisResult": result,
            "ConfidenceScore": confidence
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/check-model/")
def check_model():
    path = "models/dataset1.h5"
    return {"exists": os.path.exists(path)}
