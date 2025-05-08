from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os

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
        raise HTTPException(status_code=500, detail=str(e))
