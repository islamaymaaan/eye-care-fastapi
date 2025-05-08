import os
import tarfile
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

# ✅ فك الضغط تلقائيًا عند التشغيل لأول مرة
if not os.path.exists("models"):
    tar_path = "output.tar.gz"
    if os.path.exists(tar_path):
        print("📦 Extracting output.tar.gz...")
        try:
            with tarfile.open(tar_path) as tar:
                tar.extractall()
            print("✅ Extraction complete.")
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
    else:
        print("⚠️ output.tar.gz not found.")

# ✅ إنشاء تطبيق FastAPI
app = FastAPI()

# ✅ Endpoint رئيسي للتشخيص
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

# ✅ Endpoint بسيط للتأكد من وجود الموديل
@app.get("/check-model/")
def check_model():
    path = "models/dataset1.h5"
    return {"exists": os.path.exists(path)}
