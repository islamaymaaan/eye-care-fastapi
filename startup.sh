#!/bin/bash

# تفعيل البيئة الافتراضية (إن وُجدت)
if [ -d "antenv" ]; then
  source antenv/bin/activate
fi

# تثبيت المتطلبات
pip install -r requirements.txt

# تشغيل التطبيق
python -m uvicorn main:app --host=0.0.0.0 --port=8000
