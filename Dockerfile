FROM python:3.11-slim

# libraqm จำเป็นต่อการจัดวางสระ วรรณยุกต์ และลำดับการเขียนภาษาไทย
#
# wheel ของ Pillow มีโค้ดเรียก libraqm อยู่แล้ว แต่ไม่ได้ bundle ไลบรารีมาด้วย
# ถ้าไม่ติดตั้งสองแพ็กเกจนี้ โปรแกรมจะรันได้แต่ข้อความไทยจะซ้อนกันผิดตำแหน่ง
# โดยไม่มี error ให้เห็น จึงต้องติดตั้งไว้ตั้งแต่ตอน build
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libraqm0 \
        libfribidi0 \
    && rm -rf /var/lib/apt/lists/*

# ไม่ต้องเขียนไฟล์ .pyc ลงดิสก์ และให้ log ออกมาทันทีไม่ถูกบัฟเฟอร์
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# โค้ดที่จำเป็นเท่านั้น tests/ samples/ และโฟลเดอร์ coloring-book/
# ไม่จำเป็นต่อการทำงาน จึงไม่ copy มาเพื่อลดขนาด image
COPY app ./app
COPY static ./static
COPY fonts ./fonts

# Render กำหนดพอร์ตให้มาตอนรัน ค่านี้เป็นเพียงค่าเริ่มต้นสำหรับรันในเครื่อง
EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
