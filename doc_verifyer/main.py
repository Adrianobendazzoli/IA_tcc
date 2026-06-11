from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import shutil
import os
import uuid

from analyzer import DocumentAnalyzer

app = FastAPI(
    title="Verificador de Autenticidade de Documentos",
    description="API para detectar manipulações em documentos via ELA, EXIF e Haar Cascade",
    version="1.0.0"
)

# Libera chamadas do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

analyzer = DocumentAnalyzer()

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "application/pdf"}


@app.post("/verificar", summary="Verifica autenticidade de um documento")
async def verificar_documento(file: UploadFile = File(...)):
    """
    Recebe um arquivo (imagem ou PDF) e retorna:
    - score de autenticidade (0-100)
    - alertas encontrados
    - metadados EXIF
    - resultado da análise ELA
    - resultado da detecção facial (Haar Cascade)
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de arquivo não suportado: {file.content_type}. Use JPEG, PNG, WEBP ou PDF."
        )

    # Salva o arquivo temporariamente
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[-1].lower() or ".jpg"
    temp_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = analyzer.analisar(temp_path, file.content_type)
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar arquivo: {str(e)}")

    finally:
        # Remove arquivo temporário após análise
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/health")
def health_check():
    return {"status": "ok", "mensagem": "API de verificação rodando!"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
