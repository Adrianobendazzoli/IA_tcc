import cv2
import numpy as np
from PIL import Image, ExifTags
import os
import io
import fitz  # PyMuPDF — para ler PDFs


class DocumentAnalyzer:
    """
    Combina três técnicas de análise para verificar autenticidade:
    1. Metadados EXIF   — detecta software de edição, inconsistências de data
    2. ELA              — detecta regiões adulteradas por diferença de compressão
    3. DNN OpenCV       — detecta rostos com deep learning (mais preciso que Haar Cascade)
    """

    # URLs dos arquivos de modelo DNN do OpenCV (Caffe)
    MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    CONFIG_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"

    def __init__(self):
        self.net = self._carregar_modelo_dnn()

    def _carregar_modelo_dnn(self):
        """Baixa (se necessário) e carrega o modelo DNN de detecção facial."""
        import urllib.request

        model_path = "face_detector.caffemodel"
        config_path = "face_detector.prototxt"

        if not os.path.exists(model_path):
            print("📥 Baixando modelo DNN de detecção facial (~2MB)...")
            urllib.request.urlretrieve(self.MODEL_URL, model_path)
            print("✅ Modelo baixado!")

        if not os.path.exists(config_path):
            print("📥 Baixando configuração do modelo DNN...")
            urllib.request.urlretrieve(self.CONFIG_URL, config_path)
            print("✅ Configuração baixada!")

        return cv2.dnn.readNetFromCaffe(config_path, model_path)

    # ------------------------------------------------------------------ #
    #  PONTO DE ENTRADA                                                    #
    # ------------------------------------------------------------------ #

    def analisar(self, filepath: str, content_type: str) -> dict:
        """Orquestra todas as análises e retorna resultado consolidado."""

        alertas = []
        detalhes = {}

        if content_type == "application/pdf":
            imagem = self._pdf_para_imagem(filepath)
            detalhes["tipo_arquivo"] = "PDF"
        else:
            imagem = Image.open(filepath).convert("RGB")
            detalhes["tipo_arquivo"] = "Imagem"

        # --- 1. EXIF ---
        exif_resultado = self._analisar_exif(filepath, content_type)
        detalhes["exif"] = exif_resultado["dados"]
        alertas.extend(exif_resultado["alertas"])

        # --- 2. ELA ---
        ela_resultado = self._analisar_ela(imagem)
        detalhes["ela"] = ela_resultado["dados"]
        alertas.extend(ela_resultado["alertas"])

        # --- 3. DNN Facial ---
        haar_resultado = self._analisar_dnn(imagem)
        detalhes["facial"] = haar_resultado["dados"]
        alertas.extend(haar_resultado["alertas"])

        # --- Score final ---
        score = self._calcular_score(alertas)

        return {
            "score_autenticidade": score,
            "classificacao": self._classificar(score),
            "total_alertas": len(alertas),
            "alertas": alertas,
            "detalhes": detalhes,
        }

    # ------------------------------------------------------------------ #
    #  1. ANÁLISE DE METADADOS EXIF                                        #
    # ------------------------------------------------------------------ #

    def _analisar_exif(self, filepath: str, content_type: str) -> dict:
        alertas = []
        dados = {}

        # PDFs não têm EXIF padrão
        if content_type == "application/pdf":
            dados["disponivel"] = False
            dados["motivo"] = "PDFs não possuem metadados EXIF"
            return {"dados": dados, "alertas": alertas}

        try:
            img = Image.open(filepath)
            exif_raw = img._getexif()

            if not exif_raw:
                dados["disponivel"] = False
                alertas.append({
                    "nivel": "medio",
                    "tipo": "exif",
                    "mensagem": "Metadados EXIF ausentes — imagem pode ter sido reprocessada ou exportada por editor"
                })
                return {"dados": dados, "alertas": alertas}

            dados["disponivel"] = True
            exif = {
                ExifTags.TAGS.get(k, k): str(v)
                for k, v in exif_raw.items()
                if k in ExifTags.TAGS
            }

            # Campos relevantes para exibição
            campos_relevantes = [
                "Make", "Model", "Software", "DateTime",
                "DateTimeOriginal", "DateTimeDigitized",
                "GPSInfo", "Artist", "Copyright"
            ]
            dados["campos"] = {k: exif[k] for k in campos_relevantes if k in exif}

            # Alerta: software de edição detectado
            software = exif.get("Software", "").lower()
            softwares_suspeitos = [
                "photoshop", "gimp", "lightroom", "affinity",
                "canva", "pixlr", "paint", "dall-e", "midjourney",
                "stable diffusion", "firefly"
            ]
            for sw in softwares_suspeitos:
                if sw in software:
                    alertas.append({
                        "nivel": "alto",
                        "tipo": "exif_software",
                        "mensagem": f"Software de edição detectado nos metadados: '{exif['Software']}'"
                    })
                    break

            # Alerta: data de criação vs modificação divergem
            dt_original = exif.get("DateTimeOriginal", "")
            dt_modificado = exif.get("DateTime", "")
            if dt_original and dt_modificado and dt_original != dt_modificado:
                alertas.append({
                    "nivel": "medio",
                    "tipo": "exif_data",
                    "mensagem": f"Data original ({dt_original}) difere da data de modificação ({dt_modificado})"
                })

            # Alerta: sem câmera identificada
            if "Make" not in exif and "Model" not in exif:
                alertas.append({
                    "nivel": "baixo",
                    "tipo": "exif_camera",
                    "mensagem": "Nenhuma câmera identificada nos metadados — pode indicar imagem gerada ou muito editada"
                })

        except Exception as e:
            dados["disponivel"] = False
            dados["erro"] = str(e)

        return {"dados": dados, "alertas": alertas}

    # ------------------------------------------------------------------ #
    #  2. ANÁLISE ELA (Error Level Analysis)                               #
    # ------------------------------------------------------------------ #

    def _analisar_ela(self, imagem: Image.Image) -> dict:
        """
        Salva a imagem com qualidade reduzida e compara com o original.
        Regiões com alta diferença indicam edição posterior à compressão original.
        """
        alertas = []
        dados = {}

        try:
            # Salva versão recomprimida
            buffer = io.BytesIO()
            imagem.save(buffer, format="JPEG", quality=75)
            buffer.seek(0)
            img_recomprimida = Image.open(buffer).convert("RGB")

            # Calcula diferença pixel a pixel
            arr_original = np.array(imagem, dtype=np.float32)
            arr_recomp = np.array(img_recomprimida, dtype=np.float32)
            diff = np.abs(arr_original - arr_recomp)

            # Métricas
            media_diff = float(np.mean(diff))
            max_diff = float(np.max(diff))
            percentual_alto = float(np.mean(diff > 30) * 100)  # % de pixels com diff > 30

            dados["media_diferenca"] = round(media_diff, 2)
            dados["max_diferenca"] = round(max_diff, 2)
            dados["percentual_pixels_suspeitos"] = round(percentual_alto, 2)

            # Limiares calibrados empiricamente
            if media_diff > 15:
                alertas.append({
                    "nivel": "alto",
                    "tipo": "ela_alta_diferenca",
                    "mensagem": f"ELA detectou alta inconsistência de compressão ({media_diff:.1f}) — forte indício de edição"
                })
            elif media_diff > 8:
                alertas.append({
                    "nivel": "medio",
                    "tipo": "ela_media_diferenca",
                    "mensagem": f"ELA detectou inconsistência moderada ({media_diff:.1f}) — possível edição parcial"
                })
            else:
                dados["status"] = "Sem anomalias significativas detectadas pelo ELA"

            if percentual_alto > 20:
                alertas.append({
                    "nivel": "medio",
                    "tipo": "ela_area_suspeita",
                    "mensagem": f"{percentual_alto:.1f}% dos pixels apresentam diferença elevada — área grande pode ter sido substituída"
                })

        except Exception as e:
            dados["erro"] = str(e)

        return {"dados": dados, "alertas": alertas}

    # ------------------------------------------------------------------ #
    #  3. ANÁLISE HAAR CASCADE (Detecção Facial)                           #
    # ------------------------------------------------------------------ #

    def _analisar_dnn(self, imagem: Image.Image) -> dict:
        """
        Detecta rostos usando DNN (deep learning) do OpenCV.
        Muito mais preciso que Haar Cascade — detecta rostos em ângulos,
        com oclusão parcial e em diferentes condições de iluminação.
        """
        alertas = []
        dados = {}

        try:
            img_cv = cv2.cvtColor(np.array(imagem), cv2.COLOR_RGB2BGR)
            h, w = img_cv.shape[:2]

            # Prepara o blob para entrada na rede neural
            blob = cv2.dnn.blobFromImage(
                cv2.resize(img_cv, (300, 300)),
                scalefactor=1.0,
                size=(300, 300),
                mean=(104.0, 177.0, 123.0)
            )
            self.net.setInput(blob)
            deteccoes = self.net.forward()

            # Filtra detecções com confiança > 50%
            rostos = []
            for i in range(deteccoes.shape[2]):
                confianca = float(deteccoes[0, 0, i, 2])
                if confianca > 0.5:
                    box = deteccoes[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = box.astype("int")
                    rostos.append({
                        "x": int(x1), "y": int(y1),
                        "largura": int(x2 - x1), "altura": int(y2 - y1),
                        "confianca": round(confianca * 100, 1)
                    })

            dados["rostos_detectados"] = len(rostos)
            dados["rostos"] = rostos

            if len(rostos) == 0:
                alertas.append({
                    "nivel": "baixo",
                    "tipo": "dnn_sem_rosto",
                    "mensagem": "Nenhum rosto detectado — se for documento de identidade, isso é inesperado"
                })

            elif len(rostos) > 1:
                alertas.append({
                    "nivel": "medio",
                    "tipo": "dnn_multiplos_rostos",
                    "mensagem": f"{len(rostos)} rostos detectados — documentos de identidade normalmente contêm apenas um"
                })

            else:
                # Analisa bordas ao redor do rosto (possível colagem)
                r = rostos[0]
                cinza = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                regiao = cinza[r["y"]:r["y"]+r["altura"], r["x"]:r["x"]+r["largura"]]
                bordas = cv2.Canny(regiao, 100, 200)
                densidade_borda = float(np.mean(bordas > 0) * 100)

                dados["densidade_bordas_rosto"] = round(densidade_borda, 2)

                if densidade_borda > 12:
                    alertas.append({
                        "nivel": "medio",
                        "tipo": "dnn_borda_suspeita",
                        "mensagem": f"Alta densidade de bordas ao redor do rosto ({densidade_borda:.1f}%) — possível foto colada"
                    })
                else:
                    dados["status_rosto"] = f"Rosto detectado com {rostos[0]['confianca']}% de confiança, sem anomalias de borda"

        except Exception as e:
            dados["erro"] = str(e)

        return {"dados": dados, "alertas": alertas}

    # ------------------------------------------------------------------ #
    #  4. PDF → IMAGEM                                                     #
    # ------------------------------------------------------------------ #

    def _pdf_para_imagem(self, filepath: str) -> Image.Image:
        """Converte a primeira página do PDF em imagem para análise."""
        doc = fitz.open(filepath)
        pagina = doc[0]
        mat = fitz.Matrix(2.0, 2.0)  # escala 2x para melhor qualidade
        pix = pagina.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # ------------------------------------------------------------------ #
    #  5. SCORE E CLASSIFICAÇÃO                                            #
    # ------------------------------------------------------------------ #

    def _calcular_score(self, alertas: list) -> int:
        """
        Score de 0 a 100 (100 = totalmente autêntico).
        Penalidades por nível de alerta.
        """
        penalidades = {"alto": 25, "medio": 12, "baixo": 5}
        total_penalidade = sum(penalidades.get(a["nivel"], 0) for a in alertas)
        return max(0, 100 - total_penalidade)

    def _classificar(self, score: int) -> str:
        if score >= 80:
            return "✅ Provavelmente autêntico"
        elif score >= 55:
            return "⚠️ Suspeito — requer revisão manual"
        else:
            return "🚨 Alta probabilidade de adulteração"
