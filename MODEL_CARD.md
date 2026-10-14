---
# Metadatos según la especificación de Hugging Face:
# https://github.com/huggingface/hub-docs/blob/main/modelcard.md
license: cc-by-nc-4.0
library_name: pytorch
pipeline_tag: image-classification
tags:
  - remote-sensing
  - satellite-imagery
  - self-supervised-learning
  - masked-autoencoder
  - vision-transformer
  - sentinel-2
  - multi-spectral
  - temporal
datasets:
  - fMoW-RGB
  - fMoW-Sentinel
metrics:
  - accuracy
  - mean-average-precision
  - mIoU
---

# Ficha del modelo (Model Card): SatMAE

SatMAE es un marco de **preentrenamiento autosupervisado** basado en *Masked Autoencoders* (MAE) para imágenes satelitales **temporales** (series de tiempo de una misma ubicación) y **multiespectrales** (p. ej., las 13 bandas de Sentinel‑2). Produce un codificador Vision Transformer (ViT) cuyas representaciones se ajustan (*fine‑tuning*) a tareas de teledetección como clasificación de cobertura del suelo, clasificación multietiqueta y segmentación de edificaciones.

> Esta ficha sigue la propuesta de Mitchell et al. (2019) y la plantilla oficial de Hugging Face (`modelcard_template.md`). Documenta un modelo **de terceros** (Stanford Sustainability and AI Lab); los resultados reportados provienen del artículo y del repositorio oficiales, no de una reevaluación propia.

---

## 1. Detalles del modelo

### 1.1 Descripción

SatMAE adapta el MAE de He et al. (2022) a la estructura propia de los datos satelitales. Una imagen (o una secuencia de imágenes) se divide en parches; se enmascara aleatoriamente el 75 % de ellos y un codificador ViT procesa solo los parches visibles. Un decodificador liviano intenta reconstruir los parches ocultos y el error cuadrático medio sobre esos parches es la señal de entrenamiento. Tras el preentrenamiento se descarta el decodificador y el codificador se reutiliza como *backbone*.

Las dos contribuciones técnicas frente al MAE original son:

- **SatMAE temporal.** Cada imagen de la secuencia (longitud T = 3) se "parcheja" por separado con un *patch embedding* compartido. A la codificación posicional espacial se le concatena una **codificación temporal** sinusoidal construida con el **año, el mes y la hora** de captura (se descartan día, minuto y segundo por no aportar información visual). El enmascaramiento es **independiente** entre fechas: un parche oculto en una fecha puede estar visible en otra.
- **SatMAE multiespectral.** Las bandas se agrupan por resolución espacial y longitud de onda similares, y cada grupo tiene su propio *patch embedding* y una **codificación espectral** del índice de grupo. Configuración usada con Sentinel‑2 (se descartan B1, B9 y B10, de 60 m):
  - Grupo 1 – RGB + NIR: B2, B3, B4, B8 (10 m)
  - Grupo 2 – Borde rojo: B5, B6, B7, B8A (20 m)
  - Grupo 3 – SWIR: B11, B12 (20 m)

| Campo | Valor |
|---|---|
| **Desarrollado por** | Yezhen Cong\*, Samar Khanna\*, Chenlin Meng, Patrick Liu, Erik Rozi, Yutong He, Marshall Burke, David B. Lobell, Stefano Ermon — Stanford University (\*coautores principales) |
| **Financiado por** | ODNI / IARPA (2021‑2011000004), Stanford HAI, NSF (#1651565), AFOSR (FA95501910024), ARO (W911NF‑21‑1‑0125) y Sloan Fellowship |
| **Compartido por** | Stanford Sustainability and AI Lab (`sustainlab-group`) |
| **Tipo de modelo** | Visión por computador; aprendizaje autosupervisado (MAE) con codificador Vision Transformer (ViT‑Base / ViT‑Large) |
| **Modalidad de entrada** | Imágenes satelitales RGB de alta resolución (fMoW) y multiespectrales Sentinel‑2 (no procesa lenguaje natural) |
| **Fecha / versión** | Publicado en NeurIPS 2022 (arXiv 2207.08051, v1 jul‑2022, v3 ene‑2023) |
| **Licencia** | Creative Commons **Atribución‑NoComercial 4.0** (CC BY‑NC 4.0) — uso comercial no permitido |
| **Basado en** | MAE (He et al., 2022) y ViT (Dosovitskiy et al., 2021) |

### 1.2 Fuentes del modelo

- **Repositorio:** https://github.com/sustainlab-group/SatMAE
- **Artículo:** https://arxiv.org/abs/2207.08051 · OpenReview: https://openreview.net/forum?id=WBhqzpF6KYH
- **Página del proyecto:** https://sustainlab-group.github.io/SatMAE/
- **Pesos:** Zenodo — fMoW RGB/temporal: https://doi.org/10.5281/zenodo.7369796 · fMoW‑Sentinel: https://doi.org/10.5281/zenodo.7325338
- **Datos fMoW‑Sentinel:** https://purl.stanford.edu/vg497cb6002

---

## 2. Usos

### 2.1 Uso directo

El modelo preentrenado **no está pensado para usarse "tal cual"** sobre un problema final: su salida directa es la reconstrucción de parches enmascarados. Los usos directos razonables son:

- Extraer **representaciones (embeddings)** de imágenes satelitales para análisis exploratorio, búsqueda por similitud o agrupamiento.
- Usar los puntos de control (*checkpoints*) ya ajustados para clasificar imágenes en las **62 categorías funcionales de fMoW** (aeropuerto, astillero, estadio, zona residencial, etc.).

### 2.2 Uso derivado (*downstream*)

El caso de uso principal es como **backbone preentrenado** que se ajusta con pocos datos etiquetados para tareas de teledetección, por ejemplo:

- Clasificación de cobertura y uso del suelo (EuroSAT, NAIP).
- Clasificación multietiqueta de escenas Sentinel‑2 (BigEarthNet).
- Segmentación semántica de edificaciones (SpaceNet v1, acoplado a PSANet).
- Tareas con impacto social mencionadas por los autores: estimación de pobreza, desarrollo de infraestructura, estimación de población, monitoreo agrícola.

Usuarios previstos: investigadores y profesionales en teledetección, SIG, catastro, planificación territorial y ciencia de datos geoespaciales, con capacidad de cómputo GPU y conocimiento de *fine‑tuning*.

### 2.3 Usos fuera de alcance

- **Uso comercial** de los pesos o el código: lo prohíbe la licencia CC BY‑NC 4.0.
- Decisiones automatizadas de alto impacto sobre personas o predios (asignación de subsidios, fiscalización, avalúos o cobros) **sin validación local y supervisión humana**.
- Vigilancia o identificación de individuos: las resoluciones de entrenamiento (fMoW de alta resolución y Sentinel‑2 de 10–20 m) no están pensadas para ello y ese uso no está contemplado.
- Aplicar el modelo a **sensores distintos** (SAR, LiDAR, hiperespectral, drones) o a bandas/grupos diferentes a los del preentrenamiento sin reentrenar o adaptar.
- Detección de objetos: no se evaluó en el artículo.

---

## 3. Sesgos, riesgos y limitaciones

- **Sesgo geográfico.** fMoW y fMoW‑Sentinel están desbalanceados hacia **Norteamérica y Europa**. Los propios autores advierten que las representaciones pueden favorecer esas regiones y recomiendan cautela al aplicarlas a distribuciones geográficas distintas. Para Colombia y el resto de Latinoamérica el desempeño **no está garantizado** y debe medirse.
- **Sesgo de categorías.** Las 62 clases de fMoW describen infraestructura y usos del suelo con sesgo hacia entornos urbanos e industriales; coberturas típicas del trópico (páramo, bosque húmedo, cultivos de pequeña escala) están poco representadas.
- **Dependencia de etiquetas de calidad variable.** El conjunto fMoW‑Sentinel usa compuestos de 90 días para mitigar nubes, pero las imágenes son pequeñas (≈45 × 60 px en promedio) y se redimensionan a 96 × 96, lo que limita detalle espacial.
- **Costo computacional.** El preentrenamiento usó 8 GPU NVIDIA V100; las variantes temporal y multiespectral multiplican el número de *tokens* (×3), por lo que memoria y tiempo crecen de forma notable. Los autores lo señalan como limitación abierta.
- **Resultados de referencia, no de producción.** Las métricas provienen de *benchmarks* académicos; no reflejan robustez ante cambios de sensor, nubosidad persistente, estacionalidad tropical o errores de georreferenciación.
- **Privacidad.** Los autores consideran que las imágenes públicas usadas no exponen información personal identificable, pero reconocen que la imagen satelital puede ser objeto de abuso por actores malintencionados.

### 3.1 Recomendaciones

1. **Validar localmente** con un conjunto de prueba de la región de interés antes de cualquier uso aplicado, reportando métricas desagregadas por zona (urbana/rural, región natural, altitud).
2. Mantener la **misma configuración de bandas, normalización (medias y desviaciones de fMoW‑Sentinel) y tamaño de entrada** del preentrenamiento, o reajustar el modelo.
3. Mantener **supervisión humana** cuando los resultados alimenten decisiones catastrales, fiscales o de política pública.
4. Respetar la licencia **no comercial** y citar a los autores.
5. Reportar emisiones de cómputo si se reentrena (ver sección 8).

---

## 4. Cómo empezar a usar el modelo

```bash
git clone https://github.com/sustainlab-group/SatMAE.git
cd SatMAE
conda env create -f env.yml   # entorno sugerido por el repositorio
# Descargar un checkpoint de Zenodo, p. ej. fmow_finetune.pth (ViT-Large, fMoW RGB)
```

Ejemplo orientativo para cargar el codificador ajustado sobre fMoW RGB (usar dentro de la carpeta del repositorio):

```python
import torch
import models_vit  # módulo del repositorio oficial

# Los checkpoints publicados usan una cabeza de 1000 salidas (nota del repositorio);
# las 62 clases reales de fMoW ocupan los primeros índices.
model = models_vit.vit_large_patch16(num_classes=1000, global_pool=False)
checkpoint = torch.load("fmow_finetune.pth", map_location="cpu")
model.load_state_dict(checkpoint["model"], strict=False)
model.eval()

x = torch.randn(1, 3, 224, 224)   # imagen RGB normalizada, 224 x 224
with torch.no_grad():
    logits = model(x)
pred = logits[:, :62].argmax(dim=1)
```

Para preentrenar o ajustar las variantes temporal (`--model_type temporal`) y multiespectral (`--model_type group_c`), el README del repositorio trae los comandos completos con `torch.distributed.launch`.

---

## 5. Detalles de entrenamiento

### 5.1 Datos de entrenamiento

| Conjunto | Contenido | Tamaño |
|---|---|---|
| **fMoW RGB** (Functional Map of the World) | Imágenes RGB de alta resolución, con metadatos de ubicación y fecha; 62 categorías funcionales | Usado en versión de una imagen y en versión temporal (secuencias de 3 imágenes de la misma ubicación) |
| **fMoW‑Sentinel** (creado por los autores) | Compuestos de 90 días de reflectancia de superficie Sentinel‑2, 13 bandas, en las ubicaciones de fMoW | 712 874 entrenamiento · 84 939 validación · 84 966 prueba |

En la versión temporal, si una ubicación no tiene suficientes fechas distintas, se duplica la imagen original. La partición de fMoW garantiza que imágenes de una misma ubicación queden en la misma partición (sin fuga de datos).

### 5.2 Preprocesamiento

- **fMoW RGB:** redimensionar el lado corto a 224 px conservando proporción, recorte aleatorio de 224 × 224, normalización con media y desviación del conjunto.
- **fMoW‑Sentinel:** normalización estándar por banda, recorte aleatorio de 0,2–1,0 del área, redimensionado a 96 × 96, volteo horizontal aleatorio. Se eliminan B1, B9 y B10.

### 5.3 Hiperparámetros

| Parámetro | fMoW RGB (una imagen) | fMoW RGB temporal | fMoW‑Sentinel |
|---|---|---|---|
| Backbone | ViT‑Large | ViT‑Large | ViT‑Base / ViT‑Large |
| Tamaño de entrada / parche | 224 / 16 | 224 / 16 (×3 fechas) | 96 / 8 (×3 grupos = 432 tokens) |
| Proporción de enmascarado | 0,75 | 0,75 por imagen (independiente) | 0,75 (independiente) |
| Épocas de preentrenamiento | 800 | 100 (checkpoint publicado: 50) | 50 en el artículo; 200 en los checkpoints publicados |
| Tamaño de lote efectivo (preentr.) | 4096 | 1024 | 4096 |
| Épocas de *fine‑tuning* | 50 | 50 | 30 |
| Aumentos en *fine‑tuning* | Mixup, CutMix | Mixup, CutMix | Mixup 0,8, CutMix 1,0, drop path 0,1, reprob 0,25 |

Optimizador y programación de tasa de aprendizaje iguales a MAE (AdamW, calentamiento y decaimiento coseno). Régimen de precisión: no especificado por los autores.

### 5.4 Velocidades, tamaños y tiempos

| Experimento | Horas‑GPU (V100) |
|---|---|
| Preentrenamiento fMoW RGB temporal | 768 |
| *Fine‑tuning* fMoW RGB temporal | 192 |
| Preentrenamiento fMoW‑Sentinel | 576 |
| *Fine‑tuning* fMoW‑Sentinel | 192 |
| *Fine‑tuning* NAIP / SpaceNet / BigEarthNet / EuroSAT | 30 / 50 / 16 / 4 |

El codificador ViT‑Large tiene del orden de 300 millones de parámetros; ViT‑Base, del orden de 86 millones.

---

## 6. Evaluación

### 6.1 Datos de prueba

Conjuntos de validación de fMoW RGB, fMoW RGB temporal y fMoW‑Sentinel; y, para transferencia: **NAIP** (RGB + infrarrojo, 66 clases, 55 529 imágenes de validación), **EuroSAT** (27 000 imágenes Sentinel‑2, 10 clases), **BigEarthNet** (Sentinel‑2, 19 clases, multietiqueta; 10 % del entrenamiento) y **SpaceNet v1** (6 940 imágenes con máscaras de edificaciones; 1 940 de prueba).

### 6.2 Factores

Los autores desagregan resultados por: modalidad de entrada (RGB, temporal, multiespectral), estrategia de enmascarado (consistente vs. independiente), uso de codificación temporal/espectral, estrategia de agrupamiento de bandas, tamaño del backbone, proporción de enmascarado y tamaño de parche. **No** se reportan resultados desagregados por región geográfica, país, nivel de urbanización ni condiciones atmosféricas; esa es una brecha importante desde la perspectiva de Mitchell et al. (2019).

### 6.3 Métricas

- **Exactitud Top‑1 / Top‑5** para clasificación de una etiqueta (fMoW, NAIP, EuroSAT).
- **Precisión media promedio (mAP)** para clasificación multietiqueta (BigEarthNet).
- **mIoU** (intersección sobre unión media) para segmentación (SpaceNet v1).

### 6.4 Resultados (reportados por los autores)

| Tarea | Línea base más fuerte | SatMAE |
|---|---|---|
| fMoW RGB, *fine‑tuning*, Top‑1 | ViT‑L con pesos MAE de ImageNet: 76,91 % | **77,84 %** |
| fMoW RGB temporal, Top‑1 | MAE + aumento en prueba: 78,90 % | **81,49 %** (con aumento en prueba) |
| fMoW‑Sentinel, Top‑1 (ViT‑L, 50 ép.) | ResNet152 ImageNet: 54,46 % | **61,48 %** |
| fMoW‑Sentinel, Top‑1 (ViT‑L, 200 ép.) | — | **63,84 %** |
| NAIP, Top‑1 | ViT‑L desde cero: 69,65 % | **71,77 %** |
| EuroSAT, Top‑1 (13 bandas) | SeCo: 93,14 % | **98,98 %** |
| BigEarthNet, mAP | SeCo: 82,62 % | 82,13 % |
| SpaceNet v1, mIoU | GASSL: 78,51 % | 78,07 % |

Los checkpoints públicos en el repositorio reportan 77,78 % (fMoW RGB) y 79,99 % (fMoW temporal, **sin** aumento en prueba).

### 6.5 Resumen

SatMAE mejora el estado del arte en clasificación de escenas satelitales (hasta ≈7 % en *benchmarks* supervisados y hasta ≈14 % en transferencia de cobertura del suelo), y es competitivo pero **no superior** en BigEarthNet y SpaceNet frente a métodos contrastivos con ResNet. Las ablaciones muestran que las ganancias se deben principalmente a la codificación temporal/espectral y al enmascarado independiente, no solo al tamaño del modelo.

---

## 7. Examen del modelo (interpretabilidad)

Los autores presentan visualizaciones de reconstrucción: con enmascarado independiente, SatMAE reconstruye objetos que cambian entre fechas (p. ej., número de aviones en un aeropuerto) y aprende correlaciones entre grupos de bandas (reconstruye un punto brillante en SWIR a partir de los otros grupos). Un análisis de enmascarado de bandas muestra que RGB, borde rojo, NIR y SWIR son muy informativas, mientras que B9 y B10 aportan poco, lo que justifica descartarlas.

---

## 8. Impacto ambiental

Estimado por los autores con la calculadora *Machine Learning Impact* (Lacoste et al., 2019):

- **Hardware:** NVIDIA Tesla V100‑SXM2‑16GB (TDP 250 W), 8 GPU
- **Horas usadas:** 960 h‑GPU para el modelo temporal (preentrenamiento + *fine‑tuning*); 768 h‑GPU para el multiespectral
- **Proveedor de nube:** Google Cloud Platform
- **Región:** us‑central1 (0,57 kg CO₂eq/kWh)
- **Emisiones:** ≈136,8 kg CO₂eq (temporal) y ≈109,4 kg CO₂eq (multiespectral), compensadas al 100 % por el proveedor según los autores

---

## 9. Especificaciones técnicas

### 9.1 Arquitectura y objetivo

Autocodificador asimétrico: codificador ViT (solo parches visibles) + decodificador transformer liviano (todos los tokens con *mask token* aprendible). Objetivo: **MSE por píxel sobre los parches enmascarados**. Para el modelo temporal ViT‑L, la codificación es de 640 dimensiones espaciales + 384 temporales (año, mes, hora); para el multiespectral, 768 espaciales + 256 espectrales (D = 1024).

### 9.2 Infraestructura

- **Hardware:** 8 × NVIDIA V100 (Google Cloud). El repositorio advierte que la variante temporal consume mucha memoria.
- **Software:** Python, PyTorch, entrenamiento distribuido (`torch.distributed`), basado en el código de MAE de Meta; registro opcional con Weights & Biases.

---

## 10. Consideraciones éticas

- **Beneficio esperado:** reducir la dependencia de datos etiquetados (costosos en teledetección) para medir fenómenos económicos, sociales y ambientales que informan políticas públicas.
- **Riesgos:** uso con fines de vigilancia, decisiones automatizadas sin validación y sesgo geográfico que puede degradar el desempeño justo en regiones con menos datos —a menudo, las que más necesitan estas herramientas—.
- **Financiación:** parte del financiamiento proviene de agencias de inteligencia y defensa de EE. UU. (IARPA, AFOSR, ARO); los autores aclaran que las conclusiones no representan posiciones oficiales de esas entidades.

## 11. Advertencias para el contexto colombiano

- Antes de usar SatMAE en cartografía, catastro multipropósito o monitoreo territorial en Colombia, conviene construir un conjunto de validación local con categorías propias (p. ej., coberturas Corine Land Cover adaptada para Colombia) y medir el desempeño por región natural.
- La nubosidad persistente en zonas andinas y amazónicas puede reducir la disponibilidad de escenas Sentinel‑2 útiles; los compuestos temporales son una estrategia a evaluar.
- La licencia no comercial impide incorporar estos pesos en productos o servicios con fines de lucro.

---

## 12. Citación

**BibTeX:**

```bibtex
@inproceedings{satmae2022,
  title     = {Sat{MAE}: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery},
  author    = {Yezhen Cong and Samar Khanna and Chenlin Meng and Patrick Liu and Erik Rozi and Yutong He and Marshall Burke and David B. Lobell and Stefano Ermon},
  booktitle = {Advances in Neural Information Processing Systems},
  editor    = {Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
  year      = {2022},
  url       = {https://openreview.net/forum?id=WBhqzpF6KYH}
}
```

**APA:**

Cong, Y., Khanna, S., Meng, C., Liu, P., Rozi, E., He, Y., Burke, M., Lobell, D. B., & Ermon, S. (2022). SatMAE: Pre-training transformers for temporal and multi-spectral satellite imagery. En *Advances in Neural Information Processing Systems 35 (NeurIPS 2022)*. https://arxiv.org/abs/2207.08051

**Marco de la ficha:**

Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. En *Proceedings of the Conference on Fairness, Accountability, and Transparency* (pp. 220–229). https://arxiv.org/abs/1810.03993

---

## 13. Glosario

- **MAE (Masked Autoencoder):** método autosupervisado que oculta parches de la imagen y entrena al modelo a reconstruirlos.
- **ViT (Vision Transformer):** arquitectura transformer que procesa una imagen como secuencia de parches.
- **Enmascarado independiente / consistente:** en series temporales o grupos de bandas, ocultar parches distintos en cada imagen/grupo (independiente) o los mismos en todos (consistente).
- **Fine‑tuning:** ajuste de los pesos preentrenados con datos etiquetados de la tarea final.
- **Sondeo lineal (*linear probing*):** entrenar solo un clasificador lineal sobre las representaciones congeladas.

---

## 14. Autor de la ficha

**Juan Sebastián del Mar Aragón Ríos**
Código estudiantil: 20202025077
Ingeniería Catastral y Geodesia — Universidad Distrital Francisco José de Caldas, Bogotá, Colombia

## 15. Contacto de la ficha

GitHub: [@juandelmmar](https://github.com/juandelmmar)

*Última actualización: 14 de octubre de 2026.*
