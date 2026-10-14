# SatMAE — Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery

SatMAE es un marco de preentrenamiento autosupervisado, desarrollado en Stanford University y presentado en NeurIPS 2022, que adapta los *Masked Autoencoders* (MAE) a imágenes satelitales. El modelo oculta el 75 % de los parches de una imagen y aprende a reconstruirlos con un codificador Vision Transformer, incorporando dos ideas propias de la teledetección: una codificación temporal (año, mes y hora) con enmascarado independiente para series de imágenes de un mismo lugar, y una agrupación de bandas espectrales de Sentinel‑2 con codificación espectral propia. El codificador resultante se ajusta con pocos datos etiquetados a tareas como clasificación de cobertura del suelo, clasificación multietiqueta y segmentación de edificaciones, superando el estado del arte de su momento hasta en un 7 % en *benchmarks* supervisados y hasta en un 14 % en transferencia.

## 📄 Ficha del modelo

👉 **[Ver la ficha del modelo (MODEL_CARD.md)](MODEL_CARD.md)**

La ficha sigue el marco de Mitchell et al. (2019), *Model Cards for Model Reporting*, y la plantilla oficial de Hugging Face (`huggingface_hub/templates/modelcard_template.md`).

## 🔗 Recursos del modelo original

- Página del proyecto: https://sustainlab-group.github.io/SatMAE/
- Artículo: https://arxiv.org/abs/2207.08051
- Código oficial: https://github.com/sustainlab-group/SatMAE

## 👤 Autor de la ficha

**Juan Sebastián del Mar Aragón Ríos**
Código estudiantil: 20202025077
Ingeniería Catastral y Geodesia — Universidad Distrital Francisco José de Caldas, Bogotá, Colombia
GitHub: [@juandelmmar](https://github.com/juandelmmar)
