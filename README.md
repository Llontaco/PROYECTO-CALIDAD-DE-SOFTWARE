# PROYECTO-CALIDAD-DE-SOFTWARE
# 🎯 Clasificador de Requisitos ISO/IEC 25010

Sistema de clasificación automática de requisitos de software usando BETO (BERT en español).

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Descripción

Clasifica automáticamente requisitos de software en 8 categorías según el estándar ISO/IEC 25010:

- **Funcionalidad** - ¿Qué hace el sistema?
- **Eficiencia/Performance** - ¿Qué tan rápido?
- **Compatibilidad** - ¿Con qué se integra?
- **Usabilidad** - ¿Fácil de usar?
- **Confiabilidad** - ¿Es confiable?
- **Seguridad** - ¿Está protegido?
- **Mantenibilidad** - ¿Fácil de modificar?
- **Portabilidad** - ¿Multi-plataforma?

## 🚀 Instalación

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/clasificador-requisitos-iso25010.git
cd clasificador-requisitos-iso25010

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

## 💻 Uso

### Entrenar el modelo

```bash
python src/clasificador.py
```

### Clasificar requisitos

```bash
python src/predecir.py
```

**Ejemplo:**
```
Requisito: El sistema debe cifrar contraseñas con SHA-256
→ Categoría: Seguridad
→ Confianza: 90.0%
```

## 📊 Resultados

- **Accuracy**: 87.3%
- **F1 Macro**: 0.50
- **Dataset**: 354 requisitos en español

| Categoría | F1-Score |
|-----------|----------|
| Eficiencia | 1.00 ✅ |
| Usabilidad | 1.00 ✅ |
| Seguridad | 1.00 ✅ |
| Portabilidad | 0.90 ✅ |
| Mantenibilidad | 1.00 ✅ |
| Confiabilidad | 0.50 ✅ |
| Conpatibilidad | 1.00 ✅ |
| Funcionalidad  | 0.60 ✅ |


## 📁 Estructura

```
clasificador-requisitos-iso25010/
├── data/
│   └── raw/requisitos_ejemplo.csv
    └── raw/RespaN Equivalence.csv
├── src/
│   ├── clasificador.py          # Entrenamiento
│   └── predecir.py              # Predicción
├── models/
│   └── clasificador_requisitos/ # Modelo entrenado
├── results/
│   └── matriz_confusion.png     # Visualización
└── requirements.txt
```

## 🛠️ Tecnologías

- **Python 3.9+**
- **PyTorch** - Framework de deep learning
- **Transformers (Hugging Face)** - BETO (BERT español)
- **scikit-learn** - Métricas de evaluación
- **Pandas** - Manipulación de datos

## 📄 Paper

Ver paper completo: [Clasificación de Requisitos ISO 25010](docs/paper.pdf)

## 👥 Autores

- **[Grupo 5 ]** - [henry.llontop@unmsm.edu]

*[UNMSM] - 2025*

## 📚 Referencias

- ISO/IEC 25010:2011 - Estándar de calidad de software
- BETO: Cañete et al. (2020) - BERT en español
- Dataset PROMISE: https://www.kaggle.com/datasets/iamvaibhav100/software-requirements-dataset

## 📝 Licencia

Este proyecto está bajo la Licencia MIT

---

⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub

