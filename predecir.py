"""Predictor de Requisitos"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

CATEGORIAS = {
    0: "Funcionalidad", 1: "Eficiencia/Performance",
    2: "Compatibilidad", 3: "Usabilidad",
    4: "Confiabilidad", 5: "Seguridad",
    6: "Mantenibilidad", 7: "Portabilidad"
}

print("Cargando modelo")
if not os.path.exists('./models/clasificador_requisitos'):
    print("ERROR: Primero debes entrenar el modelo")
    print("Ejecuta: python src\clasificador.py")
    exit()

tokenizer = AutoTokenizer.from_pretrained('./models/clasificador_requisitos')
modelo = AutoModelForSequenceClassification.from_pretrained('./models/clasificador_requisitos')
print("OK - Modelo cargado\n")

print("=" * 60)
print("CLASIFICADOR DE REQUISITOS")
print("=" * 60)
print("Escribe 'salir' para terminar\n")

while True:
    req = input("Requisito: ").strip()
    
    if req.lower() in ['salir', 'exit']:
        print("Adios!")
        break
    
    if not req:
        continue
    
    inputs = tokenizer(req, return_tensors='pt', truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = modelo(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0][pred].item()
    
    print(f"  -> Categoria: {CATEGORIAS[pred]}")
    print(f"  -> Confianza: {conf:.1%}\n")
