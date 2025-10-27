"""Clasificador de Requisitos ISO/IEC 25010"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

CATEGORIAS = {
    0: "Funcionalidad",
    1: "Eficiencia/Performance",
    2: "Compatibilidad",
    3: "Usabilidad",
    4: "Confiabilidad",
    5: "Seguridad",
    6: "Mantenibilidad",
    7: "Portabilidad"
}
CATEGORIA_A_ID = {v: k for k, v in CATEGORIAS.items()}
MODELO = "dccuchile/bert-base-spanish-wwm-cased"

class DatasetRequisitos(Dataset):
    def __init__(self, textos, etiquetas, tokenizer, max_length=128):
        self.textos = textos
        self.etiquetas = etiquetas
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.textos)
    
    def __getitem__(self, idx):
        texto = str(self.textos[idx])
        etiqueta = self.etiquetas[idx]
        encoding = self.tokenizer(
            texto, add_special_tokens=True, max_length=self.max_length,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(etiqueta, dtype=torch.long)
        }

def main():
    print("=" * 60)
    print("CLASIFICADOR DE REQUISITOS ISO 25010")
    print("=" * 60)
    
    print("\n[1/7] Cargando datos")
    df = pd.read_csv('data/raw/requisitos_ejemplo.csv')
    df['label'] = df['categoria'].map(CATEGORIA_A_ID)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    print(f"OK - {len(df)} requisitos cargados")
    
    print("\n[2/7] Dividiendo datos")
    train, temp = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['label'])
    print(f"OK - Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    
    print("\n[3/7] Descargando modelo BETO ")
    tokenizer = AutoTokenizer.from_pretrained(MODELO)
    modelo = AutoModelForSequenceClassification.from_pretrained(MODELO, num_labels=8)
    print("OK - Modelo BETO cargado")
    
    print("\n[4/7] Preparando datasets")
    train_dataset = DatasetRequisitos(train['requisito'].values, train['label'].values, tokenizer)
    val_dataset = DatasetRequisitos(val['requisito'].values, val['label'].values, tokenizer)
    print("OK - Datasets listos")
    
    print("\n[5/7] Entrenando modelo (esto tarda ~30-45 min)")
    args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=7,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=50,
        weight_decay=0.01,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True
    )
    
    entrenador = Trainer(
        model=modelo,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    entrenador.train()
    print("OK - Entrenamiento completado")
    
    print("\n[6/7] Guardando modelo...")
    modelo.save_pretrained('./models/clasificador_requisitos')
    tokenizer.save_pretrained('./models/clasificador_requisitos')
    print("OK - Modelo guardado en: ./models/clasificador_requisitos")
    
    print("\n[7/7] Evaluando")
    device = torch.device('cpu')
    modelo.to(device)
    modelo.eval()
    
    predicciones = []
    for texto in test['requisito']:
        inputs = tokenizer(texto, return_tensors='pt', max_length=128, padding='max_length', truncation=True).to(device)
        with torch.no_grad():
            predicciones.append(torch.argmax(modelo(**inputs).logits, dim=1).item())
    
    print("\n" + "=" * 60)
    print("RESULTADOS:")
    print("=" * 60)
    print(classification_report(test['label'].values, predicciones, target_names=list(CATEGORIAS.values()), zero_division=0))
    
    cm = confusion_matrix(test['label'].values, predicciones)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(CATEGORIAS.values()), yticklabels=list(CATEGORIAS.values()))
    plt.title('Matriz de Confusion')
    plt.tight_layout()
    plt.savefig('results/matriz_confusion.png')
    print("\nOK - Matriz guardada: results/matriz_confusion.png")
    
    print("\n" + "=" * 60)
    print("COMPLETADO!")
    print("=" * 60)
    print("\nSiguiente paso: python src\\predecir.py")

if __name__ == "__main__":
    main()
