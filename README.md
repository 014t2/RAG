# 🤖 RAG Chatbot — Consulta de Documentos con IA Local

Sistema de **Retrieval-Augmented Generation (RAG)** que permite hacer preguntas en lenguaje natural sobre tus propios documentos (PDF y TXT), utilizando embeddings multilingües y un modelo LLM local mediante Ollama.

---

## ✨ Características

- 📄 Carga automática de documentos **PDF** y **TXT**
- 🧩 División inteligente de texto en fragmentos (*chunks*) con solapamiento
- 🌍 Embeddings **multilingües** optimizados para español (`paraphrase-multilingual-MiniLM-L12-v2`)
- ⚡ Base de datos vectorial **FAISS** para búsqueda semántica rápida
- 🦙 Modelo LLM local con **Ollama** (`llama3.2`) — sin enviar datos a la nube
- 💬 Interfaz de chat por terminal con respuestas **siempre en español**

---

## 🗂️ Estructura del proyecto

```
├── data/               # Coloca aquí tus documentos (.pdf, .txt)
├── db/                 # Base de datos vectorial generada por FAISS
├── index.py            # Indexa los documentos y crea la base vectorial
├── app.py              # Chatbot interactivo que consulta la base vectorial
└── requirements.txt    # Dependencias del proyecto
```

---

## 🛠️ Requisitos previos

- Python 3.9+
- [Ollama](https://ollama.com/) instalado y en ejecución
- Modelo `llama3.2` descargado en Ollama

```bash
ollama pull llama3.2
```

---

## 📦 Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/014t2/RAG
   cd <nombre-del-proyecto>
   ```

2. Crea y activa un entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux/macOS
   venv\Scripts\activate         # Windows
   ```

3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Uso

### 1. Indexar documentos

Coloca tus archivos `.pdf` o `.txt` en la carpeta `data/` y ejecuta:

```bash
python index.py
```

Este script:
- Carga todos los documentos de `data/`
- Los divide en fragmentos de 1000 caracteres (con 200 de solapamiento)
- Genera los embeddings con el modelo multilingüe
- Crear y guarda la base de datos vectorial en `db/`

### 2. Iniciar el chatbot

```bash
python app.py
```

El chatbot cargará la base vectorial y quedará listo para responder preguntas. Escribe `salir`, `exit` o `q` para terminar.

**Ejemplo de sesión:**

```
👩 Tú: ¿Cuáles son los plazos de entrega según la documentación?
[Info: El sistema ha analizado 5 fragmentos de la documentación]
🤖 Bot: Según la documentación, los plazos de entrega son...
```

> ⚠️ **Importante:** Ejecuta siempre `index.py` antes de `app.py`. Si modificas o añades documentos, vuelve a indexar para que los cambios se reflejen en las respuestas.

---

## ⚙️ Configuración

| Parámetro | Valor actual | Descripción |
|---|---|---|
| `chunk_size` | 1000 | Tamaño de cada fragmento de texto |
| `chunk_overlap` | 200 | Solapamiento entre fragmentos |
| `model_name` (embeddings) | `paraphrase-multilingual-MiniLM-L12-v2` | Modelo de embeddings |
| `model` (LLM) | `llama3.2` | Modelo LLM de Ollama |
| `k` (retriever) | 5 | Número de fragmentos relevantes recuperados |

Para cambiar el modelo LLM, edita esta línea en `app.py`:

```python
llm = Ollama(model="llama3.2")  # Cambia por otro modelo de Ollama
```

---

## 📋 Dependencias principales

```
langchain
langchain-community
langchain-huggingface
langchain-text-splitters
faiss-cpu
sentence-transformers
ollama
pypdf
```

---

## 🔍 Cómo funciona

```
Documentos (PDF/TXT)
        │
        ▼
   index.py
   ┌─────────────────────────────────────────┐
   │  Carga → Chunks → Embeddings → FAISS DB │
   └─────────────────────────────────────────┘
        │
        ▼
    db/ (FAISS)
        │
        ▼
    app.py
   ┌─────────────────────────────────────────────────────┐
   │  Pregunta → Embedding → Búsqueda semántica (top-5)  │
   │  → Contexto + Prompt → Ollama (llama3.2) → Respuesta│
   └─────────────────────────────────────────────────────┘
```

---

## 📄 Licencia

MIT License — libre para uso personal y comercial.
