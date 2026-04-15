import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data"
DB_PATH = "db"

def load_documents():
    docs = []
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Carpeta '{DATA_PATH}' creada. Pon tus archivos ahí.")
        return docs

    for file in os.listdir(DATA_PATH):
        file_path = os.path.join(DATA_PATH, file)
        try:
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8", autodetect_encoding=True)
            else:
                continue
            docs.extend(loader.load())
        except Exception as e:
            print(f"Error cargando {file}: {e}")
    return docs

def main():
    print("Cargando documentos de /data...")
    documents = load_documents()
    
    if not documents:
        print("No se encontraron documentos para indexar.")
        return

    print("Dividiendo texto en fragmentos (chunks)...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    print("Creando embeddings multilingües (esto puede tardar la primera vez)...")
    # MODELO MULTILINGÜE PARA MEJOR COMPRENSIÓN EN ESPAÑOL
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    print("Creando base vectorial FAISS...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    print(f"Guardando base de datos en '{DB_PATH}'...")
    vectorstore.save_local(DB_PATH)
    print("¡Indexación completada con éxito!")

if __name__ == "__main__":
    main()