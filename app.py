from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

DB_PATH = "db"

def main():
    print("Cargando configuración...")
    # DEBE SER EL MISMO QUE EN INDEX.PY
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    print("Cargando base de datos vectorial...")
    try:
        vectorstore = FAISS.load_local(
            DB_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"❌ Error al cargar la DB: {e}. ¿Has ejecutado index.py?")
        return

    print("Cargando modelo...")
    # Seleccionamos que modelo queremos utilizar
    llm = Ollama(model="llama3.2")

    template = """Usa los siguientes fragmentos de contexto para responder a la pregunta del usuario. 
Si no encuentras la respuesta en el contexto, di simplemente que no lo sabes basándote en la documentación, no inventes nada.
Responde siempre en ESPAÑOL de forma clara.

Contexto:
{context}

Pregunta: {question}

Respuesta detallada en español:"""

    PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

    # Configuramos el recuperador para que traiga los 5 mejores fragmentos
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

    print("\n Chatbot listo (Escribe 'salir' para terminar)\n")

    while True:
        query = input("👩 Tú: ")
        if query.lower() in ["salir", "exit", "quit", "q"]:
            break

        try:
            response = qa.invoke(query)
            
            # INFO DE DEPURACIÓN
            num_docs = len(response["source_documents"])
            print(f"[Info: El sistema ha analizado {num_docs} fragmentos de la documentación]")
            
            print(f"🤖 Bot: {response['result']}\n")
        except Exception as e:
            print(f"❌ Error en la consulta: {e}\n")

if __name__ == "__main__":
    main()