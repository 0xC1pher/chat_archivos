import torch
import ollama
import os
import argparse
import json
import requests
import time  # Para medir tiempos de ejecución

# Códigos ANSI para colores
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Función para abrir un archivo y retornar su contenido como una cadena
def open_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as infile:
            return infile.read()
    except FileNotFoundError:
        print(f"Error: El archivo {filepath} no fue encontrado.")
        return None
    except Exception as e:
        print(f"Error al leer el archivo {filepath}: {e}")
        return None

# Función para obtener contexto relevante del vault basado en la entrada del usuario
def get_relevant_context(rewritten_input, vault_embeddings, vault_content, top_k=3):
    start_time = time.time()  # Iniciar medición de tiempo
    if vault_embeddings is None or vault_embeddings.nelement() == 0:
        return []
    
    # Codificar la entrada reescrita
    input_embedding = ollama.embeddings(model='qwen2.5:3b', prompt=rewritten_input)["embedding"]
    # Calcular la similitud coseno entre la entrada y los embeddings del vault
    cos_scores = torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), vault_embeddings)
    # Ajustar top_k si es mayor que el número de puntuaciones disponibles
    top_k = min(top_k, len(cos_scores))
    if top_k == 0:
        return []
    
    # Ordenar las puntuaciones y obtener los índices top-k
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    # Obtener el contexto correspondiente del vault
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    
    end_time = time.time()  # Finalizar medición de tiempo
    print(f"Tiempo de ejecución de get_relevant_context: {end_time - start_time:.2f} segundos")
    return relevant_context

def rewrite_query(user_input_json, conversation_history, ollama_model):
    start_time = time.time()  # Iniciar medición de tiempo
    try:
        user_input = json.loads(user_input_json)["Query"]
        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-2:]])
        prompt = f"""Reescribe la siguiente consulta incorporando contexto relevante de la historia de la conversación.
        La consulta reescrita debe:
        
        - Preservar el objetivo principal y el significado de la consulta original
        - Expandir y aclarar la consulta para hacerla más específica e informativa para recuperar contexto relevante
        - Evitar introducir nuevos temas o consultas que se desvíen de la consulta original
        - NUNCA RESPONDER la consulta original, sino enfocarse en reescribirla y expandirla en una nueva consulta
        
        Retorna SOLO el texto de la consulta reescrita, sin ningún formato o explicaciones adicionales.
        
        Historia de la Conversación:
        {context}
        
        Consulta original: [{user_input}]
        
        Consulta reescrita: 
        """
        response = requests.post(
            'http://localhost:11434/v1/chat/completions',
            headers={'Authorization': 'Bearer llama3'},
            json={
                'model': ollama_model,
                'messages': [{"role": "system", "content": prompt}],
                'max_tokens': 200,
                'n': 1,
                'temperature': 0.1,
            }
        )
        response.raise_for_status()
        rewritten_query = response.json()['choices'][0]['message']['content'].strip()
        
        end_time = time.time()  # Finalizar medición de tiempo
        print(f"Tiempo de ejecución de rewrite_query: {end_time - start_time:.2f} segundos")
        return json.dumps({"Rewritten Query": rewritten_query})
    except requests.exceptions.RequestException as e:
        print(f"Error en la solicitud HTTP: {e}")
        return json.dumps({"Rewritten Query": user_input_json["Query"]})

def ollama_chat(user_input, system_message, vault_embeddings, vault_content, ollama_model, conversation_history):
    start_time = time.time()  # Iniciar medición de tiempo
    try:
        conversation_history.append({"role": "user", "content": user_input})

        if len(conversation_history) > 1:
            query_json = {
                "Query": user_input,
                "Rewritten Query": ""
            }
            rewritten_query_json = rewrite_query(json.dumps(query_json), conversation_history, ollama_model)
            rewritten_query_data = json.loads(rewritten_query_json)
            rewritten_query = rewritten_query_data["Rewritten Query"]
            print(PINK + "Consulta Original: " + user_input + RESET_COLOR)
            print(PINK + "Consulta Reescrita: " + rewritten_query + RESET_COLOR)
        else:
            rewritten_query = user_input

        relevant_context = get_relevant_context(rewritten_query, vault_embeddings, vault_content)
        if relevant_context:
            context_str = "\n".join(relevant_context)
            print("Contexto Extraído de los Documentos: \n\n" + CYAN + context_str + RESET_COLOR)
        else:
            print(CYAN + "No se encontró contexto relevante." + RESET_COLOR)

        user_input_with_context = user_input
        if relevant_context:
            user_input_with_context = user_input + "\n\nContexto Relevante:\n" + context_str

        conversation_history[-1]["content"] = user_input_with_context

        messages = [
            {"role": "system", "content": system_message},
            *conversation_history
        ]

        response = requests.post(
            'http://localhost:11434/v1/chat/completions',
            headers={'Authorization': 'Bearer llama3'},
            json={
                'model': ollama_model,
                'messages': messages,
                'max_tokens': 2000,
            }
        )
        response.raise_for_status()
        conversation_history.append({"role": "assistant", "content": response.json()['choices'][0]['message']['content']})

        end_time = time.time()  # Finalizar medición de tiempo
        print(f"Tiempo de ejecución de ollama_chat: {end_time - start_time:.2f} segundos")
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"Error en la solicitud HTTP: {e}")
        return "Hubo un error al procesar tu solicitud. Por favor, intenta de nuevo."

# Parsear argumentos de línea de comandos
print(NEON_GREEN + "Parseando argumentos de línea de comandos..." + RESET_COLOR)
parser = argparse.ArgumentParser(description="Chat de Al2")
parser.add_argument("--model", default="qwen2.5:3b", help="Modelo qwen2.5:3b a usar (por defecto: qwen2.5:3b)")
args = parser.parse_args()

# Cargar el contenido del vault
print(NEON_GREEN + "Cargando contenido del vault..." + RESET_COLOR)
vault_content = []
if os.path.exists("vault.txt"):
    with open("vault.txt", "r", encoding='utf-8') as vault_file:
        vault_content = vault_file.readlines()
    print(NEON_GREEN + "Contenido del vault cargado exitosamente." + RESET_COLOR)
else:
    print(NEON_GREEN + "Archivo del vault no encontrado. Asegúrate de que 'vault.txt' exista." + RESET_COLOR)

# Generar embeddings para el contenido del vault usando Ollama
if vault_content:
    print(NEON_GREEN + "Generando embeddings para el contenido del vault..." + RESET_COLOR)
    vault_embeddings = []
    for content in vault_content:
        try:
            response = ollama.embeddings(model='qwen2.5:3b', prompt=content)
            vault_embeddings.append(response["embedding"])
        except Exception as e:
            print(f"Ocurrió un error al generar los embeddings: {e}")
            break
    else:
        print(NEON_GREEN + "Embeddings generados exitosamente." + RESET_COLOR)
else:
    print(NEON_GREEN + "El vault está vacío. No se generarán embeddings." + RESET_COLOR)

# Convertir a tensor y mostrar los embeddings
if vault_embeddings:
    print("Convirtiendo embeddings a tensor...")
    vault_embeddings_tensor = torch.tensor(vault_embeddings)
    print(NEON_GREEN + "Embeddings convertidos a tensor exitosamente." + RESET_COLOR)
    print("Embeddings para cada línea en el vault:")
    print(vault_embeddings_tensor)
else:
    vault_embeddings_tensor = torch.tensor([])

# Bucle de conversación
print("Iniciando bucle de conversación...")
conversation_history = []
system_message = "Eres un asistente útil que es experto en extraer la información más útil de un texto dado. También aporta información relevante adicional a la consulta del usuario desde fuera del contexto dado."

while True:
    user_input = input(YELLOW + "Haz una consulta sobre tus documentos (o escribe 'salir' para terminar): " + RESET_COLOR)
    if user_input.lower() == 'salir':
        break

    try:
        response = ollama_chat(user_input, system_message, vault_embeddings_tensor, vault_content, args.model,
                               conversation_history)
        print(NEON_GREEN + "Respuesta: \n\n" + response + RESET_COLOR)
    except Exception as e:
        print(f"Ocurrió un error durante la conversación: {e}")

    # Limitar el tamaño del historial de conversación
    if len(conversation_history) > 10:
        conversation_history = conversation_history[-10:]

print(NEON_GREEN + "Bucle de conversación terminado." + RESET_COLOR)
