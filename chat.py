from flask import Flask, request, jsonify
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.memory import ChatMemoryBuffer
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

PERSIST_DIR = "./storage"
data = SimpleDirectoryReader(input_dir="./data").load_data()
index = VectorStoreIndex.from_documents(data)
index.storage_context.persist(persist_dir=PERSIST_DIR)

@app.route('/api/query', methods=['POST'])
def query():
    data = request.json
    user_query = data.get('query', '')
    print(user_query)

    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        system_prompt=(
            "You should not give the answer from outside only search in directory. If you don't find it, then say sorry to the user."
        ),
    )
    response = chat_engine.chat(user_query)

    
    if hasattr(response, '__iter__'):
       
        formatted_response = ' '.join(map(str, response))
    else:
        
        formatted_response = str(response)

    return jsonify({'response': formatted_response})


if __name__ == '__main__':
    app.run(debug=True)
