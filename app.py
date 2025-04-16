import os

from dotenv import load_dotenv
from flask import Flask, render_template, request
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from werkzeug.utils import secure_filename

from src.helper import download_hugging_face_embeddings, clear_data

app = Flask(__name__)

# 文件上传配置
UPLOAD_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'md'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 确保上传目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_hugging_face_embeddings()

persist_directory = "db"

# Now we can load the persisted database from disk, and use it as normal.
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embeddings)

llm = ChatOpenAI(temperature=0.4, max_tokens=500, model='gpt-3.5-turbo')
memory = ConversationSummaryMemory(llm=llm, memory_key = "chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm, retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k":8}), memory=memory)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)

    result = qa.invoke({"question": input})
    print(result['answer'])
    return str(result["answer"])

@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    clear_data()
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # 调用向量化函数
            os.system("python store_index.py")
            return f"File {filename} uploaded and indexed successfully"
    return '''
    <!doctype html>
    <title>Upload File</title>
    <h1>Upload a new file to vector index</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    ''' 

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)