from flask import Flask
app = Flask(__name__)

@app.route('/')
def home():
    return "Hey there!"

if __name__=='__main__':
    s=home()
    print(s)
    app.run(host="127.0.0.14",debug=True)