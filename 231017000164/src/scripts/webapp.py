from flask import Flask, render_template
from flask_bootstrap import Bootstrap

app = Flask(__name__)
bootstrap = Bootstrap(app)


@app.route('/')
def my_index():
    return render_template('bitcoin_all_predict.html')


@app.route('/now')
def new_data():
    return render_template('bitcoin_30_predict.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)