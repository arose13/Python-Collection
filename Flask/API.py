# Flask API Test
from flask import Flask, request, jsonify

app = Flask(__name__)
methods = ['GET', 'POST']


def square(x):
    return x ** 2


# Look how simple it is to start up!
@app.route('/', methods=methods)
def index():
    return 'Oh Hello'


# You can feed it strings
@app.route('/<name>', methods=methods)
def welcome_me(name):
    return 'Welcome {}'.format(name)


# Notice how numbers ,ust be feed in
@app.route('/math/<float:x>', methods=methods)
def do_math(x):
    return '{} * {} = {}'.format(x, x, square(x))


# Receiving Post Information
@app.route('/postman', methods=methods)
def read_post_request():
    message = None

    if request.method == 'POST':
        message = request.data

    return message


if __name__ == '__main__':
    app.run()  # Actually starts the application
