from flask import Flask, render_template, request, session, redirect, url_for
import random
from generate_captcha import CaptchaGenerator

app = Flask(__name__)
app.secret_key = 'supersecretkey'

captcha_generator = CaptchaGenerator()  # Create an instance of CaptchaGenerator

@app.route('/')
def index():
    captcha_text = captcha_generator.generate_captcha_text()  # Generate CAPTCHA text
    session['captcha'] = captcha_text
    print(captcha_text)
    captcha_generator.create_captcha_image(captcha_text)  # Generate and save CAPTCHA image
    return render_template('index.html', image_path='/static/captcha.png')

@app.route('/verify', methods=['POST'])
def verify():
    user_input = request.form['captcha']
    if user_input == session.get('captcha', ''):
        return redirect(url_for('success'))
    else:
        return redirect(url_for('failure'))

@app.route('/success')
def success():
    emoji_positions = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(50)]
    return render_template('success.html', emoji_positions=emoji_positions)

@app.route('/failure')
def failure():
    emoji_positions = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(50)]
    return render_template('failure.html', emoji_positions=emoji_positions)

@app.route('/retry')
def retry():
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
