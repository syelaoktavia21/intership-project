from flask import Flask, render_template, request, redirect, url_for, flash, make_response, send_file
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, IntegerField, RadioField
from wtforms.validators import InputRequired, Length, NumberRange
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import mysql.connector
from sqlalchemy import create_engine
from datetime import date
import datetime
import pandas as pd
import plotly.express as px
import matplotlib
matplotlib.use('Agg')
from sklearn.model_selection import cross_val_score
import pickle

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/klinik'
db = SQLAlchemy(app)
app.config['SQLALCHEMY_ECHO'] = True
bcrypt = Bcrypt(app)
app.config['SECRET_KEY'] = 'thisisasecretkey'

login_manager = LoginManager(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    __tablename__ = 'login_admin'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})
    submit = SubmitField('Login')

class KlasifikasiForm(FlaskForm):
    umur = IntegerField('Umur (Years)', validators=[InputRequired(), NumberRange(min=0)], render_kw={"placeholder": "Umur (Years)"})
    jenis_kelamin = RadioField('Jenis Kelamin', choices=[('laki-laki', 'Laki-laki'), ('perempuan', 'Perempuan')], validators=[InputRequired()])
    laju_pernapasan = IntegerField('Laju Pernapasan (breaths/min)', validators=[InputRequired(), NumberRange(min=0)], render_kw={"placeholder": "Laju Pernapasan (breaths/min)"})
    detak_jantung = IntegerField('Detak Jantung (beats/min)', validators=[InputRequired(), NumberRange(min=0)], render_kw={"placeholder": "Detak Jantung (beats/min)"})
    temperature = IntegerField('Temperature (Celcius)', validators=[InputRequired(), NumberRange(min=0)], render_kw={"placeholder": "Temperature (Celcius)"})
    denyut_jauntung = IntegerField('Denyut Jauntung (%)', validators=[InputRequired(), NumberRange(min=0)], render_kw={"placeholder": "Denyut Jauntung (%)"})
    tekanan_sistolik = IntegerField('Tekanan Darah Sistolik (mmHg)', validators=[InputRequired(), NumberRange(min=0)], render_kw={"placeholder": "Tekanan Darah Sistolik (mmHg)"})
    tekanan_diastolik = IntegerField('Tekanan Darah Diastolik (mmHg)', validators=[InputRequired(), NumberRange(min=0)], render_kw={"placeholder": "Tekanan Darah Diastolik (mmHg)"})
    tekanan_arteri = IntegerField('Tekanan Darah Arteri (mmHg)', validators=[InputRequired(), NumberRange(min=0)], render_kw={"placeholder": "Tekanan Darah Arteri (mmHg)"})
    submit = SubmitField('Prediksi')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('index'))
    return render_template('login.html', form=form)

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    return render_template('index.html')

@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/pasien')
def pasien():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute('SELECT * FROM pasien')
    patients = cursor.fetchall()
    cursor.close()
    conn.close()  # Close the connection
    return render_template('pasien.html', patients=patients)

@app.route('/jadwaldokter')
def jadwaldokter():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute('SELECT * FROM dokter')
    jadwal_dokter = cursor.fetchall()
    cursor.close()
    conn.close()  # Close the connection
    return render_template('jadwaldokter.html', jadwal_dokter=jadwal_dokter)

@app.route('/informasi')
def informasi():
    return render_template('informasi.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/toggle_sidebar', methods=['GET'])
def toggle_sidebar():
    return 'Sidebar toggled'

def get_db_connection():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="klinik"
    )
    return conn

@app.route('/surat')
def surat():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute('SELECT * FROM pasien')
    patients = cursor.fetchall()
    cursor.close()
    conn.close()  # Close the connection

    # Render halaman HTML dengan data pasien
    return render_template('surat_rujukan.html', patient=patients)

@app.route('/klasifikasi', methods=['GET', 'POST'])
@login_required
def klasifikasi():
    form = KlasifikasiForm()
    result = None
    if form.validate_on_submit():
        umur = form.umur.data
        jenis_kelamin_male = 1 if form.jenis_kelamin.data == 'male' else 0
        jenis_kelamin_female = 1 if form.jenis_kelamin.data == 'female' else 0
        laju_pernapasan = form.laju_pernapasan.data
        detak_jantung = form.detak_jantung.data
        temperature = form.temperature.data
        denyut_jauntung = form.denyut_jauntung.data
        tekanan_sistolik = form.tekanan_sistolik.data
        tekanan_diastolik = form.tekanan_diastolik.data
        tekanan_arteri = form.tekanan_arteri.data

        features = np.array([[umur, jenis_kelamin_male, jenis_kelamin_female, laju_pernapasan, detak_jantung,
                              temperature, denyut_jauntung, tekanan_sistolik, tekanan_diastolik, tekanan_arteri]])
        
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        
        prediction = model.predict(features)[0]

        if prediction == 0:
            result = 'Hijau'
        elif prediction == 1:
            result = 'Kuning'
        else:
            result = 'Merah'

    return render_template('klasifikasi.html', form=form, result=result)

if __name__ == '__main__':
    with app.app_context():
        app.run(debug=True, threaded=True)
