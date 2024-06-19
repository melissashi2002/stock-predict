from flask import Blueprint, render_template, request, flash, jsonify
from flask_login import login_required, current_user 
from . import db
from .models import Note
from . import db
import json
import yfinance as yf
from datetime import date
import streamlit as st
import prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
import os 

views = Blueprint('views',__name__)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# pd.set_option('display.width', 220)

static_folder = os.path.join(views.root_path, 'static')
file_path = os.path.join(static_folder, 'uiuc-gpa-dataset2023.csv') 
df = pd.read_csv(file_path)

@views.route('/home', methods=['GET','POST'])
@login_required
def home():
    if request.method == 'POST':
        note =request.form.get('note')
        try:
            stockCode = note
            for n in current_user.notes:
                if n.data == note.upper():
                    flash('Stock already added!', category='error')
                    return render_template("home.html", user=current_user)
            stock = yf.Ticker(stockCode)
            note = note.upper()
            new_note = Note(data=note, user_id=current_user.id, price=getPrice(note))
            db.session.add(new_note)
            db.session.commit()
            flash('Stock added!', category='success')           
        except:
            flash('Stock Code Error! The stock code doesn\'t exists', category='error')

    return render_template("home.html", user=current_user)
@views.route('/course', methods=['GET','POST'])
@login_required
def course():
    info = 'Nothing'
    if request.method == 'POST':
        try:
            if  not request.form.get('subject').upper() or not request.form.get('number'):
                flash('The course does not exist', category='error')
                return render_template("course.html", user=current_user, info=info)
            subject = request.form.get('subject').upper()
            number = int(request.form.get('number'))
            course_df = get_course_GPA(subject,number)
            info = str(course_df)
            if len(info) < 220:
                flash('The course does not exist', category='error')
        except:
            print(subject+str(number))
            flash('The course does not exist', category='error')
    
    return render_template("course.html", user=current_user, info=info)


@views.route('/delete-note', methods=['POST'])
def delete_note():
    note = json.loads(request.data)
    noteId= note['noteId']
    note = Note.query.get(noteId)
    if note:
        if note.user_id == current_user.id:
            db.session.delete(note)
            db.session.commit()
            flash('Stock removed!', category='success')
    return jsonify({})

@views.route('/refresh-stock', methods=['POST'])
def refresh_stock():
    user = current_user
    for note in user.notes:
        stockCode = note.data
        stock = yf.Ticker(stockCode)
        temp = stock.history()
        price = temp['Close'].iloc[-1]
        price = round(price, 2)
        note.price = price
        db.session.commit()
    return jsonify({})

def getPrice(code):
    stockCode = code
    stock = yf.Ticker(stockCode)
    temp = stock.history()
    price = temp['Close'].iloc[-1]
    price = round(price, 2)
    return price


def calculate_gpa(row):
    count = row['A+']+row['A']+row['A-']+row['B+']+row['B']+row['B-']+row['C+']+row['C']+row['C-']+row['D+']+row['D']+row['D-']+row['F']
    gpa = row['A+']*4+row['A']*4+row['A-']*3.67+row['B+']*3.33+row['B']*3+row['B-']*2.67+row['C+']*2.33+row['C']*2+row['C-']*1.67+row['D+']*1.33+row['D']*1+row['D-']*0.67
    return gpa/count
def calculate_ap(row):
    count = row['A+']+row['A']+row['A-']+row['B+']+row['B']+row['B-']+row['C+']+row['C']+row['C-']+row['D+']+row['D']+row['D-']+row['F']
    AP = (row['A+']+row['A'])/count
    return AP*100

def get_course_GPA(subject,number):
    Course = df[(df['Number'] == number) & (df['Subject'] == subject) & (df['Year'] >= 2014)]
    Course['GPA1'] = Course.apply(calculate_gpa, axis=1).round(2)
    Course['A%1'] = Course.apply(calculate_ap, axis=1).round(2)
    position = Course.columns.get_loc('W')
    Course.insert(position, 'GPA', Course['GPA1'])
    Course.insert(Course.columns.get_loc('W'), 'A+&A%', Course['A%1'])
    Course = Course.drop('GPA1', axis=1)
    Course = Course.drop('A%1', axis=1)
  
    return Course