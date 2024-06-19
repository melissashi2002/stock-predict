## What is this project?

This webapp is designed for people to look up stock information in a fast time.
The webapp has the following majority functions:
1. Look up a stock's real time price
2. Save stocks so user don't need to type stock code everytime they wants to look for a stock
3. Display a saved stock's recent price thrend and predict a future price

# Technical Architecture
![280d641aaf1bbdd54f417419697f76d](https://github.com/CS222-UIUC-SP24/group-project-team-55/assets/161802156/5dd2ee04-8d5b-43ec-bdec-0bb0edf126a6)

# Team

- **Xianyang Zhan**: Flask webapp full stack development
- **Melissa Shi**: Frontend design and development
- **Hangao Zhang**: Light prediction function and frontend development
- **Allen Chen**: Prediction model development

# Environment Setup

## Package Installation
To install all required packages, navigate to your source directory (cd <directory>), and run the following following in terminal.
```
pip install -r requirements.txt
```
## Run the webapp
Navigate to your source directory (cd <directory>), and run the following python file.
```
main.py
```

# Webapp Instructions

1. To access any functions of the webapp:
Register an account, then login using your registered email and password.

2. To save a stock:
Go to home page and type the stock's code in the input box and hit 'add stock code'

3. To find the price trend and predicted price:
You should be able to see all your saved stocks in the home page, click the stock code and you will be navigated to a new page that shows you a price thrend diagram and predicted stock price.
