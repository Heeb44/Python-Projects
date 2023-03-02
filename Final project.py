# Charles Rez, crez@usc.edu
# ITP 216, Fall 2021
# Final Project

# import statements for all the functions
from flask import Flask, redirect, render_template, request, session, url_for, send_file
import os
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import itertools
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import plot_decision_regions
from sklearn import preprocessing

matplotlib.use("Agg")

# set up flask
app = Flask(__name__)

# convert csv file to dataframe
df = pd.read_csv("university_rank.csv")


# display the scatter plot figure
@app.route("/fig/plots")
def request_plots():

    # get the figure from the plot generation function
    fig = plot_scatter_and_pie(session["country"])
    img = BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype="image/png")


# display the pie chart figure
@app.route("/fig/pie")
def request_pie_chart():

    # get the figure from the pie chart generation function
    fig = pie_chart_of_countries(session["country"])
    img = BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype="image/png")


@app.route("/fig/ml_visual")
def request_ml_visual():

    fig = machine_visual(session["stat1"], session["stat2"])
    img = BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype="image/png")


# original page redirects to /homepage
@app.route("/", methods=["POST", "GET"])
def reroute_to_home():
    return redirect(url_for("homepage"))


# homepage template rendering, uses list of countries and list of different qualities in universities
@app.route("/homepage", methods=["POST", "GET"])
def homepage():
    return render_template("homepage.html", universities=get_country_list(), qualities=get_university_qualities())


# app route for picking the country
@app.route("/university/data", methods=["POST", "GET"])
def university_data():
    if request.method == "POST":        # checks for a post request
        session["country"] = request.form["country"]        # saves the chosen country for the session
        if session["country"] == "Others":      # checks if they wanted a listed country or others
            return redirect(url_for("others_page"))     # redirects for others
        else:
            return redirect(url_for("university_page", country=session["country"]))     # redirects for another country
    return render_template("university.html", country=session["country"])


# app route for the chosen country
@app.route("/university/<country>", methods=["POST", "GET"])    # dynamic endpoint for country
def university_page(country):
    if request.method == "POST":       # redirecting back to home
        return redirect(url_for("homepage"))
    return render_template("university.html", country=session["country"])       # loads the university html page


# app route for other countries given
@app.route("/university/others", methods=["POST", "GET"])
def others_page():
    if request.method == "POST":        # redirect to home
        return redirect(url_for("homepage"))
    return render_template("others.html", list=get_smaller_countries_list())        # load the others template


# app route for the qualities list
@app.route("/university/qualities", methods=["POST", "GET"])
def get_qualities():
    if request.method == "POST":        # get the user inputs as stats
        session["stat1"] = request.form["stat1"]
        session["stat2"] = request.form["stat2"]
        value1 = request.form["stat_value1"]
        value2 = request.form["stat_value2"]

        # predict the college
        session["predicted_college"] = nearest_neighbor(session["stat1"], session["stat2"], value1, value2)
        return redirect(url_for("predicted_uni"))       # redirect to the prediction page


# app route for predicted universities page
@app.route("/university/predicted", methods=["POST", "GET"])
def predicted_uni():
    if request.method == "POST":        # return to homepage
        return redirect(url_for("homepage"))
    return render_template("predicted.html", predicted=session["predicted_college"])    # load the predicted html page


# gets list of all the countries that were listed in other
def get_smaller_countries_list():

    # sums the number of entries for each country
    country_counts = df['country'].value_counts()
    country_dict = dict(country_counts)     # converts country as key and sum to a dictionary
    new_dict = {}

    # sets up a new dictionary where if the sum is greater than 19, it is listed as "other"
    for key, group in itertools.groupby(country_dict, lambda k: 'Others' if (country_dict[k] > 19) else k):
        new_dict[key] = sum([country_dict[k] for k in list(group)])

    # gets the country names and converts it to a list
    labels = new_dict.keys()
    labels_list = list(labels)
    labels_list.remove("Others")    # removes the "other" category, countries with more than 19 institutions not listed
    return labels_list      # returns the new list


# gets list of all countries with enough institutions
def get_country_list():

    # creates a dictionary with country as key and number of occurrences as value
    country_counts = df['country'].value_counts()
    country_dict = dict(country_counts)

    # sets up a new dictionary where if the country had fewer than 20 institutions they count as an "other"
    new_dict = {}
    for key, group in itertools.groupby(country_dict, lambda k: 'Others' if (country_dict[k] < 20) else k):
        new_dict[key] = sum([country_dict[k] for k in list(group)])

    # gets the countries and converts it to a list, returns it
    labels = new_dict.keys()
    labels_list = list(labels)
    return labels_list


# gets a list of the different criteria that a user can prioritize
def get_university_qualities():

    # hard coded it because the csv format is different than what I wanted the text to say
    list_of_qualities = ["Quality of Education", "Alumni Employment", "Quality of Faculty", "Publications", "Influence",
                         "Citations", "Patents"]
    return list_of_qualities


# creates the pie chart with the given country exploded
# this is only used for the "other" countries
def pie_chart_of_countries(chosen):

    # creates a dictionary of each country and the number of universities from there
    country_counts = df['country'].value_counts()
    country_dict = dict(country_counts)

    # only countries with more than 20 universities are included, the rest are in the "other" category
    newdict = {}
    for key, group in itertools.groupby(country_dict, lambda k: 'Others' if (country_dict[k] < 20) else k):
        newdict[key] = sum([country_dict[k] for k in list(group)])

    # gets the labels as the keys of the dictionary and then the sizes as the count
    labels = newdict.keys()
    sizes = newdict.values()

    # creates a list from the countries labels
    labels_list = list(labels)
    explode = []

    # checks for the country being the one the user chose, sets explode value to 0.2 for that country
    for country in labels_list:
        if country == chosen:
            explode.append(0.2)
        else:
            explode.append(0)

    # makes the plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # makes a pie chart with the labels as the countries and the chosen country exploded, adding shadows
    ax.pie(sizes, labels=labels, explode=explode, autopct='%1.1f%%', startangle=0, shadow=True)
    ax.axis('equal')

    # gives a title
    plt.title("Percentage of Institutions in " + chosen, fontweight='bold', size=18)

    # tightens the chart
    plt.tight_layout()

    # returns the figure
    return fig


# creates the scatter plot and pie chart together of the universities
# this is because displaying them separately would not work with the html
def plot_scatter_and_pie(country):

    # new dataframe of just the countries and their scores
    # then converted to dictionary with country as key and list of scores as value
    new_df = df[["country", "score"]]
    country_score_dict = new_df.groupby('country')['score'].apply(list).to_dict()

    # creates a dictionary of each country and the number of universities from there
    country_counts = df['country'].value_counts()
    country_count_dict = dict(country_counts)

    # only countries with more than 20 universities are included, the rest are in the "other" category
    new_count_dict = {}
    for key, group in itertools.groupby(country_count_dict, lambda k: 'Others' if (country_count_dict[k] < 20) else k):
        new_count_dict[key] = sum([country_count_dict[k] for k in list(group)])

    # gets the labels as the keys of the dictionary and then the sizes as the count
    x1_labels = new_count_dict.keys()
    x1_sizes = new_count_dict.values()

    # creates a list from the countries labels
    x1_labels_list = list(x1_labels)
    explode = []

    # checks for the country being the one the user chose, sets explode value to 0.2 for that country
    for item in x1_labels_list:
        if item == country:
            explode.append(0.2)
        else:
            explode.append(0)

    # gets a number label for each x value based on the length of the country list
    x2_labels = []
    n = 1
    scores = country_score_dict[country]
    for item in scores:
        x2_labels.append(n)
        n += 1

    # sorts the scores in ascending order and gets the max length
    scores.sort()
    max_val = len(scores)

    # creates the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # makes a pie chart with the labels as the countries and the chosen country exploded, adding shadows
    ax1.pie(x1_sizes, labels=x1_labels, explode=explode, autopct='%1.1f%%', startangle=0, shadow=True)
    ax1.axis('equal')
    ax1.set_title("Percentage of Institutions in " + country, fontweight='bold', size=18)

    # makes a scatter plot with scores as y values and count as x values
    ax2.scatter(y=scores, x=x2_labels)

    # title for figure for scatter plot, x and y labels as well
    ax2.set_title("Aggregate Scores of Universities in " + country, fontweight='bold', size=18)
    ax2.set_xlabel("University Count", size=14)
    ax2.set_ylabel("University Scores", size=14)

    # x axis goes every 8 values from 0 up to the max number of institutions for scatter plot
    # x label is just count
    plt.xticks(np.arange(0, max_val, 8))

    # y axis goes from 20 (no score was lower than 40) up to 100, counting by 4 for scatter plot
    # y label is score
    plt.yticks(np.arange(20, 100, 4))

    # puts a dashed grid and a legend giving the total number of institutions from a given country on scatter plot
    plt.grid('on', linestyle='--', alpha=0.3)
    plt.legend(title="Total: " + str(max_val) + " institutions", loc=2, prop={'size': 30})
    plt.tight_layout()

    # return the figure
    return fig


# finding the best college given user input
def nearest_neighbor(s1, s2, v1, v2):

    # changing the parameter name for knn
    if s1 == "Quality of Education":
        s1 = "quality_of_education"
    elif s1 == "Alumni Employment":
        s1 = "alumni_employment"
    elif s1 == "Quality of Faculty":
        s1 = "quality_of_faculty"
    else:
        s1 = s1.lower()

    # changing the parameter name for knn
    if s2 == "Quality of Education":
        s2 = "quality_of_education"
    elif s2 == "Alumni Employment":
        s2 = "alumni_employment"
    elif s2 == "Quality of Faculty":
        s2 = "quality_of_faculty"
    else:
        s2 = s2.lower()

    # training the data with the parameters given, outputting a university
    x_train = df[[s1, s2]].astype(int).values
    y_train = df["institution"]

    # using the nearest neighbor and fitting the model
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(x_train, y_train)

    # getting 3 outputs. 1 output is the user input, the other is the user input with an inflated first parameter
    # last input is user input but inflated second parameter
    # this is intended on getting 3 different university outputs
    x_new = [v1, v2]
    x_newer = [v1, str(int(v2)+10)]
    x_newest = [str(int(v1)+10), v2]

    # making the 3 predictions based on the 3 inputs
    prediction_one = knn.predict([x_new])
    prediction_two = knn.predict([x_newer])
    prediction_three = knn.predict([x_newest])

    # putting the predictions in a list and returning them
    prediction_list = [prediction_one[0], prediction_two[0], prediction_three[0]]
    return prediction_list


# creates the visual for machine learning
# this part took almost a minute for my pc to load but it did eventually
# the individual model will plot if run separately
def machine_visual(s1, s2):

    # changing the parameter name for knn
    if s1 == "Quality of Education":
        stat1 = "quality_of_education"
    elif s1 == "Alumni Employment":
        stat1 = "alumni_employment"
    elif s1 == "Quality of Faculty":
        stat1 = "quality_of_faculty"
    else:
        stat1 = s1.lower()

    # changing the parameter name for knn
    if s2 == "Quality of Education":
        stat2 = "quality_of_education"
    elif s2 == "Alumni Employment":
        stat2 = "alumni_employment"
    elif s2 == "Quality of Faculty":
        stat2 = "quality_of_faculty"
    else:
        stat2 = s2.lower()

    # changing the data values to ints
    x_train = df[[stat1, stat2]].astype(int).values
    y_train = df["institution"]

    # changes the y values to classified values and not integers
    le = preprocessing.LabelEncoder()
    y_train = le.fit_transform(y_train)

    # fits model using the new y data and old x data
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(x_train, y_train)

    # sets up the graph, and plots the decision region
    fig, ax = plt.subplots(figsize=(12, 12))
    plot_decision_regions(x_train, y_train, clf=knn)

    # labels and title
    plt.xlabel(s1)
    plt.ylabel(s2)
    plt.title("KNN Model")

    # returns fig
    return fig


# runs everything
if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True)
