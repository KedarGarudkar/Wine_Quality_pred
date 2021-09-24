from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS

application = Flask(__name__)
RFscalar = pickle.load(open("New_RF_standardScalar.sav", 'rb'))
RFmodel = pickle.load(open("New_RF_modelForPrediction.sav", 'rb'))
RFpca_model = pickle.load(open("new_RF_pca_model.sav", 'rb'))

CORS(application)
application.config['DEBUG'] = True


@application.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

standard_to = StandardScaler()
@application.route("/", methods=['POST'])
def predict():
    if request.method == 'POST':
        resp = request.form

        fixed_acidity = float(resp.get('fixed acidity'))
        volatile_acidity = float(resp.get('volatile acidity'))
        citric_acid = float(resp.get('citric acid'))
        residual_sugar = float(resp.get('residual sugar'))
        chlorides = float(resp.get('chlorides'))
        free_sulfur_dioxide = float(resp.get('free sulfur dioxide'))
        total_sulfur_dioxide = float(resp.get('total sulfur dioxide'))
        density = float(resp.get('density'))
        pH = float(resp.get('pH'))
        sulphates = float(resp.get('sulphates'))
        alcohol = float(resp.get('alcohol'))

        dict_pred = [fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]
        data_df = pd.DataFrame(dict_pred)
        data_df1 = data_df.transpose()
        # scaled_data = scalar.transform(data_df1)
        # principal_data = pca_model.transform(scaled_data)
        # predict = model.predict(principal_data)
        scaled_data = RFscalar.transform(data_df1)
        principal_data = RFpca_model.transform(scaled_data)
        predict = RFmodel.predict(principal_data)
        print(predict)
        if predict[0] == 0:
            result = 'Bad'
        else:
            result = 'Good'

        return render_template('result.html', wine_quality="WINE Quality is {}".format(result))
    else:
        return render_template('index.html')
if __name__=="__main__":
    application.run(debug=True)
