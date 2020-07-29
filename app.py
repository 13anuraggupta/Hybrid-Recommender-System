import numpy as np 
import pandas as pd 
import pickle
from flask import Flask,request,jsonify,render_template

app = Flask(__name__)
corrMatrix = pickle.load(open('corrmtx.pkl','rb'))

def get_similar(itemId):
    similar_ratings = corrMatrix[itemId]
    index_list=similar_ratings.index
    value_list = similar_ratings.values

    final_df = pd.DataFrame(index=index_list)
    final_df['Corr'] = value_list
    final_df = final_df.sort_values(by='Corr',ascending=False)
    return final_df

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

	item = [x for x in request.form.values()]
	similar_items = pd.DataFrame()
	similar_item = similar_items.append(get_similar(str(item[0])),ignore_index = False)
	similar_item.drop(item,axis=0,inplace=True)
	final_list = list(similar_item.index)
	return render_template('index.html',
    	prediction_text='Recommended Items for Item : {} '.format(str(item[0])),
    	recommended_item1='First Recommended Item : {} '.format(final_list[0]),
    	recommended_item2='Second Recommended Item : {} '.format(final_list[1]),
    	recommended_item3='Third Recommended Item : {} '.format(final_list[2]),
    	)

if __name__ == '__main__':
 	app.run(debug=True)
