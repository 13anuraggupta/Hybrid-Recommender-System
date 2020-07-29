import numpy as np 
import pandas as pd 
import pickle
from flask import Flask,request,jsonify,render_template

app = Flask(__name__)
corrMatrix = pickle.load(open('corrmtx.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))
df = pd.read_csv('Users.csv')
from sklearn.preprocessing import MinMaxScaler
customer_data = pd.read_csv('ColumnCountsperCustomer.csv')


def get_similar(itemId):
    similar_ratings = corrMatrix[itemId]
    index_list=similar_ratings.index
    value_list = similar_ratings.values
    final_df = pd.DataFrame(index=index_list)
    final_df['Corr'] = value_list
    final_df = final_df.sort_values(by='Corr',ascending=False)
    return final_df
def get_item_item(test_user) :
    
#     Get the list of all items that the user has bought.
    
    user_items = []
    data = customer_data[customer_data['CustomerNumber']==test_user]
    user_items = [column for column in data.columns if data[column].iloc[0]!=0][1:]
    
    
    similar_items = pd.DataFrame()
    
#    For all items the user has bought find similar items to the  items iteratively
    for item in user_items:
        recommended_df = get_similar(item)
        items = recommended_df.index
        
#         for every item that the user has not bought append it to the similar items dataframe.
        
        for item in items:
            if item not in user_items:
                similar_items = similar_items.append(recommended_df.loc[item])
                
#   Remove all the items that have a negative correlation with the items that the user has bought.
    similar_items = similar_items[similar_items['Corr']>0]
    
#   For items with multiple occurence in the dataframe find the mean of their correlation
#   Append the item and the corr to a new final items dataframe.
    
    final_item_list = []
    final_similarity = []

    for item in np.unique(similar_items.index):
        data = similar_items.loc[item]
        final_item_list.append(item)
        final_similarity.append(data.Corr.mean())
    
#   Normalize the values of the corr in between 0-1
    items_df = pd.DataFrame()
    items_df['Item'] = final_item_list
    items_df['Similarity'] = final_similarity
    x = np.asanyarray(final_similarity,dtype=np.float32)
    x-=np.min(final_similarity)
    x/= (np.max(final_similarity)-np.min(final_similarity))
    items_df['Score'] = x
    
    
#   Sort the dataframe on the basis of the normalized score and return it
    items_df = items_df.sort_values(by='Score',ascending=False)
    return items_df

def get_item_users(test_user):
    
#     Get the distance and indices of the closest 6 neighbours of the given user
    
    distances,indices = model.kneighbors(df[df['Customer Code']==test_user][['DaysPresent','Revenue']],n_neighbors=6)
#     Remove the first entry from indices and distances as it corresponds to the same test_user    
    indices = indices.squeeze()[1:]
    distances = distances.squeeze()[1:]
    
#    Get the customer code from the given indices
    
    similar_users=[df['Customer Code'].values[i] for i in indices]
    
    
#    Find all the items that the user has bought in the past and append it to the user_items list    
    user_items = []
    data = customer_data[customer_data['CustomerNumber']==test_user]
    user_items = [column for column in data.columns if data[column].iloc[0]!=0]
    
    
    
    recommended_items= []
    similarity = []
    
#   For every similar user that has bought something in the past append all the things that 
#   the user has not bought yet and also append the distance of the similar user.
#   
    for i in range(len(similar_users)) :
        if similar_users[i] in list(customer_data['CustomerNumber']):
            data = customer_data[customer_data['CustomerNumber']==similar_users[i]]
#         print(data.head())
            for column in data.columns[1:] :
                if data[column].iloc[0]!=0:
                    if column not in user_items:
                        recommended_items.append(column)
                        similarity.append(distances[i])
                        
    users_df = pd.DataFrame()
    users_df['Item'] = recommended_items
    users_df['Similarity'] = similarity
    
#   Normalize the distance and subtract it from 1 so that we get our final similarity.
#   Lesser the distance higher should be the similarity.
    
    x = np.asanyarray(similarity,dtype=np.float32)
    x-=np.min(similarity)
    x/= (np.max(similarity)-np.min(similarity))
    x = 1-x
    users_df['Score'] = x
    
#   Sort the dataframe on the basis of the score and return it
    users_df = users_df.sort_values(by='Score',ascending=False)    
    return users_df    

def get_recommendation(userID) :

#    For the given user get the item recommendations from the item item model and user item model.    
    
    item_item_recommendation = get_item_item(userID)
    user_item_recommendation = get_item_users(userID)
    final_item_list =[]
    score = []
    confidence = []
#     
#   for every item in item item recommendation if the item has also been recommended in 
#   user item recommendation add a value of 1 to confidence indicating high confidence.
    for item in item_item_recommendation.Item.values:
        if item in user_item_recommendation.Item.values:
            final_item_list.append(item)
            confidence.append(1)
            
#   if the score of user item recommendation is greater than item item append it to the score
            
            if user_item_recommendation[user_item_recommendation['Item']==item].Score.values[0] > item_item_recommendation[item_item_recommendation['Item']==item].Score.values[0] :           
                score.append(user_item_recommendation[user_item_recommendation['Item']==item].Score.values[0])
#   else append the score of the item item recommendation          
            else:
                score.append(item_item_recommendation[item_item_recommendation['Item']==item].Score.values[0])
        
        
#   if the recommended item from item item is not in user items append 0 to confidence displaying 
#   low confidence and append the score from the item item recommendations.
        else :
            final_item_list.append(item)
            confidence.append(0)
            score.append(item_item_recommendation[item_item_recommendation['Item']==item].Score.values[0])
    recommendation_df = pd.DataFrame()
    recommendation_df['Items'] = final_item_list
    recommendation_df['Confidence'] = confidence 
    recommendation_df['Score'] = score
    
#   remove all the items that have a score of less than or equal to 0 
#   sort the dataframe by grouping them on the basis of confidence and score
    
    recommendation_df = recommendation_df[recommendation_df['Score']>0]
    recommendation_df=recommendation_df.sort_values(['Confidence', 'Score'], ascending=[False, False])
#   return the final recommendation dataframe.    
    return recommendation_df




@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

	user = [x for x in request.form.values()]
	items_df = get_recommendation(user[0])
	items_df.reset_index(inplace=True)
	items_df.drop(columns='index',axis=1,inplace=True)
	item = list(items_df['Items'].values)
	return render_template('index.html',  tables=[items_df.to_html(classes='data')], titles=items_df.columns.values)

if __name__ == '__main__':
 	app.run(debug=True)
