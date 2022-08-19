import streamlit as st
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


head=st.container()
dataset=st.container()
feature=st.container()
modelTraining=st.container()

## change background and font
st.markdown(
	"""
	<style>
	.main{
	background-color: #F5F5F5
	}
	</style>
	""",
	unsafe_allow_html=True
)


## if data is large: solution ! cacheing
@st.cache
def get_data(filename):
	taxi_data=pd.read_csv(filename)
	
	return taxi_data

with dataset:
	st.header('NYC taxi dataset')
	st.text('In this project I look into the transactions of taxis in NYC')

	taxi_data=get_data('/Users/kachunlee/Desktop/Random-Forest-apps-for-2022-NYC-taxi/data/taxi.csv')
	st.write(taxi_data.head())

	st.subheader('Tip amount distribution of head 50 ')
	fare_dist =st.bar_chart(taxi_data['tip_amount'].head(50))

with feature:
	st.header('The feature I choose')
	
	st.markdown('* **first feature:** : maybe choose tip_amount')


with modelTraining:
	st.header('Time to train the model')
	st.text('Choose the hyperparameters of the model and see how the performance changes')

	sel_col,disp_col=st.columns(2)

	max_depth=sel_col.slider('What shoud be the max_depth of the model ?', min_value=10 , max_value=100, value=20 , step=10 )

	n_estimators=sel_col.selectbox('How many tree should there be?' , options=[100,200,300,'No limit'], index=0 ) ## index zero mean the default value should be the first element 
	
	sel_col.text('Here is a list of features in my data:')
	sel_col.write(taxi_data.columns)


	input_feature= sel_col.text_input('Which feature should be used as the input feature?','tip_amount')

	if n_estimators =='No limit':
		Rfrg=RandomForestRegressor(max_depth=max_depth )
	else:
		Rfrg=RandomForestRegressor(max_depth=max_depth , n_estimators=n_estimators)
	X=taxi_data[[input_feature]][:5000:]
	y= taxi_data[['trip_distance']][:5000:]

	Rfrg.fit(X,y)
	prediction=Rfrg.predict(y)

	disp_col.subheader('Mean absolute error of the model is ')
	disp_col.write(mean_absolute_error(y,prediction))


	disp_col.subheader('Mean squared error of the model is ')
	disp_col.write(mean_squared_error(y,prediction))


	disp_col.subheader('R squared score of the model is ')
	disp_col.write(r2_score(y,prediction))

