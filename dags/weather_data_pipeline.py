import os
import json
import requests
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from joblib import dump
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator



# DAG's default arguments
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2022, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}


# Define the DAG
dag = DAG(
    'weather_data_pipeline',
    default_args=default_args,
    description='DAG for retrieving weather data',
    schedule_interval='* * * * *',
    catchup=False,
)



# Function to retrieve weather data from OpenWeatherMap API
def get_weather_data(api_key, city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
    response = requests.get(url)
    response.raise_for_status()  # This will raise an error for bad requests
    return response.json()

# Function to save data to a JSON file
def save_to_file(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)

# Main function to get and save weather data for a list of cities
def weather_data_to_File(api_key, cities):
    for city in cities:
        weather_data = get_weather_data(api_key, city)
        timestamp = datetime.now().strftime("%Y-%m-%d %H-%M")
        filename = f"/app/raw_files/{city}_{timestamp}.json"
        save_to_file(weather_data, filename)

api_key = "c799fefb4d64db284987fc9c66c34806"
cities = ['paris', 'london', 'washington','berlin','madrid','rom','tunis','Orleans','tokio'] 


# Transform JSON data from raw files into CSV format
def transform_data_into_csv(n_files=None, filename='data.csv'):
    parent_folder = '/app/raw_files'
    files = sorted([f for f in os.listdir(parent_folder) if f.endswith('.json')], reverse=True)
    if n_files:
        files = files[:n_files]

    dfs = []  # List to store data for each city

    for f in files:
        with open(os.path.join(parent_folder, f), 'r') as file:
            data_city = json.load(file)
            # Extracting date and time from filename and removing the .json extension
            datetime_str = f.split('_')[-1].replace('.json', '')
            dfs.append(
                {
                    'temperature': data_city['main']['temp'],
                    'city': data_city['name'],
                    'pressure': data_city['main']['pressure'],
                    'date': datetime_str  # Use the modified datetime_str without .json
                }
            )

    df = pd.DataFrame(dfs)
    df.to_csv(os.path.join('/app/clean_data', filename), index=False)


				   
def prepare_data(path_to_data='/app/clean_data/fulldata.csv'):
				  
    df = pd.read_csv(path_to_data)
    df = df.sort_values(['city', 'date'], ascending=True)
													  
    dfs = []
    for c in df['city'].unique():
        df_temp = df[df['city'] == c]				
        df_temp.loc[:, 'target'] = df_temp['temperature'].shift(-1)   
        for i in range(1, 10):
            df_temp.loc[:, f'temp_m-{i}'] = df_temp['temperature'].shift(i)											 
        df_temp = df_temp.dropna()					
        dfs.append(df_temp)							 
        df_final = pd.concat(dfs, ignore_index=True)
    df_final = df_final.drop(['date', 'temperature', 'city'], axis=1)
    df_final = pd.get_dummies(df_final)
    X = df_final.drop('target', axis=1)
    y = df_final['target']

    return X, y

# Function to compute model score
def compute_model_score(model_class_name, **kwargs):
    X, y = prepare_data()
    model_class = eval(model_class_name)
    model = model_class()
    score = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error').mean()
    return score

# Function to train and save the best model
def train_and_save_best_model(**kwargs):
    scores = {}
    scores['LinearRegression'] = kwargs['ti'].xcom_pull(task_ids='train_linear_regression')
    scores['DecisionTreeRegressor'] = kwargs['ti'].xcom_pull(task_ids='train_decision_tree')
    scores['RandomForestRegressor'] = kwargs['ti'].xcom_pull(task_ids='train_random_forest')

    best_model_name = max(scores, key=scores.get)
    X, y = prepare_data()

    best_model = eval(best_model_name)()
    best_model.fit(X, y)
    model_directory = '/app/raw_files/model' 
    if not os.path.exists(model_directory):
        os.makedirs(model_directory, exist_ok=True)
    
    model_path = os.path.join(model_directory, f'{best_model_name}.pickle')
    dump(best_model, model_path)
    print(f"{best_model_name} model saved at {model_path}")

# Define task  1
get_weather_task = PythonOperator(
    task_id='get_weather_data',
    python_callable=weather_data_to_File,
    op_kwargs={'api_key': api_key, 'cities': cities},
    dag=dag,
)

# Define task 2 to transform the last 20 files
transform_last_20_task = PythonOperator(
    task_id='transform_last_20',
    python_callable=transform_data_into_csv,
    op_kwargs={'n_files': 20, 'filename': 'data.csv'},
    dag=dag,
)

# Define task 3 to transform all files
transform_all_task = PythonOperator(
    task_id='transform_all',
    python_callable=transform_data_into_csv,
    op_kwargs={'filename': 'fulldata.csv'},
    dag=dag,
)

# Tasks for training models
train_linear_regression = PythonOperator(
    task_id='train_linear_regression',
    python_callable=compute_model_score,
    op_kwargs={'model_class_name': 'LinearRegression'},
    provide_context=True,
    dag=dag,
)

train_decision_tree = PythonOperator(
    task_id='train_decision_tree',
    python_callable=compute_model_score,
    op_kwargs={'model_class_name': 'DecisionTreeRegressor'},
    provide_context=True,
    dag=dag,
)

train_random_forest = PythonOperator(
    task_id='train_random_forest',
    python_callable=compute_model_score,
    op_kwargs={'model_class_name': 'RandomForestRegressor'},
    provide_context=True,
    dag=dag,
)

select_and_save_best_model = PythonOperator(
    task_id='select_and_save_best_model',
    python_callable=train_and_save_best_model,
    provide_context=True,
    dag=dag,

)

# Set the order of tasks execution
get_weather_task >> [transform_last_20_task,transform_all_task]

transform_all_task >> [train_linear_regression, train_decision_tree, train_random_forest] >> select_and_save_best_model

# Set the task
#get_weather_task


