from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, Field, ValidationError, validator
from typing import List, Union
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score, mean_squared_error as MSE
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from fastapi.responses import FileResponse, StreamingResponse
from io import BytesIO
import pandas as pd
import numpy as np
import re
import pickle
import csv

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int = Field(..., gt=0)
    selling_price: int = Field(..., gt=0)
    km_driven: int = Field(..., gt=0)
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: Union[None, str]
    engine: Union[None, str]
    max_power: Union[None, str]
    torque: Union[None, str]
    seats: Union[None, float, str]

    @validator('*')
    def empty_str_to_none(cls, v):
        if v == '':
            return None
        return v


class Items(BaseModel):
    object: List[Item]

    @validator('*')
    def empty_str_to_none(cls, v):
        if v == '':
            return None
        return v


# Загружаем информацию о модели
with open('data.pickle', 'rb') as f:
    data = pickle.load(f)


def handle_torque(x: str):
    '''Принимает на вход строку, содержащую
    сведения о крутящем моменте,
    возвращает крутящий момент в Н*м в формате float
    '''
    k = 1
    if 'nm' in x.lower():
        x = x.lower().split('nm')[0]
    if 'kgm' in x.lower():
        x = x.lower().split('kgm')[0]
        k = 9.81
    if '@' in x:
        x = x.split('@')[0]
    if '/' in x:
        x = x.split(' /')[0]
    if '(' in x:
        x = x.split('(')[0]
    return round(k * float(x), 1)


def handle_max_power(x: str):
    '''
    Принимает на вход строку, содержащую
    сведения о мощности автомобиля,
    возвращает мощность в формате float
    '''
    if len(x.replace(' bhp', '')):
        return float(x.replace(' bhp', ''))
    else:
        return np.nan


def handle_mileage(x: str):
    '''
    Принимает на вход строку, содержащую
    сведения о расходе автомобиля,
    возвращает его в kmpl в формате float
    '''
    if 'kmpl' in x:
        return float(x.replace(' kmpl', ''))
    if 'km/kg' in x:
        return 1.4 * float(x.replace(' km/kg', ''))


def handle_engine(x: str):
    '''
    Принимает на вход строку, содержащую
    сведения об объеме двигателя автомобиля,
    возвращает его в формате float
    '''
    if len(x.replace(' CC', '')):
        return float(x.replace(' CC', ''))
    else:
        return np.nan


def handle_name(x: str):
    '''
    Принимает на вход строку, содержащую сведения
    об объеме двигателя автомобиля,
    возвращает его в формате float
    '''
    if len(x.split()):
        return x.split()[0]
    else:
        return np.nan

def preprocessing(df):
    '''
    Препроцессинг поступающих данных.
    Работа с пропусками, подготовка к обработке моделью.

    input: pandas DataFrame с базовыми признаками
    return: pandas DataFrame с признаками, готовыми к предсказанию
    '''
    # Удаляем таргет
    df = df.drop('selling_price', axis=1)

    # Заполняем пустые строки NaN-ами
    df['mileage'] = df['mileage'].replace(r'^\s*$', np.nan, regex=True)
    df['engine'] = df['engine'].replace(r'^\s*$', np.nan, regex=True)
    df['max_power'] = df['max_power'].replace(r'^\s*$', np.nan, regex=True)
    df['torque'] = df['torque'].replace(r'^\s*$', np.nan, regex=True)
    df['seats'] = df['seats'].replace(r'^\s*$', np.nan, regex=True)

    # Уточняем формат для дальнейшей работы с csv
    df['year'] = df['year'].astype('Int64')
    df['km_driven'] = df['km_driven'].astype('Int64')
    df['seats'] = df['seats'].astype('float64')

    # Добавляем производителя автомобиля
    df['name'] = df['name'].apply(handle_name)

    # Приводим значения к числовым
    df['mileage'] = df['mileage'].apply(lambda x: handle_mileage(x) if not pd.isnull(x) else np.nan)
    df['engine'] = df['engine'].apply(lambda x: handle_engine(x) if not pd.isnull(x) else np.nan)
    df['max_power'] = df['max_power'].apply(lambda x: handle_max_power(x) if not pd.isnull(x) else np.nan)
    df['torque'] = df['torque'].apply(lambda x: handle_torque(x) if not pd.isnull(x) else np.nan)

    # Заполняем пропуски медианой
    df['mileage'].fillna(data['mileage_median'], inplace=True)
    df['engine'].fillna(data['engine_median'], inplace=True)
    df['max_power'].fillna(data['max_power_median'], inplace=True)
    df['seats'].fillna(data['seats_median'], inplace=True)
    df['torque'].fillna(data['torque_median'], inplace=True)

    # Добавляем One-hot encoding
    encoded = data['ohe'].transform(df[["name", "seller_type", "fuel", "transmission", "owner", "seats"]]).toarray()
    encoded = pd.DataFrame(encoded, columns=data['ohe_names'])

    X_test = pd.concat([df, encoded], axis=1).drop(columns=["name", "seller_type", "fuel", "transmission", "owner", "seats"])

    # Scale
    X_test_sc = pd.DataFrame(data['scaler'].transform(X_test),
                             columns=X_test.columns)

    return X_test_sc


@app.get('/', summary='Root')
async def root():
    hello = ["Good day to you, this is FastAPI service with super cool",
             "Linear Regresseion model inside. It can predict car prices"]
    return hello[0] + '' + hello[1]


@app.post("/predict_item", summary='Get predicitions for one item')
def predict_item(item: Item) -> float:
    '''
    На вход получает json одного объекта.
    На выход отдает предсказание для данного объекта.

    input: json с описанием объекта
    return: файл с предсказаниями
    '''
    test_item = dict(item)
    df = pd.DataFrame.from_dict([test_item])

    return data['best_model'].predict(preprocessing(df))


@app.post("/predict_items", summary='Get predicitions for csv')
def predict_items(file: UploadFile):
    '''
    На вход получает файл csv, считывает его в датафрейм.
    Добавляет в датафрейм предсказания для каждого объекта
    отдельным столбцом.
    На выход отдает файл csv со столбцом предсказаний.

    input: загружаемый файл csv
    return: файл с предсказаниями
    '''

    content = file.file.read()
    buffer = BytesIO(content)
    df = pd.read_csv(buffer)
    buffer.close()
    file.close()

    output = df

    df['predict'] = pd.Series(data['best_model'].predict(preprocessing(df)))
    output['predict'] = df['predict']
    output.to_csv('predictions.csv', index=False)
    response = FileResponse(path='predictions.csv',
                            media_type='text/csv', filename='predictions.csv')

    return response
