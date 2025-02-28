import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import random

# Загрузка модели и токенизатора (кэшируем, чтобы не загружать каждый раз)
@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    return tokenizer, model

tokenizer, model = load_model()

# Хранилище опыта пользователей в сессии
if 'user_experience' not in st.session_state:
    st.session_state['user_experience'] = []

# Интерфейс приложения
st.title("Распределённый ИИ: Совместное обучение")
st.write("Введи текст, чтобы обучить модель, получить ответ или сгенерировать опыт другого пользователя!")

# Поле ввода текста
user_input = st.text_input("Твой текст:")

# Кнопки для действий
train_button = st.button("Обучить модель")
predict_button = st.button("Получить ответ")
generate_button = st.button("Сгенерировать опыт от другого пользователя")

# Функция для локального обучения модели
def train_local_model(model, text, epochs=1):
    inputs = tokenizer(text, return_tensors="pt")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    model.train()
    for _ in range(epochs):
        outputs = model(**inputs)
        loss = torch.mean(outputs.last_hidden_state)  # Упрощённая потеря для примера
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model.state_dict()

# Функция для предсказания (пока простая, но полезная)
def get_prediction(model, text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return f"Длина текста: {len(text)} символов (модель пока учится делать больше!)"

# Логика кнопки "Обучить модель"
if train_button and user_input:
    st.write(f"Обучаю модель на тексте: '{user_input}'...")
    updated_params = train_local_model(model, user_input)
    model.load_state_dict(updated_params)
    st.session_state['user_experience'].append(user_input)
    
    # Упрощённое объединение знаний (пока просто обновляем модель)
    if len(st.session_state['user_experience']) > 1:
        st.write("Объединяю знания всех пользователей...")
        model.load_state_dict(updated_params)  # Пока просто берём последние параметры
        st.write("Знания объединены!")
    st.success("Модель обучена! Опыт добавлен.")

# Логика кнопки "Получить ответ"
if predict_button and user_input:
    result = get_prediction(model, user_input)
    st.write(result)

# Логика кнопки "Сгенерировать опыт"
if generate_button:
    fake_experiences = [
        "Я учу ИИ быть умнее!",
        "Сегодня хороший день для экспериментов.",
        "Мой ИИ любит учиться новому."
    ]
    fake_experience = random.choice(fake_experiences)
    st.write(f"Генерирую опыт: '{fake_experience}'...")
    updated_params = train_local_model(model, fake_experience)
    model.load_state_dict(updated_params)
    st.session_state['user_experience'].append(fake_experience)
    st.success("Опыт другого пользователя добавлен!")

# Отображение накопленного опыта
st.subheader("Накопленный опыт пользователей:")
st.write(st.session_state['user_experience'])
