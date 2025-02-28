import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import random
import json
import os

# Загрузка модели
@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    return tokenizer, model

tokenizer, model = load_model()

# Каталог для данных пользователей
DATA_DIR = "user_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Сохранение и загрузка данных пользователя
def save_user_data(username, data):
    with open(os.path.join(DATA_DIR, f"{username}.json"), "w") as f:
        json.dump(data, f, ensure_ascii=False)

def load_user_data(username):
    file_path = os.path.join(DATA_DIR, f"{username}.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return None

# Инициализация состояния
if 'users' not in st.session_state:
    st.session_state['users'] = {}
if 'current_user' not in st.session_state:
    st.session_state['current_user'] = None

# Регистрация и логин
st.sidebar.title("Вход / Регистрация")
action = st.sidebar.radio("Выбери действие:", ["Войти", "Зарегистрироваться"])
username = st.sidebar.text_input("Имя пользователя:")
password = st.sidebar.text_input("Пароль:", type="password")

if action == "Зарегистрироваться" and st.sidebar.button("Зарегистрироваться"):
    if username and password and username not in st.session_state['users']:
        pet_name = st.sidebar.text_input("Имя твоего питомца:")
        interests = st.sidebar.text_input("Твои интересы (например, 'кулинария, шахматы')")
        if pet_name and interests:
            st.session_state['users'][username] = password
            user_data = {
                "pet_name": pet_name,
                "interests": interests,
                "points": 0,
                "experience": [],
                "dialogue": [],
                "personality": random.choice(["Любопытный", "Оптимист", "Шутник"])
            }
            save_user_data(username, user_data)
            st.session_state['current_user'] = username
            st.sidebar.success(f"Питомец {pet_name} для {username} создан!")
        else:
            st.sidebar.error("Укажи имя питомца и интересы!")
    elif username in st.session_state['users']:
        st.sidebar.error("Пользователь уже существует!")
    else:
        st.sidebar.error("Введи имя и пароль!")

if action == "Войти" and st.sidebar.button("Войти"):
    if username in st.session_state['users'] and st.session_state['users'][username] == password:
        st.session_state['current_user'] = username
        st.sidebar.success(f"Добро пожаловать, {username}!")
    else:
        st.sidebar.error("Неверное имя или пароль!")

# Главная страница
st.title("Твой ИИ-питомец — часть децентрализованного разума!")
st.write("""
Это твой уникальный ИИ-питомец. Он живёт с тобой, учится у тебя и становится твоим отражением. 
- Дай ему имя и общайся с ним — он будет расти и помогать тебе.
- Он может использовать опыт других питомцев для быстрых ответов, а ты получишь бонусы, если твой питомец помогает другим!
- Даже без связи он всегда с тобой.
""")

# Проверка текущего пользователя
if not st.session_state['current_user']:
    st.write("Войди или зарегистрируйся в боковой панели!")
else:
    user_name = st.session_state['current_user']
    user_data = load_user_data(user_name)
    if not user_data:
        st.error("Ошибка загрузки данных питомца!")
        st.stop()
    
    pet_name = user_data["pet_name"]
    interests = user_data["interests"]
    points = user_data["points"]
    experience = user_data["experience"]
    dialogue = user_data["dialogue"]
    personality = user_data["personality"]

    # Интерфейс
    st.header(f"Твой питомец: {pet_name} (Хозяин: {user_name})")
    st.write(f"Интересы: {interests} | Очки: {points} | Уровень: {points // 50} | Характер: {personality}")

    # Диалог
    user_input = st.text_input(f"Поговори с {pet_name}:")

    # Функция обучения
    def train_model(model, text, label=None, epochs=1):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        model.train()
        for _ in range(epochs):
            outputs = model(**inputs, labels=torch.tensor([label]) if label is not None else None)
            loss = outputs.loss if label is not None else torch.mean(outputs.logits)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return model.state_dict()

    # Функция предсказания с учётом личности и настроения
    def get_prediction(model, text, personality, pet_name):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probs).item()
        base_response = "Позитивный" if prediction == 1 else "Негативный"
        # Если текст негативный, добавляем поддержку
        if prediction == 0:
            if personality == "Любопытный":
                return f"{pet_name}: Кажется, ты грустишь. Что случилось? Я с тобой!"
            elif personality == "Оптимист":
                return f"{pet_name}: Не грусти, всё наладится! Чем могу помочь?"
            else:  # Шутник
                return f"{pet_name}: Ой, это негатив? Давай я пошучу: почему грусть убежала? Потому что я пришёл!"
        else:
            if personality == "Любопытный":
                return f"{pet_name}: {base_response}. Почему ты так думаешь?"
            elif personality == "Оптимист":
                return f"{pet_name}: {base_response}. Отлично, рад за тебя!"
            else:  # Шутник
                return f"{pet_name}: {base_response}. Я всегда знал, что ты весёлый!"

    # Обновление диалога
    def update_dialogue(user_msg, pet_msg):
        dialogue.append((user_msg, pet_msg))
        if len(dialogue) > 20:
            dialogue.pop(0)
        user_data["dialogue"] = dialogue
        save_user_data(user_name, user_data)

    # Симуляция децентрализации
    def simulate_decentralized_help(user_name, pet_name):
        if random.random() < 0.3 and len(st.session_state['users']) > 1:  # 30% шанс использовать "другого"
            other_users = [u for u in st.session_state['users'] if u != user_name]
            helper = random.choice(other_users)
            helper_data = load_user_data(helper)
            helper_data["points"] += 5
            save_user_data(helper, helper_data)
            return f"{pet_name}: Я спросил у питомца {helper}. Он помог! ({helper} получил +5 очков)"
        return f"{pet_name}: Я ответил сам!"

    # Обработка ввода
    if user_input:
        # Проверка на урок
        if "— это" in user_input:
            term, meaning = user_input.split("— это", 1)
            term, meaning = term.strip(), meaning.strip()
            experience.append((term, meaning, "Урок"))
            points += 10
            pet_response = f"{pet_name}: Запомнил: '{term}' — это '{meaning}'."
            update_dialogue(user_input, pet_response)
            user_data["points"] = points
            user_data["experience"] = experience
            save_user_data(user_name, user_data)
            st.write(f"Питомец: {pet_response} +10 очков (всего: {points})")
        else:
            # Диалог с обучением
            prediction, confidence = get_prediction(model, user_input, personality, pet_name)
            decentralized_msg = simulate_decentralized_help(user_name, pet_name)
            st.write(f"Питомец: {prediction}")
            st.write(decentralized_msg)
            feedback = st.radio(f"{pet_name}: Я угадал настроение?", ["Да", "Нет"], key="feedback")
            if feedback:
                label = 1 if "Позитивный" in prediction and feedback == "Да" else 0
                updated_params = train_model(model, user_input, label)
                model.load_state_dict(updated_params)
                pet_response = f"{pet_name}: Спасибо, я учусь!" if feedback == "Да" else f"{pet_name}: Поправил себя, спасибо!"
                points += 5 if feedback == "Да" else 10
                update_dialogue(user_input, pet_response)
                user_data["points"] = points
                user_data["experience"] = experience + [(user_input, prediction.split('.')[0], "Диалог")]
                save_user_data(user_name, user_data)
                st.success(f"Питомец: {pet_response} +{5 if feedback == 'Да' else 10} очков (всего: {points})")

    # Просмотр диалога
    st.subheader(f"Последние 20 сообщений с {pet_name}:")
    for user_msg, pet_msg in dialogue:
        st.write(f"Ты: {user_msg} | {pet_name}: {pet_msg}")

    # Просмотр опыта
    st.subheader(f"Опыт {pet_name}:")
    st.write([(text, mood, topic) for text, mood, topic in experience])

    # Таблица лидеров
    st.subheader("Соревнование:")
    leaderboard = sorted([(name, load_user_data(name)["points"]) for name in st.session_state['users']], key=lambda x: x[1], reverse=True)
    for name, pts in leaderboard:
        data = load_user_data(name)
        st.write(f"{name} (Питомец: {data['pet_name']}, Интересы: {data['interests']}, Характер: {data['personality']}): {pts} очков")
