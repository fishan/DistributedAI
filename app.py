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

# Сохранение и загрузка данных
DATA_FILE = "users_data.json"

def save_data():
    with open(DATA_FILE, "w") as f:
        json.dump(st.session_state['users'], f, ensure_ascii=False)

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {}

# Инициализация состояния
if 'users' not in st.session_state:
    st.session_state['users'] = load_data()
if 'current_user' not in st.session_state:
    st.session_state['current_user'] = None

# Приветствие и описание
st.title("Твой ИИ-питомец ждёт тебя!")
st.write("""
Это твой уникальный ИИ-питомец! Он учится у тебя, отражает твои интересы и растёт вместе с тобой. 
- Расскажи о своих интересах (например, 'кулинария, шахматы'), и питомец станет в этом экспертом.
- Общайся с ним, учи его и выполняй задания. Он запомнит только последние 20 сообщений, чтобы не путаться!
- Чем дольше вы вместе, тем ценнее и умнее он становится.
""")

# Регистрация пользователя
st.sidebar.title("Регистрация")
name = st.sidebar.text_input("Введи своё имя:")
interests = st.sidebar.text_input("Твои интересы, навыки или опыт (например, 'кулинария, шахматы')")
if st.sidebar.button("Зарегистрироваться") and name and interests and name not in st.session_state['users']:
    st.session_state['users'][name] = {
        "interests": interests,
        "points": 0,
        "experience": [],
        "dialogue": [],  # Последние 20 сообщений
        "personality": random.choice(["Любопытный", "Оптимист", "Шутник"])
    }
    st.session_state['current_user'] = name
    save_data()
    st.sidebar.success(f"Привет, {name}! Твой питомец готов учиться '{interests}'")

# Проверка текущего пользователя
if not st.session_state['current_user']:
    st.write("Зарегистрируйся в боковой панели!")
else:
    user_name = st.session_state['current_user']
    user_data = st.session_state['users'][user_name]
    interests = user_data["interests"]
    points = user_data["points"]
    experience = user_data["experience"]
    dialogue = user_data["dialogue"]
    personality = user_data["personality"]

    # Интерфейс
    st.header(f"Твой ИИ-питомец: {user_name}")
    st.write(f"Интересы: {interests} | Очки: {points} | Уровень: {points // 50} | Характер: {personality}")

    # Поле для ввода текста и темы
    user_input = st.text_input("Скажи что-нибудь питомцу:")
    topic = st.text_input("О какой теме этот текст? (например, 'кулинария')")

    # Кнопки
    train_button = st.button("Обучить питомца")
    predict_button = st.button("Спросить питомца")
    generate_button = st.button("Сгенерировать опыт")

    # Проверка соответствия теме
    def check_topic_relevance(text, topic, interests):
        interest_list = [i.strip().lower() for i in interests.split(",")]
        if not topic or topic.lower() not in interest_list:
            return 0.5
        related_words = {
            "кулинария": ["готов", "еда", "рецепт", "жарить", "печь", "суп", "вкусно"],
            "шахматы": ["ход", "фигура", "партия", "шах", "мат", "доска"],
            "коты": ["кот", "кошк", "мяу", "лап", "шерсть"]
        }.get(topic.lower(), [])
        if any(word in text.lower() for word in related_words):
            return 1.5
        absurd_combinations = ["готовлю носки", "ем шахматы"]
        if any(absurd in text.lower() for absurd in absurd_combinations):
            return 0
        return 1

    # Функция обучения
    def train_model(model, text, label=None, epochs=1, topic=None, interests=interests):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        model.train()
        for _ in range(epochs):
            outputs = model(**inputs, labels=torch.tensor([label]) if label is not None else None)
            loss = outputs.loss if label is not None else torch.mean(outputs.logits)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        bonus = check_topic_relevance(text, topic, interests)
        return model.state_dict(), bonus

    # Функция предсказания с учётом личности
    def get_prediction(model, text, personality):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probs).item()
        base_response = "Позитивный" if prediction == 1 else "Негативный"
        if personality == "Любопытный":
            return f"{base_response}. А почему ты так думаешь?"
        elif personality == "Оптимист":
            return f"{base_response}. Всё будет хорошо!"
        else:  # Шутник
            return f"{base_response}. Не грусти, я не кусаюсь... пока!"

    # Обновление диалога
    def update_dialogue(user_msg, pet_msg):
        dialogue.append((user_msg, pet_msg))
        if len(dialogue) > 20:
            dialogue.pop(0)
        user_data["dialogue"] = dialogue
        save_data()

    # Логика обучения
    if train_button and user_input:
        st.write(f"Питомец: Это '{user_input}' — позитивное или негативное?")
        feedback = st.radio("Твой ответ:", ["Позитивное", "Негативное"], key="feedback")
        if feedback:
            label = 1 if feedback == "Позитивное" else 0
            updated_params, bonus = train_model(model, user_input, label, topic=topic)
            model.load_state_dict(updated_params)
            experience.append((user_input, feedback, topic))
            points += int(10 * bonus)
            pet_response = f"Я выучил! Это {feedback}."
            update_dialogue(user_input, pet_response)
            user_data["points"] = points
            user_data["experience"] = experience
            st.session_state['users'][user_name] = user_data
            save_data()
            st.success(f"Питомец: {pet_response} +{int(10 * bonus)} очков (всего: {points})")

    # Логика предсказания
    if predict_button and user_input:
        prediction, confidence = get_prediction(model, user_input, personality)
        st.write(f"Питомец: Я думаю, это '{prediction}' (уверенность: {confidence:.2f})")
        update_dialogue(user_input, prediction)
        correct = st.radio("Я угадал?", ["Да", "Нет"], key="correct")
        if correct == "Да":
            points += 15
            user_data["points"] = points
            st.session_state['users'][user_name] = user_data
            save_data()
            st.write(f"Питомец: Ура! +15 очков (всего: {points})")
        elif correct == "Нет":
            points += 5
            user_data["points"] = points
            st.session_state['users'][user_name] = user_data
            save_data()
            st.write(f"Питомец: Поправь меня! +5 очков (всего: {points})")

    # Логика генерации опыта
    if generate_button:
        fake_experiences = [("Я люблю солнце!", 1, "погода"), ("Дождь меня злит.", 0, "погода"), ("ИИ — это весело!", 1, "технологии")]
        text, label, fake_topic = random.choice(fake_experiences)
        updated_params, bonus = train_model(model, text, label, topic=fake_topic)
        model.load_state_dict(updated_params)
        experience.append((text, "Позитивное" if label == 1 else "Негативное", fake_topic))
        points += int(3 * bonus)
        pet_response = f"Кто-то сказал '{text}'. Я выучил!"
        update_dialogue("Генерация опыта", pet_response)
        user_data["points"] = points
        user_data["experience"] = experience
        st.session_state['users'][user_name] = user_data
        save_data()
        st.success(f"Питомец: {pet_response} +{int(3 * bonus)} очка (всего: {points})")

    # Таблица лидеров
    st.subheader("Соревнование: Чей питомец умнее?")
    leaderboard = sorted(st.session_state['users'].items(), key=lambda x: x[1]["points"], reverse=True)
    for name, data in leaderboard:
        st.write(f"{name} (Интересы: {data['interests']}, Характер: {data['personality']}): {data['points']} очков")

    # Показ текущего диалога
    st.subheader("Последние 20 сообщений с питомцем:")
    for user_msg, pet_msg in dialogue:
        st.write(f"Ты: {user_msg} | Питомец: {pet_msg}")

    # Показ опыта
    st.subheader(f"Весь опыт твоего питомца:")
    st.write([(text, mood, topic) for text, mood, topic in experience])
