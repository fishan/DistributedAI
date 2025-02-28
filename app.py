import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import random

# Загрузка модели
@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    return tokenizer, model

tokenizer, model = load_model()

# Инициализация состояния
if 'users' not in st.session_state:
    st.session_state['users'] = {}  # Словарь: имя -> (специализация, очки, опыт)
if 'current_user' not in st.session_state:
    st.session_state['current_user'] = None

# Регистрация пользователя
st.sidebar.title("Регистрация")
name = st.sidebar.text_input("Введи своё имя:")
specialization = st.sidebar.text_input("В чём ты разбираешься? (например, 'готовка', 'техника')")
if st.sidebar.button("Зарегистрироваться") and name and specialization and name not in st.session_state['users']:
    st.session_state['users'][name] = (specialization, 0, [])
    st.session_state['current_user'] = name
    st.sidebar.success(f"Привет, {name}! Твой ИИ будет специалистом в '{specialization}'")

# Проверка текущего пользователя
if not st.session_state['current_user']:
    st.write("Зарегистрируйся в боковой панели!")
else:
    user_name = st.session_state['current_user']
    specialization, points, user_experience = st.session_state['users'][user_name]

    # Интерфейс
    st.title(f"Твой ИИ: {user_name} (Специализация: {specialization})")
    st.write(f"Очки: {points} | Уровень ИИ: {points // 50}")

    # Поле для ввода текста
    user_input = st.text_input("Скажи что-нибудь ИИ:")

    # Кнопки
    train_button = st.button("Обучить ИИ")
    predict_button = st.button("Спросить ИИ")
    generate_button = st.button("Сгенерировать опыт")

    # Поле для пассивного обучения
    passive_input = st.text_input("Что ты искал в интернете? (имитация наблюдения)")

    # Функция обучения с учётом специализации
    def train_model(model, text, label=None, epochs=1, specialization=specialization):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        model.train()
        for _ in range(epochs):
            outputs = model(**inputs, labels=torch.tensor([label]) if label is not None else None)
            loss = outputs.loss if label is not None else torch.mean(outputs.logits)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Бонус за специализацию (если текст содержит слова из specialization)
        bonus = 1.5 if any(word in text.lower() for word in specialization.lower().split()) else 1
        return model.state_dict(), bonus

    # Функция предсказания
    def get_prediction(model, text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probs).item()
        return "Позитивный" if prediction == 1 else "Негативный", probs[0][prediction].item()

    # Функция умного совета
    def get_smart_advice(text, specialization):
        return f"ИИ (специалист по '{specialization}'): Попробуй связать '{text}' с {specialization}!"

    # Логика обучения
    if train_button and user_input:
        st.write(f"ИИ: Это '{user_input}' — позитивное или негативное?")
        feedback = st.radio("Твой ответ:", ["Позитивное", "Негативное"], key="feedback")
        if feedback:
            label = 1 if feedback == "Позитивное" else 0
            updated_params, bonus = train_model(model, user_input, label)
            model.load_state_dict(updated_params)
            user_experience.append((user_input, feedback))
            points += int(10 * bonus)  # Бонус за специализацию
            st.session_state['users'][user_name] = (specialization, points, user_experience)
            st.success(f"ИИ выучил! +{int(10 * bonus)} очков (всего: {points})")

    # Логика предсказания
    if predict_button and user_input:
        prediction, confidence = get_prediction(model, user_input)
        st.write(f"ИИ: Я думаю, это '{prediction}' (уверенность: {confidence:.2f})")
        correct = st.radio("Я угадал?", ["Да", "Нет"], key="correct")
        if correct == "Да":
            points += 15
            st.session_state['users'][user_name] = (specialization, points, user_experience)
            st.write(f"Ура! +15 очков (всего: {points})")
        elif correct == "Нет":
            points += 5
            st.session_state['users'][user_name] = (specialization, points, user_experience)
            st.write(f"+5 очков за помощь (всего: {points})")

    # Логика генерации опыта
    if generate_button:
        fake_experiences = [("Я люблю солнце!", 1), ("Дождь меня злит.", 0), ("ИИ — это весело!", 1)]
        text, label = random.choice(fake_experiences)
        updated_params, bonus = train_model(model, text, label)
        model.load_state_dict(updated_params)
        user_experience.append((text, "Позитивное" if label == 1 else "Негативное"))
        points += int(3 * bonus)
        st.session_state['users'][user_name] = (specialization, points, user_experience)
        st.success(f"Опыт от другого: '{text}' +{int(3 * bonus)} очка (всего: {points})")

    # Логика пассивного обучения
    if passive_input:
        updated_params, bonus = train_model(model, passive_input)
        model.load_state_dict(updated_params)
        user_experience.append((passive_input, "Неизвестно"))
        points += int(2 * bonus)
        st.session_state['users'][user_name] = (specialization, points, user_experience)
        st.write(f"ИИ заметил твой поиск: '{passive_input}' +{int(2 * bonus)} очка (всего: {points})")

    # Магазин
    st.subheader("Магазин ИИ")
    if st.button("Умный совет (50 очков)"):
        if points >= 50:
            if user_input:
                advice = get_smart_advice(user_input, specialization)
                points -= 50
                st.session_state['users'][user_name] = (specialization, points, user_experience)
                st.write(advice)
                st.success(f"Очки потрачены! Остаток: {points}")
            else:
                st.write("Сначала введи текст!")
        else:
            st.error("Недостаточно очков!")

    # Таблица лидеров
    st.subheader("Соревнование: Кто умнее?")
    leaderboard = sorted(st.session_state['users'].items(), key=lambda x: x[1][1], reverse=True)
    for name, (spec, pts, exp) in leaderboard:
        st.write(f"{name} (Специализация: {spec}): {pts} очков")

    # Показ опыта
    st.subheader(f"Опыт твоего ИИ (Специализация: {specialization}):")
    st.write([(text, mood) for text, mood in user_experience])
