# Установка необходимых библиотек
pip install transformers torch

# Импорт библиотек
import torch
from transformers import DistilBertTokenizer, DistilBertModel

# Загрузка токенизатора и модели DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Проверка, что модель работает
print("Модель загружена успешно!")

# Симуляция данных от 3 "пользователей"
user_data = [
    "Я люблю готовить суп!",
    "Мой кот спит весь день.",
    "Как прекрасен этот мир!"
]

# Функция для локального обучения
def train_local_model(model, text, epochs=1):
    inputs = tokenizer(text, return_tensors="pt")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # Оптимизатор

    for _ in range(epochs):
        model.train()  # Режим обучения
        outputs = model(**inputs)
        loss = torch.mean(outputs.last_hidden_state)  # Простая "потеря" для примера
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model.state_dict()  # Возвращаем обновлённые параметры

# Локальное обучение для каждого "пользователя"
user_models = []
for i, text in enumerate(user_data):
    print(f"Обучение для пользователя {i+1}...")
    updated_params = train_local_model(model, text)
    user_models.append(updated_params)

print("Локальное обучение завершено!")
