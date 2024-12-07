# Crime Prediction System (ХАКАТОН project challenge #20)
Crime Prediction System — это проект, разработанный в рамках ХАКАТОН Project Challenge #20. Он предназначен для анализа данных о преступлениях и создания модели, предсказывающей вероятность преступления в зависимости от заданных параметров. Проект включает бэкенд-логику для обработки данных и предсказаний, а также фронтенд-интерфейс для взаимодействия с пользователем.


# Структура проекта
 **Project20.ipynb:**
Содержит обработку данных, обучение модели машинного обучения и анализ. Включает визуализации и метрики качества модели.

**app.py:**
Основной бэкенд-скрипт. Реализует API для взаимодействия с моделью предсказания и отправляет данные на фронтенд.

**frontend/:**
Директория с файлами интерфейса пользователя (HTML/CSS/JS). Отвечает за отображение данных и сбор пользовательского ввода.

**data/:**
Содержит наборы данных для анализа и обучения модели.

**models/:**
Хранит сохраненные модели машинного обучения.


## Файлы проекта

- **`app.py`**: Основной файл для запуска веб-приложения.
- **`Project20.ipynb`**: Jupyter Notebook для анализа данных и обучения моделей.
- **`crime_model.pkl`**: Сохранённая модель для предсказания.
- **`scaler.pkl`**: Сохранённый StandardScaler для обработки данных.
- **`crimedata.csv`**: Данные о преступности, используемые для обучения.
- **`index.html`**: HTML-файл для интерфейса веб-приложения.
- **`README.md`**: Документация проекта.

---

## Структура проекта

```plaintext
project-folder/
├── app.py                # Веб-приложение
├── Project20.ipynb       # Jupyter Notebook с кодом анализа и обучения
├── crime_model.pkl       # Сохранённая модель
├── scaler.pkl            # Сохранённый StandardScaler
├── crimedata.csv         # Данные для обучения
├── index.html            # Интерфейс веб-приложения
├── README.md             # Документация проекта


# Используемые технологии
**Python:** 
Основной язык программирования.

**Pandas, NumPy:** 
Для обработки данных.

**Scikit-learn:** 
Для обучения и оценки моделей машинного обучения.

**Matplotlib, Seaborn:** 
Для визуализации данных.

**Flask:** 
Для создания API и взаимодействия с фронтендом.

**HTML, CSS, JS:**
Для создания пользовательского интерфейса.


## Инструкция по запуску

1. **Клонируйте репозиторий**:
   ```bash
   git@github.com:Bublik-05/crime-prediction.git

2. **Установите зависимости**:
   ```bash
   pip install -r requirements.txt
3. **Запустите Jupyter Notebook (все функции по порядку)**
4. **Запустите app.py**
5. **Запустите index.html**



## Команда № 18
### Paзработка бэкенда: Зинетов Алихан Дарханович, Щудро Александр Александрович.
### Разработка фронтенда: Пернебек Абылай Абайұлы.




