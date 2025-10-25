# Iris Classifier Project

Простой проект по обучению модели `RandomForestClassifier` на наборе данных Iris с автоматическим CI/CD в Docker Hub.

---

## Описание

Проект включает:

1. **Препроцессинг данных** с юнит-тестами для проверки корректности.
2. **Обучение модели** `RandomForestClassifier`.
3. **Dockerfile** для контейнеризации модели.
4. **CI/CD workflow** для автоматического билда и пуша Docker-образа на Docker Hub.

---

## Структура проекта

└───iris_ml
    │   Dockerfile
    │   LICENSE
    |   README.md
    │   requirements.txt
    │
    ├───.github
    │   └───workflows
    │           ci_cd.yml
    │
    ├───models
    │
    │
    ├───src
    │       predict.py
    │       train.py
    │       utils.py
    │       __init__.py
    │
    └───tests
            test_utils.py
            __init__.py
