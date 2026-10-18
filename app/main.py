from transformers import pipeline
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
token = os.getenv("TOKEN")

if token:
    os.environ["HF_TOKEN"] = token


is_wrote_reviews = False


def main():
    global is_wrote_reviews
    analyzer = None
    reviews = get_reviews()

    while True:
        if analyzer is None:
            analyzer = load_analyzer()

        try:
            menu = int(input(
                "===== Review Analyzer =====\n"
                "1. Аналіз одного відгуку\n"
                "2. аналіз усього списку\n"
                "0. Вихід\n"
            ))
        except ValueError:
            print("Введіть номер пункту.")
            continue

        match menu:
            case 0:
                break
            case 1:
                analyze_one_review(analyzer)
            case 2:
                analyze_reviews(analyzer(reviews), reviews)
                is_wrote_reviews = True
            case _:
                print("Немає такого пункту.")

        input("Press Enter to continue: ")


def get_reviews():
    with open('reviews.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return lines


def write_reviews(review, mark, result):
    with open("results.txt", "a", encoding='utf-8') as f:
        f.write(
            f"\nВідгук: {review}\n"
            f"Оцінка: {mark}\n"
            f"Впевненість моделі: {result['score']:.2%}\n"
            f"==============================\n\n"
        )


def load_analyzer():
    print("Завантаження моделі. Перший запуск завантажує її з Hugging Face...")
    return pipeline("text-classification", model="federicopascual/finetuning-sentiment-model-3000-samples", max_length=512)


def analyze_reviews(results, reviews):
    for review, r in zip(reviews, results):
        mark = None

        if r["label"] == "LABEL_0":
            mark = "негативний"
        elif r["label"] == "LABEL_1":
            mark = "позитивний"

        if mark is not None:
            if not is_wrote_reviews:
                write_reviews(review, mark, r)
            print(
                f"Відгук: {review.strip()}\n"
                f"Оцінка: {mark}\n"
                f"Впевненість моделі: {r['score']:.2%}\n"
                f"=============================="
            )
        else:
            print(f"Помилка з відгуком: {review.strip()}")


def analyze_one_review(analyzer):
    review = input("Введіть відгук: ").strip()

    if not review:
        print("Ви не ввели відгук.")
        return

    result = analyzer(review)[0]

    if result["label"] == "LABEL_0":
        mark = "негативний"
    elif result["label"] == "LABEL_1":
        mark = "позитивний"
    else:
        mark = "невідомий"

    write_reviews(review, mark, result)
    print(
        f"\nВідгук: {review}\n"
        f"Оцінка: {mark}\n"
        f"Впевненість моделі: {result['score']:.2%}\n"
        f"=============================="
    )


if __name__ == "__main__":
    main()