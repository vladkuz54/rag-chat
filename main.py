from dotenv import load_dotenv

load_dotenv()

from graph.graph import app

if __name__ == "__main__":
    print(
        app.invoke(
            input={
                "question": "Я втратив доступ до аккаунту через підозрілий вхід. Мені треба подзвонити за українським номером підтримки 0-800-AI-HELP, щоб розблокувати його?"
            }
        )
    )
