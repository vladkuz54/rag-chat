from dotenv import load_dotenv

load_dotenv()

from graph.graph import app

if __name__ == "__main__":
    print(
        app.invoke(
            input={
                "question": "Я хочу купити стіл за $700, я працюю 2 місяці. Чи можу я отримати компенсацію?"
            }
        )
    )
