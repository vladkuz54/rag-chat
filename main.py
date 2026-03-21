from dotenv import load_dotenv
load_dotenv()

from pprint import pprint
from workflow import app

inputs = {
    "question": "some question",
    "counter": 0,
    "transform_counter": 0,
}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Node '{key}':")
    pprint("\n---\n")


pprint(value.get("generation"))
pprint(f"Total generations: {value.get('counter', 0)}")
pprint(f"Total transforms: {value.get('transform_counter', 0)}")