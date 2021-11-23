from utils import load_yml
import pprint

d = load_yml("20211013T133839-1/Trainer.yml")
pprint.pprint(d["params"])
