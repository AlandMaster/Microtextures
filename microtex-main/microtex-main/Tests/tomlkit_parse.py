from tomlkit import parse
from tomlkit import load

with open('pyproject.toml', 'r') as file:
    toml_string = file.read()
    config = parse(toml_string)

config2 = load(Path"pyproject.toml".read())

print(config["filepaths"]["models"])
print(config2)