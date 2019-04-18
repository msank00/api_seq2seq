import yaml

with open("../config.yaml", 'r') as stream:
    config = yaml.load(stream)


print(config['directory']['data'])
print(config['directory']['model'])
print(config['directory']['output'])
