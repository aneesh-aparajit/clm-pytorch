from glob import glob

import pandas as pd

txt_files = glob('../data/tamil/*.txt')
output_txt = '../data/ta_valid.txt'
output_csv = '../data/ta_valid.csv'

for file in txt_files:
    with open(file, 'r') as f:
        txt = f.read()
        with open(output_txt, 'a') as out:
            out.write(txt + "\n")

text = []
# id = []
for file in txt_files:
    with open(file, 'r') as f:
        txt = f.readlines()
        text.extend(txt)
        # id.extend(list(range(len(id), len(txt))))
df = pd.DataFrame({
    'id': range(len(text)),
    'text': text
})

print(df.head())
df.to_csv(output_csv, index=False)
