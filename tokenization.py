import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
print(word_tokenize("Don't ne fooled by the dark sounding name,"
                    "Mr. Jone's Orphanage is as cheery goes for a pastry shop"))