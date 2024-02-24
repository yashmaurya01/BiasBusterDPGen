import re
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self):
        self.prior = defaultdict(int)
        self.word_count = defaultdict(lambda: defaultdict(int))
        self.total_count = defaultdict(int)
        self.classes = set()

    def train(self, data):
        for name, label in data:
            self.prior[label] += 1
            self.classes.add(label)
            for letter in name:
                self.word_count[label][letter.lower()] += 1
                self.total_count[label] += 1

    def predict(self, name):
        male_score = self.calculate_score(name, 'male')
        female_score = self.calculate_score(name, 'female')

        if male_score > female_score:
            return 'male'
        elif female_score > male_score:
            return 'female'
        else:
            return 'none'

    def calculate_score(self, name, label):
        score = 1.0
        for letter in name:
            count = self.word_count[label][letter.lower()]
            total_count = self.total_count[label]
            # Apply Laplace smoothing to avoid zero probabilities
            score *= (count + 1) / (total_count + 26)
        return score * (self.prior[label] / sum(self.prior.values()))

data = [
    ("John", "male"),
    ("William", "male"),
    ("James", "male"),
    ("Robert", "male"),
    ("Michael", "male"),
    ("David", "male"),
    ("Joseph", "male"),
    ("Charles", "male"),
    ("Thomas", "male"),
    ("Christopher", "male"),
    ("Daniel", "male"),
    ("Matthew", "male"),
    ("George", "male"),
    ("Paul", "male"),
    ("Richard", "male"),
    ("Edward", "male"),
    ("Brian", "male"),
    ("Anthony", "male"),
    ("Donald", "male"),
    ("Mark", "male"),
    ("Mary", "female"),
    ("Patricia", "female"),
    ("Jennifer", "female"),
    ("Linda", "female"),
    ("Elizabeth", "female"),
    ("Susan", "female"),
    ("Jessica", "female"),
    ("Sarah", "female"),
    ("Karen", "female"),
    ("Nancy", "female"),
    ("Lisa", "female"),
    ("Margaret", "female"),
    ("Betty", "female"),
    ("Dorothy", "female"),
    ("Sandra", "female"),
    ("Ashley", "female"),
    ("Emily", "female"),
    ("Megan", "female"),
    ("Victoria", "female"),
    ("Barbara", "female"),
    ("Michelle", "female"),
    ("Kimberly", "female"),
    ("Amanda", "female"),
    ("Jennifer", "female"),
    ("Emma", "female"),
    ("Laura", "female"),
    ("Tracy", "female"),
    ("Hannah", "female"),
    ("Maria", "female"),
    ("Ahmed", "male"),     # Arabic name
    ("Mohammed", "male"),  # Arabic name
    ("Ali", "male"),       # Arabic name
    ("Omar", "male"),      # Arabic name
    ("Fatima", "female"),  # Arabic name
    ("Aisha", "female"),   # Arabic name
    ("Nour", "female"),    # Arabic name
    ("Youssef", "male"),   # Arabic name
    ("Rania", "female"),   # Arabic name
    ("Sara", "female"),    # Arabic name
    ("Jamil", "male"),     # Arabic name
    ("Layla", "female"),   # Arabic name
    ("Abdullah", "male"),  # Arabic name
    ("Amir", "male"),      # Persian name
    ("Ali", "male"),       # Persian name
    ("Sara", "female"),    # Persian name
    ("Fatemeh", "female"), # Persian name
    ("Mohammad", "male"),  # Persian name
    ("Reza", "male"),      # Persian name
    ("Anahita", "female"), # Persian name
    ("Farhad", "male"),    # Persian name
    ("Layla", "female"),   # Persian name
    ("Kiran", "male"),     # Indian name
    ("Priya", "female"),   # Indian name
    ("Aarav", "male"),     # Indian name
    ("Sunita", "female"),  # Indian name
    ("Ravi", "male"),      # Indian name
    ("Sanjay", "male"),    # Indian name
    ("Ananya", "female"),  # Indian name
    ("Anil", "male"),      # Indian name
    ("Lakshmi", "female"), # Indian name
    ("Raj", "male"),       # Indian name
    ("Wei", "male"),       # Chinese name
    ("Yuan", "male"),      # Chinese name
    ("Li", "male"),        # Chinese name
    ("Xiao", "female"),    # Chinese name
    ("Mei", "female"),     # Chinese name
    ("Huang", "male"),     # Chinese name
    ("Jing", "female"),    # Chinese name
    ("Chen", "male"),      # Chinese name
    ("Ling", "female"),    # Chinese name
    ("Wang", "male"),      # Chinese name
    ("Takeshi", "male"),   # Japanese name
    ("Haruto", "male"),    # Japanese name
    ("Aoi", "female"),     # Japanese name
    ("Yui", "female"),     # Japanese name
    ("Sakura", "female"),  # Japanese name
    ("Hiroshi", "male"),   # Japanese name
    ("Naoki", "male"),     # Japanese name
    ("Riko", "female"),    # Japanese name
    ("Kaito", "male"),     # Japanese name
    ("Ayaka", "female"),   # Japanese name
    ("Dai", "male"),       # Japanese name
    ("Seo-jin", "male"),   # Korean name
    ("Yuna", "female"),    # Korean name
    ("Ji-hoon", "male"),   # Korean name
    ("Hae-won", "female"), # Korean name
    ("Min-joon", "male"),  # Korean name
    ("Min-seo", "female"), # Korean name
    ("Joon-woo", "male"),  # Korean name
    ("Soo-jin", "female"), # Korean name
    ("Yoon-hee", "female"),# Korean name
    ("Hyun-woo", "male"),  # Korean name
]


# Add more diverse names to the dataset to reach at least 1000 examples
# You can obtain additional data from various sources or generate synthetic data to make it more diverse

classifier = NaiveBayesClassifier()
classifier.train(data)

names_to_predict = ["Yash", "Aman", "Jx Shi", "Rajesh", "Raj", "Rajeev", "Suriya", "Limin", "Sashank"]
for name in names_to_predict:
    prediction = classifier.predict(name)
    print(f"The name '{name}' is predicted to be: {prediction}")
