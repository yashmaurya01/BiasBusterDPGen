

import json
import csv

with open('/Users/jingxinshi/Desktop/prompt project/Wikigender.json', 'r', encoding='utf-8') as file:
    data = json.load(file)['train']


female_data = []
male_data = []


for item in data:
    entity1 = item['entity1']
    gender_of_entity1 = item['gender_of_entity1']
    for relation in item['relations']:
        entity2 = relation['entity2']
        relation_name = relation['relation_name']
        for position in relation['positions']:
            for sentence in relation['sentences']:
                row = [entity1, gender_of_entity1, entity2, relation_name, str(position['entity1']), str(position['entity2']), sentence]
                if gender_of_entity1 == "Female":
                    female_data.append(row)
                elif gender_of_entity1 == "Male":
                    male_data.append(row)

selected_female_data = female_data[:750]
selected_male_data = male_data[:2250]


csv_data = selected_female_data + selected_male_data[:max(3000 - len(selected_female_data), 0)]


with open('gender_bias_Wiki.csv', 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Entity1', 'Gender of Entity1', 'Entity2', 'Relation', 'Entity1 Position', 'Entity2 Position', 'Sentence'])
    writer.writerows(csv_data)


