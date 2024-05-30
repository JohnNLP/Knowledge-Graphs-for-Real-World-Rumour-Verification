from allennlp.predictors.predictor import Predictor
import pickle

predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz")

with open("1_original_data.pkl", 'rb') as f:
    all_data = pickle.load(f)


for v, i in enumerate(all_data):
    print(v,"/",len(all_data))
    evidence = all_data[i]["articles"]
    disambiguated_articles = {}
    for article in evidence:
        print(evidence[article])
        disambiguated_articles[article] = {}

        if "title" in evidence[article]:
            disambiguated_articles[article]["title"] = evidence[article]["title"] #title is not coref'd, has chance of directly matching hearsay on Twitter etc

        if "paras" in evidence[article]:
            disambiguated_articles[article]["paras"] = []
            for paragraph in evidence[article]["paras"]:
                try:
                    if len(paragraph) <= 20:
                        continue
                    if len(paragraph) > 400:
                        disambiguated_articles[article]["paras"].append(paragraph)
                        continue
                    predictor.predict(document=paragraph)
                    prediction = predictor.coref_resolved(paragraph)
                    disambiguated_articles[article]["paras"].append(prediction)
                except:
                    print("Error")

        print(disambiguated_articles[article])
    all_data[i]["articles"] = disambiguated_articles


with open('2_disambiguated_data.pkl', 'wb') as f:
    pickle.dump(all_data, f, protocol=pickle.HIGHEST_PROTOCOL)

#if this takes too long, consider skipping the step





