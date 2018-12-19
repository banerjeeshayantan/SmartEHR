from flask import Flask, render_template,request,url_for
from flask_bootstrap import Bootstrap 
import re
import pickle
import string
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from tabulate import tabulate
from nltk import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def index():
	return render_template('index.html')
def tokenizer_better(text):
		tokens=word_tokenize(text)
		return tokens	

def _uppercase_for_dict_keys(lower_dict):
    upper_dict = {}
    for k, v in lower_dict.items():
        if isinstance(v, dict):
            v = _uppercase_for_dict_keys(v)
        upper_dict[k.upper()] = v
    return upper_dict

def preprocess_textual(df):
		
    		df =re.sub('\n',' ',df)
    		df =re.sub('\r',' ',df)
    		df = re.sub('\d+', '',df)
    		df=re.sub('\[\*\*[^\]]*\*\*\]', '',df)
    		df=re.sub('<[^>]*>','',df)
    		df=df.lower()
    		df=re.sub(r'[^a-z0-9]+', ' ',df)
    		df=  re.sub(r"\b[a-zA-Z]\b", "", df)
    		return df
def extract_topn_from_vector(feature_names, sorted_items, topn=20):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)    		

@app.route('/analyse',methods=['POST'])
def analyse():
	if request.method == 'POST':
		rawtext = request.form['rawtext']
		blob=preprocess_textual(rawtext)
		received_tex2=blob
		transformer =CountVectorizer(max_features=20000,tokenizer=tokenizer_better,stop_words=stopwords.words('english'),decode_error="replace",vocabulary=pickle.load(open("feature123.pkl", "rb")))	
		trained=transformer.transform([preprocess_textual(rawtext)])
		log=pickle.load(open('logreg.pkl','rb'))
		y=log.predict(trained)
		if(y==1):
			op='There is a high chance that the  patient will return to the hospital within 30-days'
		else:
			op='Its very unlikely that the patient will be readmitted within 30 days'
	
		l=dict()
	#vect= TfidfVectorizer(max_features = 20000, tokenizer = tokenizer_better, stop_words=stopwords.words('english'),decode_error="replace", vocabulary=pickle.load(open('new_feature.pkl','rb')))
		labels=['4019','4280','42731','41401','5849','25000','2724','51881','5990','53081','2859','2449','486','2851','2762','496','99592','V5861','5070','0389','5859','40390','311','3051','412','2875','41071','2761']
		vectorizer=pickle.load(open('feature.pkl','rb'))
		for i in range(len(labels)):		
			v=pickle.load(open('feature'+labels[i]+'.pkl','rb'))
			l.update({labels[i]:(v.predict_proba(vectorizer.transform([preprocess_textual(rawtext)]))[0][0])})
		
		sor=sorted(l.items(), key=lambda x: x[1],reverse=True)[0:10]
		sort=dict(sor)
		trained=vectorizer.transform([preprocess_textual(rawtext)])
	#print(trained.shape)
		sorted_items=sort_coo(trained.tocoo())
		feature_names=vectorizer.get_feature_names()
		keywords=extract_topn_from_vector(feature_names,sorted_items,20)	
		k=keywords.keys()

	return render_template('index.html',keywords=_uppercase_for_dict_keys(keywords),diag_codes=sort.keys(),thirty=op)

if __name__ == '__main__':
	app.run(debug=True)
